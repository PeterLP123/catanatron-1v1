"""Policy-guided same-turn search with an explicit hidden-information boundary."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from catanatron.features import (
    build_production_features,
    create_sample,
    reachability_features,
)
from catanatron.gym.envs.action_space import to_action_space
from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    read_model_schema,
    validate_model_schema,
)
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT, ActionType
from catanatron.models.player import Color, Player
from catanatron.players.checkpoint_manifest import verify_checkpoint
from catanatron.players.learned import (
    TorchBcCheckpointPlayer,
    _preserve_inference_loader_rng,
)
from catanatron.players.leaf_evaluation import leaf_win_probability
from catanatron.players.value import DEFAULT_WEIGHTS, value_production
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_num_dev_cards,
    player_num_resource_cards,
)

if TYPE_CHECKING:
    from catanatron.game import Game
    from catanatron.models.enums import Action


# Every expanded successor is determined by public state plus the acting player's
# own cards. Chance draws and transfers from an opponent's hidden hand stay out of
# the tree. If any legal root action is outside this set, the retained policy owns
# the whole decision so search cannot bias against an unmodelled action.
VISIBLE_DETERMINISTIC_ACTIONS = frozenset(
    {
        ActionType.END_TURN,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_ROAD,
        ActionType.BUILD_CITY,
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.MARITIME_TRADE,
        ActionType.DISCARD_RESOURCE,
    }
)
FORBIDDEN_SEARCH_ACTIONS = frozenset(
    {
        ActionType.ROLL,
        ActionType.BUY_DEVELOPMENT_CARD,
        ActionType.MOVE_ROBBER,
        ActionType.PLAY_MONOPOLY,
    }
)
_PUBLIC_PRODUCTION_FEATURES = build_production_features(True)


def public_f_position_value(game: "Game", color: Color) -> float:
    """F positional score with no dependence on hidden resource composition.

    Resource and development-card *counts* are public in physical play, but the
    opponent's resource mix is not. This deliberately omits F's hand-synergy
    feature for every player while retaining its public board and count terms.
    """

    params = DEFAULT_WEIGHTS
    production_sample = _PUBLIC_PRODUCTION_FEATURES(game, color)
    production = value_production(production_sample, "P0")
    enemy_production = value_production(production_sample, "P1", False)

    reachability_sample = reachability_features(game, color, 2)
    reachable_zero = sum(
        reachability_sample[f"P0_0_ROAD_REACHABLE_{resource}"] for resource in RESOURCES
    )
    reachable_one = sum(
        reachability_sample[f"P0_1_ROAD_REACHABLE_{resource}"] for resource in RESOURCES
    )

    num_in_hand = player_num_resource_cards(game.state, color)
    discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0
    buildings = game.state.buildings_by_color[color]
    owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
    owned_tiles = set()
    for node in owned_nodes:
        owned_tiles.update(game.state.board.map.adjacent_tiles[node])
    num_buildable_nodes = len(game.state.board.buildable_node_ids(color))
    longest_road_length = get_longest_road_length(game.state, color)
    longest_road_factor = params["longest_road"] if num_buildable_nodes == 0 else 0.1

    return float(
        production * params["production"]
        + enemy_production * params["enemy_production"]
        + reachable_zero * params["reachable_production_0"]
        + reachable_one * params["reachable_production_1"]
        + num_buildable_nodes * params["buildable_nodes"]
        + len(owned_tiles) * params["num_tiles"]
        + num_in_hand * params["hand_resources"]
        + discard_penalty
        + longest_road_length * longest_road_factor
        + player_num_dev_cards(game.state, color) * params["hand_devs"]
        + get_played_dev_cards(game.state, color, "KNIGHT") * params["army_size"]
    )


def public_f_leaf_value(game: "Game", color: Color) -> float:
    """Public/own-information positional leaf value in ``[0, 1]``."""

    return leaf_win_probability(game, color, public_f_position_value)


class _PuctNode:
    def __init__(
        self,
        game: "Game",
        *,
        parent: "_PuctNode | None" = None,
        action: "Action | None" = None,
        action_index: int = -1,
        prior: float = 1.0,
    ) -> None:
        self.game = game
        self.parent = parent
        self.action = action
        self.action_index = int(action_index)
        self.prior = float(prior)
        self.children: list[_PuctNode] = []
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.5


class VisibleSameTurnPuctPlayer(Player):
    """Frozen policy/critic PUCT that never searches hidden-state transitions."""

    MANIFEST_KIND = "visible_same_turn_puct"
    VISIBLE_ACTION_TYPES = VISIBLE_DETERMINISTIC_ACTIONS
    SEARCH_ACTION_TYPES = VISIBLE_DETERMINISTIC_ACTIONS
    FORBIDDEN_ACTION_TYPES = FORBIDDEN_SEARCH_ACTIONS

    def __init__(self, color: Color, manifest: Union[str, Path]):
        super().__init__(color)
        import torch
        from catanatron.gym.model_architectures import FactoredOutcomeCritic

        manifest_path = Path(manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("kind") != self.MANIFEST_KIND:
            raise ValueError(f"Unsupported visible PUCT manifest: {manifest_path}")

        policy_path = verify_checkpoint(payload, manifest_path, "policy")
        critic_path = verify_checkpoint(payload, manifest_path, "critic")
        self.num_simulations = int(payload["num_simulations"])
        self.c_puct = float(payload["c_puct"])
        self.leaf_evaluator = payload.get("leaf_evaluator", "outcome_critic")
        if self.num_simulations < 1:
            raise ValueError("Visible PUCT num_simulations must be positive")
        if not math.isfinite(self.c_puct) or self.c_puct <= 0:
            raise ValueError("Visible PUCT c_puct must be finite and positive")
        if self.leaf_evaluator not in {"outcome_critic", "public_f"}:
            raise ValueError(
                "Visible PUCT leaf_evaluator must be outcome_critic or public_f"
            )

        declared_visible = frozenset(
            ActionType[name] for name in payload.get("visible_action_types", [])
        )
        if declared_visible != self.VISIBLE_ACTION_TYPES:
            raise ValueError("Visible PUCT action boundary differs from audited set")
        declared_forbidden = frozenset(
            ActionType[name] for name in payload.get("forbidden_action_types", [])
        )
        if declared_forbidden != self.FORBIDDEN_ACTION_TYPES:
            raise ValueError(
                "Visible PUCT forbidden-action boundary differs from audit"
            )

        self.policy = TorchBcCheckpointPlayer(color, policy_path)
        critic_meta_path = critic_path.with_suffix(".meta.json")
        critic_meta = json.loads(critic_meta_path.read_text(encoding="utf-8"))
        if critic_meta.get("kind") != "factored_outcome_critic":
            raise ValueError(f"Unsupported outcome critic metadata: {critic_meta_path}")
        critic_schema = read_model_schema(checkpoint_schema_path(critic_path))
        if critic_schema is None:
            raise FileNotFoundError(f"Missing critic schema for {critic_path}")
        expected_schema = build_model_schema()
        validate_model_schema(
            expected_schema, critic_schema, context="Visible same-turn PUCT critic"
        )
        self.feature_columns = tuple(critic_meta["feature_columns"])
        self.sample_features = tuple(
            name.removeprefix("F_") for name in self.feature_columns
        )
        if self.sample_features != tuple(critic_schema["observation"]["features"]):
            raise ValueError("Critic metadata feature columns do not match its schema")
        if self.sample_features != tuple(self.policy._inner.features):
            raise ValueError("Policy and critic observation order differs")
        self.critic = None
        if self.leaf_evaluator == "outcome_critic":
            with _preserve_inference_loader_rng():
                self.critic = FactoredOutcomeCritic(
                    self.feature_columns,
                    embedding_dim=int(critic_meta["embedding_dim"]),
                )
                state = torch.load(critic_path, map_location="cpu", weights_only=True)
                self.critic.load_state_dict(state)
                self.critic.eval()

        self.decision_stats = {
            "decisions": 0,
            "choice_decisions": 0,
            "search_decisions": 0,
            "fallback_decisions": 0,
            "changed_decisions": 0,
            "multi_ply_decisions": 0,
            "expanded_nodes": 0,
            "expanded_actions": 0,
            "forbidden_action_expansions": 0,
            "opponent_turn_expansions": 0,
            "value_evaluations": 0,
            "critic_evaluations": 0,
            "public_f_evaluations": 0,
            "latencies_ms": [],
            "fallback_action_types": {},
        }
        self.last_decision_stats = None

    def _is_visible_action_set(self, actions: list["Action"]) -> bool:
        return bool(actions) and all(
            action.action_type in self.SEARCH_ACTION_TYPES for action in actions
        )

    def _observation(self, game: "Game") -> np.ndarray:
        sample = create_sample(game, self.color, feature_profile="raw")
        return np.asarray(
            [sample[name] for name in self.sample_features], dtype=np.float32
        )

    def _policy_priors(
        self, game: "Game", actions: list["Action"]
    ) -> tuple[list[tuple[int, "Action", float]], "Action"]:
        import torch

        observation = self._observation(game)
        with torch.no_grad():
            logits = (
                self.policy._inner.torch_policy(
                    torch.from_numpy(observation).unsqueeze(0)
                )
                .squeeze(0)
                .numpy()
            )
        indexed = sorted(
            (
                to_action_space(
                    action,
                    self.policy._inner.player_colors,
                    self.policy._inner.map_type,
                ),
                action,
            )
            for action in actions
        )
        legal_logits = np.asarray([logits[index] for index, _ in indexed], dtype=float)
        legal_logits -= float(np.max(legal_logits))
        weights = np.exp(legal_logits)
        total = float(weights.sum())
        priors = (
            weights / total if total > 0 else np.full(len(indexed), 1 / len(indexed))
        )
        entries = [
            (index, action, float(prior))
            for (index, action), prior in zip(indexed, priors)
        ]
        champion = max(entries, key=lambda item: (item[2], -item[0]))[1]
        return entries, champion

    def _leaf_value(self, game: "Game") -> float:
        import torch

        winner = game.winning_color()
        if winner is not None:
            return 1.0 if winner == self.color else 0.0
        self.decision_stats["value_evaluations"] += 1
        if self.leaf_evaluator == "public_f":
            self.decision_stats["public_f_evaluations"] += 1
            return public_f_leaf_value(game, self.color)
        observation = self._observation(game)
        assert self.critic is not None
        with torch.no_grad():
            output = self.critic(torch.from_numpy(observation).unsqueeze(0))
        self.decision_stats["critic_evaluations"] += 1
        return float(torch.sigmoid(output.win_logit).item())

    def _expand(self, node: _PuctNode) -> "Action | None":
        if node.game.state.current_color() != self.color:
            self.decision_stats["opponent_turn_expansions"] += 1
            return None
        actions = list(node.game.playable_actions)
        if not self._is_visible_action_set(actions):
            return None
        entries, champion = self._policy_priors(node.game, actions)
        children = []
        for action_index, action, prior in entries:
            if action.action_type not in self.VISIBLE_ACTION_TYPES:
                self.decision_stats["forbidden_action_expansions"] += 1
                continue
            child_game = node.game.copy()
            child_game.execute(action, validate_action=False)
            children.append(
                _PuctNode(
                    child_game,
                    parent=node,
                    action=action,
                    action_index=action_index,
                    prior=prior,
                )
            )
        node.children = children
        self.decision_stats["expanded_nodes"] += 1
        self.decision_stats["expanded_actions"] += len(children)
        return champion

    def _select_child(self, node: _PuctNode) -> _PuctNode:
        scale = math.sqrt(node.visits + 1)

        def score(child: _PuctNode) -> tuple[float, float, int]:
            exploration = self.c_puct * child.prior * scale / (1 + child.visits)
            return child.q + exploration, child.prior, -child.action_index

        return max(node.children, key=score)

    def _record_fallback(self, actions: list["Action"]) -> None:
        self.decision_stats["fallback_decisions"] += 1
        counts = self.decision_stats["fallback_action_types"]
        for action_type in sorted(
            {action.action_type.name for action in actions}
            - {item.name for item in self.SEARCH_ACTION_TYPES}
        ):
            counts[action_type] = int(counts.get(action_type, 0)) + 1

    def decide(self, game: "Game", playable_actions: list["Action"]) -> "Action":
        self.decision_stats["decisions"] += 1
        if len(playable_actions) <= 1:
            return playable_actions[0]
        self.decision_stats["choice_decisions"] += 1
        actions = list(playable_actions)
        if not self._is_visible_action_set(actions):
            self._record_fallback(actions)
            return self.policy.decide(game, actions)

        started = time.perf_counter()
        root = _PuctNode(game.copy())
        champion = self._expand(root)
        if champion is None or not root.children:
            self._record_fallback(actions)
            return self.policy.decide(game, actions)

        maximum_depth = 0
        for _ in range(self.num_simulations):
            node = root
            path = [root]
            while node.children:
                node = self._select_child(node)
                path.append(node)
            maximum_depth = max(maximum_depth, len(path) - 1)
            if (
                node.game.winning_color() is None
                and node.game.state.current_color() == self.color
                and self._is_visible_action_set(list(node.game.playable_actions))
            ):
                self._expand(node)
            value = self._leaf_value(node.game)
            for ancestor in path:
                ancestor.visits += 1
                ancestor.value_sum += value

        selected_node = max(
            root.children,
            key=lambda child: (
                child.visits,
                child.q,
                child.prior,
                -child.action_index,
            ),
        )
        selected = selected_node.action
        changed = selected != champion
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.decision_stats["search_decisions"] += 1
        self.decision_stats["changed_decisions"] += int(changed)
        self.decision_stats["multi_ply_decisions"] += int(maximum_depth >= 2)
        self.decision_stats["latencies_ms"].append(elapsed_ms)
        self.last_decision_stats = {
            "searched": True,
            "changed": changed,
            "maximum_depth": maximum_depth,
            "root_actions": len(root.children),
            "selected_visits": selected_node.visits,
            "selected_q": selected_node.q,
            "selected_prior": selected_node.prior,
            "latency_ms": elapsed_ms,
        }
        return selected

    def stats_summary(self) -> dict[str, object]:
        latencies = np.asarray(self.decision_stats["latencies_ms"], dtype=float)
        searches = int(self.decision_stats["search_decisions"])
        changed = int(self.decision_stats["changed_decisions"])
        return {
            **{
                key: value
                for key, value in self.decision_stats.items()
                if key not in {"latencies_ms", "fallback_action_types"}
            },
            "change_rate": changed / searches if searches else 0.0,
            "latency_mean_ms": float(latencies.mean()) if len(latencies) else None,
            "latency_p95_ms": (
                float(np.percentile(latencies, 95)) if len(latencies) else None
            ),
            "latency_max_ms": float(latencies.max()) if len(latencies) else None,
            "fallback_action_types": dict(self.decision_stats["fallback_action_types"]),
        }

"""Visible-state PUCT with public dice and development-card chance nodes."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Union

from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    Action,
    ActionRecord,
    ActionType,
)
from catanatron.models.map import number_probability
from catanatron.players.tree_search_utils import _execute_balanced_roll_spectrum
from catanatron.players.visible_puct import (
    VISIBLE_DETERMINISTIC_ACTIONS,
    VisibleSameTurnPuctPlayer,
)
from catanatron.state_functions import (
    get_dev_cards_in_hand,
    get_enemy_colors,
    player_key,
)

if TYPE_CHECKING:
    from catanatron.game import Game
    from catanatron.models.player import Color


PUBLIC_CHANCE_ACTIONS = frozenset({ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD})
PUBLIC_CHANCE_SEARCH_ACTIONS = VISIBLE_DETERMINISTIC_ACTIONS | PUBLIC_CHANCE_ACTIONS
PUBLIC_CHANCE_FORBIDDEN_ACTIONS = frozenset(
    {ActionType.MOVE_ROBBER, ActionType.PLAY_MONOPOLY}
)


def public_unseen_development_distribution(
    game: "Game", color: "Color"
) -> tuple[tuple[str, float], ...]:
    """Distribution of cards unseen by ``color``, independent of hidden partition."""

    counts = Counter(game.state.development_listdeck)
    for opponent in get_enemy_colors(game.state.colors, color):
        for card in DEVELOPMENT_CARDS:
            counts[card] += get_dev_cards_in_hand(game.state, opponent, card)
    total = sum(counts.values())
    if total <= 0:
        return ()
    return tuple(
        (card, counts[card] / total) for card in DEVELOPMENT_CARDS if counts[card] > 0
    )


def _swap_development_card_into_deck(
    game: "Game", color: "Color", selected_card: str
) -> None:
    """Determinize an unseen card while preserving every public count."""

    deck = game.state.development_listdeck
    if selected_card in deck:
        return
    replacement = next((card for card in DEVELOPMENT_CARDS if card in deck), None)
    if replacement is None:
        raise ValueError("Cannot determinize a development card from an empty deck")
    for opponent in get_enemy_colors(game.state.colors, color):
        key = player_key(game.state, opponent)
        selected_key = f"{key}_{selected_card}_IN_HAND"
        if game.state.player_state[selected_key] <= 0:
            continue
        replacement_key = f"{key}_{replacement}_IN_HAND"
        game.state.player_state[selected_key] -= 1
        game.state.player_state[replacement_key] += 1
        deck.remove(replacement)
        deck.append(selected_card)
        # Deliberately preserve ACTUAL_VICTORY_POINTS. It is hidden and the
        # public-F leaf must not infer it from a belief-particle type swap.
        return
    raise ValueError(f"Unseen development card is unavailable: {selected_card}")


def public_buy_development_spectrum(
    game: "Game", action: Action
) -> tuple[tuple["Game", float], ...]:
    """Enumerate public-belief development-card outcomes for one purchase."""

    distribution = public_unseen_development_distribution(game, action.color)
    outcomes = []
    for card, probability in distribution:
        option_game = game.copy()
        _swap_development_card_into_deck(option_game, action.color, card)
        option_game.execute(
            action,
            validate_action=False,
            action_record=ActionRecord(action=action, result=card),
        )
        outcomes.append((option_game, probability))
    return tuple(outcomes)


def public_roll_spectrum(
    game: "Game", action: Action
) -> tuple[tuple["Game", float], ...]:
    """Enumerate dice outcomes using only the public controller/history."""

    if (
        getattr(game.state, "dice_mode", "uniform") == "balanced"
        and game.state.dice_controller is not None
    ):
        return tuple(_execute_balanced_roll_spectrum(game, action))
    outcomes = []
    for total in range(2, 13):
        dice = (total // 2, math.ceil(total / 2))
        option_game = game.copy()
        option_game.execute(
            action,
            validate_action=False,
            action_record=ActionRecord(action=action, result=dice),
        )
        outcomes.append((option_game, number_probability(total)))
    return tuple(outcomes)


def public_action_spectrum(
    game: "Game", action: Action
) -> tuple[tuple["Game", float], ...]:
    """Public-only successor distribution; never calls generic execute_spectrum."""

    if action.action_type in VISIBLE_DETERMINISTIC_ACTIONS:
        option_game = game.copy()
        option_game.execute(action, validate_action=False)
        return ((option_game, 1.0),)
    if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return public_buy_development_spectrum(game, action)
    if action.action_type == ActionType.ROLL:
        return public_roll_spectrum(game, action)
    raise ValueError(f"Action is outside the public chance boundary: {action}")


class _ChanceStateNode:
    def __init__(self, game: "Game") -> None:
        self.game = game
        self.children: list[_ChanceActionEdge] = []
        self.visits = 0
        self.value_sum = 0.0


class _ChanceOutcome:
    def __init__(
        self, child: _ChanceStateNode, probability: float, ordinal: int
    ) -> None:
        self.child = child
        self.probability = float(probability)
        self.ordinal = int(ordinal)
        self.visits = 0


class _ChanceActionEdge:
    def __init__(
        self,
        action: Action,
        *,
        action_index: int,
        prior: float,
        outcomes: list[_ChanceOutcome],
    ) -> None:
        self.action = action
        self.action_index = int(action_index)
        self.prior = float(prior)
        self.outcomes = outcomes
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.5


class VisibleChancePuctPlayer(VisibleSameTurnPuctPlayer):
    """Run-55 PUCT widened to public dice and unseen-dev-card beliefs."""

    MANIFEST_KIND = "visible_public_chance_puct"
    VISIBLE_ACTION_TYPES = VISIBLE_DETERMINISTIC_ACTIONS
    SEARCH_ACTION_TYPES = PUBLIC_CHANCE_SEARCH_ACTIONS
    FORBIDDEN_ACTION_TYPES = PUBLIC_CHANCE_FORBIDDEN_ACTIONS

    def __init__(self, color: "Color", manifest: Union[str, Path]):
        from catanatron.gym.provenance import sha256_file

        super().__init__(color, manifest)
        manifest_path = Path(manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.leaf_evaluator != "public_f":
            raise ValueError("Visible chance PUCT requires the public_f leaf")
        declared_chance = frozenset(
            ActionType[name] for name in payload.get("chance_action_types", [])
        )
        if declared_chance != PUBLIC_CHANCE_ACTIONS:
            raise ValueError("Visible chance PUCT chance boundary differs from audit")
        if payload.get("chance_allocation") != "deterministic_probability_deficit":
            raise ValueError("Visible chance PUCT allocation rule differs from audit")

        raw_parent = Path(payload["parent_manifest"])
        parent_path = (
            raw_parent
            if raw_parent.is_absolute()
            else manifest_path.parent / raw_parent
        ).resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(
                f"Missing visible chance PUCT parent: {parent_path}"
            )
        actual_parent_hash = sha256_file(parent_path)
        if actual_parent_hash != payload.get("parent_manifest_sha256"):
            raise ValueError("Visible chance PUCT parent manifest hash mismatch")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if parent.get("kind") != "visible_same_turn_puct":
            raise ValueError("Visible chance PUCT parent has the wrong kind")
        for key in (
            "policy_checkpoint_sha256",
            "policy_metadata_sha256",
            "policy_schema_sha256",
            "critic_checkpoint_sha256",
            "critic_metadata_sha256",
            "critic_schema_sha256",
            "num_simulations",
            "c_puct",
            "leaf_evaluator",
            "final_move_rule",
        ):
            if payload.get(key) != parent.get(key):
                raise ValueError(
                    f"Visible chance PUCT changed frozen parent field: {key}"
                )

        self.decision_stats.update(
            {
                "chance_actions_expanded": 0,
                "chance_outcomes_expanded": 0,
                "chance_root_decisions": 0,
                "probability_sum_violations": 0,
            }
        )

    def _expand(self, node: _ChanceStateNode) -> "Action | None":
        if node.game.state.current_color() != self.color:
            self.decision_stats["opponent_turn_expansions"] += 1
            return None
        actions = list(node.game.playable_actions)
        if not self._is_visible_action_set(actions):
            return None
        entries, champion = self._policy_priors(node.game, actions)
        edges = []
        for action_index, action, prior in entries:
            outcomes = public_action_spectrum(node.game, action)
            probability_sum = sum(probability for _, probability in outcomes)
            if not outcomes or not math.isclose(
                probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9
            ):
                self.decision_stats["probability_sum_violations"] += 1
                continue
            chance_outcomes = [
                _ChanceOutcome(_ChanceStateNode(game), probability, ordinal)
                for ordinal, (game, probability) in enumerate(outcomes)
            ]
            edges.append(
                _ChanceActionEdge(
                    action,
                    action_index=action_index,
                    prior=prior,
                    outcomes=chance_outcomes,
                )
            )
            if action.action_type in PUBLIC_CHANCE_ACTIONS:
                self.decision_stats["chance_actions_expanded"] += 1
                self.decision_stats["chance_outcomes_expanded"] += len(outcomes)
        node.children = edges
        self.decision_stats["expanded_nodes"] += 1
        self.decision_stats["expanded_actions"] += len(edges)
        return champion

    def _select_action_edge(self, node: _ChanceStateNode) -> _ChanceActionEdge:
        scale = math.sqrt(node.visits + 1)

        def score(edge: _ChanceActionEdge) -> tuple[float, float, int]:
            exploration = self.c_puct * edge.prior * scale / (1 + edge.visits)
            return edge.q + exploration, edge.prior, -edge.action_index

        return max(node.children, key=score)

    @staticmethod
    def _select_outcome(edge: _ChanceActionEdge) -> _ChanceOutcome:
        next_total = edge.visits + 1
        return max(
            edge.outcomes,
            key=lambda outcome: (
                outcome.probability * next_total - outcome.visits,
                outcome.probability,
                -outcome.ordinal,
            ),
        )

    def decide(self, game: "Game", playable_actions: list[Action]) -> Action:
        self.decision_stats["decisions"] += 1
        if len(playable_actions) <= 1:
            return playable_actions[0]
        self.decision_stats["choice_decisions"] += 1
        actions = list(playable_actions)
        if not self._is_visible_action_set(actions):
            self._record_fallback(actions)
            return self.policy.decide(game, actions)

        started = time.perf_counter()
        root = _ChanceStateNode(game.copy())
        champion = self._expand(root)
        if champion is None or not root.children:
            self._record_fallback(actions)
            return self.policy.decide(game, actions)
        if any(action.action_type in PUBLIC_CHANCE_ACTIONS for action in actions):
            self.decision_stats["chance_root_decisions"] += 1

        maximum_depth = 0
        for _ in range(self.num_simulations):
            node = root
            state_path = [root]
            edge_path: list[_ChanceActionEdge] = []
            outcome_path: list[_ChanceOutcome] = []
            while node.children:
                edge = self._select_action_edge(node)
                outcome = self._select_outcome(edge)
                edge_path.append(edge)
                outcome_path.append(outcome)
                node = outcome.child
                state_path.append(node)
            maximum_depth = max(maximum_depth, len(edge_path))
            if (
                node.game.winning_color() is None
                and node.game.state.current_color() == self.color
                and self._is_visible_action_set(list(node.game.playable_actions))
            ):
                self._expand(node)
            value = self._leaf_value(node.game)
            for state_node in state_path:
                state_node.visits += 1
                state_node.value_sum += value
            for edge in edge_path:
                edge.visits += 1
                edge.value_sum += value
            for outcome in outcome_path:
                outcome.visits += 1

        selected_edge = max(
            root.children,
            key=lambda edge: (
                edge.visits,
                edge.q,
                edge.prior,
                -edge.action_index,
            ),
        )
        selected = selected_edge.action
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
            "selected_visits": selected_edge.visits,
            "selected_q": selected_edge.q,
            "selected_prior": selected_edge.prior,
            "latency_ms": elapsed_ms,
        }
        return selected

"""
Learned policies as :class:`catanatron.models.player.Player` (SB3 or Torch BC).

Used for self-play wrappers and ``catanatron-play``-style evaluation when registered.
"""

from __future__ import annotations

import json
import random
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, Union

import numpy as np

from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.action_space import from_action_space, to_action_space
from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    read_model_schema,
    validate_model_schema,
)
from catanatron.models.player import Color, Player
from catanatron.players.checkpoint_manifest import verify_checkpoint

if TYPE_CHECKING:
    from catanatron.game import Game
    from catanatron.models.actions import Action


@contextmanager
def _preserve_inference_loader_rng() -> Iterator[None]:
    """Make inference-only checkpoint construction invisible to trainer RNGs."""

    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class Colonist1v1Player(Player):
    """
    Pick actions with an SB3 Maskable model or a Torch policy module.

    ``player_colors`` must match the environment order (P0 ``Color.BLUE``, enemy ``Color.RED`` in 1v1).
    Observations use :func:`catanatron.features.create_sample` from the acting player's perspective.
    """

    def __init__(
        self,
        color: Color,
        *,
        map_type: str = "BASE",
        num_players: int = 2,
        model=None,
        torch_policy=None,
        player_colors: Sequence[Color] = (Color.BLUE, Color.RED),
        deterministic: bool = True,
        feature_profile: str = "raw",
        human_visible_obs: bool = False,
    ):
        super().__init__(color)
        if (model is None) == (torch_policy is None):
            raise ValueError(
                "Provide exactly one of: model= (SB3), torch_policy= (nn.Module)"
            )

        self.map_type = map_type
        self.num_players = num_players
        self.feature_profile = feature_profile
        self.human_visible_obs = human_visible_obs
        self.features = get_feature_ordering(
            num_players, map_type, feature_profile=feature_profile
        )
        self.player_colors = tuple(player_colors)
        self.model = model
        self.torch_policy = torch_policy
        self.deterministic = deterministic

        from catanatron.gym.envs.action_space import get_action_array

        self._action_array_len = len(get_action_array(self.player_colors, map_type))

    def decide(self, game: "Game", playable_actions: list["Action"]) -> "Action":
        sample = create_sample(game, self.color, feature_profile=self.feature_profile)
        if self.human_visible_obs and "P0_ACTUAL_VPS" in sample:
            from catanatron.state_functions import get_visible_victory_points

            sample["P0_ACTUAL_VPS"] = get_visible_victory_points(game.state, self.color)
        obs = np.array([sample[k] for k in self.features], dtype=np.float32)

        mask = np.zeros(self._action_array_len, dtype=bool)
        for a in playable_actions:
            idx = to_action_space(a, self.player_colors, self.map_type)
            mask[idx] = True

        if self.model is not None:
            action_arr, _ = self.model.predict(
                obs, action_masks=mask, deterministic=self.deterministic
            )
            action_int = int(action_arr)
        else:
            import torch

            with torch.no_grad():
                logits = self.torch_policy(torch.as_tensor(obs).unsqueeze(0))
                logits = logits.squeeze(0).numpy()
            logits = np.where(mask, logits, -1e9)
            action_int = int(np.argmax(logits))

        return from_action_space(
            action_int, self.color, self.player_colors, self.map_type
        )


def load_sb3_player(
    checkpoint: Union[str, os.PathLike[str]],
    color: Color,
    *,
    map_type: str = "BASE",
    player_colors: Sequence[Color] = (Color.BLUE, Color.RED),
    deterministic: bool = True,
    feature_profile: Optional[str] = None,
    human_visible_obs: Optional[bool] = None,
) -> Colonist1v1Player:
    """Load :class:`sb3_contrib.ppo_mask.MaskablePPO` from disk and wrap as a Player."""
    from sb3_contrib import MaskablePPO  # type: ignore[import-untyped]

    with _preserve_inference_loader_rng():
        model = MaskablePPO.load(str(checkpoint))
    stored_schema = getattr(model, "catanatron_model_schema", None)
    if stored_schema is None:
        stored_schema = read_model_schema(checkpoint_schema_path(checkpoint))
    stored_observation = (stored_schema or {}).get("observation", {})
    resolved_profile = feature_profile or stored_observation.get(
        "feature_profile", "raw"
    )
    resolved_visibility = (
        bool(human_visible_obs)
        if human_visible_obs is not None
        else bool(stored_observation.get("human_visible_obs", False))
    )
    if stored_schema is not None:
        expected_schema = build_model_schema(
            map_type=map_type,
            player_colors=player_colors,
            feature_profile=resolved_profile,
            human_visible_obs=resolved_visibility,
        )
        validate_model_schema(expected_schema, stored_schema, context="SB3 inference")
    return Colonist1v1Player(
        color,
        map_type=map_type,
        model=model,
        player_colors=player_colors,
        deterministic=deterministic,
        feature_profile=resolved_profile,
        human_visible_obs=resolved_visibility,
    )


def load_torch_bc_player(
    checkpoint: Union[str, os.PathLike[str]],
    color: Color,
    obs_dim: int,
    n_actions: int,
    hidden_sizes: Sequence[int] = (256, 256),
    *,
    architecture: str = "mlp",
    embedding_dim: int = 128,
    feature_names: Optional[Sequence[str]] = None,
    map_type: str = "BASE",
    player_colors: Sequence[Color] = (Color.BLUE, Color.RED),
    feature_profile: str = "raw",
    human_visible_obs: bool = False,
) -> Colonist1v1Player:
    """Load a Torch ``state_dict`` saved by ``examples/colonist_1v1_bc.py``."""
    import torch
    from catanatron.gym.model_architectures import build_bc_policy

    with _preserve_inference_loader_rng():
        resolved_features = tuple(
            feature_names
            or get_feature_ordering(
                len(player_colors), map_type, feature_profile=feature_profile
            )
        )
        if len(resolved_features) != obs_dim:
            raise ValueError(
                f"BC feature schema has {len(resolved_features)} features; "
                f"metadata declares {obs_dim}"
            )
        net = build_bc_policy(
            architecture,
            resolved_features,
            n_actions,
            hidden_sizes=hidden_sizes,
            embedding_dim=embedding_dim,
        )
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
    return Colonist1v1Player(
        color,
        map_type=map_type,
        torch_policy=net,
        player_colors=player_colors,
        deterministic=True,
        feature_profile=feature_profile,
        human_visible_obs=human_visible_obs,
    )


class TorchBcCheckpointPlayer(Player):
    """
    CLI helper for Torch BC checkpoints from ``examples/colonist_1v1_bc.py``.

    Expects a sibling ``.meta.json`` (written by BC training) beside the ``.pt`` file.
    """

    def __init__(self, color: Color, checkpoint: Union[str, os.PathLike[str]]):
        super().__init__(color)
        from catanatron.gym.colonist_training import load_bc_checkpoint_meta

        ckpt = Path(checkpoint)
        meta_path = ckpt.with_suffix(".meta.json")
        meta = load_bc_checkpoint_meta(meta_path)
        if meta is None:
            raise FileNotFoundError(
                f"Missing BC metadata {meta_path}. Re-run colonist_1v1_bc.py to generate it."
            )
        stored_schema = read_model_schema(checkpoint_schema_path(ckpt))
        if stored_schema is None and meta.model_schema:
            stored_schema = meta.model_schema
        observation_schema = (stored_schema or {}).get("observation", {})
        feature_profile = observation_schema.get("feature_profile", "raw")
        human_visible_obs = bool(observation_schema.get("human_visible_obs", False))
        if stored_schema is not None:
            expected_schema = build_model_schema(
                feature_profile=feature_profile,
                human_visible_obs=human_visible_obs,
            )
            validate_model_schema(
                expected_schema, stored_schema, context="Torch BC inference"
            )
            if meta.obs_dim != len(expected_schema["observation"]["features"]):
                raise ValueError(
                    "BC metadata obs_dim does not match its feature schema"
                )
            if meta.n_actions != len(expected_schema["actions"]):
                raise ValueError(
                    "BC metadata n_actions does not match its action schema"
                )
        self._inner = load_torch_bc_player(
            ckpt,
            color,
            obs_dim=meta.obs_dim,
            n_actions=meta.n_actions,
            hidden_sizes=meta.hidden_sizes,
            architecture=meta.architecture,
            embedding_dim=meta.embedding_dim,
            feature_names=observation_schema.get("features"),
            feature_profile=feature_profile,
            human_visible_obs=human_visible_obs,
        )

    def decide(self, game: "Game", playable_actions: list["Action"]) -> "Action":
        return self._inner.decide(game, playable_actions)


class OpeningSpecialistCheckpointPlayer(Player):
    """Frozen Torch BC policy with a deterministic setup-only value fallback."""

    OPENING_PROMPTS = (
        "BUILD_INITIAL_SETTLEMENT",
        "BUILD_INITIAL_ROAD",
    )

    def __init__(self, color: Color, manifest: Union[str, os.PathLike[str]]):
        super().__init__(color)
        from catanatron.players.value import ValueFunctionPlayer

        manifest_path = Path(manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("kind") != "opening_specialist":
            raise ValueError(
                f"Unsupported opening-specialist manifest: {manifest_path}"
            )
        if tuple(payload.get("opening_prompts", ())) != self.OPENING_PROMPTS:
            raise ValueError(
                "Opening-specialist prompts must be exactly "
                f"{list(self.OPENING_PROMPTS)}"
            )
        if payload.get("opening_evaluator") != "value_function_default":
            raise ValueError(
                "Opening-specialist evaluator must be value_function_default"
            )
        if payload.get("policy_frozen") is not True:
            raise ValueError("Opening-specialist policy must be frozen")

        policy_path = verify_checkpoint(payload, manifest_path, "policy")

        self.manifest_path = manifest_path
        self.policy_checkpoint = policy_path
        self.policy = TorchBcCheckpointPlayer(color, policy_path)
        self.opening = ValueFunctionPlayer(color)
        self.decision_stats = {
            "decisions": 0,
            "choice_decisions": 0,
            "opening_decisions": 0,
            "opening_choice_decisions": 0,
            "policy_decisions": 0,
            "prompt_counts": {},
            "opening_prompt_counts": {prompt: 0 for prompt in self.OPENING_PROMPTS},
            "opening_latencies_ms": [],
        }
        self.last_decision_stats = None

    def decide(self, game: "Game", playable_actions: list["Action"]) -> "Action":
        prompt = getattr(game.state, "current_prompt", None)
        prompt_name = getattr(prompt, "name", str(prompt))
        self.decision_stats["decisions"] += 1
        if len(playable_actions) > 1:
            self.decision_stats["choice_decisions"] += 1
        prompt_counts = self.decision_stats["prompt_counts"]
        prompt_counts[prompt_name] = prompt_counts.get(prompt_name, 0) + 1

        if prompt_name in self.OPENING_PROMPTS:
            started = time.perf_counter()
            selected = self.opening.decide(game, playable_actions)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.decision_stats["opening_decisions"] += 1
            if len(playable_actions) > 1:
                self.decision_stats["opening_choice_decisions"] += 1
            self.decision_stats["opening_prompt_counts"][prompt_name] += 1
            self.decision_stats["opening_latencies_ms"].append(elapsed_ms)
            route = "opening_value_function"
        else:
            selected = self.policy.decide(game, playable_actions)
            elapsed_ms = None
            self.decision_stats["policy_decisions"] += 1
            route = "frozen_policy"
        self.last_decision_stats = {
            "prompt": prompt_name,
            "route": route,
            "num_playable_actions": len(playable_actions),
            "latency_ms": elapsed_ms,
        }
        return selected

    def stats_summary(self) -> dict[str, object]:
        latencies = np.asarray(self.decision_stats["opening_latencies_ms"], dtype=float)
        return {
            "decisions": int(self.decision_stats["decisions"]),
            "choice_decisions": int(self.decision_stats["choice_decisions"]),
            "opening_decisions": int(self.decision_stats["opening_decisions"]),
            "opening_choice_decisions": int(
                self.decision_stats["opening_choice_decisions"]
            ),
            "policy_decisions": int(self.decision_stats["policy_decisions"]),
            "prompt_counts": dict(self.decision_stats["prompt_counts"]),
            "opening_prompt_counts": dict(self.decision_stats["opening_prompt_counts"]),
            "opening_latency_mean_ms": (
                float(latencies.mean()) if len(latencies) else None
            ),
            "opening_latency_p95_ms": (
                float(np.percentile(latencies, 95)) if len(latencies) else None
            ),
            "opening_latency_max_ms": (
                float(latencies.max()) if len(latencies) else None
            ),
        }


class OutcomeRerankerCheckpointPlayer(Player):
    """Frozen Torch BC policy with a bounded top-k outcome-critic reranker."""

    def __init__(self, color: Color, manifest: Union[str, os.PathLike[str]]):
        super().__init__(color)
        import torch
        from catanatron.gym.model_architectures import FactoredOutcomeCritic

        manifest_path = Path(manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("kind") != "outcome_critic_reranker":
            raise ValueError(f"Unsupported reranker manifest: {manifest_path}")
        expected_chance_handling = "public_only_spectrum_with_policy_fallback"
        if payload.get("chance_handling") != expected_chance_handling:
            raise ValueError(
                "Outcome reranker manifest must declare hidden-safe chance handling: "
                f"{expected_chance_handling}"
            )

        policy_path = verify_checkpoint(payload, manifest_path, "policy")
        critic_path = verify_checkpoint(payload, manifest_path, "critic")
        self.top_k = int(payload["top_k"])
        self.minimum_win_probability_improvement = float(
            payload["minimum_win_probability_improvement"]
        )
        if self.top_k < 1:
            raise ValueError("Reranker top_k must be positive")
        if not 0 <= self.minimum_win_probability_improvement <= 1:
            raise ValueError(
                "minimum_win_probability_improvement must be between 0 and 1"
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
            expected_schema, critic_schema, context="Outcome critic reranker"
        )
        self.feature_columns = tuple(critic_meta["feature_columns"])
        expected_features = tuple(critic_schema["observation"]["features"])
        self.sample_features = tuple(
            name.removeprefix("F_") for name in self.feature_columns
        )
        if self.sample_features != expected_features:
            raise ValueError("Critic metadata feature columns do not match its schema")
        policy_features = tuple(self.policy._inner.features)
        if self.sample_features != policy_features:
            raise ValueError("Policy and critic observation order differs")
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
            "reranked_decisions": 0,
            "fallback_decisions": 0,
            "candidate_actions_evaluated": 0,
            "latencies_ms": [],
            "accepted_improvements": [],
        }
        self.last_decision_stats = None

    def _critic_action_value(self, game: "Game", action: "Action") -> float:
        import torch
        from catanatron.players.visible_chance_puct import public_action_spectrum

        total_probability = 0.0
        weighted_probability = 0.0
        for outcome_game, probability in public_action_spectrum(game, action):
            winner = outcome_game.winning_color()
            if winner is not None:
                win_probability = 1.0 if winner == self.color else 0.0
            else:
                sample = create_sample(outcome_game, self.color, feature_profile="raw")
                observation = np.asarray(
                    [sample[name] for name in self.sample_features], dtype=np.float32
                )
                with torch.no_grad():
                    output = self.critic(torch.from_numpy(observation).unsqueeze(0))
                    win_probability = float(torch.sigmoid(output.win_logit).item())
            weighted_probability += float(probability) * win_probability
            total_probability += float(probability)
        return (
            weighted_probability / total_probability if total_probability > 0 else 0.5
        )

    def decide(self, game: "Game", playable_actions: list["Action"]) -> "Action":
        import torch

        self.decision_stats["decisions"] += 1
        if len(playable_actions) <= 1:
            return playable_actions[0]
        from catanatron.players.visible_chance_puct import (
            PUBLIC_CHANCE_SEARCH_ACTIONS,
        )

        if any(
            action.action_type not in PUBLIC_CHANCE_SEARCH_ACTIONS
            for action in playable_actions
        ):
            self.decision_stats["fallback_decisions"] += 1
            selected = self.policy.decide(game, playable_actions)
            self.last_decision_stats = {
                "reranked": False,
                "fallback": "outside_public_chance_boundary",
            }
            return selected
        started = time.perf_counter()
        self.decision_stats["choice_decisions"] += 1
        sample = create_sample(game, self.color, feature_profile="raw")
        observation = np.asarray(
            [sample[name] for name in self.sample_features], dtype=np.float32
        )
        with torch.no_grad():
            logits = (
                self.policy._inner.torch_policy(
                    torch.from_numpy(observation).unsqueeze(0)
                )
                .squeeze(0)
                .numpy()
            )
        indexed = [
            (
                to_action_space(
                    action,
                    self.policy._inner.player_colors,
                    self.policy._inner.map_type,
                ),
                action,
            )
            for action in playable_actions
        ]
        indexed.sort(key=lambda item: (-float(logits[item[0]]), item[0]))
        shortlist = indexed[: min(self.top_k, len(indexed))]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            scored = [
                (self._critic_action_value(game, action), action_index, action)
                for action_index, action in shortlist
            ]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self.decision_stats["candidate_actions_evaluated"] += len(scored)
        champion_score = scored[0][0]
        best_score, _, best_action = max(scored, key=lambda item: (item[0], -item[1]))
        improvement = best_score - champion_score
        reranked = (
            best_action != shortlist[0][1]
            and improvement >= self.minimum_win_probability_improvement
        )
        selected = best_action if reranked else shortlist[0][1]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.decision_stats["latencies_ms"].append(elapsed_ms)
        if reranked:
            self.decision_stats["reranked_decisions"] += 1
            self.decision_stats["accepted_improvements"].append(improvement)
        self.last_decision_stats = {
            "reranked": reranked,
            "shortlist": len(shortlist),
            "champion_score": champion_score,
            "best_score": best_score,
            "improvement": improvement,
            "latency_ms": elapsed_ms,
        }
        return selected

    def stats_summary(self) -> dict[str, float | int | None]:
        latencies = np.asarray(self.decision_stats["latencies_ms"], dtype=float)
        improvements = np.asarray(
            self.decision_stats["accepted_improvements"], dtype=float
        )
        choices = int(self.decision_stats["choice_decisions"])
        reranked = int(self.decision_stats["reranked_decisions"])
        return {
            "decisions": int(self.decision_stats["decisions"]),
            "choice_decisions": choices,
            "reranked_decisions": reranked,
            "fallback_decisions": int(self.decision_stats["fallback_decisions"]),
            "rerank_rate": reranked / choices if choices else 0.0,
            "candidate_actions_evaluated": int(
                self.decision_stats["candidate_actions_evaluated"]
            ),
            "latency_mean_ms": float(latencies.mean()) if len(latencies) else None,
            "latency_p95_ms": (
                float(np.percentile(latencies, 95)) if len(latencies) else None
            ),
            "latency_max_ms": float(latencies.max()) if len(latencies) else None,
            "accepted_improvement_mean": (
                float(improvements.mean()) if len(improvements) else None
            ),
        }


class Sb3CheckpointPlayer(Player):
    """
    CLI / scripting helper: ``Sb3CheckpointPlayer(color, "path/to/maskable_ppo.zip")``.

    Prefer :func:`load_sb3_player` when you want a :class:`Colonist1v1Player` directly.
    """

    def __init__(self, color: Color, checkpoint: Union[str, os.PathLike[str]]):
        super().__init__(color)
        self._inner = load_sb3_player(
            checkpoint,
            color,
            map_type="BASE",
            player_colors=(Color.BLUE, Color.RED),
        )

    def decide(self, game: "Game", playable_actions: list["Action"]) -> "Action":
        return self._inner.decide(game, playable_actions)


def pick_league_checkpoint(
    paths: Sequence[Union[str, os.PathLike[str]]],
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Return one path from a league pool (uniform)."""
    r = rng or np.random.default_rng()
    return str(r.choice(list(paths)))

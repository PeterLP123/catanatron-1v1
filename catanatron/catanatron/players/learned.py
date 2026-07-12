"""
Learned policies as :class:`catanatron.models.player.Player` (SB3 MaskablePPO or Torch BC MLP).

Used for self-play wrappers and ``catanatron-play``-style evaluation when registered.
"""

from __future__ import annotations

import random
import os
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
    map_type: str = "BASE",
    player_colors: Sequence[Color] = (Color.BLUE, Color.RED),
    feature_profile: str = "raw",
    human_visible_obs: bool = False,
) -> Colonist1v1Player:
    """Load a Torch ``state_dict`` saved by ``examples/colonist_1v1_bc.py``."""
    import torch
    from torch import nn

    with _preserve_inference_loader_rng():
        layers = []
        d_in = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(d_in, h), nn.ReLU()])
            d_in = h
        layers.append(nn.Linear(d_in, n_actions))
        net = nn.Sequential(*layers)
        state = torch.load(str(checkpoint), map_location="cpu")
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
            feature_profile=feature_profile,
            human_visible_obs=human_visible_obs,
        )

    def decide(self, game: "Game", playable_actions: list["Action"]) -> "Action":
        return self._inner.decide(game, playable_actions)


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

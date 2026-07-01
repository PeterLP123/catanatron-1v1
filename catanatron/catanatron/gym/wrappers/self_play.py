"""
Self-play / league opponent wrapper for :class:`~catanatron.gym.envs.catanatron_env.CatanatronEnv`.

**Usage (league — last K checkpoints)**

.. code-block:: python

    import gymnasium as gym
    from catanatron.gym.wrappers.self_play import SelfPlayEnv

    base = gym.make(
        "catanatron/Catanatron-v0",
        config={"colonist_1v1": True, "representation": "vector", ...},
    )
    env = SelfPlayEnv(
        base,
        opponent_checkpoints=["ckpt/a.zip", "ckpt/b.zip", "ckpt/c.zip"],
    )

On each ``reset``, a checkpoint path is chosen uniformly. SB3 models are cached per path.

**Callable opponent**

.. code-block:: python

    env = SelfPlayEnv(env, opponent_factory=lambda: MyPlayer(Color.RED), sample_each_reset=True)

**Dependencies**: ``stable-baselines3`` + ``sb3-contrib`` when using checkpoint paths (lazy-loaded
by ``catanatron.players.learned.load_sb3_player``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np

from catanatron.models.player import Player

from catanatron.players.learned import load_sb3_player, pick_league_checkpoint

OpponentSource = Union[
    str,
    Path,
    Sequence[Union[str, Path]],
    Player,
    Callable[[], Player],
]


class SelfPlayEnv(gym.Wrapper):
    """
    Replace ``env.unwrapped.enemies[0]`` with an SB3-trained opponent or a custom :class:`Player`.

    Parameters
    ----------
    opponent
        Ready-to-use opponent player (fixed for the lifetime of the env).
    opponent_checkpoint
        Path to a saved ``MaskablePPO`` (SB3-contrib) policy.
    opponent_checkpoints
        Several checkpoints; one is sampled on every ``reset`` (league / self-play pool).
    opponent_factory
        Returns a new :class:`Player` when called.
    sample_each_reset
        If True and ``opponent_factory`` is set, build a fresh opponent after each ``reset``.
    league_rng
        Optional ``numpy.random.Generator`` for league sampling.
    opponent_source
        Convenience: one value that can be a path, list of paths, :class:`Player`, or factory.

    Pass **at most one** primary mechanism (explicit kwargs *or* ``opponent_source``).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        opponent: Optional[Player] = None,
        opponent_checkpoint: Optional[Union[str, Path]] = None,
        opponent_checkpoints: Optional[Sequence[Union[str, Path]]] = None,
        opponent_factory: Optional[Callable[[], Player]] = None,
        sample_each_reset: bool = False,
        league_rng: Optional[np.random.Generator] = None,
        opponent_source: Optional[OpponentSource] = None,
    ):
        super().__init__(env)
        self._league_rng = league_rng or np.random.default_rng()
        self._sample_each_reset = sample_each_reset
        self._factory = opponent_factory
        self._fixed_opponent = opponent
        self._single_ckpt = (
            os.fspath(opponent_checkpoint) if opponent_checkpoint is not None else None
        )
        self._league_paths: Optional[list[str]]
        if opponent_checkpoints is not None:
            self._league_paths = [os.fspath(p) for p in opponent_checkpoints]
        else:
            self._league_paths = None
        self._sb3_cache: dict[str, Player] = {}

        if opponent_source is not None:
            self._apply_opponent_source(opponent_source)

        self._count_configs()

        self._apply_opponent(self._resolve_opponent())

    def _apply_opponent_source(self, arg: OpponentSource) -> None:
        if isinstance(arg, Player):
            self._fixed_opponent = arg
        elif callable(arg):
            self._factory = arg  # type: ignore[assignment]
            self._sample_each_reset = True
        elif isinstance(arg, (str, Path)):
            self._single_ckpt = os.fspath(arg)
        elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes, bytearray)):
            self._league_paths = [os.fspath(p) for p in arg]
        else:
            raise TypeError(f"Unsupported opponent_source type: {type(arg)!r}")

    def _count_configs(self) -> None:
        n = sum(
            1
            for x in (
                self._fixed_opponent,
                self._single_ckpt,
                self._league_paths,
                self._factory,
            )
            if x is not None
        )
        if n == 0:
            raise ValueError(
                "SelfPlayEnv requires opponent configuration (opponent_source= or kwargs)."
            )
        if n > 1:
            raise ValueError(
                "Pass at most one of: opponent=, opponent_checkpoint=, "
                "opponent_checkpoints=, opponent_factory= / opponent_source="
            )
        if self._league_paths is not None and len(self._league_paths) == 0:
            raise ValueError("opponent_checkpoints= must be non-empty")

    def _p1_color(self):
        return self.env.unwrapped.enemies[0].color

    def _map_and_colors(self) -> tuple[str, tuple]:
        u = self.env.unwrapped
        return u.map_type, u.player_colors

    def _get_sb3(self, path: str) -> Player:
        if path not in self._sb3_cache:
            map_type, player_colors = self._map_and_colors()
            self._sb3_cache[path] = load_sb3_player(
                path,
                self._p1_color(),
                map_type=map_type,
                player_colors=player_colors,
            )
        return self._sb3_cache[path]

    def _resolve_opponent(self) -> Player:
        if self._fixed_opponent is not None:
            return self._fixed_opponent
        if self._factory is not None:
            return self._factory()
        if self._single_ckpt is not None:
            return self._get_sb3(self._single_ckpt)
        if self._league_paths:
            return self._get_sb3(
                pick_league_checkpoint(self._league_paths, self._league_rng)
            )
        raise RuntimeError("SelfPlayEnv: failed to resolve opponent")

    def _apply_opponent(self, opponent: Player) -> None:
        u = self.env.unwrapped
        u.enemies = [opponent]
        # Rebuild via the env's own helper so any seat order already drawn
        # (randomize_seats=True) is respected instead of always putting p0
        # first. The next reset() redraws the seat if randomization is on.
        u._rebuild_players()

    def reset(self, **kwargs: Any):
        if self._league_paths:
            path = pick_league_checkpoint(self._league_paths, self._league_rng)
            self._apply_opponent(self._get_sb3(path))
        elif self._factory is not None and self._sample_each_reset:
            self._apply_opponent(self._factory())
        return self.env.reset(**kwargs)

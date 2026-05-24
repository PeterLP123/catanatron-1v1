"""
Shaped rewards for Colonist.io-style 1v1 RL (dense VP progress + terminal outcome).

Depends only on the post-step ``game`` passed from ``CatanatronEnv.reward_function``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from catanatron.state_functions import get_actual_victory_points, get_visible_victory_points

if TYPE_CHECKING:
    from catanatron.models.actions import Action
    from catanatron.models.player import Color
    from catanatron.game import Game


_COLONIST_PREV_VP_ATTR = "_colonist_shaped_prev_vp_actual"

# Scale for intermediate VP deltas (terminal reward is always ±1).
COLONIST_SHAPED_VP_SCALE = 0.02


def colonist_shaped_reward(action: "Action", game: "Game", p0_color: "Color") -> float:
    """
    Sparse terminal signal (same sign as :func:`catanatron.gym.utils.simple_total_return`)
    plus a small reward for victory-point progress for ``p0_color`` since its last turn.

    Uses **actual** VP from the full simulator state. For public-only VP shaping, use
    :func:`make_colonist_shaped_reward` with ``use_visible_vp=True``.
    """
    winning_color = game.winning_color()
    if winning_color == p0_color:
        return 1.0
    if winning_color is not None:
        return -1.0

    vp = get_actual_victory_points(game.state, p0_color)
    prev = getattr(game, _COLONIST_PREV_VP_ATTR, 0)
    setattr(game, _COLONIST_PREV_VP_ATTR, vp)
    return float(COLONIST_SHAPED_VP_SCALE) * (vp - prev)


def make_colonist_shaped_reward(
    *,
    vp_scale: float = 0.02,
    use_visible_vp: bool = False,
) -> Callable[["Action", "Game", "Color"], float]:
    """
    Build a reward function with custom scaling and VP visibility.

    When ``use_visible_vp`` is True, shaping uses public victory points (no hidden VP cards);
    default uses actual VP for full-state training.
    """
    attr = (
        "_colonist_shaped_prev_vp_visible"
        if use_visible_vp
        else "_colonist_shaped_prev_vp_actual_custom"
    )

    def reward_fn(action: "Action", game: "Game", p0_color: "Color") -> float:
        winning_color = game.winning_color()
        if winning_color == p0_color:
            return 1.0
        if winning_color is not None:
            return -1.0

        get_vp = (
            get_visible_victory_points if use_visible_vp else get_actual_victory_points
        )
        vp = get_vp(game.state, p0_color)
        prev = getattr(game, attr, 0)
        setattr(game, attr, vp)
        return float(vp_scale) * (vp - prev)

    return reward_fn

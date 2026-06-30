"""Heuristic leaf evaluation for search (a trusted replacement for random playouts).

Search benefits from a fast, reliable estimate of how good a non-terminal state
is. Long random rollouts are slow and noisy; the hand-crafted ``F`` value
function (:mod:`catanatron.players.value`) already ranks Catan positions well, so
we reuse it as the search leaf evaluator.

:func:`f_leaf_value` turns that value function into a bounded win-probability
proxy in ``[0, 1]`` from a chosen color's perspective:

* terminal states return an exact ``1.0`` (we won) or ``0.0`` (we lost);
* non-terminal states return a scale-free, symmetric squash of our value
  advantage over the strongest opponent, so the huge raw magnitudes in the F
  weights (e.g. ``public_vps`` is ``3e14``) cannot blow up the leaf value.

The squash satisfies ``f(me) + f(opp) == 1`` in a two-player game and is
guaranteed to stay in ``[0, 1]`` because ``|v_me - v_opp| <= |v_me| + |v_opp|``.
"""

from __future__ import annotations

from typing import Callable, Optional

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.players.value import get_value_fn

EPSILON = 1e-9

# Type of an F-style value function: (game, color) -> scalar, higher is better.
ValueFn = Callable[[Game, Color], float]


def make_f_value_fn(value_fn_builder_name: str = "base_fn", params=None) -> ValueFn:
    """Build an F value function (defaults to the ``base_fn`` weights).

    Pass ``"contender_fn"`` (or the CLI ``"C"`` alias is resolved upstream) to use
    the tuned contender weights instead.
    """
    return get_value_fn(value_fn_builder_name, params)


def f_leaf_value(game: Game, color: Color, value_fn: Optional[ValueFn] = None) -> float:
    """Estimate ``color``'s win probability in ``[0, 1]`` for the given state.

    Terminal states are scored exactly. Non-terminal states are scored by the F
    value advantage over the strongest opponent, squashed to ``[0, 1]``.
    """
    winner = game.winning_color()
    if winner is not None:
        return 1.0 if winner == color else 0.0

    if value_fn is None:
        value_fn = make_f_value_fn()

    v_me = value_fn(game, color)
    opponents = [c for c in game.state.colors if c != color]
    if not opponents:
        return 0.5
    v_opp = max(value_fn(game, c) for c in opponents)

    diff = v_me - v_opp
    denom = abs(v_me) + abs(v_opp) + EPSILON
    return 0.5 + 0.5 * (diff / denom)

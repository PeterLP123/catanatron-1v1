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

import math
from typing import Callable, Optional

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.players.value import get_value_fn
from catanatron.state_functions import player_key

EPSILON = 1e-9

# Type of an F-style value function: (game, color) -> scalar, higher is better.
ValueFn = Callable[[Game, Color], float]

# Calibration of the leaf win-probability (see leaf_win_probability). The value
# is sigmoid(W_VP * public_vp_margin + W_POSITION * position_advantage):
#   - one victory point of lead shifts the logit by W_VP (a strong but not
#     decisive edge), and
#   - position_advantage = tanh((pos_me - pos_opp) / POS_SCALE) in (-1, 1) shifts
#     it by up to W_POSITION, so same-VP candidates stay well separated instead
#     of all collapsing to 0.5 the way the bounded f_leaf_value proxy does.
# POS_SCALE (~half the base production weight) is an external, fixed scale so the
# position signal does not saturate when the opponent's position is near zero
# (early game), which a magnitude-relative normalization would.
W_VP = 0.6
W_POSITION = 1.0
POS_SCALE = 5e7


def make_position_value_fn(
    value_fn_builder_name: str = "base_fn", params=None
) -> ValueFn:
    """Build an F value function with the victory-point term removed.

    The result scores only the *positional* part of F (production, reachability,
    hand synergy, longest road, dev cards, ...). Used by
    :func:`leaf_win_probability` so the VP signal can be handled separately and
    not drown out the sub-VP differences that distinguish same-score candidates.
    """
    from catanatron.players.value import (
        base_fn,
        CONTENDER_WEIGHTS,
        DEFAULT_WEIGHTS,
    )

    if value_fn_builder_name == "contender_fn":
        weights = dict(params or CONTENDER_WEIGHTS)
    else:
        weights = dict(DEFAULT_WEIGHTS)
        if params:
            weights.update(params)
    weights["public_vps"] = 0.0
    return base_fn(weights)


def _public_vps(game: Game, color: Color) -> float:
    """Public victory points of ``color`` (no hidden VP dev cards)."""
    key = player_key(game.state, color)
    return float(game.state.player_state[f"{key}_VICTORY_POINTS"])


def value_target_components(
    game: Game, color: Color, pos_value_fn: Optional[ValueFn] = None
) -> dict:
    """The value-target heads for ``color``: outcome, VP margin, F potential.

    Returns a dict with:
      * ``outcome`` -- 1.0/0.0 at terminal states, else ``None``;
      * ``vp_margin`` -- public VP lead over the strongest opponent;
      * ``position_advantage`` -- scale-free F position advantage in ``[-1, 1]``;
      * ``win_prob`` -- the combined leaf win probability in ``[0, 1]``.

    These are the "outcome + VP margin + F potential" targets the plan calls for;
    ``win_prob`` is the scalar the search uses as its leaf value.
    """
    winner = game.winning_color()
    opponents = [c for c in game.state.colors if c != color]

    vp_me = _public_vps(game, color)
    vp_opp = max((_public_vps(game, c) for c in opponents), default=0.0)
    vp_margin = vp_me - vp_opp

    if pos_value_fn is None:
        pos_value_fn = make_position_value_fn()
    pos_me = pos_value_fn(game, color)
    pos_opp = max((pos_value_fn(game, c) for c in opponents), default=0.0)
    position_advantage = math.tanh((pos_me - pos_opp) / POS_SCALE)

    if winner is not None:
        win_prob = 1.0 if winner == color else 0.0
        outcome = win_prob
    else:
        logit = W_VP * vp_margin + W_POSITION * position_advantage
        win_prob = 1.0 / (1.0 + math.exp(-logit))
        outcome = None

    return {
        "outcome": outcome,
        "vp_margin": vp_margin,
        "position_advantage": position_advantage,
        "win_prob": win_prob,
    }


def leaf_win_probability(
    game: Game, color: Color, pos_value_fn: Optional[ValueFn] = None
) -> float:
    """Win-probability leaf value in ``[0, 1]`` that preserves sub-VP resolution.

    Terminal states score an exact 1/0. Non-terminal states combine the public
    victory-point margin (the dominant, calibrated term) with a scale-free F
    position advantage, so candidates with the same victory points still spread
    across the interval instead of collapsing to ~0.5 (the failure mode of the
    magnitude-normalized :func:`f_leaf_value`). Symmetric: the two perspectives
    sum to 1.
    """
    return value_target_components(game, color, pos_value_fn)["win_prob"]


def make_f_value_fn(value_fn_builder_name: str = "base_fn", params=None) -> ValueFn:
    """Build an F value function (defaults to the ``base_fn`` weights).

    Pass ``"contender_fn"`` (or the CLI ``"C"`` alias is resolved upstream) to use
    the tuned contender weights instead.
    """
    return get_value_fn(value_fn_builder_name, params)


def state_signature(game: Game, color: Color):
    """A cheap, hashable key capturing everything ``f_leaf_value`` reads.

    The F value function depends only on the board (settlements, cities, roads,
    robber) and the per-player scalar state (victory points, hands, dev cards,
    played knights, longest road). Two states with the same signature therefore
    produce the same leaf value, so this is a collision-free key for caching F
    evaluations across transposed positions. It deliberately excludes hidden
    deck *ordering*, which the value function does not use.
    """
    state = game.state
    board = state.board
    return (
        color,
        tuple(sorted(state.player_state.items())),
        frozenset(board.buildings.items()),
        frozenset(board.roads.items()),
        board.robber_coordinate,
    )


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


def action_value(
    game: Game, action, color: Color, value_fn: Optional[ValueFn] = None
) -> float:
    """Chance-weighted *raw* F value of playing ``action``, from ``color``'s view.

    This is the value the teacher F itself maximizes (``value_fn(successor,
    color)``), averaged over chance outcomes with :func:`execute_spectrum`. It is
    deliberately **not** squashed to ``[0, 1]``: the bounded :func:`f_leaf_value`
    proxy is dominated by the victory-point weight and collapses same-VP
    candidates to ~0.5, erasing exactly the decision-margin signal these labels
    exist to capture. Magnitudes are large and only meaningful *relative to other
    candidates of the same decision*.
    """
    # Imported here to avoid a heavy import at module load; tree_search_utils
    # pulls in the feature stack.
    from catanatron.players.tree_search_utils import execute_spectrum

    if value_fn is None:
        value_fn = make_f_value_fn()
    outcomes = execute_spectrum(game, action)
    total = 0.0
    weighted = 0.0
    for outcome_game, proba in outcomes:
        weighted += proba * value_fn(outcome_game, color)
        total += proba
    return weighted / total if total > 0 else value_fn(game, color)


def candidate_values(
    game: Game, color: Color, value_fn: Optional[ValueFn] = None
) -> list:
    """Raw F value of every legal action, aligned with ``game.playable_actions``.

    Entry ``i`` is the F value (for ``color``) of the position after
    ``playable_actions[i]``. Used to label teacher decisions with the value of
    each legal candidate so training can target the decision margin and report
    regret against the best legal option. Values are comparable only within one
    decision (see :func:`action_value`).
    """
    if value_fn is None:
        value_fn = make_f_value_fn()
    return [action_value(game, a, color, value_fn) for a in game.playable_actions]

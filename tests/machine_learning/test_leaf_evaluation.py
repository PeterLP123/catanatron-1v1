import random

from catanatron import Game, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import get_value_fn
from catanatron.players.leaf_evaluation import (
    action_value,
    candidate_values,
    f_leaf_value,
    leaf_win_probability,
    make_f_value_fn,
    make_position_value_fn,
    state_signature,
    value_target_components,
)


def make_midgame_1v1(seed=42, ticks=25):
    """A non-terminal mid-game 1v1 state (deterministic given the seed)."""
    random.seed(seed)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    assert game.winning_color() is None, "expected a non-terminal mid-game state"
    return game


def make_terminal_1v1(seed=7):
    """A finished 1v1 game with a definite winner."""
    random.seed(seed)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    game.play()
    assert game.winning_color() is not None, "expected the game to finish"
    return game


def test_leaf_value_within_unit_interval():
    game = make_midgame_1v1()
    for color in (Color.RED, Color.BLUE):
        v = f_leaf_value(game, color)
        assert 0.0 <= v <= 1.0


def test_leaf_value_is_symmetric_between_players():
    game = make_midgame_1v1()
    v_red = f_leaf_value(game, Color.RED)
    v_blue = f_leaf_value(game, Color.BLUE)
    # Zero-sum proxy: the two perspectives must sum to 1.
    assert abs((v_red + v_blue) - 1.0) < 1e-9


def test_leaf_value_terminal_is_exact():
    game = make_terminal_1v1()
    winner = game.winning_color()
    loser = Color.RED if winner == Color.BLUE else Color.BLUE
    assert f_leaf_value(game, winner) == 1.0
    assert f_leaf_value(game, loser) == 0.0


def test_leaf_value_orders_with_value_function():
    """The leaf proxy crosses 0.5 exactly where the F value advantage flips sign."""
    game = make_midgame_1v1()
    value_fn = get_value_fn("base_fn", None)
    v_red = value_fn(game, Color.RED)
    v_blue = value_fn(game, Color.BLUE)

    leaf_red = f_leaf_value(game, Color.RED, value_fn)
    assert (leaf_red > 0.5) == (v_red > v_blue)
    assert (leaf_red < 0.5) == (v_red < v_blue)


def test_make_f_value_fn_is_callable():
    game = make_midgame_1v1()
    value_fn = make_f_value_fn()
    assert isinstance(value_fn(game, Color.RED), float)


def test_state_signature_is_hashable_and_stable_under_copy():
    game = make_midgame_1v1()
    sig = state_signature(game, Color.RED)
    hash(sig)  # must be usable as a cache key
    assert sig == state_signature(game.copy(), Color.RED)


def test_state_signature_differs_by_color_and_state():
    game = make_midgame_1v1()
    assert state_signature(game, Color.RED) != state_signature(game, Color.BLUE)

    later = make_midgame_1v1(seed=42, ticks=45)
    assert state_signature(game, Color.RED) != state_signature(later, Color.RED)


def test_state_signature_determines_leaf_value():
    """Equal signatures must imply equal leaf values (cache correctness)."""
    game = make_midgame_1v1()
    value_fn = make_f_value_fn()
    a = f_leaf_value(game, Color.RED, value_fn)
    b = f_leaf_value(game.copy(), Color.RED, value_fn)
    assert state_signature(game, Color.RED) == state_signature(game.copy(), Color.RED)
    assert a == b


def make_choice_1v1(seed=4, max_ticks=200):
    """A state where the current player has more than one legal action."""
    random.seed(seed)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    for _ in range(max_ticks):
        if game.winning_color() is not None:
            break
        if len(game.playable_actions) > 1:
            return game
        game.play_tick()
    raise AssertionError("no multi-action decision state found")


def test_candidate_values_align_with_legal_actions():
    import math

    game = make_choice_1v1()
    color = game.state.current_color()
    values = candidate_values(game, color)
    assert len(values) == len(game.playable_actions)
    assert all(math.isfinite(v) for v in values)


def test_action_value_matches_candidate_values_entry():
    game = make_choice_1v1()
    color = game.state.current_color()
    value_fn = make_f_value_fn()
    values = candidate_values(game, color, value_fn)
    first = action_value(game, game.playable_actions[0], color, value_fn)
    assert first == values[0]


# --- resolution-preserving leaf win probability --------------------------------


def test_leaf_win_probability_within_unit_interval_and_symmetric():
    game = make_midgame_1v1()
    v_red = leaf_win_probability(game, Color.RED)
    v_blue = leaf_win_probability(game, Color.BLUE)
    assert 0.0 <= v_red <= 1.0 and 0.0 <= v_blue <= 1.0
    assert abs((v_red + v_blue) - 1.0) < 1e-9


def test_leaf_win_probability_terminal_is_exact():
    game = make_terminal_1v1()
    winner = game.winning_color()
    loser = Color.RED if winner == Color.BLUE else Color.BLUE
    assert leaf_win_probability(game, winner) == 1.0
    assert leaf_win_probability(game, loser) == 0.0


def test_leaf_win_probability_resolves_same_vp_candidates():
    """The new leaf separates candidates that the bounded proxy collapses to 0.5."""
    game = make_choice_1v1()
    color = game.state.current_color()
    pos_fn = make_position_value_fn()

    leaf_vals = []
    proxy_vals = []
    for action in game.playable_actions:
        successor = game.copy()
        try:
            successor.execute(action, validate_action=False)
        except Exception:
            continue
        leaf_vals.append(round(leaf_win_probability(successor, color, pos_fn), 6))
        proxy_vals.append(round(f_leaf_value(successor, color), 6))

    assert all(0.0 <= v <= 1.0 for v in leaf_vals)
    # The repaired leaf discriminates between same-VP candidates...
    assert len(set(leaf_vals)) > 1
    # ...where the old magnitude-normalized proxy is far flatter.
    assert len(set(leaf_vals)) >= len(set(proxy_vals))


def test_value_target_components_shape():
    game = make_midgame_1v1()
    comps = value_target_components(game, Color.RED)
    assert comps["outcome"] is None  # non-terminal
    assert -1.0 <= comps["position_advantage"] <= 1.0
    assert 0.0 <= comps["win_prob"] <= 1.0
    assert isinstance(comps["vp_margin"], float)

    terminal = make_terminal_1v1()
    winner = terminal.winning_color()
    assert value_target_components(terminal, winner)["outcome"] == 1.0

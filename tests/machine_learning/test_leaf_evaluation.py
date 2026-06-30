import random

from catanatron import Game, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import get_value_fn
from catanatron.players.leaf_evaluation import f_leaf_value, make_f_value_fn


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

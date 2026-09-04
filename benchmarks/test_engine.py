"""Optional engine timings; run with ``make benchmark``."""

import json

import pytest

from catanatron.features import create_sample
from catanatron.game import Game
from catanatron.json import GameEncoder
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer


@pytest.fixture
def game():
    return Game([RandomPlayer(color) for color in Color], seed=0)


def test_to_json_speed(benchmark, game):
    result = benchmark(json.dumps, game, cls=GameEncoder)
    assert isinstance(result, str)


def test_copy_speed(benchmark, game):
    result = benchmark(game.copy)
    assert result.seed == game.seed


def test_create_sample_speed(benchmark, game):
    for _ in range(30):
        game.play_tick()
    sample = benchmark(create_sample, game, Color.BLUE)
    assert sample


@pytest.mark.parametrize(
    "player_class",
    [RandomPlayer, WeightedRandomPlayer, AlphaBetaPlayer, SameTurnAlphaBetaPlayer],
    ids=["random", "weighted", "alpha-beta", "same-turn-alpha-beta"],
)
def test_player_speed(benchmark, player_class):
    def setup():
        players = [
            RandomPlayer(color) for color in (Color.RED, Color.BLUE, Color.WHITE)
        ]
        players.append(player_class(Color.ORANGE))
        return (Game(players, seed=0),), {}

    def play_ticks(game):
        for _ in range(100):
            game.play_tick()
        return game

    result = benchmark.pedantic(play_ticks, setup=setup, rounds=5)
    assert len(result.state.action_records) == 100

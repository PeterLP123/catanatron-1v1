from typing import Iterable

from catanatron import Action, Color, Game, GameAccumulator, Player, RandomPlayer


class FirstActionPlayer(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        return next(iter(playable_actions))


def test_top_level_imports_work():
    class MyAccumulator(GameAccumulator):
        pass

    players = [
        FirstActionPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players)
    game.play(accumulators=[MyAccumulator()])

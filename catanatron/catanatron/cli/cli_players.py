from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from rich.table import Table

from catanatron.models.player import Color, HumanPlayer, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer


@dataclass(frozen=True)
class CliPlayer:
    code: str
    name: str
    description: str
    factory: Callable | str

    @property
    def import_fn(self) -> Callable:
        """Resolve optional players only when selected, keeping CLI help lightweight."""
        if isinstance(self.factory, str):
            module, name = self.factory.rsplit(".", 1)
            return getattr(import_module(module), name)
        return self.factory


CLI_PLAYERS = [
    CliPlayer(
        "H", "HumanPlayer", "Human player, uses input() to get action.", HumanPlayer
    ),
    CliPlayer("R", "RandomPlayer", "Chooses actions at random.", RandomPlayer),
    CliPlayer(
        "W",
        "WeightedRandomPlayer",
        "Like RandomPlayer, but favors buying cities, settlements, and dev cards when possible.",
        WeightedRandomPlayer,
    ),
    CliPlayer(
        "VP",
        "VictoryPointPlayer",
        "Chooses randomly from actions that increase victory points immediately if possible, else at random.",
        VictoryPointPlayer,
    ),
    CliPlayer(
        "G",
        "GreedyPlayoutsPlayer",
        "For each action, will play N random 'playouts'. "
        + "Takes the action that led to best winning percent. "
        + "First param is NUM_PLAYOUTS",
        GreedyPlayoutsPlayer,
    ),
    CliPlayer(
        "M",
        "MCTSPlayer",
        "Decides according to the MCTS algorithm. First param is NUM_SIMULATIONS.",
        MCTSPlayer,
    ),
    CliPlayer(
        "F",
        "ValueFunctionPlayer",
        "Chooses the action that leads to the most immediate reward, based on a hand-crafted value function.",
        ValueFunctionPlayer,
    ),
    CliPlayer(
        "AB",
        "AlphaBetaPlayer",
        "Implements alpha-beta algorithm. That is, looks ahead a couple "
        + "levels deep evaluating leafs with hand-crafted value function. "
        + "Params are DEPTH, PRUNNING",
        AlphaBetaPlayer,
    ),
    CliPlayer(
        "SAB",
        "SameTurnAlphaBetaPlayer",
        "AlphaBeta but searches only within turn",
        SameTurnAlphaBetaPlayer,
    ),
    CliPlayer(
        "L",
        "Sb3CheckpointPlayer",
        "MaskablePPO policy (sb3-contrib). Pass checkpoint path, e.g. L:runs/ppo.zip",
        "catanatron.players.learned.Sb3CheckpointPlayer",
    ),
    CliPlayer(
        "T",
        "TorchBcCheckpointPlayer",
        "Torch BC policy (.pt + .meta.json). e.g. T:runs/colonist_bc_policy.pt",
        "catanatron.players.learned.TorchBcCheckpointPlayer",
    ),
    CliPlayer(
        "C",
        "OutcomeRerankerCheckpointPlayer",
        "Frozen Torch BC policy plus bounded outcome-critic reranker manifest.",
        "catanatron.players.learned.OutcomeRerankerCheckpointPlayer",
    ),
    CliPlayer(
        "O",
        "OpeningSpecialistCheckpointPlayer",
        "Frozen Torch BC policy plus deterministic setup-only value fallback manifest.",
        "catanatron.players.learned.OpeningSpecialistCheckpointPlayer",
    ),
    CliPlayer(
        "N",
        "VisibleSameTurnPuctPlayer",
        "Frozen policy-guided same-turn PUCT with a hidden-information boundary.",
        "catanatron.players.visible_puct.VisibleSameTurnPuctPlayer",
    ),
    CliPlayer(
        "Q",
        "VisibleChancePuctPlayer",
        "Visible same-turn PUCT with public dice and development-card chance nodes.",
        "catanatron.players.visible_chance_puct.VisibleChancePuctPlayer",
    ),
]


def parse_cli_string(player_string: str):
    """Build players in seat order, rejecting invalid specs before construction."""
    player_keys = [key.strip() for key in player_string.split(",")]
    colors = list(Color)
    if len(player_keys) > len(colors):
        raise ValueError(f"At most {len(colors)} players are supported")
    specifications = []
    for key in player_keys:
        code, *params = key.split(":")
        entry = next((player for player in CLI_PLAYERS if player.code == code), None)
        if entry is None:
            raise ValueError(
                f"Unknown player code {code!r}; use --help-players for supported codes"
            )
        specifications.append((entry, params))
    players = []
    for color, (entry, params) in zip(colors, specifications):
        try:
            players.append(entry.import_fn(color, *params))
        except ModuleNotFoundError as exc:
            if isinstance(entry.factory, str) and exc.name in {
                "numpy",
                "pandas",
                "gymnasium",
                "torch",
                "stable_baselines3",
                "sb3_contrib",
            }:
                raise ValueError(
                    f"Player {entry.code} requires {exc.name}; install training dependencies "
                    "with pip install 'catanatron-1v1[gym,colonist]'"
                ) from exc
            raise
    return players


def register_cli_player(code, player_class):
    CLI_PLAYERS.append(
        CliPlayer(
            code,
            player_class.__name__,
            player_class.__doc__,
            player_class,
        ),
    )


CUSTOM_ACCUMULATORS = []


def register_cli_accumulator(accumulator_class):
    CUSTOM_ACCUMULATORS.append(accumulator_class)


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("DESCRIPTION")
    for player in CLI_PLAYERS:
        table.add_row(player.code, player.name, player.description)
    return table

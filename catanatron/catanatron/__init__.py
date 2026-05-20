"""
This is to allow an API like:

from catanatron import Game, Player, Color, Accumulator
"""

from catanatron.game import Game, GameAccumulator
from catanatron.colonist_1v1 import (
    COLONIST_1V1_SETTINGS,
    Colonist1v1Settings,
    create_colonist_1v1_game,
)
from catanatron.models.player import Player, Color, RandomPlayer
from catanatron.models.enums import (
    Action,
    ActionType,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    RESOURCES,
)

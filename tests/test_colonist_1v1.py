from catanatron import COLONIST_1V1_SETTINGS, create_colonist_1v1_game
from catanatron.colonist_1v1 import validate_colonist_1v1_players
from catanatron.game import Game
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.player import Color, RandomPlayer
from catanatron.apply_action import apply_roll
from catanatron.state_functions import player_num_resource_cards
import pytest


def test_colonist_1v1_settings():
    s = COLONIST_1V1_SETTINGS
    assert s.num_players == 2
    assert s.vps_to_win == 15
    assert s.dice_mode == "balanced"
    assert s.friendly_robber is True
    assert s.discard_limit == 9
    assert s.friendly_robber_vp_threshold == 2


def test_create_colonist_1v1_game():
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = create_colonist_1v1_game(players, seed=1)
    assert game.vps_to_win == 15
    assert game.state.discard_limit == 9
    assert game.state.friendly_robber is True
    assert game.state.dice_mode == "balanced"
    assert game.state.dice_controller is not None
    assert game.state.friendly_robber_vp_threshold == 2
    assert game.state.friendly_robber_use_visible_vp is True


def test_game_colonist_1v1_flag():
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=1, colonist_1v1=True)
    assert game.colonist_1v1 is True
    assert game.vps_to_win == 15


def test_rejects_wrong_player_count():
    with pytest.raises(ValueError):
        validate_colonist_1v1_players([RandomPlayer(Color.RED)])


def test_safe_hand_limit_on_seven():
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = create_colonist_1v1_game(players, seed=1)
    red_key = "P0"
    for resource in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]:
        game.state.player_state[f"{red_key}_{resource}_IN_HAND"] = 2

    assert player_num_resource_cards(game.state, Color.RED) == 10
    apply_roll(
        game.state,
        Action(Color.RED, ActionType.ROLL, None),
        action_record=ActionRecord(
            action=Action(Color.RED, ActionType.ROLL, None), result=(3, 4)
        ),
    )
    assert game.state.discard_counts[0] == 5

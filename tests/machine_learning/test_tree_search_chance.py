import pytest

from catanatron import Color, Game, RandomPlayer
from catanatron.models.dice_controller_balanced import DiceControllerBalanced
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.state_functions import get_player_freqdeck, player_key
from catanatron.players.tree_search_utils import execute_spectrum


def _game():
    return Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])


def test_balanced_roll_spectrum_uses_observable_remaining_deck():
    game = _game()
    game.state.dice_mode = "balanced"
    controller = DiceControllerBalanced(number_of_players=2)
    seven_deck = controller.weighted_dice_deck[7 - controller.INDEX_OFFSET]
    removed_sevens = len(seven_deck.dice_pairs)
    seven_deck.dice_pairs.clear()
    controller.cards_left_in_deck -= removed_sevens
    game.state.dice_controller = controller

    action = Action(game.state.current_color(), ActionType.ROLL, None)
    outcomes = execute_spectrum(game, action)

    assert sum(probability for _, probability in outcomes) == pytest.approx(1.0)
    totals = {
        child.state.action_records[-1].result[0]
        + child.state.action_records[-1].result[1]
        for child, _ in outcomes
    }
    assert 7 not in totals
    assert len(outcomes) == 10
    assert controller.cards_left_in_deck == 30  # the source game is untouched
    assert all(
        child.state.dice_controller.cards_left_in_deck == 29 for child, _ in outcomes
    )


def test_robber_spectrum_weights_resources_by_opponent_hand_counts():
    game = _game()
    actor = game.state.current_color()
    opponent = next(color for color in game.state.colors if color != actor)
    opponent_key = player_key(game.state, opponent)
    actor_key = player_key(game.state, actor)
    for resource, count in zip(RESOURCES, (2, 0, 3, 0, 0)):
        game.state.player_state[f"{opponent_key}_{resource}_IN_HAND"] = count
        game.state.player_state[f"{actor_key}_{resource}_IN_HAND"] = 0

    coordinate = next(
        coordinate
        for coordinate in game.state.board.map.land_tiles
        if coordinate != game.state.board.robber_coordinate
    )
    action = Action(actor, ActionType.MOVE_ROBBER, (coordinate, opponent))
    outcomes = execute_spectrum(game, action)

    probabilities = {
        child.state.action_records[-1].result: probability
        for child, probability in outcomes
    }
    assert probabilities == {
        RESOURCES[0]: pytest.approx(0.4),
        RESOURCES[2]: pytest.approx(0.6),
    }
    assert get_player_freqdeck(game.state, opponent) == [2, 0, 3, 0, 0]
    for child, _ in outcomes:
        stolen = child.state.action_records[-1].result
        resource_index = RESOURCES.index(stolen)
        assert get_player_freqdeck(child.state, opponent)[resource_index] in {1, 2}


def test_development_card_spectrum_conditions_the_replayed_draw():
    game = _game()
    actor = game.state.current_color()
    actor_key = player_key(game.state, actor)
    for resource, count in zip(RESOURCES, (0, 0, 1, 1, 1)):
        game.state.player_state[f"{actor_key}_{resource}_IN_HAND"] = count
    game.state.development_listdeck = ["KNIGHT", "VICTORY_POINT"]
    action = Action(actor, ActionType.BUY_DEVELOPMENT_CARD, None)

    outcomes = execute_spectrum(game, action)

    assert sum(probability for _, probability in outcomes) == pytest.approx(1.0)
    assert {child.state.action_records[-1].result for child, _ in outcomes} == {
        "KNIGHT",
        "VICTORY_POINT",
    }

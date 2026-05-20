from catanatron.models.dice_controller_balanced import DiceControllerBalanced
from catanatron.models.player import Color


def test_standard_deck_has_36_outcomes():
    deck = DiceControllerBalanced.get_standard_dice_deck()
    assert sum(len(bucket.dice_pairs) for bucket in deck) == 36


def test_reshuffle_restores_full_deck():
    controller = DiceControllerBalanced(4)
    controller.cards_left_in_deck = 0
    for bucket in controller.weighted_dice_deck:
        bucket.dice_pairs.clear()

    controller.reshuffle_weighted_dice_deck()
    assert controller.cards_left_in_deck == 36
    assert sum(len(bucket.dice_pairs) for bucket in controller.weighted_dice_deck) == 36


def test_reshuffles_when_deck_drops_below_minimum():
    controller = DiceControllerBalanced(4)
    # 36 - 23 = 13, the minimum before reshuffle
    for _ in range(23):
        controller.throw_dice(Color.RED)
    assert controller.cards_left_in_deck == 13

    # Next roll drops to 12, then the following roll triggers reshuffle
    controller.throw_dice(Color.RED)
    assert controller.cards_left_in_deck == 12

    controller.throw_dice(Color.RED)
    assert controller.cards_left_in_deck == 35


def test_copy_preserves_deck_state():
    controller = DiceControllerBalanced(2)
    controller.throw_dice(Color.RED)
    cloned = controller.copy()
    assert cloned.cards_left_in_deck == controller.cards_left_in_deck
    assert cloned.recent_rolls == controller.recent_rolls

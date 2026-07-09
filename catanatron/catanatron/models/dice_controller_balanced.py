"""
Colonist-style balanced dice deck: 36 physical outcomes drawn without replacement,
with probability adjustments for recent totals and seven distribution.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from catanatron.models.player import Color

DicePair = Tuple[int, int]


@dataclass
class StandardDiceDeck:
    total_dice: int
    dice_pairs: List[DicePair]


@dataclass
class WeightedDiceDeck:
    total_dice: int
    dice_pairs: List[DicePair] = field(default_factory=list)
    probability_weighting: float = 0.0
    recently_rolled_count: int = 0


class DiceControllerBalanced:
    """Weighted deck of all 36 two-dice outcomes with reshuffle and bias adjustments."""

    MINIMUM_CARDS_BEFORE_RESHUFFLING = 13
    PROBABILITY_REDUCTION_FOR_RECENTLY_ROLLED = 0.34
    PROBABILITY_REDUCTION_FOR_SEVEN_STREAKS = 0.4
    MAXIMUM_RECENT_ROLL_MEMORY = 5
    INDEX_OFFSET = 2  # totals 2-12 map to deck indices 0-10
    TOTAL_COMBINATIONS = 36

    def __init__(self, number_of_players: int):
        self.number_of_players = number_of_players
        self.weighted_dice_deck: List[WeightedDiceDeck] = []
        self.cards_left_in_deck = 0
        self.recent_rolls: List[int] = []
        self.seven_streak_count: Dict = {"player_color": None, "streak_count": 0}
        self.total_sevens_rolled_by_player: Dict[Color, int] = {}
        self._init_weighted_dice_deck()
        self.reshuffle_weighted_dice_deck()
        self.update_weighted_dice_deck_probabilities()

    def throw_dice(self, player_color: Color) -> DicePair:
        self._init_total_sevens(player_color)
        return self._draw_weighted_card(player_color)

    def copy(self) -> "DiceControllerBalanced":
        cloned = DiceControllerBalanced.__new__(DiceControllerBalanced)
        cloned.number_of_players = self.number_of_players
        cloned.weighted_dice_deck = [
            WeightedDiceDeck(
                total_dice=deck.total_dice,
                dice_pairs=deck.dice_pairs.copy(),
                probability_weighting=deck.probability_weighting,
                recently_rolled_count=deck.recently_rolled_count,
            )
            for deck in self.weighted_dice_deck
        ]
        cloned.cards_left_in_deck = self.cards_left_in_deck
        cloned.recent_rolls = self.recent_rolls.copy()
        cloned.seven_streak_count = copy.copy(self.seven_streak_count)
        cloned.total_sevens_rolled_by_player = self.total_sevens_rolled_by_player.copy()
        return cloned

    def _init_weighted_dice_deck(self) -> None:
        self.weighted_dice_deck = [
            WeightedDiceDeck(total_dice=total, dice_pairs=[], probability_weighting=0.0)
            for total in range(2, 13)
        ]

    def reshuffle_weighted_dice_deck(self) -> None:
        standard_dice_deck = self.get_standard_dice_deck()
        for total_dice_index, dice_pairs_for_total_dice in enumerate(
            standard_dice_deck
        ):
            self.weighted_dice_deck[total_dice_index].dice_pairs = (
                dice_pairs_for_total_dice.dice_pairs.copy()
            )
            self.weighted_dice_deck[total_dice_index].recently_rolled_count = 0
        self.cards_left_in_deck = self.TOTAL_COMBINATIONS

    def update_weighted_dice_deck_probabilities(self) -> None:
        for dice_deck_for_total_dice in self.weighted_dice_deck:
            if self.cards_left_in_deck == 0:
                dice_deck_for_total_dice.probability_weighting = 0.0
            else:
                dice_deck_for_total_dice.probability_weighting = (
                    len(dice_deck_for_total_dice.dice_pairs) / self.cards_left_in_deck
                )

    def _draw_weighted_card(self, player_color: Color) -> DicePair:
        if self.cards_left_in_deck < self.MINIMUM_CARDS_BEFORE_RESHUFFLING:
            self.reshuffle_weighted_dice_deck()
        self.update_weighted_dice_deck_probabilities()
        self._adjust_weighted_dice_deck_based_on_recent_rolls()
        self._adjust_seven_probability_based_on_sevens(player_color)
        return self._get_weighted_dice(player_color)

    def _get_weighted_dice(self, player_color: Color) -> DicePair:
        total_probability_weight = self._get_total_probability_weight()
        if total_probability_weight <= 0:
            return (3, 4)

        target_random_number = random.random() * total_probability_weight
        for dice_deck_for_total_dice in self.weighted_dice_deck:
            if target_random_number <= dice_deck_for_total_dice.probability_weighting:
                if not dice_deck_for_total_dice.dice_pairs:
                    break
                drawn_card = random.choice(dice_deck_for_total_dice.dice_pairs)
                dice_deck_for_total_dice.dice_pairs.remove(drawn_card)

                self.recent_rolls.append(dice_deck_for_total_dice.total_dice)
                dice_deck_for_total_dice.recently_rolled_count += 1
                self.cards_left_in_deck -= 1

                if len(self.recent_rolls) > self.MAXIMUM_RECENT_ROLL_MEMORY:
                    self._update_recently_rolled()
                if dice_deck_for_total_dice.total_dice == 7:
                    self._update_seven_rolls(player_color)
                return drawn_card
            target_random_number -= dice_deck_for_total_dice.probability_weighting

        return (3, 4)

    def _get_total_probability_weight(self) -> float:
        return sum(deck.probability_weighting for deck in self.weighted_dice_deck)

    def _update_recently_rolled(self) -> None:
        total_dice_five_rolls_ago = self.recent_rolls.pop(0)
        index = total_dice_five_rolls_ago - self.INDEX_OFFSET
        self.weighted_dice_deck[index].recently_rolled_count -= 1

    def _adjust_weighted_dice_deck_based_on_recent_rolls(self) -> None:
        for dice_deck_for_total_dice in self.weighted_dice_deck:
            probability_reduction = (
                dice_deck_for_total_dice.recently_rolled_count
                * self.PROBABILITY_REDUCTION_FOR_RECENTLY_ROLLED
            )
            probability_multiplier = 1 - probability_reduction
            dice_deck_for_total_dice.probability_weighting *= probability_multiplier
            if dice_deck_for_total_dice.probability_weighting < 0:
                dice_deck_for_total_dice.probability_weighting = 0.0

    def _init_total_sevens(self, player_color: Color) -> None:
        if player_color in self.total_sevens_rolled_by_player:
            return
        self.total_sevens_rolled_by_player[player_color] = 0

    def _update_seven_rolls(self, player_color: Color) -> None:
        sevens_rolled_by_player = self.total_sevens_rolled_by_player.get(
            player_color, 0
        )
        self.total_sevens_rolled_by_player[player_color] = sevens_rolled_by_player + 1

        if player_color == self.seven_streak_count["player_color"]:
            self.seven_streak_count["streak_count"] += 1
            return

        self.seven_streak_count["player_color"] = player_color
        self.seven_streak_count["streak_count"] = 1

    def _adjust_seven_probability_based_on_sevens(self, player_color: Color) -> None:
        if self.number_of_players < 2:
            return

        streak_adjustment_percentage = self._get_streak_adjustment_constant(
            player_color
        )
        player_sevens_adjustment_percentage = self._get_seven_imbalance_adjustment(
            player_color
        )

        seven_probability_adjustment = (
            1 * player_sevens_adjustment_percentage + streak_adjustment_percentage
        )

        minimum_adjustment = 0.0
        maximum_adjustment = 2.0
        seven_probability_adjustment = max(
            minimum_adjustment, min(maximum_adjustment, seven_probability_adjustment)
        )

        seven_index = 7 - self.INDEX_OFFSET
        self.weighted_dice_deck[
            seven_index
        ].probability_weighting *= seven_probability_adjustment

    def _get_streak_adjustment_constant(self, player: Color) -> float:
        is_streak_for_or_against_player = (
            -1 if self.seven_streak_count["player_color"] == player else 1
        )
        return (
            self.PROBABILITY_REDUCTION_FOR_SEVEN_STREAKS
            * self.seven_streak_count["streak_count"]
            * is_streak_for_or_against_player
        )

    def _get_seven_imbalance_adjustment(self, player_color: Color) -> float:
        total_sevens = self._get_total_sevens_rolled()
        if total_sevens < len(self.total_sevens_rolled_by_player):
            return 1.0

        sevens_per_player = self.total_sevens_rolled_by_player.get(player_color, 0)
        percentage_of_total_sevens = sevens_per_player / total_sevens
        ideal_percentage_of_total_sevens = 1 / len(self.total_sevens_rolled_by_player)

        return 1 + (
            (ideal_percentage_of_total_sevens - percentage_of_total_sevens)
            / ideal_percentage_of_total_sevens
        )

    def _get_total_sevens_rolled(self) -> int:
        return sum(self.total_sevens_rolled_by_player.values())

    @staticmethod
    def get_standard_dice_deck() -> List[StandardDiceDeck]:
        return [
            StandardDiceDeck(2, [(1, 1)]),
            StandardDiceDeck(3, [(1, 2), (2, 1)]),
            StandardDiceDeck(4, [(1, 3), (2, 2), (3, 1)]),
            StandardDiceDeck(5, [(1, 4), (2, 3), (3, 2), (4, 1)]),
            StandardDiceDeck(6, [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]),
            StandardDiceDeck(
                7,
                [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)],
            ),
            StandardDiceDeck(8, [(2, 6), (3, 5), (4, 4), (5, 3), (6, 2)]),
            StandardDiceDeck(9, [(3, 6), (4, 5), (5, 4), (6, 3)]),
            StandardDiceDeck(10, [(4, 6), (5, 5), (6, 4)]),
            StandardDiceDeck(11, [(5, 6), (6, 5)]),
            StandardDiceDeck(12, [(6, 6)]),
        ]

"""Tests for Colonist 1v1 training helpers (rewards, VP cap)."""

from unittest.mock import MagicMock, patch

from catanatron import Color
from catanatron.gym.colonist_rewards import colonist_shaped_reward, make_colonist_shaped_reward
from catanatron.gym.utils import (
    get_tournament_total_return,
    get_victory_points_total_return,
    infer_vps_cap,
)


def test_infer_vps_cap_defaults_to_10_without_attribute():
    g = MagicMock(spec=[])  # no vps_to_win
    assert infer_vps_cap(g) == 10


def test_infer_vps_cap_uses_game_attribute():
    g = MagicMock()
    g.vps_to_win = 15
    assert infer_vps_cap(g) == 15


def test_victory_points_return_not_clamped_at_ten_for_15_vp_game():
    game = MagicMock()
    game.vps_to_win = 15
    game.state.num_turns = 1
    with patch(
        "catanatron.gym.utils.get_actual_victory_points",
        return_value=14,
    ):
        r = get_victory_points_total_return(game, Color.BLUE)
    assert abs(r - 14 * 0.9999**1) < 1e-9


def test_tournament_return_respects_vp_cap():
    game = MagicMock()
    game.vps_to_win = 15
    game.state.num_turns = 0
    game.winning_color.return_value = Color.BLUE
    with patch(
        "catanatron.gym.utils.get_actual_victory_points",
        return_value=14,
    ):
        r = get_tournament_total_return(game, Color.BLUE)
    assert r == 1000 + min(14, 15) * (0.9999**0)


def test_colonist_shaped_terminal_and_vp_delta():
    from catanatron.models.actions import Action
    from catanatron.models.enums import ActionType

    game = MagicMock()
    game.winning_color.return_value = None
    game.state = MagicMock()
    a = Action(Color.BLUE, ActionType.END_TURN, None)
    with patch(
        "catanatron.gym.colonist_rewards.get_actual_victory_points",
        side_effect=[2, 3],
    ):
        r0 = colonist_shaped_reward(a, game, Color.BLUE)
        r1 = colonist_shaped_reward(a, game, Color.BLUE)
    assert r0 != r1  # delta differs between steps

    game2 = MagicMock()
    game2.winning_color.return_value = Color.BLUE
    win_r = colonist_shaped_reward(
        Action(Color.BLUE, ActionType.END_TURN, None),
        game2,
        Color.BLUE,
    )
    assert win_r == 1.0


def test_make_colonist_shaped_visible_factory():
    from catanatron.models.actions import Action
    from catanatron.models.enums import ActionType

    fn = make_colonist_shaped_reward(vp_scale=1.0, use_visible_vp=True)
    game = MagicMock()
    game.winning_color.return_value = None
    with patch(
        "catanatron.gym.colonist_rewards.get_visible_victory_points",
        return_value=4,
    ) as vis:
        with patch(
            "catanatron.gym.colonist_rewards.get_actual_victory_points",
        ) as act:
            fn(Action(Color.BLUE, ActionType.END_TURN, None), game, Color.BLUE)
    vis.assert_called_once()
    act.assert_not_called()

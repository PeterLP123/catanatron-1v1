from pathlib import Path

from catanatron.state_functions import get_actual_victory_points


def infer_vps_cap(game):
    """
    Victory-point ceiling used when scaling returns to match game length.

    Uses ``game.vps_to_win`` when present (all current ``Game`` instances set it).
    Falls back to ``10`` for backward compatibility with older pickled games.
    """
    vps = getattr(game, "vps_to_win", None)
    if vps is not None:
        return int(vps)
    return 10


# DISCOUNT_FACTOR = 0 would mean only focus on immediate reward. Must be < 1. The closer to 1, the more
#   important the future is. 0.99 means future is 100 times more important than immediate reward.
DISCOUNT_FACTOR = 0.99


def simple_total_return(game, color):
    """
    Get the final return for the given color.
    Args:
        game: The game object.
        color: The color of the player.
    Returns:
        float: The final return.
    """
    if game.winning_color() == color:
        return 1.0
    elif game.winning_color() is None:
        return 0.0
    else:
        return -1.0


def get_tournament_total_return(game, p0_color):
    """
    Winning is worth 1000 points, and the number of victory points
    is worth 1 point. The factor (0.9999) ensures a game
    won in less turns is better, and still a Game with 9vps is less
    than 10vps, no matter turns.
    """
    sign = simple_total_return(game, p0_color)
    cap = infer_vps_cap(game)
    points = get_actual_victory_points(game.state, p0_color)
    return sign * 1000 + min(points, cap) * 0.9999**game.state.num_turns


def get_victory_points_total_return(game, p0_color):
    """
    The final reward will be the number of victory points, no matter
    if the game is won or not.
    """
    # This discount factor (0.9999) ensures a game won in less turns
    #   is better, and still a Game with 9vps is less than 10vps,
    #   no matter turns.
    cap = infer_vps_cap(game)
    points = get_actual_victory_points(game.state, p0_color)
    episode_return = min(points, cap)
    return episode_return * 0.9999**game.state.num_turns


def get_victory_point_margin_total_return(game, p0_color):
    """Terminal VP margin from the acting player's perspective.

    In multiplayer data the comparison is against the strongest opponent; for
    this project's 1v1 data that is exactly own VP minus the other player's VP.
    """
    own_points = get_actual_victory_points(game.state, p0_color)
    opponent_points = [
        get_actual_victory_points(game.state, color)
        for color in game.state.colors
        if color != p0_color
    ]
    if not opponent_points:
        raise ValueError("Victory-point margin requires at least two players")
    margin = own_points - max(opponent_points)
    return margin * 0.9999**game.state.num_turns


def populate_matrices(
    samples_df, board_tensors_df, actions_df, rewards_df, main_df, games_directory
):
    directory = Path(games_directory)
    directory.mkdir(parents=True, exist_ok=True)
    first_write = not (directory / "samples.csv.gz").is_file()
    for name, frame in (
        ("samples", samples_df),
        ("board_tensors", board_tensors_df),
        ("actions", actions_df),
        ("rewards", rewards_df),
        ("main", main_df),
    ):
        if name == "board_tensors" and frame is None:
            continue
        frame.to_csv(
            directory / f"{name}.csv.gz",
            mode="a",
            header=first_write,
            index=False,
            compression="gzip",
        )

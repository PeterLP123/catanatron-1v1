"""
Colonist.io-style 1v1 game settings for simulation and model training.

Use ``create_colonist_1v1_game`` or pass ``colonist_1v1=True`` to ``CatanatronEnv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from catanatron.game import Game
from catanatron.models.map import MapType, NumberPlacement, build_map
from catanatron.models.player import Color, Player
from catanatron.state import DiceMode


@dataclass(frozen=True)
class Colonist1v1Settings:
    """Rules aligned with Colonist.io 1v1."""

    num_players: int = 2
    vps_to_win: int = 15
    dice_mode: DiceMode = "balanced"
    friendly_robber: bool = True
    friendly_robber_vp_threshold: int = 2
    friendly_robber_use_visible_vp: bool = True
    discard_limit: int = 9
    map_type: MapType = "BASE"
    number_placement: NumberPlacement = "official_spiral"


COLONIST_1V1_SETTINGS = Colonist1v1Settings()


@dataclass(frozen=True)
class Colonist1v1TrainConfig:
    """Minimal defaults for Colonist 1v1 data generation and RL training."""

    map_type: MapType = COLONIST_1V1_SETTINGS.map_type
    number_placement: NumberPlacement = COLONIST_1V1_SETTINGS.number_placement
    vps_to_win: int = COLONIST_1V1_SETTINGS.vps_to_win
    seed: Optional[int] = None
    # ``catanatron-play`` player strings, e.g. "F,F" or "VP,F"
    teacher_players: str = "F,F"
    output_dir: str = "data/colonist_1v1_parquet"


def validate_colonist_1v1_players(players: Sequence[Player]) -> None:
    if len(players) != COLONIST_1V1_SETTINGS.num_players:
        raise ValueError(
            f"Colonist 1v1 requires exactly {COLONIST_1V1_SETTINGS.num_players} players, "
            f"got {len(players)}"
        )


def create_colonist_1v1_game(
    players: Sequence[Player],
    seed: Optional[int] = None,
    *,
    settings: Colonist1v1Settings = COLONIST_1V1_SETTINGS,
) -> Game:
    """Create a ``Game`` configured for Colonist.io 1v1."""
    validate_colonist_1v1_players(players)
    catan_map = build_map(settings.map_type, settings.number_placement)
    return Game(
        players,
        seed=seed,
        vps_to_win=settings.vps_to_win,
        discard_limit=settings.discard_limit,
        friendly_robber=settings.friendly_robber,
        friendly_robber_vp_threshold=settings.friendly_robber_vp_threshold,
        friendly_robber_use_visible_vp=settings.friendly_robber_use_visible_vp,
        dice_mode=settings.dice_mode,
        catan_map=catan_map,
    )


def colonist_1v1_game_kwargs(
    settings: Colonist1v1Settings = COLONIST_1V1_SETTINGS,
) -> dict:
    """Keyword arguments to pass to ``Game`` for Colonist 1v1."""
    return dict(
        vps_to_win=settings.vps_to_win,
        discard_limit=settings.discard_limit,
        friendly_robber=settings.friendly_robber,
        friendly_robber_vp_threshold=settings.friendly_robber_vp_threshold,
        friendly_robber_use_visible_vp=settings.friendly_robber_use_visible_vp,
        dice_mode=settings.dice_mode,
    )


def default_colonist_1v1_players(
    p0: Player,
    p1: Player,
) -> list[Player]:
    """Return a two-player list (colors must differ)."""
    if p0.color == p1.color:
        raise ValueError("Colonist 1v1 players must have different colors")
    return [p0, p1]

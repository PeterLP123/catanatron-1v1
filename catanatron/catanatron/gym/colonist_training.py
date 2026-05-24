"""
Shared Colonist 1v1 training utilities: MLP builder, BC warm-start, checkpoint league.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from catanatron.models.player import Color, Player

RUN_MARKER_NAME = ".colonist_run_started"


def touch_run_marker(output_dir: Path) -> Path:
    """Mark the start of a data-generation run (used to filter parquet files for BC)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / RUN_MARKER_NAME
    marker.write_text("", encoding="utf-8")
    return marker


def resolve_teacher_parquet_paths(data_dir: Path) -> list[Path]:
    """
    Select parquet files for BC/PPO dataset loading.

    When ``dataset_meta.json`` exists, keep only the newest ``num_games`` files
    (one parquet per game). If ``.colonist_run_started`` exists, prefer files
    written at or after that marker. This avoids mixing stale runs in one folder.
    """
    data_dir = Path(data_dir)
    paths = list(data_dir.glob("*.parquet"))
    if not paths:
        return []

    marker = data_dir / RUN_MARKER_NAME
    if marker.exists():
        since = marker.stat().st_mtime - 1.0
        paths = [p for p in paths if p.stat().st_mtime >= since]

    meta_path = data_dir / "dataset_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        num_games = meta.get("num_games")
        if isinstance(num_games, int) and num_games > 0 and len(paths) > num_games:
            paths = sorted(paths, key=lambda p: p.stat().st_mtime)[-num_games:]

    return sorted(paths, key=lambda p: p.stat().st_mtime)


def load_teacher_parquet(data_dir: Path, *, progress: bool = True):
    """Load teacher parquet logs into a single DataFrame."""
    import pandas as pd

    paths = resolve_teacher_parquet_paths(data_dir)
    if not paths:
        raise FileNotFoundError(f"No .parquet files under {data_dir}")

    if progress:
        print(f"Loading {len(paths)} parquet file(s) from {data_dir} ...")

    frames = []
    for i, path in enumerate(paths, start=1):
        frames.append(pd.read_parquet(path))
        if progress and i % 250 == 0:
            print(f"  loaded {i}/{len(paths)} files")

    df = pd.concat(frames, ignore_index=True)
    if progress:
        print(f"  dataset rows={len(df):,}  cols={len(df.columns)}")
    return df


def build_mlp_layers(
    obs_dim: int,
    n_actions: int,
    hidden_sizes: Sequence[int] = (512, 512),
):
    """Return ``nn.Sequential`` MLP used by BC and Torch eval (requires torch)."""
    from torch import nn

    layers: list[nn.Module] = []
    d_in = obs_dim
    for h in hidden_sizes:
        layers.extend([nn.Linear(d_in, h), nn.ReLU()])
        d_in = h
    layers.append(nn.Linear(d_in, n_actions))
    return nn.Sequential(*layers)


def warmstart_bc_into_maskable_ppo(policy: Any, bc_state: dict) -> int:
    """
    Copy BC ``Sequential`` weights into SB3 ``MaskableActorCriticPolicy``.

    Maps BC layers ``0``, ``2`` to ``mlp_extractor.policy_net`` and ``4`` to ``action_net``.
    Returns number of parameter tensors loaded.
    """
    import torch

    def _copy_linear(dst: torch.nn.Linear, prefix: str) -> int:
        n = 0
        w = bc_state.get(f"{prefix}.weight")
        b = bc_state.get(f"{prefix}.bias")
        if w is not None and dst.weight.shape == w.shape:
            dst.weight.data.copy_(w)
            n += 1
        if b is not None and dst.bias is not None and dst.bias.shape == b.shape:
            dst.bias.data.copy_(b)
            n += 1
        return n

    loaded = 0
    pi_linears = [
        m
        for m in policy.mlp_extractor.policy_net
        if isinstance(m, torch.nn.Linear)
    ]
    for pi_layer, bc_idx in zip(pi_linears, (0, 2)):
        loaded += _copy_linear(pi_layer, str(bc_idx))

    if hasattr(policy, "action_net"):
        loaded += _copy_linear(policy.action_net, "4")

    return loaded


@dataclass
class BcCheckpointMeta:
    obs_dim: int
    n_actions: int
    hidden_sizes: list[int]
    epochs: int
    val_accuracy: Optional[float] = None
    train_rows: int = 0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


def load_bc_checkpoint_meta(path: Path) -> Optional[BcCheckpointMeta]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return BcCheckpointMeta(**data)


class CheckpointLeague:
    """
    Manage last-K PPO checkpoints for league self-play during training.

    Checkpoints are copied into ``<run_dir>/league/`` with stable names.
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        *,
        max_checkpoints: int = 8,
    ):
        self.run_dir = Path(run_dir)
        self.max_checkpoints = max_checkpoints
        self.league_dir = self.run_dir / "league"
        self.league_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.league_dir / "index.json"
        self._entries: list[dict[str, Any]] = self._load_index()

    def _load_index(self) -> list[dict[str, Any]]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return []

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._entries, indent=2))

    def register(self, checkpoint_path: Union[str, Path], *, label: Optional[str] = None) -> str:
        """Copy checkpoint into league pool; prune to ``max_checkpoints``."""
        src = Path(checkpoint_path)
        if not src.exists():
            raise FileNotFoundError(src)

        label = label or src.stem
        dest = self.league_dir / f"{label}{src.suffix}"
        shutil.copy2(src, dest)

        entry = {"path": str(dest), "label": label}
        self._entries = [e for e in self._entries if e["path"] != str(dest)]
        self._entries.append(entry)
        while len(self._entries) > self.max_checkpoints:
            old = self._entries.pop(0)
            try:
                Path(old["path"]).unlink(missing_ok=True)
            except OSError:
                pass

        self._save_index()
        return str(dest)

    def paths(self) -> list[str]:
        return [e["path"] for e in self._entries if Path(e["path"]).exists()]

    def sample_path(self, rng: Optional[np.random.Generator] = None) -> Optional[str]:
        paths = self.paths()
        if not paths:
            return None
        r = rng or np.random.default_rng()
        return str(r.choice(paths))


# Teacher / baseline opponent codes for mixed league sampling.
DEFAULT_LEAGUE_TEACHER_CODES: tuple[str, ...] = ("F", "VP", "W")


def make_mixed_opponent_factory(
    *,
    league: CheckpointLeague,
    teacher_codes: Sequence[str] = DEFAULT_LEAGUE_TEACHER_CODES,
    league_weight: float = 0.5,
    teacher_weight: float = 0.35,
    baseline_weight: float = 0.15,
    baseline_code: str = "W",
    map_type: str = "BASE",
    p1_color: Color = Color.RED,
    rng: Optional[np.random.Generator] = None,
) -> Callable[[], Player]:
    """
    Factory for ``SelfPlayEnv(opponent_factory=..., sample_each_reset=True)``.

    Samples league SB3 checkpoints, classical teachers, or a weak baseline.
    """
    from catanatron.cli.cli_players import parse_cli_string
    from catanatron.players.learned import load_sb3_player

    r = rng or np.random.default_rng()
    player_colors = (Color.BLUE, p1_color)

    def _factory() -> Player:
        roll = r.random()
        if roll < league_weight:
            ckpt = league.sample_path(r)
            if ckpt is not None:
                return load_sb3_player(
                    ckpt,
                    p1_color,
                    map_type=map_type,
                    player_colors=player_colors,
                )
        if roll < league_weight + teacher_weight and teacher_codes:
            code = str(r.choice(list(teacher_codes)))
            return parse_cli_string(f"{code},R")[0]
        return parse_cli_string(f"{baseline_code},R")[0]

    return _factory


def write_dataset_metadata(
    output_dir: Path,
    *,
    teachers: str,
    num_games: int,
    command: str,
    parquet_files: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Write ``dataset_meta.json`` beside generated parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if parquet_files is None:
        parquet_files = len(resolve_teacher_parquet_paths(output_dir))
    meta = {
        "teachers": teachers,
        "num_games": num_games,
        "parquet_files": parquet_files,
        "command": command,
        "colonist_1v1": True,
    }
    if extra:
        meta.update(extra)
    (output_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))

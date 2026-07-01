"""
Shared Colonist 1v1 training utilities: MLP builder, BC warm-start, checkpoint league.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import socket
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

import numpy as np

from catanatron.models.player import Color, Player

RUN_MARKER_NAME = ".colonist_run_started"
MANIFEST_NAME = "run_manifest.json"
EVENTS_NAME = "training_events.jsonl"
MODEL_REGISTRY_NAME = "models_index.jsonl"

# Teacher / baseline opponent codes for mixed league sampling.
DEFAULT_LEAGUE_TEACHER_CODES: tuple[str, ...] = ("F", "VP", "W")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    paths = [p for p in data_dir.glob("*.parquet") if not p.name.startswith(".")]
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


def load_teacher_parquet(
    data_dir: Union[Path, Sequence[Path]], *, progress: bool = True
):
    """Load teacher parquet logs from one or more directories into a single DataFrame."""
    import pandas as pd

    data_dirs = [Path(p) for p in data_dir] if isinstance(data_dir, Sequence) and not isinstance(data_dir, (str, bytes, Path)) else [Path(data_dir)]  # type: ignore[arg-type]
    paths: list[Path] = []
    for d in data_dirs:
        meta_path = d / "dataset_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") not in {None, "complete"}:
                raise ValueError(
                    f"Dataset {d} is {meta.get('status')!r}; resume generation before training"
                )
        paths.extend(resolve_teacher_parquet_paths(d))
    if not paths:
        raise FileNotFoundError(f"No .parquet files under {data_dirs}")

    if progress:
        joined = ", ".join(os.fspath(d) for d in data_dirs)
        print(f"Loading {len(paths)} parquet file(s) from {joined} ...")

    frames = []
    for i, path in enumerate(paths, start=1):
        frames.append(pd.read_parquet(path))
        if progress and i % 250 == 0:
            print(f"  loaded {i}/{len(paths)} files")

    df = pd.concat(frames, ignore_index=True)
    if progress:
        print(f"  dataset rows={len(df):,}  cols={len(df.columns)}")
    return df


# Dataset v2 column names (see catanatron.gym.accumulators).
GAME_ID_COLUMN = "GAME_ID"
SEAT_COLUMN = "SEAT"
PHASE_COLUMN = "PHASE"
NUM_LEGAL_COLUMN = "NUM_LEGAL"
LEGAL_ACTIONS_COLUMN = "LEGAL_ACTIONS"
CANDIDATE_VALUES_COLUMN = "CANDIDATE_VALUES"


def grouped_split_masks(
    game_ids: Sequence,
    val_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 0,
):
    """Split row indices into train/val/test by game so no game leaks across splits.

    Returns three boolean ``numpy`` masks aligned with ``game_ids``. Splitting on
    whole games (not individual rows) is what makes behavioral-cloning validation
    honest: rows from one game are highly correlated, so a row-level split inflates
    the headline accuracy.
    """
    game_ids = np.asarray(game_ids)
    unique = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    n = len(unique)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_games = set(unique[:n_test].tolist())
    val_games = set(unique[n_test : n_test + n_val].tolist())

    test_mask = np.array([g in test_games for g in game_ids], dtype=bool)
    val_mask = np.array([g in val_games for g in game_ids], dtype=bool)
    train_mask = ~(test_mask | val_mask)
    return train_mask, val_mask, test_mask


def _legal_to_list(legal):
    """Normalize a stored LEGAL_ACTIONS cell to a python list of ints."""
    if legal is None:
        return []
    return [int(x) for x in legal]


def decision_metrics(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    action_types: Optional[np.ndarray] = None,
    num_legal: Optional[np.ndarray] = None,
    legal_actions: Optional[Sequence] = None,
    candidate_values: Optional[Sequence] = None,
    topk: Sequence[int] = (1, 3, 5),
):
    """Decision-quality metrics that do not reward forced moves.

    ``logits`` are raw policy scores over the full action space. When the legal
    action set is supplied, accuracy is measured *within the legal candidates*
    (a legal-masked argmax) and restricted to genuine choices (``NUM_LEGAL > 1``),
    so ROLL/END_TURN-style forced rows cannot inflate the score. Without the v2
    columns it degrades to plain top-1 accuracy.

    When ``candidate_values`` (the raw F value of each legal action, aligned with
    ``legal_actions``) is supplied, ``mean_regret`` reports how much value the
    model's legal-masked pick leaves on the table versus the best legal action --
    the Phase 02 exit-gate metric. Regret is normalized per decision to the value
    range of that decision's candidates (0 = picked the best, 1 = picked the
    worst), so it is scale-free and comparable across rows. Lower is better.
    """
    logits = np.asarray(logits)
    y_true = np.asarray(y_true)
    n = len(y_true)
    metrics: dict[str, Any] = {"rows": int(n)}
    if n == 0:
        return metrics

    raw_pred = logits.argmax(axis=1)
    metrics["accuracy"] = float((raw_pred == y_true).mean())

    if num_legal is None:
        return metrics

    num_legal = np.asarray(num_legal)
    choice = num_legal > 1
    metrics["choice_rows"] = int(choice.sum())
    metrics["forced_fraction"] = float((~choice).mean())

    if legal_actions is None:
        if choice.any():
            metrics["choice_accuracy"] = float(
                (raw_pred[choice] == y_true[choice]).mean()
            )
        return metrics

    legal_lists = [_legal_to_list(legal_actions[i]) for i in range(n)]
    max_k = max(topk)
    topk_hits = {k: [] for k in topk}
    masked_correct = []
    per_family_correct: dict[int, list] = {}
    regrets: list = []

    for i in range(n):
        if not choice[i]:
            continue
        legal = legal_lists[i]
        if not legal:
            continue
        legal_arr = np.asarray(legal)
        legal_scores = logits[i, legal_arr]
        order = np.argsort(-legal_scores)
        ranked = legal_arr[order]
        masked_pred = int(ranked[0])
        is_correct = masked_pred == int(y_true[i])
        masked_correct.append(is_correct)
        for k in topk:
            topk_hits[k].append(int(y_true[i]) in ranked[:k].tolist())
        if action_types is not None:
            fam = int(action_types[i])
            per_family_correct.setdefault(fam, []).append(is_correct)
        if candidate_values is not None:
            cand = candidate_values[i]
            # Aligned with legal_arr; order[0] is the model's pick position.
            if cand is not None and len(cand) == len(legal_arr) and len(cand) > 1:
                cand = np.asarray(cand, dtype=float)
                value_range = float(cand.max() - cand.min())
                if value_range > 0:
                    chosen = float(cand[int(order[0])])
                    regrets.append((float(cand.max()) - chosen) / value_range)
                else:
                    regrets.append(0.0)

    if masked_correct:
        metrics["legal_choice_accuracy"] = float(np.mean(masked_correct))
        for k in topk:
            if k <= max_k and topk_hits[k]:
                metrics[f"legal_top{k}_accuracy"] = float(np.mean(topk_hits[k]))
    if per_family_correct:
        metrics["per_action_family_accuracy"] = {
            str(fam): float(np.mean(vals)) for fam, vals in per_family_correct.items()
        }
    if regrets:
        metrics["mean_regret"] = float(np.mean(regrets))
        metrics["regret_rows"] = len(regrets)
    return metrics


# Per-decision-family sampling weights for hard-state training. ROLL is always
# forced (dropped by require_choice anyway); END_TURN is downweighted because it
# is rarely a strategically rich choice; the genuine strategy families are
# oversampled. Unlisted families default to 1.0.
DEFAULT_FAMILY_WEIGHTS = {
    "ROLL": 0.0,
    "END_TURN": 0.25,
    "DISCARD_RESOURCE": 1.0,
    "MOVE_ROBBER": 2.0,
    "BUILD_ROAD": 1.5,
    "BUILD_SETTLEMENT": 2.0,
    "BUILD_CITY": 2.0,
    "MARITIME_TRADE": 2.0,
    "BUY_DEVELOPMENT_CARD": 1.5,
    "PLAY_KNIGHT_CARD": 1.5,
    "PLAY_YEAR_OF_PLENTY": 1.5,
    "PLAY_MONOPOLY": 1.5,
    "PLAY_ROAD_BUILDING": 1.5,
}

SETUP_PHASES = ("BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD")


def hard_state_sample_weights(
    df,
    *,
    family_weights: Optional[dict] = None,
    setup_boost: float = 1.5,
    require_choice: bool = True,
    require_distinct_candidates: bool = False,
):
    """Per-row sampling weights that focus training on real decisions.

    Forced rows (``NUM_LEGAL <= 1``) get weight 0, ROLL/END_TURN are downweighted,
    and the strategy families (robber, build location, maritime trade, dev-card
    timing) plus the high-leverage initial-placement phases are oversampled.
    With ``require_distinct_candidates``, scored rows whose candidates are all
    equal (no value distinction) are dropped too.
    """
    from catanatron.gym.envs.action_space import ACTION_TYPES

    n = len(df)
    fam_w = {**DEFAULT_FAMILY_WEIGHTS, **(family_weights or {})}
    name_by_index = {i: at.name for i, at in enumerate(ACTION_TYPES)}

    action_types = df["ACTION_TYPE"].to_numpy()
    weights = np.array(
        [fam_w.get(name_by_index.get(int(a)), 1.0) for a in action_types],
        dtype=float,
    )

    if setup_boost != 1.0 and PHASE_COLUMN in df.columns:
        is_setup = np.isin(df[PHASE_COLUMN].to_numpy(), SETUP_PHASES)
        weights[is_setup] *= setup_boost

    if require_choice and NUM_LEGAL_COLUMN in df.columns:
        weights[df[NUM_LEGAL_COLUMN].to_numpy() <= 1] = 0.0

    if require_distinct_candidates and CANDIDATE_VALUES_COLUMN in df.columns:
        cands = df[CANDIDATE_VALUES_COLUMN].to_numpy()
        for i in range(n):
            c = cands[i]
            if c is not None and len(c) > 1 and len({float(v) for v in c}) <= 1:
                weights[i] = 0.0

    return weights


def sample_hard_states(df, *, n: Optional[int] = None, seed: int = 0, **kwargs):
    """Resample ``df`` rows in proportion to :func:`hard_state_sample_weights`.

    Returns a new DataFrame (sampling with replacement, so strategy families are
    oversampled). Defaults to as many rows as carry positive weight.
    """
    weights = hard_state_sample_weights(df, **kwargs)
    total = float(weights.sum())
    if total <= 0:
        return df.iloc[[]].copy()
    probs = weights / total
    if n is None:
        n = int((weights > 0).sum())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=True, p=probs)
    return df.iloc[idx].reset_index(drop=True)


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
        m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)
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
    data_dirs: list[str] = field(default_factory=list)
    val_loss: Optional[float] = None

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
            try:
                return json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, OSError):
                return self._entries if hasattr(self, "_entries") else []
        return []

    def _save_index(self) -> None:
        tmp = self._index_path.with_name(f".{self._index_path.name}.tmp")
        tmp.write_text(json.dumps(self._entries, indent=2))
        os.replace(tmp, self._index_path)

    def register(
        self,
        checkpoint_path: Union[str, Path],
        *,
        label: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> str:
        """Copy checkpoint into league pool; prune to ``max_checkpoints``."""
        src = Path(checkpoint_path)
        if not src.exists():
            raise FileNotFoundError(src)

        label = label or src.stem
        dest = self.league_dir / f"{label}{src.suffix}"
        shutil.copy2(src, dest)

        entry = {
            "path": str(dest),
            "label": label,
            "created_at": utc_now_iso(),
            "metrics": metrics or {},
        }
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
        # Subprocess workers hold their own CheckpointLeague object. Reload the
        # file-backed index so newly promoted checkpoints become visible.
        self._entries = self._load_index()
        return [e["path"] for e in self._entries if Path(e["path"]).exists()]

    def sample_path(self, rng: Optional[np.random.Generator] = None) -> Optional[str]:
        paths = self.paths()
        if not paths:
            return None
        r = rng or np.random.default_rng()
        return str(r.choice(paths))

    def entries(self) -> list[dict[str, Any]]:
        self._entries = self._load_index()
        return [dict(e) for e in self._entries if Path(e["path"]).exists()]


@dataclass(frozen=True)
class CurriculumStage:
    """Opponent mix beginning at ``start_step``."""

    start_step: int
    league_weight: float
    teacher_weight: float
    baseline_weight: float
    teacher_codes: tuple[str, ...] = DEFAULT_LEAGUE_TEACHER_CODES
    baseline_code: str = "W"


@dataclass(frozen=True)
class CurriculumSchedule:
    stages: tuple[CurriculumStage, ...]

    def stage_for(self, step: int) -> CurriculumStage:
        active = self.stages[0]
        for stage in self.stages:
            if step >= stage.start_step:
                active = stage
            else:
                break
        return active

    def to_dict(self) -> dict[str, Any]:
        return {"stages": [asdict(stage) for stage in self.stages]}


CURRICULUM_PRESETS: dict[str, CurriculumSchedule] = {
    "balanced": CurriculumSchedule(
        stages=(
            CurriculumStage(
                0, league_weight=0.0, teacher_weight=0.65, baseline_weight=0.35
            ),
            CurriculumStage(
                100_000, league_weight=0.30, teacher_weight=0.55, baseline_weight=0.15
            ),
            CurriculumStage(
                500_000, league_weight=0.55, teacher_weight=0.40, baseline_weight=0.05
            ),
        )
    ),
    "strong": CurriculumSchedule(
        stages=(
            CurriculumStage(
                0,
                league_weight=0.0,
                teacher_weight=0.75,
                baseline_weight=0.25,
                teacher_codes=("VP", "F"),
            ),
            CurriculumStage(
                250_000,
                league_weight=0.35,
                teacher_weight=0.60,
                baseline_weight=0.05,
                teacher_codes=("F", "VP"),
            ),
            CurriculumStage(
                1_000_000,
                league_weight=0.60,
                teacher_weight=0.40,
                baseline_weight=0.0,
                teacher_codes=("F", "G:25"),
            ),
        )
    ),
    "self_play": CurriculumSchedule(
        stages=(
            CurriculumStage(
                0, league_weight=0.25, teacher_weight=0.55, baseline_weight=0.20
            ),
            CurriculumStage(
                250_000, league_weight=0.70, teacher_weight=0.30, baseline_weight=0.0
            ),
        )
    ),
}


def curriculum_from_name(name: str) -> CurriculumSchedule:
    if name not in CURRICULUM_PRESETS:
        valid = ", ".join(sorted(CURRICULUM_PRESETS))
        raise ValueError(f"Unknown curriculum {name!r}; expected one of: {valid}")
    return CURRICULUM_PRESETS[name]


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
    curriculum: Optional[CurriculumSchedule] = None,
    step_getter: Optional[Callable[[], int]] = None,
    telemetry: Optional["TrainingRunTracker"] = None,
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
        nonlocal teacher_codes, league_weight, teacher_weight, baseline_weight, baseline_code
        if curriculum is not None:
            stage = curriculum.stage_for(step_getter() if step_getter else 0)
            teacher_codes = stage.teacher_codes
            league_weight = stage.league_weight
            teacher_weight = stage.teacher_weight
            baseline_weight = stage.baseline_weight
            baseline_code = stage.baseline_code
        roll = r.random()
        if roll < league_weight:
            ckpt = league.sample_path(r)
            if ckpt is not None:
                if telemetry:
                    telemetry.event(
                        "opponent_sampled", source="league", checkpoint=ckpt
                    )
                return load_sb3_player(
                    ckpt,
                    p1_color,
                    map_type=map_type,
                    player_colors=player_colors,
                )
        if roll < league_weight + teacher_weight and teacher_codes:
            code = str(r.choice(list(teacher_codes)))
            if telemetry:
                telemetry.event("opponent_sampled", source="teacher", code=code)
            player = parse_cli_string(code)[0]
            player.color = p1_color
            return player
        if telemetry:
            telemetry.event("opponent_sampled", source="baseline", code=baseline_code)
        player = parse_cli_string(baseline_code)[0]
        player.color = p1_color
        return player

    return _factory


@dataclass
class TrainingPreset:
    timesteps: int
    save_freq: int
    eval_freq: int
    eval_games: int
    n_envs: int
    curriculum: str


TRAINING_PRESETS: dict[str, TrainingPreset] = {
    "smoke": TrainingPreset(20_000, 10_000, 10_000, 10, 1, "balanced"),
    "standard": TrainingPreset(500_000, 50_000, 50_000, 50, 4, "balanced"),
    "strong": TrainingPreset(5_000_000, 100_000, 250_000, 100, 8, "strong"),
    "overnight": TrainingPreset(20_000_000, 250_000, 500_000, 150, 8, "strong"),
}


class TrainingRunTracker:
    """JSON manifest + JSONL event writer used by scripts and the Rich TUI."""

    def __init__(
        self,
        run_dir: Union[str, Path],
        *,
        run_id: Optional[str] = None,
        preset: Optional[str] = None,
        command: Optional[Sequence[str]] = None,
        extra: Optional[dict[str, Any]] = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = (
            run_id
            or f"c1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        )
        self.manifest_path = self.run_dir / MANIFEST_NAME
        self.events_path = self.run_dir / EVENTS_NAME
        self.registry_path = self.run_dir / MODEL_REGISTRY_NAME
        self._manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "host": socket.gethostname(),
            "run_dir": os.fspath(self.run_dir),
            "preset": preset,
            "command": list(command) if command else None,
            "phase": "created",
            "artifacts": {},
        }
        if extra:
            self._manifest.update(extra)
        if self.manifest_path.exists():
            try:
                previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                previous.update(self._manifest)
                self._manifest = previous
            except json.JSONDecodeError:
                pass
        self.write_manifest()

    def write_manifest(self) -> None:
        self._manifest["updated_at"] = utc_now_iso()
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True)
        )

    def update_manifest(self, **kwargs: Any) -> None:
        self._manifest.update(kwargs)
        self.write_manifest()

    def phase(self, name: str, **data: Any) -> None:
        self.update_manifest(phase=name)
        self.event("phase", phase=name, **data)

    def event(self, event_type: str, **data: Any) -> dict[str, Any]:
        row = {
            "time": utc_now_iso(),
            "run_id": self.run_id,
            "type": event_type,
            **data,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        return row


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
        "schema_version": "2.0",
        "status": "complete",
        "teachers": teachers,
        "num_games": num_games,
        "requested_games": num_games,
        "completed_games": num_games,
        "parquet_files": parquet_files,
        "command": command,
        "colonist_1v1": True,
    }
    if extra:
        meta.update(extra)
    path = output_dir / "dataset_meta.json"
    tmp = output_dir / ".dataset_meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.replace(tmp, path)

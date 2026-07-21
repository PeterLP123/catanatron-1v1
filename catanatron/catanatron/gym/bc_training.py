"""Memory-conscious behavioral-cloning datasets, losses, and metrics.

The public training script intentionally stays thin.  This module owns the
parts that need focused tests: grouped shard inspection, batched Parquet
streaming, legal-action losses, deterministic setup, and online validation
metrics.
"""

from __future__ import annotations

import random
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

try:  # Torch is an optional dependency of the core simulator.
    import torch

    _IterableDatasetBase = torch.utils.data.IterableDataset
except ImportError:  # pragma: no cover - exercised by core-only installations
    torch = None

    class _IterableDatasetBase:  # type: ignore[no-redef]
        pass


GAME_ID_COLUMN = "GAME_ID"
NUM_LEGAL_COLUMN = "NUM_LEGAL"
LEGAL_ACTIONS_COLUMN = "LEGAL_ACTIONS"
CANDIDATE_VALUES_COLUMN = "CANDIDATE_VALUES"
DISTILLATION_TARGET_COLUMN = "TEACHER_ACTION"
DISTILLATION_CANDIDATE_VALUES_COLUMN = "CANDIDATE_SCORES"

LossName = Literal["cross_entropy", "legal_ce", "listwise", "hybrid"]


def hash_parquet_shards(
    paths: Sequence[Path], *, progress: bool = True
) -> tuple[list[dict[str, Any]], str]:
    """Hash every selected shard and return a location-independent set hash."""
    from catanatron.gym.provenance import sha256_file

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(paths, start=1):
        path = Path(source).resolve()
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if progress and index % 250 == 0:
            print(f"  hashed {index}/{len(paths)} input shards")
    content_identity = sorted(
        ({key: row[key] for key in ("name", "bytes", "sha256")} for row in rows),
        key=lambda row: (row["name"], row["sha256"], row["bytes"]),
    )
    combined = hashlib.sha256(
        json.dumps(content_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return rows, combined


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch without importing Torch at module import."""
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # ``warn_only`` keeps uncommon unsupported kernels from making the CLI
        # unusable while still exposing nondeterminism to the operator.
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_torch_device(requested: str = "auto"):
    """Resolve ``auto`` to CUDA, then MPS, then CPU."""
    import torch

    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown Torch device {requested!r}")
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def _candidate_logits(logits, targets, legal_indices, legal_mask):
    """Gather legal logits and locate each demonstrated action among them."""
    import torch

    if logits.ndim != 2 or legal_indices.ndim != 2:
        raise ValueError("logits and legal_indices must both be rank-2")
    if legal_indices.shape != legal_mask.shape:
        raise ValueError("legal_indices and legal_mask must have the same shape")
    if logits.shape[0] != legal_indices.shape[0] or targets.shape[0] != logits.shape[0]:
        raise ValueError("batch dimensions do not match")
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("Every row must contain at least one legal action")

    safe_indices = legal_indices.clamp_min(0)
    gathered = logits.gather(1, safe_indices)
    gathered = gathered.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    target_matches = legal_mask & legal_indices.eq(targets.unsqueeze(1))
    counts = target_matches.sum(dim=1)
    if not bool(counts.eq(1).all()):
        bad = counts.ne(1).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Target action must occur exactly once in each legal set; bad rows={bad}"
        )
    local_targets = target_matches.to(torch.int64).argmax(dim=1)
    return gathered, local_targets


def legal_masked_cross_entropy(
    logits,
    targets,
    legal_indices,
    legal_mask,
    *,
    reduction: str = "mean",
):
    """Cross entropy over legal actions only.

    Illegal logits receive no gradient.  A demonstrated action missing from its
    legal set is a corrupt training row and raises instead of silently falling
    back to full-space cross entropy.
    """
    from torch.nn import functional as F

    gathered, local_targets = _candidate_logits(
        logits, targets, legal_indices, legal_mask
    )
    return F.cross_entropy(gathered, local_targets, reduction=reduction)


def candidate_listwise_loss(
    logits,
    legal_indices,
    legal_mask,
    candidate_values,
    candidate_mask,
    *,
    temperature: float = 0.25,
    tie_tolerance: float = 1e-6,
    reduction: str = "mean",
):
    """ListNet-style loss against per-decision candidate values.

    Candidate values are min/max normalized independently for every decision,
    making the target invariant to affine value scale. Values within
    ``tie_tolerance`` are snapped together before the target softmax, so tied
    actions receive exactly equal probability. Rows need at least two scored
    legal actions; unscored/forced rows are excluded and reported in the valid
    mask returned alongside the loss.
    """
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if candidate_values.shape != legal_indices.shape:
        raise ValueError("candidate_values must align with legal_indices")
    if candidate_mask.shape != legal_mask.shape:
        raise ValueError("candidate_mask must align with legal_mask")
    if bool((candidate_mask & ~legal_mask).any()):
        raise ValueError("candidate values cannot exist for padded legal actions")

    safe_indices = legal_indices.clamp_min(0)
    predicted = logits.gather(1, safe_indices)
    valid = candidate_mask.sum(dim=1).ge(2) & candidate_mask.eq(legal_mask).all(dim=1)
    if not bool(valid.any()):
        # Preserve a differentiable scalar so callers can aggregate batches and
        # decide whether the whole epoch had any scored rows.
        return logits.sum() * 0.0, valid

    pred = predicted[valid]
    values = candidate_values[valid]
    mask = candidate_mask[valid]
    large = torch.finfo(values.dtype).max
    row_min = values.masked_fill(~mask, large).min(dim=1, keepdim=True).values
    row_max = values.masked_fill(~mask, -large).max(dim=1, keepdim=True).values
    span = (row_max - row_min).clamp_min(torch.finfo(values.dtype).eps)
    normalized = (values - row_min) / span

    # Quantization is deliberate tie handling: values closer than the declared
    # tolerance become identical soft targets instead of receiving arbitrary
    # ordering from floating-point noise.
    if tie_tolerance > 0:
        normalized = torch.round(normalized / tie_tolerance) * tie_tolerance
    target_logits = (normalized / temperature).masked_fill(~mask, -large)
    target_probs = torch.softmax(target_logits, dim=1)
    pred_log_probs = torch.log_softmax(pred.masked_fill(~mask, -large), dim=1)
    row_loss = -(target_probs * pred_log_probs).sum(dim=1)

    if reduction == "none":
        return row_loss, valid
    if reduction == "sum":
        return row_loss.sum(), valid
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction {reduction!r}")
    return row_loss.mean(), valid


def padded_decision_columns(
    legal_actions: Sequence,
    candidate_values: Optional[Sequence] = None,
):
    """Convert variable-length legal/candidate columns to padded tensors."""
    import torch

    legal = [list(map(int, xs)) if xs is not None else [] for xs in legal_actions]
    width = max((len(xs) for xs in legal), default=0)
    if width == 0:
        width = 1
    legal_indices = torch.full((len(legal), width), -1, dtype=torch.long)
    legal_mask = torch.zeros((len(legal), width), dtype=torch.bool)
    values = torch.zeros((len(legal), width), dtype=torch.float32)
    value_mask = torch.zeros((len(legal), width), dtype=torch.bool)
    candidate_values = (
        candidate_values if candidate_values is not None else [()] * len(legal)
    )

    for row, actions in enumerate(legal):
        if actions:
            legal_indices[row, : len(actions)] = torch.as_tensor(actions)
            legal_mask[row, : len(actions)] = True
        candidates = candidate_values[row]
        if candidates is None:
            continue
        candidates = list(map(float, candidates))
        if candidates and len(candidates) != len(actions):
            raise ValueError(
                f"Candidate values must align with legal actions at row {row}: "
                f"{len(candidates)} != {len(actions)}"
            )
        if candidates:
            candidate_tensor = torch.as_tensor(candidates, dtype=torch.float32)
            finite = torch.isfinite(candidate_tensor)
            values[row, : len(candidates)] = torch.where(
                finite, candidate_tensor, torch.zeros_like(candidate_tensor)
            )
            value_mask[row, : len(candidates)] = finite
    return legal_indices, legal_mask, values, value_mask


@dataclass(frozen=True)
class ParquetDatasetPlan:
    """Immutable grouped split and schema discovered without loading features."""

    paths: tuple[Path, ...]
    feature_columns: tuple[str, ...]
    available_columns: frozenset[str]
    has_game_ids: bool
    train_groups: frozenset[str]
    val_groups: frozenset[str]
    test_groups: frozenset[str]
    rows_by_group: dict[str, int]
    path_groups: dict[Path, frozenset[str]]
    path_target_columns: dict[Path, str]
    path_candidate_value_columns: dict[Path, str]

    def groups_for(self, split: str) -> frozenset[str]:
        if split == "train":
            return self.train_groups
        if split == "val":
            return self.val_groups
        if split == "test":
            return self.test_groups
        raise ValueError(f"Unknown split {split!r}")

    def rows_for(self, split: str) -> int:
        return sum(self.rows_by_group.get(group, 0) for group in self.groups_for(split))


def combine_parquet_dataset_plans(
    plans: Sequence[ParquetDatasetPlan],
) -> ParquetDatasetPlan:
    """Combine independently split corpora without reshuffling earlier games.

    DAgger adds new student-visited games over time. Splitting the entire aggregate
    again would move frozen base games between train/validation/test whenever an
    iteration is appended. Inspecting each corpus independently and then combining
    their plans preserves the base split while giving each augmentation its own
    deterministic whole-game holdouts.
    """
    plans = tuple(plans)
    if not plans:
        raise ValueError("At least one Parquet dataset plan is required")
    first = plans[0]
    paths: list[Path] = []
    available_columns = set(first.available_columns)
    train_groups: set[str] = set()
    val_groups: set[str] = set()
    test_groups: set[str] = set()
    rows_by_group: dict[str, int] = {}
    path_groups: dict[Path, frozenset[str]] = {}
    path_target_columns: dict[Path, str] = {}
    path_candidate_value_columns: dict[Path, str] = {}

    for plan in plans:
        if plan.feature_columns != first.feature_columns:
            raise ValueError("Cannot combine plans with different feature schemas")
        if plan.has_game_ids != first.has_game_ids:
            raise ValueError("Cannot combine grouped and legacy Parquet plans")
        duplicate_paths = set(paths) & set(plan.paths)
        if duplicate_paths:
            raise ValueError(f"Duplicate Parquet paths across plans: {duplicate_paths}")
        duplicate_groups = set(rows_by_group) & set(plan.rows_by_group)
        if duplicate_groups:
            raise ValueError(
                "Game IDs collide across independently split corpora: "
                f"{sorted(duplicate_groups)[:5]}"
            )
        paths.extend(plan.paths)
        available_columns.intersection_update(plan.available_columns)
        train_groups.update(plan.train_groups)
        val_groups.update(plan.val_groups)
        test_groups.update(plan.test_groups)
        rows_by_group.update(plan.rows_by_group)
        path_groups.update(plan.path_groups)
        path_target_columns.update(plan.path_target_columns)
        path_candidate_value_columns.update(plan.path_candidate_value_columns)

    return ParquetDatasetPlan(
        paths=tuple(paths),
        feature_columns=first.feature_columns,
        available_columns=frozenset(available_columns),
        has_game_ids=first.has_game_ids,
        train_groups=frozenset(train_groups),
        val_groups=frozenset(val_groups),
        test_groups=frozenset(test_groups),
        rows_by_group=rows_by_group,
        path_groups=path_groups,
        path_target_columns=path_target_columns,
        path_candidate_value_columns=path_candidate_value_columns,
    )


def inspect_parquet_dataset(
    paths: Sequence[Path],
    *,
    val_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 0,
) -> ParquetDatasetPlan:
    """Inspect Parquet metadata and GAME_ID only; never concatenate features."""
    import pandas as pd
    import pyarrow.parquet as pq

    paths = tuple(Path(path) for path in paths)
    if not paths:
        raise FileNotFoundError("No Parquet shards supplied")
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    first_columns = tuple(pq.ParquetFile(paths[0]).schema_arrow.names)
    feature_columns = tuple(sorted(c for c in first_columns if c.startswith("F_")))
    if not feature_columns:
        raise ValueError("No F_* feature columns found (vector teacher logs)")
    has_game_ids = GAME_ID_COLUMN in first_columns
    path_groups: dict[Path, frozenset[str]] = {}
    path_target_columns: dict[Path, str] = {}
    path_candidate_value_columns: dict[Path, str] = {}
    rows_by_group: Counter[str] = Counter()
    common_columns: set[str] | None = None

    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = tuple(parquet.schema_arrow.names)
        shard_features = tuple(sorted(c for c in columns if c.startswith("F_")))
        if shard_features != feature_columns:
            raise ValueError(f"Feature schema mismatch in {path}")
        if (GAME_ID_COLUMN in columns) != has_game_ids:
            raise ValueError("Cannot mix legacy and GAME_ID-aware Parquet shards")
        target_column = next(
            (
                candidate
                for candidate in ("ACTION", DISTILLATION_TARGET_COLUMN)
                if candidate in columns
            ),
            None,
        )
        if target_column is None:
            raise ValueError(
                f"Parquet must include ACTION or {DISTILLATION_TARGET_COLUMN}: {path}"
            )
        path_target_columns[path] = target_column

        candidate_column = next(
            (
                candidate
                for candidate in (
                    CANDIDATE_VALUES_COLUMN,
                    DISTILLATION_CANDIDATE_VALUES_COLUMN,
                )
                if candidate in columns
            ),
            None,
        )
        if candidate_column is not None:
            path_candidate_value_columns[path] = candidate_column

        logical_columns = set(columns)
        logical_columns.add("ACTION")
        if candidate_column is not None:
            logical_columns.add(CANDIDATE_VALUES_COLUMN)
        if common_columns is None:
            common_columns = logical_columns
        else:
            common_columns.intersection_update(logical_columns)
        if has_game_ids:
            game_ids = pd.read_parquet(path, columns=[GAME_ID_COLUMN])[GAME_ID_COLUMN]
            counts = game_ids.astype(str).value_counts()
            groups = frozenset(counts.index.tolist())
            rows_by_group.update(
                {str(group): int(count) for group, count in counts.items()}
            )
        else:
            group = str(path.resolve())
            groups = frozenset((group,))
            rows_by_group[group] += parquet.metadata.num_rows
        path_groups[path] = groups

    groups = np.asarray(sorted(rows_by_group), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    requested_holdouts = int(test_fraction > 0) + int(val_fraction > 0)
    minimum_groups = 1 + requested_holdouts
    if len(groups) < minimum_groups:
        raise ValueError(
            f"Dataset has {len(groups)} independent group(s), but the requested "
            f"validation/test fractions need at least {minimum_groups}. "
            "Add games or explicitly set the unavailable holdout fraction to 0."
        )
    n_test = max(1, int(len(groups) * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, int(len(groups) * val_fraction)) if val_fraction > 0 else 0
    test_groups = frozenset(map(str, groups[:n_test]))
    val_groups = frozenset(map(str, groups[n_test : n_test + n_val]))
    train_groups = frozenset(map(str, groups[n_test + n_val :]))
    return ParquetDatasetPlan(
        paths=paths,
        feature_columns=feature_columns,
        available_columns=frozenset(common_columns or ()),
        has_game_ids=has_game_ids,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        rows_by_group=dict(rows_by_group),
        path_groups=path_groups,
        path_target_columns=path_target_columns,
        path_candidate_value_columns=path_candidate_value_columns,
    )


def _identity(value):
    return value


class ParquetDecisionBatches(_IterableDatasetBase):
    """IterableDataset that loads at most one Parquet shard plus one batch."""

    def __init__(
        self,
        plan: ParquetDatasetPlan,
        split: Literal["train", "val", "test"],
        *,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = False,
        sample_weight_fn: Optional[Callable[[Any], np.ndarray]] = None,
        path_weights: Optional[Mapping[Path, float]] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if torch is None:
            raise ImportError("ParquetDecisionBatches requires the 'colonist' extra")
        super().__init__()
        self.plan = plan
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.sample_weight_fn = sample_weight_fn
        self.path_weights = {
            Path(path): float(weight) for path, weight in (path_weights or {}).items()
        }
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in self.path_weights.values()
        ):
            raise ValueError("Parquet path weights must be positive and finite")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _paths_for_worker(self) -> list[Path]:
        import torch

        wanted = self.plan.groups_for(self.split)
        paths = [p for p in self.plan.paths if self.plan.path_groups[p] & wanted]
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            paths = paths[worker.id :: worker.num_workers]
        if self.shuffle:
            rng = np.random.default_rng(
                self.seed + self.epoch + (worker.id if worker else 0)
            )
            rng.shuffle(paths)
        return paths

    def __iter__(self) -> Iterable[dict[str, Any]]:
        import pandas as pd
        import torch

        wanted = self.plan.groups_for(self.split)
        rng = np.random.default_rng(self.seed + self.epoch)
        for path in self._paths_for_worker():
            target_column = self.plan.path_target_columns[path]
            candidate_column = self.plan.path_candidate_value_columns.get(path)
            read_columns = [*self.plan.feature_columns, target_column]
            read_columns.extend(
                c
                for c in (
                    "ACTION_TYPE",
                    "PHASE",
                    GAME_ID_COLUMN,
                    NUM_LEGAL_COLUMN,
                    LEGAL_ACTIONS_COLUMN,
                )
                if c in self.plan.available_columns
            )
            if candidate_column is not None and CANDIDATE_VALUES_COLUMN in (
                self.plan.available_columns
            ):
                read_columns.append(candidate_column)
            frame = pd.read_parquet(path, columns=read_columns)
            rename_columns = {}
            if target_column != "ACTION":
                rename_columns[target_column] = "ACTION"
            if (
                candidate_column is not None
                and candidate_column != CANDIDATE_VALUES_COLUMN
            ):
                rename_columns[candidate_column] = CANDIDATE_VALUES_COLUMN
            if rename_columns:
                frame = frame.rename(columns=rename_columns)
            if self.plan.has_game_ids:
                frame = frame[frame[GAME_ID_COLUMN].astype(str).isin(wanted)]
            elif str(path.resolve()) not in wanted:
                continue
            if frame.empty:
                continue
            if self.sample_weight_fn is None:
                weights = np.ones(len(frame), dtype=np.float32)
            else:
                weights = np.asarray(self.sample_weight_fn(frame), dtype=np.float32)
                keep = weights > 0
                frame = frame.iloc[np.flatnonzero(keep)]
                weights = weights[keep]
            weights *= self.path_weights.get(path, 1.0)
            if frame.empty:
                continue
            order = np.arange(len(frame))
            if self.shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                positions = order[start : start + self.batch_size]
                chunk = frame.iloc[positions]
                legal = (
                    chunk[LEGAL_ACTIONS_COLUMN].tolist()
                    if LEGAL_ACTIONS_COLUMN in chunk
                    else [[int(action)] for action in chunk["ACTION"]]
                )
                candidates = (
                    chunk[CANDIDATE_VALUES_COLUMN].tolist()
                    if CANDIDATE_VALUES_COLUMN in chunk
                    else None
                )
                legal_indices, legal_mask, values, value_mask = padded_decision_columns(
                    legal, candidates
                )
                yield {
                    "features": torch.from_numpy(
                        chunk.loc[:, self.plan.feature_columns]
                        .to_numpy(np.float32)
                        .copy()
                    ),
                    "targets": torch.from_numpy(
                        chunk["ACTION"].to_numpy(np.int64).copy()
                    ),
                    "action_types": (
                        chunk["ACTION_TYPE"].to_numpy()
                        if "ACTION_TYPE" in chunk
                        else None
                    ),
                    "num_legal": (
                        chunk[NUM_LEGAL_COLUMN].to_numpy()
                        if NUM_LEGAL_COLUMN in chunk
                        else np.asarray([len(actions) for actions in legal])
                    ),
                    "legal_actions": legal,
                    "candidate_values_raw": candidates,
                    "has_decision_metadata": LEGAL_ACTIONS_COLUMN in chunk,
                    "legal_indices": legal_indices,
                    "legal_mask": legal_mask,
                    "candidate_values": values,
                    "candidate_mask": value_mask,
                    "sample_weights": torch.from_numpy(weights[positions].copy()),
                }

    def loader(self, *, num_workers: int = 0):
        """Return a DataLoader while preserving already-batched dictionaries."""
        if torch is None:  # pragma: no cover - guarded by __init__
            raise ImportError("ParquetDecisionBatches requires the 'colonist' extra")
        return torch.utils.data.DataLoader(
            self,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=_identity,
        )


class DecisionMetricAccumulator:
    """Online equivalent of ``decision_metrics`` for batched validation."""

    def __init__(self, *, topk: Sequence[int] = (1, 3, 5)):
        self.topk = tuple(topk)
        self.rows = 0
        self.raw_correct = 0
        self.choice_rows = 0
        self.masked_rows = 0
        self.masked_correct = 0
        self.topk_hits = Counter()
        self.family_correct: Counter[int] = Counter()
        self.family_rows: Counter[int] = Counter()
        self.regret_sum = 0.0
        self.regret_rows = 0

    def update(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        *,
        action_types: Optional[np.ndarray] = None,
        num_legal: Optional[np.ndarray] = None,
        legal_actions: Optional[Sequence] = None,
        candidate_values: Optional[Sequence] = None,
    ) -> None:
        logits = np.asarray(logits)
        y_true = np.asarray(y_true)
        predictions = logits.argmax(axis=1)
        self.rows += len(y_true)
        self.raw_correct += int((predictions == y_true).sum())
        if num_legal is None:
            return
        choice = np.asarray(num_legal) > 1
        self.choice_rows += int(choice.sum())
        if legal_actions is None:
            return
        for i in np.flatnonzero(choice):
            legal = list(map(int, legal_actions[i]))
            if not legal:
                continue
            legal_arr = np.asarray(legal)
            order = np.argsort(-logits[i, legal_arr])
            ranked = legal_arr[order]
            correct = int(ranked[0]) == int(y_true[i])
            self.masked_rows += 1
            self.masked_correct += int(correct)
            for k in self.topk:
                self.topk_hits[k] += int(int(y_true[i]) in ranked[:k])
            if action_types is not None:
                family = int(action_types[i])
                self.family_correct[family] += int(correct)
                self.family_rows[family] += 1
            if candidate_values is not None:
                candidates = candidate_values[i]
                if (
                    candidates is not None
                    and len(candidates) == len(legal)
                    and len(legal) > 1
                ):
                    values = np.asarray(candidates, dtype=float)
                    span = float(values.max() - values.min())
                    chosen = float(values[int(order[0])])
                    self.regret_sum += (
                        (float(values.max()) - chosen) / span if span > 0 else 0.0
                    )
                    self.regret_rows += 1

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {"rows": self.rows}
        if not self.rows:
            return result
        result["accuracy"] = self.raw_correct / self.rows
        result["choice_rows"] = self.choice_rows
        result["forced_fraction"] = (self.rows - self.choice_rows) / self.rows
        if self.masked_rows:
            result["legal_choice_accuracy"] = self.masked_correct / self.masked_rows
            for k in self.topk:
                result[f"legal_top{k}_accuracy"] = self.topk_hits[k] / self.masked_rows
        if self.family_rows:
            result["per_action_family_accuracy"] = {
                str(family): self.family_correct[family] / rows
                for family, rows in self.family_rows.items()
            }
        if self.regret_rows:
            result["mean_regret"] = self.regret_sum / self.regret_rows
            result["regret_rows"] = self.regret_rows
        return result

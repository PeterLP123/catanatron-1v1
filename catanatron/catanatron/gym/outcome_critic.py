"""Streaming whole-game data and metrics for a separate outcome critic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from catanatron.gym.bc_training import (
    GAME_ID_COLUMN,
    ParquetDatasetPlan,
    inspect_parquet_corpora,
)
from catanatron.gym.outcome_target_audit import load_outcome_target_index

try:
    import torch

    _IterableDatasetBase = torch.utils.data.IterableDataset
except ImportError:  # pragma: no cover - core-only installation
    torch = None

    class _IterableDatasetBase:  # type: ignore[no-redef]
        pass


def _identity(value):
    return value


def build_outcome_dataset_plan(
    corpora: Sequence[Sequence[Path]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[
    ParquetDatasetPlan,
    dict[tuple[str, str, int], tuple[float, float]],
]:
    """Resolve targets, then reproduce the BC whole-game split contract."""
    corpus_paths = []
    targets: dict[tuple[str, str, int], tuple[float, float]] = {}
    for inputs in corpora:
        paths, current = load_outcome_target_index(inputs)
        corpus_paths.append(paths)
        duplicate = set(targets) & set(current)
        if duplicate:
            raise ValueError(
                f"Duplicate outcome target identities: {sorted(duplicate)[:5]}"
            )
        targets.update(current)
    plan = inspect_parquet_corpora(
        corpus_paths,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return plan, targets


class OutcomeCriticBatches(_IterableDatasetBase):
    """Stream feature batches with manifest/direct terminal targets."""

    def __init__(
        self,
        plan: ParquetDatasetPlan,
        targets: Mapping[tuple[str, str, int], tuple[float, float]],
        split: Literal["train", "val", "test"],
        *,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("OutcomeCriticBatches requires the 'colonist' extra")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        super().__init__()
        self.plan = plan
        self.targets = dict(targets)
        self.split = split
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        try:
            self.public_p0_index = plan.feature_columns.index("F_P0_PUBLIC_VPS")
            self.public_p1_index = plan.feature_columns.index("F_P1_PUBLIC_VPS")
        except ValueError as exc:
            raise ValueError("Outcome critic requires public VP features") from exc

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _paths_for_worker(self) -> list[Path]:
        import torch

        wanted = self.plan.groups_for(self.split)
        paths = [
            path for path in self.plan.paths if self.plan.path_groups[path] & wanted
        ]
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
            columns = [*self.plan.feature_columns, GAME_ID_COLUMN, "SEAT"]
            frame = pd.read_parquet(path, columns=columns)
            frame = frame[frame[GAME_ID_COLUMN].astype(str).isin(wanted)]
            if frame.empty:
                continue
            path_key = str(path.resolve())
            target_rows = [
                self.targets.get((path_key, str(game_id), int(seat)))
                for game_id, seat in zip(frame[GAME_ID_COLUMN], frame["SEAT"])
            ]
            if any(target is None for target in target_rows):
                missing = sum(target is None for target in target_rows)
                raise ValueError(
                    f"Missing outcome targets for {missing} rows in {path}"
                )
            targets = np.asarray(target_rows, dtype=np.float32)
            order = np.arange(len(frame))
            if self.shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                positions = order[start : start + self.batch_size]
                features = (
                    frame.iloc[positions]
                    .loc[:, self.plan.feature_columns]
                    .to_numpy(np.float32)
                    .copy()
                )
                batch_targets = targets[positions]
                win_raw = batch_targets[:, 0]
                win_mask = np.isfinite(win_raw)
                win_binary = np.where(win_mask, (win_raw + 1.0) / 2.0, 0.0)
                margin_raw = batch_targets[:, 1]
                margin_mask = np.isfinite(margin_raw)
                yield {
                    "features": torch.from_numpy(features),
                    "win_targets": torch.from_numpy(win_binary.astype(np.float32)),
                    "win_mask": torch.from_numpy(win_mask),
                    "margin_targets": torch.from_numpy(
                        np.where(margin_mask, margin_raw, 0.0).astype(np.float32)
                    ),
                    "margin_mask": torch.from_numpy(margin_mask),
                    "public_vp_diff": torch.from_numpy(
                        (
                            features[:, self.public_p0_index]
                            - features[:, self.public_p1_index]
                        ).astype(np.float32)
                    ),
                }

    def loader(self, *, num_workers: int = 0):
        import torch

        return torch.utils.data.DataLoader(
            self,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=_identity,
        )


def outcome_critic_loss(
    output,
    batch: Mapping[str, Any],
    *,
    margin_weight: float,
):
    """Masked BCE plus Smooth-L1 margin loss."""
    from torch.nn import functional as F

    if margin_weight < 0:
        raise ValueError("margin_weight must be non-negative")
    win_mask = batch["win_mask"]
    if not bool(win_mask.any()):
        raise ValueError("Outcome critic batch has no win targets")
    win_loss = F.binary_cross_entropy_with_logits(
        output.win_logit[win_mask], batch["win_targets"][win_mask]
    )
    margin_mask = batch["margin_mask"]
    if bool(margin_mask.any()):
        margin_loss = F.smooth_l1_loss(
            output.vp_margin[margin_mask], batch["margin_targets"][margin_mask]
        )
    else:
        margin_loss = output.vp_margin.sum() * 0.0
    return win_loss + margin_weight * margin_loss, win_loss, margin_loss


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    import pandas as pd

    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if not n_positive or not n_negative:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


class OutcomeMetricAccumulator:
    """Collect bounded holdout arrays and compare against public VP baselines."""

    def __init__(self) -> None:
        self.win_logits: list[np.ndarray] = []
        self.win_targets: list[np.ndarray] = []
        self.win_public: list[np.ndarray] = []
        self.margin_predictions: list[np.ndarray] = []
        self.margin_targets: list[np.ndarray] = []
        self.margin_public: list[np.ndarray] = []
        self.rows = 0

    def update(self, output, batch: Mapping[str, Any]) -> None:
        self.rows += len(batch["features"])
        public = batch["public_vp_diff"].detach().cpu().numpy()
        win_mask = batch["win_mask"].detach().cpu().numpy().astype(bool)
        if bool(win_mask.any()):
            self.win_logits.append(output.win_logit.detach().cpu().numpy()[win_mask])
            self.win_targets.append(
                batch["win_targets"].detach().cpu().numpy()[win_mask]
            )
            self.win_public.append(public[win_mask])
        margin_mask = batch["margin_mask"].detach().cpu().numpy().astype(bool)
        if bool(margin_mask.any()):
            self.margin_predictions.append(
                output.vp_margin.detach().cpu().numpy()[margin_mask]
            )
            self.margin_targets.append(
                batch["margin_targets"].detach().cpu().numpy()[margin_mask]
            )
            self.margin_public.append(public[margin_mask])

    def finalize(self) -> dict[str, Any]:
        result: dict[str, Any] = {"rows": self.rows}
        if self.win_targets:
            logits = np.concatenate(self.win_logits).astype(float)
            targets = np.concatenate(self.win_targets).astype(int)
            public = np.concatenate(self.win_public).astype(float)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            public_probabilities = 1.0 / (1.0 + np.exp(-np.clip(public, -30.0, 30.0)))
            prevalence = float(targets.mean())
            epsilon = np.finfo(float).eps
            result["win"] = {
                "rows": len(targets),
                "auc": _binary_auc(targets, logits),
                "brier": float(np.mean((probabilities - targets) ** 2)),
                "log_loss": float(
                    -np.mean(
                        targets * np.log(np.clip(probabilities, epsilon, 1.0))
                        + (1 - targets)
                        * np.log(np.clip(1.0 - probabilities, epsilon, 1.0))
                    )
                ),
                "accuracy": float(((probabilities >= 0.5) == targets).mean()),
                "public_vp_baseline": {
                    "auc": _binary_auc(targets, public),
                    "brier": float(np.mean((public_probabilities - targets) ** 2)),
                },
                "constant_prevalence_brier": float(
                    np.mean((np.full(len(targets), prevalence) - targets) ** 2)
                ),
            }
        else:
            result["win"] = None
        if self.margin_targets:
            predictions = np.concatenate(self.margin_predictions).astype(float)
            targets = np.concatenate(self.margin_targets).astype(float)
            public = np.concatenate(self.margin_public).astype(float)
            errors = predictions - targets
            public_errors = public - targets
            result["vp_margin"] = {
                "rows": len(targets),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "public_vp_baseline": {
                    "mae": float(np.mean(np.abs(public_errors))),
                    "rmse": float(np.sqrt(np.mean(public_errors**2))),
                },
            }
        else:
            result["vp_margin"] = None
        return result


def critic_gate_deltas(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return critic-minus-public deltas; negative error and positive AUC win."""
    win = metrics["win"]
    margin = metrics["vp_margin"]
    if win is None or margin is None:
        raise ValueError("Critic gates require both win and margin holdout targets")
    return {
        "win_auc": float(win["auc"] - win["public_vp_baseline"]["auc"]),
        "win_brier": float(win["brier"] - win["public_vp_baseline"]["brier"]),
        "vp_margin_mae": float(margin["mae"] - margin["public_vp_baseline"]["mae"]),
        "vp_margin_rmse": float(margin["rmse"] - margin["public_vp_baseline"]["rmse"]),
    }

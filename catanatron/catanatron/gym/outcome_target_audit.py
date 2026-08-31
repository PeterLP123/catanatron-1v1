"""Audit whether BC corpora can support leakage-safe outcome learning."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from catanatron.gym.bc_training import (
    VP_MARGIN_TARGET_COLUMN,
    WIN_VALUE_TARGET_COLUMN,
    hash_parquet_shards,
)
from catanatron.gym.provenance import sha256_file

OWN_VP_RETURN_COLUMN = "VICTORY_POINTS_RETURN"
PUBLIC_VP_COLUMNS = ("F_P0_PUBLIC_VPS", "F_P1_PUBLIC_VPS")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iteration_manifests(path: Path) -> list[Path]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = _read_json(manifest_path)
    if "shards" in manifest and "iteration" in manifest:
        return [manifest_path]
    manifests = []
    for item in manifest.get("iterations", []):
        candidate = path / item["manifest"]
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing iteration manifest: {candidate}")
        expected = item.get("manifest_sha256")
        if expected is not None and sha256_file(candidate) != expected:
            raise ValueError(f"Iteration manifest hash mismatch: {candidate}")
        manifests.append(candidate)
    return manifests


def _resolve_input(path: Path) -> tuple[list[Path], list[Path]]:
    path = path.resolve()
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"Outcome audit input is not Parquet: {path}")
        return [path], []
    if not path.is_dir():
        raise FileNotFoundError(f"Outcome audit input does not exist: {path}")

    manifests = _iteration_manifests(path)
    if manifests:
        shards: list[Path] = []
        for manifest_path in manifests:
            manifest = _read_json(manifest_path)
            root = manifest_path.parent.parent
            for item in manifest.get("shards", []):
                shard = root / item["path"]
                if not shard.is_file():
                    raise FileNotFoundError(f"Missing distillation shard: {shard}")
                expected = item.get("sha256")
                if expected is not None and sha256_file(shard) != expected:
                    raise ValueError(f"Distillation shard hash mismatch: {shard}")
                shards.append(shard.resolve())
        if not shards:
            raise ValueError(f"Distillation input has no shards: {path}")
        return shards, manifests

    shards = sorted(item.resolve() for item in path.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found under {path}")
    return shards, []


def _manifest_games(manifests: Sequence[Path]) -> dict[tuple[int, int], dict[str, Any]]:
    games: dict[tuple[int, int], dict[str, Any]] = {}
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        iteration = int(manifest["iteration"])
        for game in manifest.get("games", []):
            key = (iteration, int(game["game_index"]))
            if key in games:
                raise ValueError(f"Duplicate distillation game identity: {key}")
            games[key] = dict(game)
    return games


def _finite_numeric(series):
    import pandas as pd

    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(np.isfinite(values), np.nan)


def _load_input_frame(path: Path):
    import pandas as pd
    import pyarrow.parquet as pq

    shards, manifests = _resolve_input(path)
    wanted = {
        "GAME_ID",
        "GAME_INDEX",
        "ITERATION",
        "SEAT",
        "COLOR",
        WIN_VALUE_TARGET_COLUMN,
        VP_MARGIN_TARGET_COLUMN,
        OWN_VP_RETURN_COLUMN,
        *PUBLIC_VP_COLUMNS,
    }
    frames = []
    for shard in shards:
        columns = set(pq.ParquetFile(shard).schema_arrow.names)
        selected = sorted(wanted & columns)
        frame = pd.read_parquet(shard, columns=selected)
        frame["_SOURCE_SHARD"] = str(shard)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "GAME_ID" in frame:
        split_ids = frame["GAME_ID"].astype(str)
    else:
        split_ids = frame["_SOURCE_SHARD"].astype(str)
    source_namespace = str(path.resolve())
    frame["_SPLIT_GROUP"] = split_ids
    frame["_TRAJECTORY_GROUP"] = source_namespace + "::" + split_ids

    frame["_WIN_TARGET"] = np.nan
    frame["_WIN_SOURCE"] = "missing"
    if WIN_VALUE_TARGET_COLUMN in frame:
        direct = _finite_numeric(frame[WIN_VALUE_TARGET_COLUMN])
        valid = direct.notna()
        frame.loc[valid, "_WIN_TARGET"] = direct[valid]
        frame.loc[valid, "_WIN_SOURCE"] = "direct"

    frame["_MARGIN_TARGET"] = np.nan
    frame["_MARGIN_SOURCE"] = "missing"
    if VP_MARGIN_TARGET_COLUMN in frame:
        direct = _finite_numeric(frame[VP_MARGIN_TARGET_COLUMN])
        valid = direct.notna()
        frame.loc[valid, "_MARGIN_TARGET"] = direct[valid]
        frame.loc[valid, "_MARGIN_SOURCE"] = "direct"

    games = _manifest_games(manifests)
    if games:
        if "GAME_INDEX" not in frame:
            raise ValueError(
                "Distillation rows lack GAME_INDEX required by their manifest"
            )
        default_iteration = next(iter(games))[0] if len(manifests) == 1 else None
        iterations = (
            frame["ITERATION"].astype(int)
            if "ITERATION" in frame
            else np.full(len(frame), default_iteration)
        )
        game_indices = frame["GAME_INDEX"].astype(int)
        keys = list(zip(iterations, game_indices))
        unknown = sorted(set(keys) - set(games))
        if unknown:
            raise ValueError(
                f"Rows reference games absent from manifest: {unknown[:5]}"
            )
        if "COLOR" in frame:
            mismatches = sum(
                str(color) != str(games[key]["student_color"])
                for color, key in zip(frame["COLOR"], keys)
            )
            if mismatches:
                raise ValueError(
                    f"Distillation row perspective differs from manifest: {mismatches} rows"
                )
        derived_win = []
        derived_margin = []
        for key in keys:
            game = games[key]
            truncated = bool(game.get("truncated"))
            winner = game.get("winner")
            recorded_return = game.get("student_return")
            if recorded_return is not None:
                derived_win.append(float(recorded_return))
            elif truncated or winner is None:
                derived_win.append(math.nan)
            else:
                derived_win.append(1.0 if winner == game.get("student_color") else -1.0)
            recorded_margin = game.get("student_vp_margin_return")
            derived_margin.append(
                float(recorded_margin)
                if recorded_margin is not None and not truncated
                else math.nan
            )
        missing = frame["_WIN_TARGET"].isna()
        derived_win = pd.Series(derived_win, index=frame.index, dtype=float)
        fill = missing & derived_win.notna()
        frame.loc[fill, "_WIN_TARGET"] = derived_win[fill]
        frame.loc[fill, "_WIN_SOURCE"] = "manifest"
        missing = frame["_MARGIN_TARGET"].isna()
        derived_margin = pd.Series(derived_margin, index=frame.index, dtype=float)
        fill = missing & derived_margin.notna()
        frame.loc[fill, "_MARGIN_TARGET"] = derived_margin[fill]
        frame.loc[fill, "_MARGIN_SOURCE"] = "manifest"

    if OWN_VP_RETURN_COLUMN in frame and "SEAT" in frame:
        own = _finite_numeric(frame[OWN_VP_RETURN_COLUMN])
        perspective = pd.DataFrame(
            {
                "trajectory": frame["_TRAJECTORY_GROUP"],
                "seat": frame["SEAT"],
                "own": own,
            }
        ).dropna(subset=["own"])
        grouped = perspective.groupby(["trajectory", "seat"])["own"]
        if bool((grouped.nunique() > 1).any()):
            raise ValueError(
                "Terminal VP return changes within one game/seat trajectory"
            )
        own_by_perspective = grouped.first()
        margin_by_perspective: dict[tuple[str, Any], float] = {}
        for trajectory, values in own_by_perspective.groupby(level=0):
            seats = values.droplevel(0)
            if len(seats) != 2:
                continue
            for seat, value in seats.items():
                other = next(
                    float(other_value)
                    for other_seat, other_value in seats.items()
                    if other_seat != seat
                )
                margin_by_perspective[(str(trajectory), seat)] = float(value) - other
        derived = pd.Series(
            [
                margin_by_perspective.get((str(trajectory), seat), math.nan)
                for trajectory, seat in zip(frame["_TRAJECTORY_GROUP"], frame["SEAT"])
            ],
            index=frame.index,
            dtype=float,
        )
        missing = frame["_MARGIN_TARGET"].isna()
        fill = missing & derived.notna()
        frame.loc[fill, "_MARGIN_TARGET"] = derived[fill]
        frame.loc[fill, "_MARGIN_SOURCE"] = "paired_vp_return"

    if all(column in frame for column in PUBLIC_VP_COLUMNS):
        frame["_PUBLIC_VP_DIFF"] = _finite_numeric(
            frame[PUBLIC_VP_COLUMNS[0]]
        ) - _finite_numeric(frame[PUBLIC_VP_COLUMNS[1]])
    else:
        frame["_PUBLIC_VP_DIFF"] = np.nan
    return frame, shards


def resolve_outcome_shards(path: Path) -> list[Path]:
    """Resolve and integrity-check one outcome corpus input."""
    shards, _ = _resolve_input(Path(path))
    return shards


def load_outcome_target_index(
    inputs: Sequence[Path],
) -> tuple[list[Path], dict[tuple[str, str, int], tuple[float, float]]]:
    """Build constant per-shard/game/seat targets for streaming critic data."""
    paths: list[Path] = []
    index: dict[tuple[str, str, int], tuple[float, float]] = {}
    for source in inputs:
        frame, source_paths = _load_input_frame(Path(source))
        if "SEAT" not in frame:
            raise ValueError(f"Outcome corpus lacks SEAT: {source}")
        paths.extend(source_paths)
        grouped = frame.groupby(["_SOURCE_SHARD", "_SPLIT_GROUP", "SEAT"])
        for (shard, game_id, seat), rows in grouped:
            win_values = rows["_WIN_TARGET"].dropna().unique()
            margin_values = rows["_MARGIN_TARGET"].dropna().unique()
            if len(win_values) > 1 or len(margin_values) > 1:
                raise ValueError(
                    "Outcome target changes within one shard/game/seat trajectory"
                )
            win = float(win_values[0]) if len(win_values) else math.nan
            margin = float(margin_values[0]) if len(margin_values) else math.nan
            key = (str(Path(shard).resolve()), str(game_id), int(seat))
            existing = index.get(key)
            target = (win, margin)
            if existing is not None and not all(
                (math.isnan(left) and math.isnan(right)) or math.isclose(left, right)
                for left, right in zip(existing, target)
            ):
                raise ValueError(f"Conflicting outcome targets for {key}")
            index[key] = target
    return paths, index


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    import pandas as pd

    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if not n_positive or not n_negative:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def _baseline_metrics(frame) -> dict[str, Any]:
    win = frame[["_WIN_TARGET", "_PUBLIC_VP_DIFF"]].dropna()
    win = win[win["_WIN_TARGET"].isin((-1.0, 1.0))]
    win_metrics: dict[str, Any] | None = None
    if not win.empty:
        labels = (win["_WIN_TARGET"].to_numpy(float) > 0).astype(int)
        scores = win["_PUBLIC_VP_DIFF"].to_numpy(float)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
        predictions = np.sign(scores)
        non_ties = predictions != 0
        accuracy = (
            float(
                (
                    predictions[non_ties] == win["_WIN_TARGET"].to_numpy()[non_ties]
                ).mean()
            )
            if bool(non_ties.any())
            else None
        )
        prevalence = float(labels.mean())
        win_metrics = {
            "rows": int(len(win)),
            "auc": _binary_auc(labels, scores),
            "non_tie_accuracy": accuracy,
            "tie_fraction": float((~non_ties).mean()),
            "sigmoid_brier": float(np.mean((probabilities - labels) ** 2)),
            "constant_prevalence_brier": float(
                np.mean((np.full(len(labels), prevalence) - labels) ** 2)
            ),
        }

    margin = frame[["_MARGIN_TARGET", "_PUBLIC_VP_DIFF"]].dropna()
    margin_metrics: dict[str, Any] | None = None
    if not margin.empty:
        errors = margin["_PUBLIC_VP_DIFF"].to_numpy(float) - margin[
            "_MARGIN_TARGET"
        ].to_numpy(float)
        margin_metrics = {
            "rows": int(len(margin)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
        }
    return {"public_vp_win": win_metrics, "public_vp_margin": margin_metrics}


def _target_summary(frame, target: str, source: str) -> dict[str, Any]:
    valid = frame[target].notna()
    source_counts = Counter(frame.loc[valid, source].astype(str))
    covered_trajectories = frame.loc[valid, "_TRAJECTORY_GROUP"].nunique()
    return {
        "rows": int(valid.sum()),
        "row_coverage": float(valid.mean()) if len(frame) else 0.0,
        "trajectories": int(covered_trajectories),
        "trajectory_coverage": (
            float(covered_trajectories / frame["_TRAJECTORY_GROUP"].nunique())
            if len(frame)
            else 0.0
        ),
        "sources": dict(sorted(source_counts.items())),
    }


def _corpus_summary(frame, paths: Sequence[Path], label: str) -> dict[str, Any]:
    win = _target_summary(frame, "_WIN_TARGET", "_WIN_SOURCE")
    valid_win = frame["_WIN_TARGET"].dropna()
    win["value_counts"] = {
        str(float(value)): int(count)
        for value, count in sorted(valid_win.value_counts().items())
    }
    margin = _target_summary(frame, "_MARGIN_TARGET", "_MARGIN_SOURCE")
    return {
        "label": label,
        "rows": int(len(frame)),
        "shards": len(paths),
        "split_groups": int(frame["_SPLIT_GROUP"].nunique()),
        "trajectories": int(frame["_TRAJECTORY_GROUP"].nunique()),
        "win_target": win,
        "vp_margin_target": margin,
        "baselines": _baseline_metrics(frame),
    }


def audit_outcome_targets(
    corpora: Sequence[Sequence[Path]],
    *,
    expected_dataset_sha256: str | None = None,
    expected_shards: int | None = None,
    minimum_win_row_coverage: float | None = None,
    minimum_margin_row_coverage: float | None = None,
    minimum_split_groups: int | None = None,
    minimum_minority_fraction: float | None = None,
) -> dict[str, Any]:
    """Audit outcome coverage and public-score baselines without fitting a model."""
    import pandas as pd

    if not corpora or any(not corpus for corpus in corpora):
        raise ValueError("At least one non-empty logical corpus is required")
    for name, value in (
        ("minimum_win_row_coverage", minimum_win_row_coverage),
        ("minimum_margin_row_coverage", minimum_margin_row_coverage),
        ("minimum_minority_fraction", minimum_minority_fraction),
    ):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if minimum_split_groups is not None and minimum_split_groups < 1:
        raise ValueError("minimum_split_groups must be positive")

    corpus_frames = []
    corpus_paths = []
    summaries = []
    split_sets = []
    for index, inputs in enumerate(corpora):
        frames = []
        paths = []
        for source in inputs:
            frame, source_paths = _load_input_frame(Path(source))
            frames.append(frame)
            paths.extend(source_paths)
        corpus_frame = pd.concat(frames, ignore_index=True, sort=False)
        label = "+".join(str(Path(item)) for item in inputs)
        corpus_frames.append(corpus_frame)
        corpus_paths.extend(paths)
        split_sets.append(set(corpus_frame["_SPLIT_GROUP"].astype(str)))
        summaries.append(_corpus_summary(corpus_frame, paths, label))

    collision_rows = []
    for left in range(len(split_sets)):
        for right in range(left + 1, len(split_sets)):
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                collision_rows.append(
                    {
                        "left_corpus": left,
                        "right_corpus": right,
                        "count": len(overlap),
                        "examples": sorted(overlap)[:5],
                    }
                )

    combined = pd.concat(corpus_frames, ignore_index=True, sort=False)
    shard_rows, dataset_sha256 = hash_parquet_shards(corpus_paths, progress=False)
    combined_summary = _corpus_summary(combined, corpus_paths, "combined")
    binary = combined["_WIN_TARGET"].dropna()
    binary = binary[binary.isin((-1.0, 1.0))]
    counts = binary.value_counts()
    minority_fraction = float(counts.min() / counts.sum()) if len(counts) == 2 else 0.0

    gates = []

    def add_gate(name: str, passed: bool, actual: Any, threshold: Any) -> None:
        gates.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": actual,
                "threshold": threshold,
            }
        )

    if expected_dataset_sha256 is not None:
        add_gate(
            "expected_dataset_sha256",
            dataset_sha256 == expected_dataset_sha256,
            dataset_sha256,
            expected_dataset_sha256,
        )
    if expected_shards is not None:
        add_gate(
            "expected_shards",
            len(corpus_paths) == expected_shards,
            len(corpus_paths),
            expected_shards,
        )
    add_gate(
        "cross_corpus_split_group_collisions",
        not collision_rows,
        len(collision_rows),
        0,
    )
    if minimum_win_row_coverage is not None:
        actual = combined_summary["win_target"]["row_coverage"]
        add_gate(
            "minimum_win_row_coverage",
            actual >= minimum_win_row_coverage,
            actual,
            minimum_win_row_coverage,
        )
    if minimum_margin_row_coverage is not None:
        actual = combined_summary["vp_margin_target"]["row_coverage"]
        add_gate(
            "minimum_margin_row_coverage",
            actual >= minimum_margin_row_coverage,
            actual,
            minimum_margin_row_coverage,
        )
    if minimum_split_groups is not None:
        actual = min(summary["split_groups"] for summary in summaries)
        add_gate(
            "minimum_split_groups_per_corpus",
            actual >= minimum_split_groups,
            actual,
            minimum_split_groups,
        )
    if minimum_minority_fraction is not None:
        add_gate(
            "minimum_win_minority_fraction",
            minority_fraction >= minimum_minority_fraction,
            minority_fraction,
            minimum_minority_fraction,
        )
    baselines = combined_summary["baselines"]
    add_gate(
        "finite_public_score_baselines",
        baselines["public_vp_win"] is not None
        and baselines["public_vp_margin"] is not None,
        {
            "win": baselines["public_vp_win"] is not None,
            "margin": baselines["public_vp_margin"] is not None,
        },
        {"win": True, "margin": True},
    )
    return {
        "schema_version": "1.0",
        "kind": "outcome_target_feasibility_audit",
        "dataset": {
            "sha256": dataset_sha256,
            "shards": len(corpus_paths),
            "rows": int(len(combined)),
            "input_shards": shard_rows,
            "logical_corpora": summaries,
            "cross_corpus_split_group_collisions": collision_rows,
        },
        "combined": combined_summary,
        "win_minority_fraction": minority_fraction,
        "gates": gates,
        "all_gates_passed": bool(gates) and all(gate["passed"] for gate in gates),
    }

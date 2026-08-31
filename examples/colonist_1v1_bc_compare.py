#!/usr/bin/env python3
"""Compare two BC checkpoints on one exactly matched grouped holdout.

The training metadata of an older checkpoint is not a valid comparator after a
DAgger corpus is appended: its recorded validation/test metrics were computed
before the new corpus existed.  This command rebuilds one immutable split plan,
loads both checkpoints, and evaluates them on the same validation and test rows.
Each augmentation directory is split independently so earlier DAgger holdouts
remain frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from catanatron.gym.bc_training import (
    ParquetDecisionBatches,
    combine_parquet_dataset_plans,
    hash_parquet_shards,
    inspect_parquet_corpora,
    inspect_parquet_dataset,
    resolve_torch_device,
)
from catanatron.gym.colonist_training import load_bc_checkpoint_meta
from catanatron.gym.model_architectures import build_bc_policy
from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    read_model_schema,
    validate_model_schema,
)
from catanatron.gym.provenance import sha256_file

try:
    from examples.colonist_1v1_bc import _evaluate, _resolve_dataset_paths
except ModuleNotFoundError:  # Direct execution: sys.path starts at examples/.
    from colonist_1v1_bc import _evaluate, _resolve_dataset_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Base teacher corpora, combined before their grouped split.",
    )
    parser.add_argument(
        "--augmentation-data-dir",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Additional corpora split independently in argument order. Pass "
            "each DAgger iteration directory as a separate argument."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--feature-profile", choices=("raw", "public_derived"), default="raw"
    )
    parser.add_argument(
        "--allow-legacy-dataset-schema",
        action="store_true",
        help="Allow schema-less datasets only after manual compatibility review.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_checkpoint(
    checkpoint: Path,
    *,
    expected_schema: dict[str, Any],
    feature_columns: tuple[str, ...],
    device,
):
    import torch

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    meta_path = checkpoint.with_suffix(".meta.json")
    meta = load_bc_checkpoint_meta(meta_path)
    if meta is None:
        raise FileNotFoundError(f"Checkpoint metadata does not exist: {meta_path}")

    schema_path = checkpoint_schema_path(checkpoint)
    stored_schema = read_model_schema(schema_path)
    if stored_schema is None and meta.model_schema:
        stored_schema = meta.model_schema
    if stored_schema is None:
        raise ValueError(f"Checkpoint has no model schema: {checkpoint}")
    validate_model_schema(
        expected_schema, stored_schema, context=f"checkpoint {checkpoint}"
    )
    if meta.obs_dim != len(feature_columns):
        raise ValueError(
            f"Checkpoint {checkpoint} declares obs_dim={meta.obs_dim}, "
            f"but the matched dataset has {len(feature_columns)} features"
        )
    if meta.n_actions != len(expected_schema["actions"]):
        raise ValueError(
            f"Checkpoint {checkpoint} declares n_actions={meta.n_actions}, "
            f"but the schema has {len(expected_schema['actions'])}"
        )

    net = build_bc_policy(
        meta.architecture,
        feature_columns,
        meta.n_actions,
        hidden_sizes=tuple(meta.hidden_sizes),
        embedding_dim=meta.embedding_dim,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    if meta.parameter_count and meta.parameter_count != parameter_count:
        raise ValueError(
            f"Checkpoint {checkpoint} parameter count disagrees with metadata: "
            f"{parameter_count} != {meta.parameter_count}"
        )
    return net, {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "metadata_path": str(meta_path),
        "metadata_sha256": sha256_file(meta_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path) if schema_path.is_file() else None,
        "architecture": meta.architecture,
        "hidden_sizes": meta.hidden_sizes,
        "embedding_dim": meta.embedding_dim,
        "parameter_count": parameter_count,
        "training_dataset_sha256": meta.dataset_sha256,
        "training_selection_metric": meta.selection_metric,
        "training_selection_value": meta.selection_value,
    }


def _numeric_metric_deltas(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in sorted(candidate.keys() & baseline.keys()):
        left = candidate[key]
        right = baseline[key]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isfinite(float(left))
            and math.isfinite(float(right))
        ):
            deltas[key] = float(left) - float(right)
    return deltas


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.candidate.resolve() == args.baseline.resolve():
        raise ValueError("Candidate and baseline checkpoints must differ")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("Matched comparison requires non-empty val and test splits")

    expected_schema = build_model_schema(feature_profile=args.feature_profile)
    base_paths = _resolve_dataset_paths(
        args.data_dir,
        expected_schema=expected_schema,
        allow_legacy_schema=args.allow_legacy_dataset_schema,
    )
    base_plan = inspect_parquet_dataset(
        base_paths,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    augmentation_corpora = [
        _resolve_dataset_paths(
            [directory],
            expected_schema=expected_schema,
            allow_legacy_schema=args.allow_legacy_dataset_schema,
        )
        for directory in (args.augmentation_data_dir or [])
    ]
    plan = base_plan
    if augmentation_corpora:
        augmentation_plan = inspect_parquet_corpora(
            augmentation_corpora,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
        plan = combine_parquet_dataset_plans((base_plan, augmentation_plan))

    expected_features = tuple(
        f"F_{name}" for name in expected_schema["observation"]["features"]
    )
    if plan.feature_columns != expected_features:
        raise ValueError(
            "Dataset feature order does not match the requested model schema: "
            f"dataset={len(plan.feature_columns)} schema={len(expected_features)}"
        )
    if not plan.rows_for("val") or not plan.rows_for("test"):
        raise ValueError("Matched comparison produced an empty holdout split")

    paths = [*base_paths, *(path for corpus in augmentation_corpora for path in corpus)]
    print(f"Hashing {len(paths):,} matched input shards ...")
    input_shards, dataset_sha256 = hash_parquet_shards(paths)
    device = resolve_torch_device(args.device)
    print(
        f"dataset_sha256={dataset_sha256} shards={len(plan.paths):,} "
        f"val_rows={plan.rows_for('val'):,} test_rows={plan.rows_for('test'):,} "
        f"device={device}"
    )

    evaluation_args = SimpleNamespace(
        num_workers=args.num_workers,
        listwise_temperature=0.25,
        tie_tolerance=1e-6,
        hybrid_listwise_weight=0.0,
        win_value_weight=0.0,
        vp_margin_weight=0.0,
    )
    datasets = {
        split: ParquetDecisionBatches(
            plan,
            split,
            batch_size=args.batch_size,
            seed=args.split_seed,
        )
        for split in ("val", "test")
    }
    checkpoint_records: dict[str, dict[str, Any]] = {}
    for label, checkpoint in (
        ("candidate", args.candidate),
        ("baseline", args.baseline),
    ):
        net, record = _load_checkpoint(
            checkpoint,
            expected_schema=expected_schema,
            feature_columns=plan.feature_columns,
            device=device,
        )
        record["splits"] = {}
        for split, dataset in datasets.items():
            loss, metrics = _evaluate(net, dataset, "legal_ce", device, evaluation_args)
            if not math.isfinite(loss):
                raise ValueError(f"{label} produced non-finite {split} loss")
            record["splits"][split] = {"loss": loss, **metrics}
            print(
                f"{label} {split}: loss={loss:.6f} "
                f"mean_regret={metrics.get('mean_regret', float('nan')):.6f} "
                f"legal_choice_accuracy="
                f"{metrics.get('legal_choice_accuracy', float('nan')):.6f}"
            )
        checkpoint_records[label] = record

    deltas = {
        split: _numeric_metric_deltas(
            checkpoint_records["candidate"]["splits"][split],
            checkpoint_records["baseline"]["splits"][split],
        )
        for split in ("val", "test")
    }
    report = {
        "schema_version": "1.0",
        "kind": "matched_bc_holdout_comparison",
        "protocol": {
            "loss": "legal_ce",
            "split_seed": args.split_seed,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "batch_size": args.batch_size,
            "feature_profile": args.feature_profile,
            "augmentation_split": "each_argument_independently",
            "delta_definition": "candidate_minus_baseline",
        },
        "dataset": {
            "sha256": dataset_sha256,
            "shards": len(plan.paths),
            "input_shards": input_shards,
            "base_data_dirs": [str(path.resolve()) for path in args.data_dir],
            "augmentation_data_dirs": [
                str(path.resolve()) for path in (args.augmentation_data_dir or [])
            ],
            "train_rows": plan.rows_for("train"),
            "val_rows": plan.rows_for("val"),
            "test_rows": plan.rows_for("test"),
        },
        "checkpoints": checkpoint_records,
        "deltas": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

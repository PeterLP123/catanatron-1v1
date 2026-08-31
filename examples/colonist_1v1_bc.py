#!/usr/bin/env python3
"""Behavioral cloning on Colonist 1v1 Parquet decision logs.

The trainer streams one game shard at a time, splits whole games, and supports
three objectives: full-space legacy cross entropy, legal-masked cross entropy,
and candidate-value listwise learning.

Example::

    python examples/colonist_1v1_bc.py --data-dir data/c1_teachers \
        --loss listwise --epochs 10 --out runs/colonist_bc_policy.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from functools import partial
from pathlib import Path
from typing import Any

from catanatron.gym.bc_training import (
    CANDIDATE_VALUES_COLUMN,
    LEGAL_ACTIONS_COLUMN,
    VP_MARGIN_TARGET_COLUMN,
    WIN_VALUE_TARGET_COLUMN,
    DecisionMetricAccumulator,
    ParquetDecisionBatches,
    candidate_listwise_loss,
    combine_parquet_dataset_plans,
    hash_parquet_shards,
    inspect_parquet_corpora,
    inspect_parquet_dataset,
    legal_masked_cross_entropy,
    resolve_torch_device,
    seed_everything,
)
from catanatron.gym.colonist_training import (
    BcCheckpointMeta,
    TrainingRunTracker,
    hard_state_sample_weights,
    load_bc_checkpoint_meta,
    outcome_deficit_sample_weights,
    resolve_teacher_parquet_paths,
)
from catanatron.gym.model_architectures import build_bc_policy
from catanatron.gym.distillation import verify_distillation_dataset
from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    read_model_schema,
    validate_model_schema,
    write_model_schema,
)
from catanatron.gym.provenance import sha256_file

DEFAULT_BC_CHECKPOINT_PATH = Path("runs/colonist_bc_policy.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        help=(
            "One or more directories of game *.parquet files. The option may "
            "be repeated; all occurrences accumulate."
        ),
    )
    parser.add_argument(
        "--augmentation-data-dir",
        type=Path,
        nargs="+",
        action="extend",
        default=None,
        help=(
            "Additional teacher-labelled corpora, such as DAgger iterations. "
            "Each directory is split independently; repeated option occurrences "
            "accumulate so earlier holdouts remain frozen."
        ),
    )
    parser.add_argument(
        "--augmentation-weight",
        type=float,
        default=1.0,
        help="Training-only sample weight for rows from --augmentation-data-dir.",
    )
    parser.add_argument(
        "--outcome-weighted-augmentation-data-dir",
        type=Path,
        nargs="+",
        action="extend",
        default=None,
        help=(
            "Fresh DAgger corpora whose training rows receive bounded loss/VP-deficit "
            "weights. Each directory is split independently and must contain complete "
            "native terminal targets."
        ),
    )
    parser.add_argument(
        "--outcome-loss-bonus",
        type=float,
        default=1.0,
        help="Maximum extra training weight for a student loss (default: 1.0).",
    )
    parser.add_argument(
        "--outcome-vp-deficit-bonus",
        type=float,
        default=0.5,
        help="Maximum extra training weight for a terminal VP deficit (default: 0.5).",
    )
    parser.add_argument(
        "--outcome-vp-deficit-scale",
        type=float,
        default=10.0,
        help="VP deficit receiving the full deficit bonus (default: 10).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Held-out test split (by game). Reported once at the end.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for the grouped (by-game) train/val/test split.",
    )
    parser.add_argument(
        "--expected-dataset-sha256",
        default=None,
        help="Abort before training unless the selected shard-set hash matches.",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=None,
        help="Abort before training unless this many input shards are selected.",
    )
    for split in ("train", "val", "test"):
        parser.add_argument(
            f"--expected-{split}-rows",
            type=int,
            default=None,
            help=f"Abort before training unless the frozen {split} split has this many rows.",
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Python/NumPy/Torch seed (defaults to --split-seed).",
    )
    parser.add_argument(
        "--hard-states",
        action="store_true",
        help="Weight TRAIN rows toward genuine strategic decisions. Validation/test remain honest.",
    )
    parser.add_argument(
        "--architecture",
        choices=(
            "mlp",
            "action_conditioned",
            "factored_policy_value",
            "spatial_edge_residual",
            "spatial_robber_residual",
        ),
        default="mlp",
        help="Checkpoint architecture (default keeps legacy MLP compatibility).",
    )
    parser.add_argument("--hidden", type=int, nargs=2, default=(512, 512))
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help=(
            "Shared embedding width for action_conditioned, "
            "factored_policy_value, or either spatial residual head."
        ),
    )
    parser.add_argument(
        "--win-value-weight",
        type=float,
        default=None,
        help="Auxiliary terminal win-value loss weight (factored default: 0.25).",
    )
    parser.add_argument(
        "--vp-margin-weight",
        type=float,
        default=None,
        help="Auxiliary terminal VP-margin loss weight (factored default: 0.05).",
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=332,
        help="Policy action head size. Must match the recorded action schema.",
    )
    parser.add_argument(
        "--loss",
        choices=("auto", "cross_entropy", "legal_ce", "listwise", "hybrid"),
        default="auto",
        help=(
            "auto selects legal_ce for dataset-v2 logs and cross_entropy for legacy "
            "logs; hybrid uses legal_ce plus a weighted listwise regularizer."
        ),
    )
    parser.add_argument(
        "--listwise-temperature",
        type=float,
        default=0.25,
        help="Soft-target temperature for --loss listwise or hybrid.",
    )
    parser.add_argument(
        "--hybrid-listwise-weight",
        type=float,
        default=0.1,
        help="Listwise regularizer weight for --loss hybrid (default: 0.1).",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-6,
        help="Candidate values closer than this are treated as exact ties.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--feature-profile",
        choices=("raw", "public_derived"),
        default="raw",
        help="Schema identity of the F_* observation columns.",
    )
    parser.add_argument(
        "--allow-legacy-dataset-schema",
        action="store_true",
        help="Allow datasets without action/rules schema hashes after manual verification.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_BC_CHECKPOINT_PATH)
    parser.add_argument("--tensorboard", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize from a schema-compatible BC checkpoint and include its "
            "unchanged epoch-0 validation result in best-checkpoint selection."
        ),
    )
    parser.add_argument(
        "--freeze-base-policy",
        action="store_true",
        help=(
            "For a spatial residual architecture, freeze the MLP loaded from "
            "--init-checkpoint and train only the targeted residual."
        ),
    )
    return parser


def _resolve_dataset_paths(
    data_dirs: list[Path],
    *,
    expected_schema: dict[str, Any],
    allow_legacy_schema: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for directory in data_dirs:
        aggregate_path = directory / "manifest.json"
        aggregate = (
            json.loads(aggregate_path.read_text(encoding="utf-8"))
            if aggregate_path.is_file()
            else None
        )
        if (
            isinstance(aggregate, dict)
            and isinstance(aggregate.get("iteration"), int)
            and isinstance(aggregate.get("shards"), list)
        ):
            distillation_root = directory.parent
            problems = verify_distillation_dataset(distillation_root)
            if problems:
                raise ValueError(
                    f"Distillation dataset {distillation_root} failed integrity checks: "
                    + "; ".join(problems)
                )
            root_manifest_path = distillation_root / "manifest.json"
            root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
            relative_manifest = str(
                (directory / "manifest.json").relative_to(distillation_root)
            )
            if relative_manifest not in {
                iteration.get("manifest")
                for iteration in root_manifest.get("iterations", [])
            }:
                raise ValueError(
                    f"Distillation iteration {directory} is not indexed by "
                    f"{root_manifest_path}"
                )
            schema = aggregate.get("metadata", {}).get("schema")
            if not isinstance(schema, dict):
                raise ValueError(
                    f"Distillation manifest {aggregate_path} has no model schema"
                )
            validate_model_schema(
                expected_schema,
                schema,
                context=f"distillation iteration {directory}",
            )
            paths.extend(
                distillation_root / shard["path"]
                for shard in aggregate.get("shards", [])
            )
            continue
        if isinstance(aggregate, dict) and isinstance(
            aggregate.get("iterations"), list
        ):
            problems = verify_distillation_dataset(directory)
            if problems:
                raise ValueError(
                    f"Distillation dataset {directory} failed integrity checks: "
                    + "; ".join(problems)
                )
            distillation_paths: list[Path] = []
            for iteration in aggregate["iterations"]:
                manifest_path = directory / iteration["manifest"]
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                schema = manifest.get("metadata", {}).get("schema")
                if not isinstance(schema, dict):
                    raise ValueError(
                        f"Distillation manifest {manifest_path} has no model schema"
                    )
                validate_model_schema(
                    expected_schema,
                    schema,
                    context=f"distillation dataset {directory}",
                )
                distillation_paths.extend(
                    directory / shard["path"] for shard in manifest.get("shards", [])
                )
            paths.extend(distillation_paths)
            continue

        meta_path = directory / "dataset_meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") not in {None, "complete"}:
                raise ValueError(
                    f"Dataset {directory} is {meta.get('status')!r}; "
                    "resume generation before training"
                )
        schema = read_model_schema(directory / "dataset_schema.json")
        if schema is None:
            if not allow_legacy_schema:
                raise ValueError(
                    f"Dataset {directory} has no dataset_schema.json. "
                    "Regenerate it or use --allow-legacy-dataset-schema only "
                    "after manually confirming feature/action/rules compatibility."
                )
        else:
            validate_model_schema(
                expected_schema, schema, context=f"dataset {directory}"
            )
            for key in (
                "model_schema_hash",
                "feature_hash",
                "action_hash",
                "rules_hash",
            ):
                expected_key = "schema_hash" if key == "model_schema_hash" else key
                if meta.get(key) is not None and meta[key] != schema[expected_key]:
                    raise ValueError(
                        f"Dataset {directory} metadata {key} disagrees with its schema"
                    )
        directory_paths = resolve_teacher_parquet_paths(directory)
        if meta.get("dataset_sha256"):
            _, actual_dataset_hash = hash_parquet_shards(
                directory_paths, progress=False
            )
            if actual_dataset_hash != meta["dataset_sha256"]:
                raise ValueError(
                    f"Dataset {directory} shard hash does not match dataset_meta.json"
                )
        paths.extend(directory_paths)
    if not paths:
        raise FileNotFoundError(f"No .parquet files under {data_dirs}")
    return paths


def _load_initial_checkpoint(
    checkpoint: Path,
    net,
    *,
    expected_schema: dict[str, Any],
    architecture: str,
    hidden_sizes: tuple[int, ...],
    embedding_dim: int,
    obs_dim: int,
    n_actions: int,
    device,
) -> dict[str, Any]:
    """Load a parent only after its schema and architecture contract match."""
    import torch

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Initial checkpoint does not exist: {checkpoint}")
    meta_path = checkpoint.with_suffix(".meta.json")
    meta = load_bc_checkpoint_meta(meta_path)
    if meta is None:
        raise FileNotFoundError(f"Initial checkpoint metadata is missing: {meta_path}")
    schema_path = checkpoint_schema_path(checkpoint)
    stored_schema = read_model_schema(schema_path)
    if stored_schema is None and meta.model_schema:
        stored_schema = meta.model_schema
    if stored_schema is None:
        raise ValueError(f"Initial checkpoint has no model schema: {checkpoint}")
    validate_model_schema(
        expected_schema, stored_schema, context=f"initial checkpoint {checkpoint}"
    )

    expected_contract = {
        "architecture": architecture,
        "hidden_sizes": tuple(hidden_sizes),
        "embedding_dim": embedding_dim,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
    }
    actual_contract = {
        "architecture": meta.architecture,
        "hidden_sizes": tuple(meta.hidden_sizes),
        "embedding_dim": meta.embedding_dim,
        "obs_dim": meta.obs_dim,
        "n_actions": meta.n_actions,
    }
    base_policy_initialization = (
        architecture in {"spatial_edge_residual", "spatial_robber_residual"}
        and meta.architecture == "mlp"
        and hasattr(net, "load_base_policy_state_dict")
    )
    comparison_keys = (
        ("hidden_sizes", "obs_dim", "n_actions")
        if base_policy_initialization
        else tuple(expected_contract)
    )
    mismatches = {
        key: (expected_contract[key], actual_contract[key])
        for key in comparison_keys
        if expected_contract[key] != actual_contract[key]
    }
    if mismatches:
        raise ValueError(
            f"Initial checkpoint architecture contract mismatch: {mismatches}"
        )

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if base_policy_initialization:
        net.load_base_policy_state_dict(state)
        loaded_parameter_count = sum(
            parameter.numel() for parameter in net.base_policy.parameters()
        )
    else:
        net.load_state_dict(state)
        loaded_parameter_count = sum(
            parameter.numel() for parameter in net.parameters()
        )
    if meta.parameter_count and meta.parameter_count != loaded_parameter_count:
        raise ValueError(
            "Initial checkpoint parameter count disagrees with metadata: "
            f"{loaded_parameter_count} != {meta.parameter_count}"
        )
    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "metadata_path": str(meta_path),
        "metadata_sha256": sha256_file(meta_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path) if schema_path.is_file() else None,
        "initialization_mode": (
            "mlp_base_policy" if base_policy_initialization else "exact_architecture"
        ),
    }


def _weighted_mean(losses, weights):
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("Training batch has no positive sample weight")
    return (losses * weights).sum() / denominator


def _validate_dataset_contract(
    *,
    dataset_sha256: str,
    shard_count: int,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    expected_dataset_sha256: str | None = None,
    expected_shards: int | None = None,
    expected_train_rows: int | None = None,
    expected_val_rows: int | None = None,
    expected_test_rows: int | None = None,
) -> None:
    """Reject corpus or split drift before the optimizer can take a step."""
    actual = {
        "dataset_sha256": dataset_sha256,
        "shards": shard_count,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
    }
    expected = {
        "dataset_sha256": expected_dataset_sha256,
        "shards": expected_shards,
        "train_rows": expected_train_rows,
        "val_rows": expected_val_rows,
        "test_rows": expected_test_rows,
    }
    mismatches = {
        name: {"expected": value, "actual": actual[name]}
        for name, value in expected.items()
        if value is not None and value != actual[name]
    }
    if mismatches:
        raise ValueError(f"Dataset contract mismatch before training: {mismatches}")


def _batch_loss(net, batch, loss_name: str, device, args):
    from torch.nn import functional as F

    features = batch["features"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    policy_value = net.policy_value(features) if hasattr(net, "policy_value") else None
    logits = policy_value.policy_logits if policy_value is not None else net(features)
    weights = batch["sample_weights"].to(device, non_blocking=True)
    if loss_name == "cross_entropy":
        row_losses = F.cross_entropy(logits, targets, reduction="none")
        loss = _weighted_mean(row_losses, weights)
        used_rows = len(targets)
    else:
        legal_indices = batch["legal_indices"].to(device, non_blocking=True)
        legal_mask = batch["legal_mask"].to(device, non_blocking=True)
        if loss_name in {"legal_ce", "hybrid"}:
            legal_row_losses = legal_masked_cross_entropy(
                logits,
                targets,
                legal_indices,
                legal_mask,
                reduction="none",
            )
            loss = _weighted_mean(legal_row_losses, weights)
            used_rows = len(targets)
        else:
            loss = logits.sum() * 0.0
            used_rows = 0

    if loss_name in {"hybrid", "listwise"}:
        values = batch["candidate_values"].to(device, non_blocking=True)
        value_mask = batch["candidate_mask"].to(device, non_blocking=True)
        listwise_row_losses, valid = candidate_listwise_loss(
            logits,
            legal_indices,
            legal_mask,
            values,
            value_mask,
            temperature=args.listwise_temperature,
            tie_tolerance=args.tie_tolerance,
            reduction="none",
        )
        if listwise_row_losses.ndim > 0 and len(listwise_row_losses):
            listwise_loss = _weighted_mean(listwise_row_losses, weights[valid])
            if loss_name == "hybrid":
                loss = loss + args.hybrid_listwise_weight * listwise_loss
            else:
                loss = listwise_loss
                used_rows = len(listwise_row_losses)

    if policy_value is not None:
        win_weight = float(getattr(args, "win_value_weight", 0.0) or 0.0)
        win_mask = batch["win_value_mask"].to(device, non_blocking=True)
        if win_weight > 0 and bool(win_mask.any()):
            win_targets = batch["win_value_targets"].to(device, non_blocking=True)
            win_rows = F.mse_loss(
                policy_value.win_value[win_mask],
                win_targets[win_mask],
                reduction="none",
            )
            loss = loss + win_weight * _weighted_mean(win_rows, weights[win_mask])
            used_rows = max(used_rows, int(win_mask.sum().item()))

        margin_weight = float(getattr(args, "vp_margin_weight", 0.0) or 0.0)
        margin_mask = batch["vp_margin_mask"].to(device, non_blocking=True)
        if margin_weight > 0 and bool(margin_mask.any()):
            margin_targets = batch["vp_margin_targets"].to(device, non_blocking=True)
            margin_rows = F.smooth_l1_loss(
                policy_value.vp_margin[margin_mask],
                margin_targets[margin_mask],
                reduction="none",
            )
            loss = loss + margin_weight * _weighted_mean(
                margin_rows, weights[margin_mask]
            )
            used_rows = max(used_rows, int(margin_mask.sum().item()))
    return loss, logits, used_rows


def _evaluate(
    net, dataset, loss_name: str, device, args
) -> tuple[float, dict[str, Any]]:
    import torch

    accumulator = DecisionMetricAccumulator()
    loss_total = 0.0
    loss_rows = 0
    win_squared_error = 0.0
    win_correct = 0
    win_rows = 0
    margin_absolute_error = 0.0
    margin_rows = 0
    net.eval()
    with torch.no_grad():
        for batch in dataset.loader(num_workers=args.num_workers):
            loss, logits, used_rows = _batch_loss(net, batch, loss_name, device, args)
            if used_rows:
                loss_total += float(loss.detach().cpu()) * used_rows
                loss_rows += used_rows
            accumulator.update(
                logits.detach().cpu().numpy(),
                batch["targets"].numpy(),
                action_types=batch["action_types"],
                num_legal=(
                    batch["num_legal"] if batch["has_decision_metadata"] else None
                ),
                legal_actions=(
                    batch["legal_actions"] if batch["has_decision_metadata"] else None
                ),
                candidate_values=batch["candidate_values_raw"],
            )
            if hasattr(net, "policy_value"):
                outputs = net.policy_value(
                    batch["features"].to(device, non_blocking=True)
                )
                win_mask = batch["win_value_mask"].to(device, non_blocking=True)
                if bool(win_mask.any()):
                    targets = batch["win_value_targets"].to(device, non_blocking=True)[
                        win_mask
                    ]
                    predictions = outputs.win_value[win_mask]
                    win_squared_error += float(
                        ((predictions - targets) ** 2).sum().detach().cpu()
                    )
                    win_correct += int(
                        (predictions.ge(0) == targets.ge(0)).sum().detach().cpu()
                    )
                    win_rows += int(win_mask.sum().item())
                margin_mask = batch["vp_margin_mask"].to(device, non_blocking=True)
                if bool(margin_mask.any()):
                    targets = batch["vp_margin_targets"].to(device, non_blocking=True)[
                        margin_mask
                    ]
                    predictions = outputs.vp_margin[margin_mask]
                    margin_absolute_error += float(
                        (predictions - targets).abs().sum().detach().cpu()
                    )
                    margin_rows += int(margin_mask.sum().item())
    if not accumulator.rows:
        return float("nan"), {}
    metrics = accumulator.compute()
    if win_rows:
        metrics["win_value_mse"] = win_squared_error / win_rows
        metrics["win_value_accuracy"] = win_correct / win_rows
        metrics["win_value_rows"] = win_rows
    if margin_rows:
        metrics["vp_margin_mae"] = margin_absolute_error / margin_rows
        metrics["vp_margin_rows"] = margin_rows
    return loss_total / loss_rows if loss_rows else float("nan"), metrics


def _selection_value(metrics: dict[str, Any], val_loss: float) -> tuple[str, float]:
    regret = metrics.get("mean_regret")
    if regret is not None:
        name, value = "mean_regret", float(regret)
    else:
        name, value = "val_loss", float(val_loss)
    if not math.isfinite(value):
        raise ValueError(
            f"Validation produced no finite {name}; verify that held-out rows "
            "contain the metadata required by the selected loss"
        )
    return name, value


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.hybrid_listwise_weight < 0:
        raise ValueError("hybrid_listwise_weight must be non-negative")
    if not math.isfinite(args.augmentation_weight) or args.augmentation_weight <= 0:
        raise ValueError("augmentation_weight must be positive and finite")
    outcome_parameters = (
        args.outcome_loss_bonus,
        args.outcome_vp_deficit_bonus,
        args.outcome_vp_deficit_scale,
    )
    if any(not math.isfinite(value) for value in outcome_parameters):
        raise ValueError("outcome weighting parameters must be finite")
    if args.outcome_loss_bonus < 0 or args.outcome_vp_deficit_bonus < 0:
        raise ValueError("outcome weighting bonuses must be non-negative")
    if args.outcome_vp_deficit_scale <= 0:
        raise ValueError("outcome_vp_deficit_scale must be positive")
    if args.outcome_weighted_augmentation_data_dir and not (
        args.outcome_loss_bonus > 0 or args.outcome_vp_deficit_bonus > 0
    ):
        raise ValueError("outcome weighting requires at least one positive bonus")
    if args.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if args.win_value_weight is None:
        args.win_value_weight = (
            0.25 if args.architecture == "factored_policy_value" else 0.0
        )
    if args.vp_margin_weight is None:
        args.vp_margin_weight = (
            0.05 if args.architecture == "factored_policy_value" else 0.0
        )
    if args.win_value_weight < 0 or args.vp_margin_weight < 0:
        raise ValueError("value loss weights must be non-negative")
    if args.architecture != "factored_policy_value" and (
        args.win_value_weight or args.vp_margin_weight
    ):
        raise ValueError(
            "Auxiliary value losses require --architecture factored_policy_value"
        )
    if args.freeze_base_policy and args.architecture not in {
        "spatial_edge_residual",
        "spatial_robber_residual",
    }:
        raise ValueError(
            "--freeze-base-policy requires a spatial residual architecture"
        )
    seed = args.split_seed if args.seed is None else args.seed
    seed_everything(seed)

    import torch

    command_args = list(argv) if argv is not None else sys.argv[1:]
    regular_augmentation_dirs = list(args.augmentation_data_dir or [])
    outcome_augmentation_dirs = list(args.outcome_weighted_augmentation_data_dir or [])
    augmentation_dirs = [*regular_augmentation_dirs, *outcome_augmentation_dirs]
    resolved_dirs = [path.resolve() for path in [*args.data_dir, *augmentation_dirs]]
    if len(resolved_dirs) != len(set(resolved_dirs)):
        raise ValueError("Each base or augmentation data directory must be unique")
    tracker = (
        TrainingRunTracker(args.run_dir, command=["colonist_1v1_bc.py", *command_args])
        if args.run_dir
        else None
    )
    if tracker:
        tracker.phase(
            "bc_training",
            data_dirs=[str(path) for path in args.data_dir],
            augmentation_data_dirs=[str(path) for path in augmentation_dirs],
            outcome_weighted_augmentation_data_dirs=[
                str(path) for path in outcome_augmentation_dirs
            ],
        )

    model_schema = build_model_schema(feature_profile=args.feature_profile)
    base_paths = _resolve_dataset_paths(
        args.data_dir,
        expected_schema=model_schema,
        allow_legacy_schema=args.allow_legacy_dataset_schema,
    )
    base_plan = inspect_parquet_dataset(
        base_paths,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    augmentation_paths: list[Path] = []
    outcome_augmentation_paths: list[Path] = []
    plan = base_plan
    if augmentation_dirs:
        augmentation_corpora = [
            _resolve_dataset_paths(
                [data_dir],
                expected_schema=model_schema,
                allow_legacy_schema=args.allow_legacy_dataset_schema,
            )
            for data_dir in augmentation_dirs
        ]
        augmentation_paths = [
            path for corpus_paths in augmentation_corpora for path in corpus_paths
        ]
        outcome_start = len(regular_augmentation_dirs)
        outcome_augmentation_paths = [
            path
            for corpus_paths in augmentation_corpora[outcome_start:]
            for path in corpus_paths
        ]
        augmentation_plan = inspect_parquet_corpora(
            augmentation_corpora,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
        plan = combine_parquet_dataset_plans((base_plan, augmentation_plan))
    paths = [*base_paths, *augmentation_paths]
    print(f"Hashing {len(paths):,} selected input shards ...")
    input_shards, dataset_sha256 = hash_parquet_shards(paths)
    print(f"dataset_sha256={dataset_sha256}")
    if tracker:
        tracker.event(
            "bc_dataset_hashed",
            dataset_sha256=dataset_sha256,
            input_shards=input_shards,
        )
        tracker.update_manifest(
            bc_dataset={
                "dataset_sha256": dataset_sha256,
                "input_shards": input_shards,
            }
        )
    expected_dataset_contract = {
        name: value
        for name, value in {
            "dataset_sha256": args.expected_dataset_sha256,
            "shards": args.expected_shards,
            "train_rows": args.expected_train_rows,
            "val_rows": args.expected_val_rows,
            "test_rows": args.expected_test_rows,
        }.items()
        if value is not None
    }
    _validate_dataset_contract(
        dataset_sha256=dataset_sha256,
        shard_count=len(paths),
        train_rows=plan.rows_for("train"),
        val_rows=plan.rows_for("val"),
        test_rows=plan.rows_for("test"),
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_shards=args.expected_shards,
        expected_train_rows=args.expected_train_rows,
        expected_val_rows=args.expected_val_rows,
        expected_test_rows=args.expected_test_rows,
    )
    has_legal = LEGAL_ACTIONS_COLUMN in plan.available_columns
    has_candidates = CANDIDATE_VALUES_COLUMN in plan.available_columns
    loss_name = args.loss
    if loss_name == "auto":
        loss_name = "legal_ce" if has_legal else "cross_entropy"
    if loss_name in {"legal_ce", "listwise", "hybrid"} and not has_legal:
        raise ValueError(
            f"--loss {loss_name} requires dataset-v2 {LEGAL_ACTIONS_COLUMN}; "
            "regenerate teacher data or use --loss cross_entropy for legacy logs"
        )
    if loss_name in {"listwise", "hybrid"} and not has_candidates:
        raise ValueError(
            f"--loss {loss_name} requires scored {CANDIDATE_VALUES_COLUMN} rows"
        )
    has_win_values = any(
        WIN_VALUE_TARGET_COLUMN in columns
        for columns in plan.path_value_target_columns.values()
    )
    has_vp_margins = any(
        VP_MARGIN_TARGET_COLUMN in columns
        for columns in plan.path_value_target_columns.values()
    )
    if args.win_value_weight and not has_win_values:
        print(f"WARNING: no {WIN_VALUE_TARGET_COLUMN}; disabling win-value loss")
        args.win_value_weight = 0.0
    if args.vp_margin_weight and not has_vp_margins:
        print(f"WARNING: no {VP_MARGIN_TARGET_COLUMN}; disabling VP-margin loss")
        args.vp_margin_weight = 0.0

    schema_features = tuple(
        f"F_{name}" for name in model_schema["observation"]["features"]
    )
    if plan.feature_columns != schema_features:
        raise ValueError(
            "Dataset feature order does not match the requested model schema: "
            f"dataset={len(plan.feature_columns)} schema={len(schema_features)}"
        )
    schema_actions = len(model_schema["actions"])
    if args.n_actions != schema_actions:
        raise ValueError(
            f"--n-actions {args.n_actions} does not match action schema {schema_actions}"
        )

    device = resolve_torch_device(args.device)
    print(
        f"dataset shards={len(plan.paths):,} train_rows={plan.rows_for('train'):,} "
        f"val_rows={plan.rows_for('val'):,} test_rows={plan.rows_for('test'):,}"
    )
    if not plan.has_game_ids:
        print(
            "WARNING: legacy dataset has no GAME_ID; splitting by whole Parquet shard. "
            "Regenerate data for explicit by-game identities and legal-action learning."
        )
    print(
        f"architecture={args.architecture} loss={loss_name} device={device} seed={seed} "
        f"win_value_weight={args.win_value_weight:g} "
        f"vp_margin_weight={args.vp_margin_weight:g}"
    )

    sample_weight_fn = hard_state_sample_weights if args.hard_states else None
    outcome_weight_fn = partial(
        outcome_deficit_sample_weights,
        loss_bonus=args.outcome_loss_bonus,
        vp_deficit_bonus=args.outcome_vp_deficit_bonus,
        vp_deficit_scale=args.outcome_vp_deficit_scale,
    )
    outcome_weighting_summary: dict[str, Any] = {}
    if outcome_augmentation_paths:
        import pandas as pd

        row_count = 0
        weight_sum = 0.0
        weight_min = math.inf
        weight_max = -math.inf
        game_targets: dict[tuple[str, str], tuple[float, float]] = {}
        for path in outcome_augmentation_paths:
            frame = pd.read_parquet(
                path,
                columns=["GAME_ID", WIN_VALUE_TARGET_COLUMN, VP_MARGIN_TARGET_COLUMN],
            )
            weights = outcome_weight_fn(frame)
            row_count += len(frame)
            weight_sum += float(weights.sum())
            weight_min = min(weight_min, float(weights.min()))
            weight_max = max(weight_max, float(weights.max()))
            corpus_id = str(path.parent.resolve())
            for game_id, outcome, margin in zip(
                frame["GAME_ID"],
                frame[WIN_VALUE_TARGET_COLUMN],
                frame[VP_MARGIN_TARGET_COLUMN],
            ):
                key = (corpus_id, str(game_id))
                target = (float(outcome), float(margin))
                previous = game_targets.setdefault(key, target)
                if previous != target:
                    raise ValueError(
                        f"Inconsistent terminal targets within outcome game {game_id}"
                    )
        outcomes = [target[0] for target in game_targets.values()]
        outcome_weighting_summary = {
            "rows": row_count,
            "games": len(game_targets),
            "wins": sum(value > 0 for value in outcomes),
            "losses": sum(value < 0 for value in outcomes),
            "draws": sum(value == 0 for value in outcomes),
            "weight_min": weight_min,
            "weight_mean": weight_sum / row_count,
            "weight_max": weight_max,
        }
        print(f"outcome_weighting={outcome_weighting_summary}")
        if tracker:
            tracker.event(
                "bc_outcome_weighting_verified",
                outcome_weighted_augmentation_data_dirs=[
                    str(path) for path in outcome_augmentation_dirs
                ],
                loss_bonus=args.outcome_loss_bonus,
                vp_deficit_bonus=args.outcome_vp_deficit_bonus,
                vp_deficit_scale=args.outcome_vp_deficit_scale,
                **outcome_weighting_summary,
            )
            tracker.update_manifest(
                bc_outcome_weighting={
                    "data_dirs": [str(path) for path in outcome_augmentation_dirs],
                    "loss_bonus": args.outcome_loss_bonus,
                    "vp_deficit_bonus": args.outcome_vp_deficit_bonus,
                    "vp_deficit_scale": args.outcome_vp_deficit_scale,
                    "summary": outcome_weighting_summary,
                }
            )
    augmentation_path_weights = {
        path: args.augmentation_weight for path in augmentation_paths
    }
    train_data = ParquetDecisionBatches(
        plan,
        "train",
        batch_size=args.batch_size,
        seed=seed,
        shuffle=True,
        sample_weight_fn=sample_weight_fn,
        path_weights=augmentation_path_weights,
        path_sample_weight_fns={
            path: outcome_weight_fn for path in outcome_augmentation_paths
        },
    )
    val_data = ParquetDecisionBatches(
        plan, "val", batch_size=args.batch_size, seed=seed
    )
    test_data = ParquetDecisionBatches(
        plan, "test", batch_size=args.batch_size, seed=seed
    )

    hidden = tuple(args.hidden)
    net = build_bc_policy(
        args.architecture,
        plan.feature_columns,
        args.n_actions,
        hidden_sizes=hidden,
        embedding_dim=args.embedding_dim,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    initial_checkpoint: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        if not plan.rows_for("val"):
            raise ValueError("--init-checkpoint requires a non-empty validation split")
        initial_checkpoint = _load_initial_checkpoint(
            args.init_checkpoint,
            net,
            expected_schema=model_schema,
            architecture=args.architecture,
            hidden_sizes=hidden,
            embedding_dim=args.embedding_dim,
            obs_dim=len(plan.feature_columns),
            n_actions=args.n_actions,
            device=device,
        )
        print(
            "initialized_from="
            f"{initial_checkpoint['path']} sha256={initial_checkpoint['sha256']}"
        )
    if args.freeze_base_policy:
        if initial_checkpoint is None:
            raise ValueError("--freeze-base-policy requires --init-checkpoint")
        net.freeze_base_policy()
    trainable_parameter_count = sum(
        parameter.numel() for parameter in net.parameters() if parameter.requires_grad
    )
    win_value_target_shards = sum(
        WIN_VALUE_TARGET_COLUMN in columns
        for columns in plan.path_value_target_columns.values()
    )
    vp_margin_target_shards = sum(
        VP_MARGIN_TARGET_COLUMN in columns
        for columns in plan.path_value_target_columns.values()
    )
    print(
        f"parameters={parameter_count:,} trainable={trainable_parameter_count:,} "
        "value_target_shards="
        f"win:{win_value_target_shards}/{len(plan.paths)} "
        f"vp_margin:{vp_margin_target_shards}/{len(plan.paths)}"
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in net.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    writer = None
    if args.tensorboard is not None:
        from torch.utils.tensorboard import SummaryWriter

        args.tensorboard.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(args.tensorboard))

    best_state = None
    best_epoch = None
    best_metric_name = None
    best_metric_value = float("inf")
    best_val_loss = float("nan")
    best_val_metrics: dict[str, Any] = {}
    initial_val_loss: float | None = None
    initial_val_metrics: dict[str, Any] = {}
    last_train_rows = 0
    if initial_checkpoint is not None:
        initial_val_loss, initial_val_metrics = _evaluate(
            net, val_data, loss_name, device, args
        )
        initial_metric_name, initial_metric_value = _selection_value(
            initial_val_metrics, initial_val_loss
        )
        best_state = {
            key: value.detach().cpu().clone() for key, value in net.state_dict().items()
        }
        best_epoch = 0
        best_metric_name = initial_metric_name
        best_metric_value = initial_metric_value
        best_val_loss = initial_val_loss
        best_val_metrics = copy.deepcopy(initial_val_metrics)
        print(
            f"epoch 0/{args.epochs} init_val_loss={initial_val_loss:.4f} "
            f"{initial_metric_name}={initial_metric_value:.4f}"
        )
        if tracker:
            tracker.event(
                "bc_initial_validation",
                init_checkpoint=initial_checkpoint,
                val_loss=initial_val_loss,
                selection_metric=initial_metric_name,
                selection_value=initial_metric_value,
                **{
                    key: value
                    for key, value in initial_val_metrics.items()
                    if key != "accuracy"
                },
            )
    try:
        for epoch in range(args.epochs):
            train_data.set_epoch(epoch)
            net.train()
            epoch_loss_total = 0.0
            epoch_loss_rows = 0
            seen_rows = 0
            for batch in train_data.loader(num_workers=args.num_workers):
                optimizer.zero_grad(set_to_none=True)
                loss, _, used_rows = _batch_loss(net, batch, loss_name, device, args)
                seen_rows += len(batch["targets"])
                if not used_rows:
                    continue
                loss.backward()
                optimizer.step()
                epoch_loss_total += float(loss.detach().cpu()) * used_rows
                epoch_loss_rows += used_rows
            if not epoch_loss_rows:
                raise ValueError(
                    f"No usable rows for {loss_name}; candidate-valued choice rows are required"
                )
            last_train_rows = seen_rows
            train_loss = epoch_loss_total / epoch_loss_rows
            if plan.rows_for("val"):
                val_loss, val_metrics = _evaluate(
                    net, val_data, loss_name, device, args
                )
            else:
                val_loss, val_metrics = train_loss, {}
            metric_name, metric_value = _selection_value(val_metrics, val_loss)
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                best_metric_name = metric_name
                best_epoch = epoch + 1
                best_val_loss = val_loss
                best_val_metrics = copy.deepcopy(val_metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in net.state_dict().items()
                }

            val_acc = val_metrics.get("accuracy", float("nan"))
            choice_acc = val_metrics.get(
                "legal_choice_accuracy", val_metrics.get("choice_accuracy")
            )
            choice_text = (
                f" choice_acc={choice_acc:.4f}" if choice_acc is not None else ""
            )
            regret = val_metrics.get("mean_regret")
            regret_text = f" regret={regret:.4f}" if regret is not None else ""
            print(
                f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                f"{choice_text}{regret_text}"
            )
            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("accuracy/val", val_acc, epoch)
                for key in (
                    "legal_choice_accuracy",
                    "legal_top3_accuracy",
                    "mean_regret",
                ):
                    if key in val_metrics:
                        writer.add_scalar(f"metrics/{key}", val_metrics[key], epoch)
            if tracker:
                tracker.event(
                    "bc_epoch",
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    selection_metric=metric_name,
                    selection_value=metric_value,
                    **{
                        key: value
                        for key, value in val_metrics.items()
                        if key != "accuracy"
                    },
                )
    finally:
        if writer is not None:
            writer.close()

    if best_state is None:  # Defensive; positive epochs and usable rows set this.
        best_state = {
            key: value.detach().cpu() for key, value in net.state_dict().items()
        }
        best_epoch = args.epochs
        best_metric_name = "train_loss"
        best_metric_value = train_loss
        best_val_loss = train_loss
    net.load_state_dict(best_state)
    net.to(device)

    test_metrics: dict[str, Any] = {}
    if plan.rows_for("test"):
        test_loss, test_metrics = _evaluate(net, test_data, loss_name, device, args)
        test_metrics = {"loss": test_loss, **test_metrics}
        print(f"test {test_metrics}")
        if tracker:
            tracker.event("bc_test", **test_metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)
    schema_path = write_model_schema(checkpoint_schema_path(args.out), model_schema)
    meta = BcCheckpointMeta(
        obs_dim=len(plan.feature_columns),
        n_actions=args.n_actions,
        hidden_sizes=list(hidden),
        epochs=args.epochs,
        architecture=args.architecture,
        embedding_dim=args.embedding_dim,
        win_value_weight=args.win_value_weight,
        vp_margin_weight=args.vp_margin_weight,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        base_policy_frozen=args.freeze_base_policy,
        initialization_mode=(
            initial_checkpoint["initialization_mode"]
            if initial_checkpoint is not None
            else None
        ),
        win_value_target_shards=win_value_target_shards,
        vp_margin_target_shards=vp_margin_target_shards,
        val_accuracy=best_val_metrics.get("accuracy"),
        train_rows=last_train_rows,
        data_dirs=[
            *[str(path) for path in args.data_dir],
            *[str(path) for path in augmentation_dirs],
        ],
        augmentation_data_dirs=[str(path) for path in augmentation_dirs],
        augmentation_weight=args.augmentation_weight,
        outcome_weighted_augmentation_data_dirs=[
            str(path) for path in outcome_augmentation_dirs
        ],
        outcome_loss_bonus=(
            args.outcome_loss_bonus if outcome_augmentation_dirs else 0.0
        ),
        outcome_vp_deficit_bonus=(
            args.outcome_vp_deficit_bonus if outcome_augmentation_dirs else 0.0
        ),
        outcome_vp_deficit_scale=args.outcome_vp_deficit_scale,
        outcome_weighting_summary=outcome_weighting_summary,
        init_checkpoint=(
            initial_checkpoint["path"] if initial_checkpoint is not None else None
        ),
        init_checkpoint_sha256=(
            initial_checkpoint["sha256"] if initial_checkpoint is not None else None
        ),
        initial_val_loss=initial_val_loss,
        initial_val_metrics=initial_val_metrics,
        val_loss=best_val_loss,
        loss_name=loss_name,
        listwise_temperature=(
            args.listwise_temperature if loss_name in {"listwise", "hybrid"} else None
        ),
        hybrid_listwise_weight=(
            args.hybrid_listwise_weight if loss_name == "hybrid" else None
        ),
        seed=seed,
        device=str(device),
        best_epoch=best_epoch,
        selection_metric=best_metric_name,
        selection_value=best_metric_value,
        val_rows=plan.rows_for("val"),
        test_rows=plan.rows_for("test"),
        val_metrics=(
            {"loss": best_val_loss, **best_val_metrics} if plan.rows_for("val") else {}
        ),
        test_metrics=test_metrics,
        model_schema=model_schema,
        input_shards=input_shards,
        dataset_sha256=dataset_sha256,
        expected_dataset_contract=expected_dataset_contract,
    )
    meta_path = args.out.with_suffix(".meta.json")
    meta.save(meta_path)
    if tracker:
        tracker.event(
            "bc_complete",
            checkpoint=str(args.out),
            meta_path=str(meta_path),
            schema_path=str(schema_path),
            best_epoch=best_epoch,
            selection_metric=best_metric_name,
            selection_value=best_metric_value,
        )
        tracker.update_manifest(
            bc_checkpoint=str(args.out),
            bc_schema=str(schema_path),
            bc_model={
                "architecture": args.architecture,
                "embedding_dim": args.embedding_dim,
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "base_policy_frozen": args.freeze_base_policy,
                "initialization_mode": (
                    initial_checkpoint["initialization_mode"]
                    if initial_checkpoint is not None
                    else None
                ),
                "win_value_weight": args.win_value_weight,
                "vp_margin_weight": args.vp_margin_weight,
                "win_value_target_shards": win_value_target_shards,
                "vp_margin_target_shards": vp_margin_target_shards,
                "outcome_weighted_augmentation_data_dirs": [
                    str(path) for path in outcome_augmentation_dirs
                ],
                "outcome_loss_bonus": (
                    args.outcome_loss_bonus if outcome_augmentation_dirs else 0.0
                ),
                "outcome_vp_deficit_bonus": (
                    args.outcome_vp_deficit_bonus if outcome_augmentation_dirs else 0.0
                ),
                "outcome_vp_deficit_scale": args.outcome_vp_deficit_scale,
                "outcome_weighting_summary": outcome_weighting_summary,
            },
            bc_expected_dataset_contract=expected_dataset_contract,
            bc_init_checkpoint=initial_checkpoint,
            phase="bc_complete",
        )
    print(
        f"Wrote {args.out}, {meta_path}, and {schema_path} "
        f"(best_epoch={best_epoch}, {best_metric_name}={best_metric_value:.6f})"
    )


if __name__ == "__main__":
    main()

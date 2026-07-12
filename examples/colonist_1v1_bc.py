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
from pathlib import Path
from typing import Any

from catanatron.gym.bc_training import (
    CANDIDATE_VALUES_COLUMN,
    LEGAL_ACTIONS_COLUMN,
    DecisionMetricAccumulator,
    ParquetDecisionBatches,
    candidate_listwise_loss,
    hash_parquet_shards,
    inspect_parquet_dataset,
    legal_masked_cross_entropy,
    resolve_torch_device,
    seed_everything,
)
from catanatron.gym.colonist_training import (
    BcCheckpointMeta,
    TrainingRunTracker,
    build_mlp_layers,
    hard_state_sample_weights,
    resolve_teacher_parquet_paths,
)
from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    read_model_schema,
    validate_model_schema,
    write_model_schema,
)

DEFAULT_BC_CHECKPOINT_PATH = Path("runs/colonist_bc_policy.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories of game *.parquet files.",
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
    parser.add_argument("--hidden", type=int, nargs=2, default=(512, 512))
    parser.add_argument(
        "--n-actions",
        type=int,
        default=332,
        help="Policy action head size. Must match the recorded action schema.",
    )
    parser.add_argument(
        "--loss",
        choices=("auto", "cross_entropy", "legal_ce", "listwise"),
        default="auto",
        help="auto selects legal_ce for dataset-v2 logs and cross_entropy for legacy logs.",
    )
    parser.add_argument(
        "--listwise-temperature",
        type=float,
        default=0.25,
        help="Soft-target temperature for --loss listwise.",
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
    return parser


def _resolve_dataset_paths(
    data_dirs: list[Path],
    *,
    expected_schema: dict[str, Any],
    allow_legacy_schema: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for directory in data_dirs:
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


def _weighted_mean(losses, weights):
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("Training batch has no positive sample weight")
    return (losses * weights).sum() / denominator


def _batch_loss(net, batch, loss_name: str, device, args):
    from torch.nn import functional as F

    features = batch["features"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    logits = net(features)
    weights = batch["sample_weights"].to(device, non_blocking=True)
    if loss_name == "cross_entropy":
        row_losses = F.cross_entropy(logits, targets, reduction="none")
        return _weighted_mean(row_losses, weights), logits, len(targets)

    legal_indices = batch["legal_indices"].to(device, non_blocking=True)
    legal_mask = batch["legal_mask"].to(device, non_blocking=True)
    if loss_name == "legal_ce":
        row_losses = legal_masked_cross_entropy(
            logits,
            targets,
            legal_indices,
            legal_mask,
            reduction="none",
        )
        return _weighted_mean(row_losses, weights), logits, len(targets)

    values = batch["candidate_values"].to(device, non_blocking=True)
    value_mask = batch["candidate_mask"].to(device, non_blocking=True)
    row_losses, valid = candidate_listwise_loss(
        logits,
        legal_indices,
        legal_mask,
        values,
        value_mask,
        temperature=args.listwise_temperature,
        tie_tolerance=args.tie_tolerance,
        reduction="none",
    )
    valid_weights = weights[valid]
    if not len(row_losses):
        return logits.sum() * 0.0, logits, 0
    return _weighted_mean(row_losses, valid_weights), logits, len(row_losses)


def _evaluate(
    net, dataset, loss_name: str, device, args
) -> tuple[float, dict[str, Any]]:
    import torch

    accumulator = DecisionMetricAccumulator()
    loss_total = 0.0
    loss_rows = 0
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
    if not accumulator.rows:
        return float("nan"), {}
    metrics = accumulator.compute()
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
    seed = args.split_seed if args.seed is None else args.seed
    seed_everything(seed)

    import torch

    command_args = list(argv) if argv is not None else sys.argv[1:]
    tracker = (
        TrainingRunTracker(args.run_dir, command=["colonist_1v1_bc.py", *command_args])
        if args.run_dir
        else None
    )
    if tracker:
        tracker.phase("bc_training", data_dirs=[str(path) for path in args.data_dir])

    model_schema = build_model_schema(feature_profile=args.feature_profile)
    paths = _resolve_dataset_paths(
        args.data_dir,
        expected_schema=model_schema,
        allow_legacy_schema=args.allow_legacy_dataset_schema,
    )
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
    plan = inspect_parquet_dataset(
        paths,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    has_legal = LEGAL_ACTIONS_COLUMN in plan.available_columns
    has_candidates = CANDIDATE_VALUES_COLUMN in plan.available_columns
    loss_name = args.loss
    if loss_name == "auto":
        loss_name = "legal_ce" if has_legal else "cross_entropy"
    if loss_name in {"legal_ce", "listwise"} and not has_legal:
        raise ValueError(
            f"--loss {loss_name} requires dataset-v2 {LEGAL_ACTIONS_COLUMN}; "
            "regenerate teacher data or use --loss cross_entropy for legacy logs"
        )
    if loss_name == "listwise" and not has_candidates:
        raise ValueError(
            f"--loss listwise requires scored {CANDIDATE_VALUES_COLUMN} rows"
        )

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
    print(f"loss={loss_name} device={device} seed={seed}")

    sample_weight_fn = hard_state_sample_weights if args.hard_states else None
    train_data = ParquetDecisionBatches(
        plan,
        "train",
        batch_size=args.batch_size,
        seed=seed,
        shuffle=True,
        sample_weight_fn=sample_weight_fn,
    )
    val_data = ParquetDecisionBatches(
        plan, "val", batch_size=args.batch_size, seed=seed
    )
    test_data = ParquetDecisionBatches(
        plan, "test", batch_size=args.batch_size, seed=seed
    )

    hidden = tuple(args.hidden)
    net = build_mlp_layers(len(plan.feature_columns), args.n_actions, hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
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
    last_train_rows = 0
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
        val_accuracy=best_val_metrics.get("accuracy"),
        train_rows=last_train_rows,
        data_dirs=[str(path) for path in args.data_dir],
        val_loss=best_val_loss,
        loss_name=loss_name,
        listwise_temperature=(
            args.listwise_temperature if loss_name == "listwise" else None
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
            phase="bc_complete",
        )
    print(
        f"Wrote {args.out}, {meta_path}, and {schema_path} "
        f"(best_epoch={best_epoch}, {best_metric_name}={best_metric_value:.6f})"
    )


if __name__ == "__main__":
    main()

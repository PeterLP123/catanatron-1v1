#!/usr/bin/env python3
"""Train one value-only critic and gate it against public-score baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from catanatron.gym.bc_training import (
    hash_parquet_shards,
    resolve_torch_device,
    seed_everything,
)
from catanatron.gym.model_architectures import FactoredOutcomeCritic
from catanatron.gym.model_schema import build_model_schema, write_model_schema
from catanatron.gym.outcome_critic import (
    OutcomeCriticBatches,
    OutcomeMetricAccumulator,
    build_outcome_dataset_plan,
    critic_gate_deltas,
    outcome_critic_loss,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        nargs="+",
        type=Path,
        required=True,
        help="One logical corpus; repeat for independently split DAgger iterations.",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin-weight", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=101)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--expected-train-rows", type=int)
    parser.add_argument("--expected-val-rows", type=int)
    parser.add_argument("--expected-test-rows", type=int)
    parser.add_argument("--minimum-auc-improvement", type=float, default=0.01)
    parser.add_argument("--minimum-brier-improvement", type=float, default=0.01)
    parser.add_argument("--minimum-margin-mae-improvement", type=float, default=0.25)
    parser.add_argument("--minimum-margin-rmse-improvement", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _move_batch(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _evaluate(model, dataset: OutcomeCriticBatches, *, device, num_workers: int):
    import torch

    model.eval()
    metrics = OutcomeMetricAccumulator()
    with torch.no_grad():
        for raw_batch in dataset.loader(num_workers=num_workers):
            batch = _move_batch(raw_batch, device)
            output = model(batch["features"])
            metrics.update(output, batch)
    return metrics.finalize()


def _selection_value(metrics: Mapping[str, Any]) -> float:
    win = metrics["win"]
    margin = metrics["vp_margin"]
    if win is None or margin is None:
        return math.inf
    return float(win["brier"] + 0.01 * margin["mae"])


def _offline_gates(
    val_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    *,
    minimum_auc_improvement: float,
    minimum_brier_improvement: float,
    minimum_margin_mae_improvement: float,
    minimum_margin_rmse_improvement: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    deltas = {
        "val": critic_gate_deltas(val_metrics),
        "test": critic_gate_deltas(test_metrics),
    }
    gates = []

    def add(name: str, passed: bool, actual: float, threshold: float) -> None:
        gates.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": float(actual),
                "threshold": float(threshold),
            }
        )

    for split, current in deltas.items():
        add(
            f"{split}_win_auc_improvement",
            current["win_auc"] >= minimum_auc_improvement,
            current["win_auc"],
            minimum_auc_improvement,
        )
        brier_improvement = -current["win_brier"]
        add(
            f"{split}_win_brier_improvement",
            brier_improvement >= minimum_brier_improvement,
            brier_improvement,
            minimum_brier_improvement,
        )
        margin_mae_improvement = -current["vp_margin_mae"]
        add(
            f"{split}_margin_mae_improvement",
            margin_mae_improvement >= minimum_margin_mae_improvement,
            margin_mae_improvement,
            minimum_margin_mae_improvement,
        )
        margin_rmse_improvement = -current["vp_margin_rmse"]
        add(
            f"{split}_margin_rmse_improvement",
            margin_rmse_improvement >= minimum_margin_rmse_improvement,
            margin_rmse_improvement,
            minimum_margin_rmse_improvement,
        )
    return gates, deltas


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.embedding_dim <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("embedding_dim, batch_size, and epochs must be positive")
    if args.lr <= 0 or args.margin_weight < 0:
        raise ValueError("lr must be positive and margin_weight non-negative")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    import torch

    seed_everything(args.seed)
    device = resolve_torch_device(args.device)
    plan, target_index = build_outcome_dataset_plan(
        args.corpus,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    shard_rows, dataset_sha256 = hash_parquet_shards(plan.paths)
    split_rows = {split: plan.rows_for(split) for split in ("train", "val", "test")}
    expected_contract = {
        "dataset_sha256": args.expected_dataset_sha256,
        "shards": args.expected_shards,
        "train_rows": args.expected_train_rows,
        "val_rows": args.expected_val_rows,
        "test_rows": args.expected_test_rows,
    }
    observed_contract = {
        "dataset_sha256": dataset_sha256,
        "shards": len(plan.paths),
        **{f"{split}_rows": rows for split, rows in split_rows.items()},
    }
    mismatches = {
        key: {"expected": expected, "observed": observed_contract[key]}
        for key, expected in expected_contract.items()
        if expected is not None and expected != observed_contract[key]
    }
    if mismatches:
        raise ValueError(f"Dataset contract mismatch before training: {mismatches}")
    if not split_rows["val"] or not split_rows["test"]:
        raise ValueError("Outcome critic requires non-empty validation and test splits")

    model = FactoredOutcomeCritic(
        plan.feature_columns, embedding_dim=args.embedding_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_data = OutcomeCriticBatches(
        plan,
        target_index,
        "train",
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
    )
    val_data = OutcomeCriticBatches(
        plan,
        target_index,
        "val",
        batch_size=args.batch_size,
    )
    test_data = OutcomeCriticBatches(
        plan,
        target_index,
        "test",
        batch_size=args.batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_tmp = args.out.with_name(f".{args.out.name}.tmp")
    best_epoch = 0
    best_selection = math.inf
    best_val_metrics = None
    events = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_data.set_epoch(epoch)
        loss_sum = 0.0
        win_loss_sum = 0.0
        margin_loss_sum = 0.0
        rows = 0
        for raw_batch in train_data.loader(num_workers=args.num_workers):
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["features"])
            loss, win_loss, margin_loss = outcome_critic_loss(
                output, batch, margin_weight=args.margin_weight
            )
            loss.backward()
            optimizer.step()
            batch_rows = len(batch["features"])
            rows += batch_rows
            loss_sum += float(loss.detach()) * batch_rows
            win_loss_sum += float(win_loss.detach()) * batch_rows
            margin_loss_sum += float(margin_loss.detach()) * batch_rows
        val_metrics = _evaluate(
            model, val_data, device=device, num_workers=args.num_workers
        )
        selection = _selection_value(val_metrics)
        event = {
            "epoch": epoch,
            "train_rows": rows,
            "train_loss": loss_sum / rows,
            "train_win_loss": win_loss_sum / rows,
            "train_margin_loss": margin_loss_sum / rows,
            "selection_value": selection,
            "val_metrics": val_metrics,
        }
        events.append(event)
        print(
            f"epoch={epoch} train_loss={event['train_loss']:.6f} "
            f"val_brier={val_metrics['win']['brier']:.6f} "
            f"val_auc={val_metrics['win']['auc']:.6f} "
            f"val_margin_mae={val_metrics['vp_margin']['mae']:.6f}"
        )
        if selection < best_selection:
            best_epoch = epoch
            best_selection = selection
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), checkpoint_tmp)

    if best_val_metrics is None or not checkpoint_tmp.is_file():
        raise RuntimeError("Outcome critic training produced no finite selected epoch")
    os.replace(checkpoint_tmp, args.out)
    state_dict = torch.load(args.out, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    test_metrics = _evaluate(
        model, test_data, device=device, num_workers=args.num_workers
    )
    gates, deltas = _offline_gates(
        best_val_metrics,
        test_metrics,
        minimum_auc_improvement=args.minimum_auc_improvement,
        minimum_brier_improvement=args.minimum_brier_improvement,
        minimum_margin_mae_improvement=args.minimum_margin_mae_improvement,
        minimum_margin_rmse_improvement=args.minimum_margin_rmse_improvement,
    )
    model_schema = build_model_schema(feature_profile="raw")
    schema_path = args.out.with_suffix(".schema.json")
    write_model_schema(schema_path, model_schema)
    metadata = {
        "schema_version": "1.0",
        "kind": "factored_outcome_critic",
        "architecture": "factored_outcome_critic",
        "checkpoint": str(args.out),
        "schema_path": str(schema_path),
        "dataset_sha256": dataset_sha256,
        "input_shards": shard_rows,
        "expected_dataset_contract": expected_contract,
        "observed_dataset_contract": observed_contract,
        "feature_columns": list(plan.feature_columns),
        "embedding_dim": args.embedding_dim,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "selection_metric": "win_brier_plus_0.01_margin_mae",
        "selection_value": best_selection,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "margin_weight": args.margin_weight,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "device": str(device),
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "critic_minus_public_deltas": deltas,
        "gates": gates,
        "all_offline_gates_passed": all(gate["passed"] for gate in gates),
    }
    metadata_path = args.out.with_suffix(".meta.json")
    _atomic_json(metadata_path, metadata)
    _atomic_json(args.run_dir / "training_events.json", {"events": events})
    _atomic_json(args.run_dir / "offline_gate_report.json", metadata)
    print(
        f"selected_epoch={best_epoch} test_brier={test_metrics['win']['brier']:.6f} "
        f"test_auc={test_metrics['win']['auc']:.6f} "
        f"test_margin_mae={test_metrics['vp_margin']['mae']:.6f} "
        f"gates={'PASS' if metadata['all_offline_gates_passed'] else 'FAIL'}"
    )
    return 0 if metadata["all_offline_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

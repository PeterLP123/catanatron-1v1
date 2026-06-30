#!/usr/bin/env python3
"""
Behavioral cloning on Colonist 1v1 parquet logs (``ParquetDataAccumulator`` output).

**Dependencies** (install through this repository's extras)::

    pip install -e '.[gym,colonist]'

Example::

    python examples/colonist_1v1_bc.py --data-dir data/c1_teachers --epochs 10 \\
        --out runs/colonist_bc_policy.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from catanatron.gym.colonist_training import (
    BcCheckpointMeta,
    GAME_ID_COLUMN,
    LEGAL_ACTIONS_COLUMN,
    NUM_LEGAL_COLUMN,
    TrainingRunTracker,
    build_mlp_layers,
    decision_metrics,
    grouped_split_masks,
    load_teacher_parquet,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories of game *.parquet files.",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Held-out test split (by game). Reported once at the end.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for the grouped (by-game) train/val/test split.",
    )
    p.add_argument("--hidden", type=int, nargs=2, default=(512, 512))
    p.add_argument(
        "--n-actions",
        type=int,
        default=332,
        help="Policy action head size; keep full action space even if rare actions are absent.",
    )
    p.add_argument("--out", type=Path, default=Path("colonist_bc_policy.pt"))
    p.add_argument("--tensorboard", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    args = p.parse_args(argv)

    import torch
    from torch import nn

    tracker = (
        TrainingRunTracker(args.run_dir, command=["colonist_1v1_bc.py", *argv])
        if args.run_dir and argv is not None
        else TrainingRunTracker(args.run_dir, command=None) if args.run_dir else None
    )
    if tracker:
        tracker.phase("bc_training", data_dirs=[str(p) for p in args.data_dir])

    df = load_teacher_parquet(args.data_dir)
    feat_cols = sorted(c for c in df.columns if c.startswith("F_"))
    if not feat_cols:
        raise ValueError("No F_* feature columns found (vector teacher logs).")
    if "ACTION" not in df.columns:
        raise ValueError("Parquet must include ACTION column.")

    x = torch.as_tensor(df[feat_cols].to_numpy(np.float32))
    y = torch.as_tensor(df["ACTION"].to_numpy(np.int64))

    n = x.shape[0]

    # Honest split: group whole games into train/val/test so rows from one game
    # never leak across splits (dataset v2). Legacy logs without GAME_ID fall
    # back to the old row-level split with a warning.
    has_v2 = GAME_ID_COLUMN in df.columns
    if has_v2:
        train_mask, val_mask, test_mask = grouped_split_masks(
            df[GAME_ID_COLUMN].to_numpy(),
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
    else:
        print(
            f"WARNING: no {GAME_ID_COLUMN} column (legacy dataset); falling back to a "
            "row-level split. Regenerate data for honest by-game validation."
        )
        rng = np.random.default_rng(args.split_seed)
        shuffled = rng.permutation(n)
        n_val = int(n * args.val_fraction)
        n_test = int(n * args.test_fraction)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        val_mask[shuffled[:n_val]] = True
        test_mask[shuffled[n_val : n_val + n_test]] = True
        train_mask = ~(val_mask | test_mask)

    train_t = torch.as_tensor(np.flatnonzero(train_mask))
    val_t = torch.as_tensor(np.flatnonzero(val_mask))
    x_train, y_train = x[train_t], y[train_t]
    x_val, y_val = x[val_t], y[val_t]
    n_val = int(val_mask.sum())

    # Decision-quality columns for honest validation metrics (v2 only).
    if has_v2:
        val_action_types = df["ACTION_TYPE"].to_numpy()[val_mask]
        val_num_legal = df[NUM_LEGAL_COLUMN].to_numpy()[val_mask]
        val_legal_actions = df[LEGAL_ACTIONS_COLUMN].to_numpy()[val_mask]
    else:
        val_action_types = val_num_legal = val_legal_actions = None

    n_actions = max(args.n_actions, int(y.max().item() + 1))
    obs_dim = x.shape[1]
    hidden = tuple(args.hidden)

    net = build_mlp_layers(obs_dim, n_actions, hidden)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    writer = None
    if args.tensorboard is not None:
        from torch.utils.tensorboard import SummaryWriter

        args.tensorboard.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(args.tensorboard))

    def evaluate(xb, yb, action_types, num_legal, legal_actions):
        with torch.no_grad():
            logits = net(xb)
            loss = float(loss_fn(logits, yb).item())
        m = decision_metrics(
            logits.cpu().numpy(),
            yb.cpu().numpy(),
            action_types=action_types,
            num_legal=num_legal,
            legal_actions=legal_actions,
        )
        return loss, m

    val_metrics: dict = {}
    for epoch in range(args.epochs):
        perm_t = torch.randperm(x_train.shape[0])
        losses = []
        for start in range(0, x_train.shape[0], args.batch_size):
            idx = perm_t[start : start + args.batch_size]
            logits = net(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        if n_val:
            val_loss, val_metrics = evaluate(
                x_val, y_val, val_action_types, val_num_legal, val_legal_actions
            )
        else:
            val_loss, val_metrics = train_loss, {}
        val_acc = val_metrics.get("accuracy", float("nan"))
        # Honest headline: accuracy on genuine choices within the legal set.
        choice_acc = val_metrics.get(
            "legal_choice_accuracy", val_metrics.get("choice_accuracy")
        )
        choice_str = f"  choice_acc={choice_acc:.4f}" if choice_acc is not None else ""
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{choice_str}"
        )
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("accuracy/val", val_acc, epoch)
            for key in ("legal_choice_accuracy", "legal_top3_accuracy"):
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
                **{k: v for k, v in val_metrics.items() if k != "accuracy"},
            )
    if writer is not None:
        writer.close()

    # Final held-out test report (by game), if requested.
    if int(test_mask.sum()) > 0:
        test_t = torch.as_tensor(np.flatnonzero(test_mask))
        if has_v2:
            test_action_types = df["ACTION_TYPE"].to_numpy()[test_mask]
            test_num_legal = df[NUM_LEGAL_COLUMN].to_numpy()[test_mask]
            test_legal_actions = df[LEGAL_ACTIONS_COLUMN].to_numpy()[test_mask]
        else:
            test_action_types = test_num_legal = test_legal_actions = None
        _, test_metrics = evaluate(
            x[test_t],
            y[test_t],
            test_action_types,
            test_num_legal,
            test_legal_actions,
        )
        print(f"test  {test_metrics}")
        if tracker:
            tracker.event("bc_test", **test_metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    meta = BcCheckpointMeta(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_sizes=list(hidden),
        epochs=args.epochs,
        val_accuracy=val_acc if n_val else None,
        train_rows=int(x_train.shape[0]),
        data_dirs=[str(p) for p in args.data_dir],
        val_loss=val_loss if n_val else None,
    )
    meta.save(args.out.with_suffix(".meta.json"))
    if tracker:
        tracker.event(
            "bc_complete",
            checkpoint=str(args.out),
            meta_path=str(args.out.with_suffix(".meta.json")),
            val_accuracy=val_acc if n_val else None,
        )
        tracker.update_manifest(bc_checkpoint=str(args.out), phase="bc_complete")
    print(
        f"Wrote {args.out} and {args.out.with_suffix('.meta.json')}  "
        f"(obs_dim={obs_dim}, n_actions={n_actions})"
    )


if __name__ == "__main__":
    main()

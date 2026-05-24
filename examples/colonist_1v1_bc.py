#!/usr/bin/env python3
"""
Behavioral cloning on Colonist 1v1 parquet logs (``ParquetDataAccumulator`` output).

**Dependencies** (not in base ``catanatron`` — install explicitly)::

    pip install torch pyarrow

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
    TrainingRunTracker,
    build_mlp_layers,
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
        else TrainingRunTracker(args.run_dir, command=None)
        if args.run_dir
        else None
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
    n_val = int(n * args.val_fraction)
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

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

    def accuracy(xb, yb) -> float:
        with torch.no_grad():
            pred = net(xb).argmax(dim=1)
            return float((pred == yb).float().mean().item())

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
        val_loss = float(loss_fn(net(x_val), y_val).item()) if n_val else train_loss
        val_acc = accuracy(x_val, y_val) if n_val else accuracy(x_train, y_train)
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("accuracy/val", val_acc, epoch)
        if tracker:
            tracker.event(
                "bc_epoch",
                epoch=epoch + 1,
                epochs=args.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
    if writer is not None:
        writer.close()

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

#!/usr/bin/env python3
"""
Generate teacher trajectories for Colonist.io-style 1v1 (parquet) via ``catanatron-play``.

Recommended teachers (stronger than ``W,W``)::

    python examples/colonist_1v1_generate_data.py --num 5000 --teachers F,F --output data/c1_ff
    python examples/colonist_1v1_generate_data.py --num 2000 --teachers VP,F --output data/c1_vpf

Install the project extras first: ``pip install -e '.[gym,colonist]'``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from catanatron.colonist_1v1 import Colonist1v1TrainConfig
from catanatron.gym.colonist_training import touch_run_marker


def _write_meta(path: Path, meta: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def main(argv: list[str] | None = None) -> int:
    cfg = Colonist1v1TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num", type=int, default=100, help="Number of games.")
    p.add_argument("--seed", type=int, default=0, help="Base deterministic seed.")
    p.add_argument(
        "--shard-games",
        type=int,
        default=100,
        help="Games per atomic Parquet shard (default: 100).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted dataset after validating its configuration.",
    )
    p.add_argument(
        "--choices-only",
        action="store_true",
        help="Store only genuine decisions. Off by default so value targets retain all states.",
    )
    p.add_argument(
        "--output",
        default=cfg.output_dir,
        help="Output directory for ParquetDataAccumulator.",
    )
    p.add_argument(
        "--teachers",
        default=cfg.teacher_players,
        help='Two player codes for --players=..., e.g. "F,F" or "VP,F".',
    )
    p.add_argument(
        "--include-board-tensor",
        action="store_true",
        help="Also store flattened board tensors (slower).",
    )
    p.add_argument(
        "--score-candidates",
        action="store_true",
        help="Label each genuine decision's legal actions with F candidate "
        "values (parquet only, slower). Enables regret metrics and value targets.",
    )
    args = p.parse_args(argv)
    if args.num <= 0:
        p.error("--num must be positive")
    if args.shard_games <= 0:
        p.error("--shard-games must be positive")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_tmp in output_dir.glob(".shard-*.tmp.parquet"):
        stale_tmp.unlink(missing_ok=True)
    meta_path = output_dir / "dataset_meta.json"
    expected = {
        "schema_version": "2.0",
        "teachers": args.teachers,
        "num_games": args.num,
        "requested_games": args.num,
        "seed": args.seed,
        "shard_games": args.shard_games,
        "choices_only": args.choices_only,
        "score_candidates": args.score_candidates,
        "include_board_tensor": args.include_board_tensor,
        "colonist_1v1": True,
    }
    existing_files = list(output_dir.glob("*.parquet"))
    if args.resume:
        meta = _read_meta(meta_path)
        if not meta:
            p.error(f"--resume requires {meta_path}")
        mismatches = {
            key: (meta.get(key), value)
            for key, value in expected.items()
            if meta.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: stored={old!r} requested={new!r}"
                for key, (old, new) in mismatches.items()
            )
            p.error(f"resume configuration mismatch: {details}")
    else:
        if existing_files or meta_path.exists():
            p.error(
                f"{output_dir} already contains dataset artifacts; use --resume or a new output directory"
            )
        touch_run_marker(output_dir)
        meta = {
            **expected,
            "status": "in_progress",
            "completed_games": 0,
            "next_seed": args.seed,
            "rows": 0,
            "parquet_files": 0,
            "command": None,
        }
        _write_meta(meta_path, meta)

    completed = int(meta.get("completed_games", 0))
    if completed > args.num:
        p.error(
            f"dataset metadata reports {completed} completed games for a {args.num}-game request"
        )
    remaining = args.num - completed
    if remaining == 0:
        meta["status"] = "complete"
        _write_meta(meta_path, meta)
        print(f"Dataset already complete: {output_dir}")
        return 0

    cmd = [sys.executable, "-m", "catanatron.cli.play"]

    cmd.extend(
        [
            "--colonist-1v1",
            "--players",
            args.teachers,
            "--num",
            str(remaining),
            "--output",
            args.output,
            "--output-format",
            "parquet",
            "--quiet",
            "--seed",
            str(args.seed + completed),
            "--parquet-shard-games",
            str(args.shard_games),
            "--parquet-start-shard",
            str(int(meta.get("parquet_files", 0))),
            "--dataset-meta",
            str(meta_path),
        ]
    )
    if args.include_board_tensor:
        cmd.append("--include-board-tensor")
    if args.score_candidates:
        cmd.append("--score-candidates")
    if args.choices_only:
        cmd.append("--choices-only")

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0:
        meta = _read_meta(meta_path)
        if int(meta.get("completed_games", 0)) != args.num:
            meta["status"] = "in_progress"
            _write_meta(meta_path, meta)
            print(
                "Generation exited without accounting for every requested game; resume the dataset.",
                file=sys.stderr,
            )
            return 1
        meta["status"] = "complete"
        meta["command"] = " ".join(cmd)
        _write_meta(meta_path, meta)
        print(f"Wrote {meta_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

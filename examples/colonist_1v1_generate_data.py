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
import shutil
import subprocess
import sys
from pathlib import Path

from catanatron.colonist_1v1 import Colonist1v1TrainConfig
from catanatron.gym.colonist_training import touch_run_marker, write_dataset_metadata


def main(argv: list[str] | None = None) -> int:
    cfg = Colonist1v1TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num", type=int, default=100, help="Number of games.")
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
    args = p.parse_args(argv)
    output_dir = Path(args.output)
    touch_run_marker(output_dir)

    exe = shutil.which("catanatron-play")
    if exe:
        cmd = [exe]
    else:
        cmd = [sys.executable, "-m", "catanatron.cli.play"]

    cmd.extend(
        [
            "--colonist-1v1",
            "--players",
            args.teachers,
            "--num",
            str(args.num),
            "--output",
            args.output,
            "--output-format",
            "parquet",
            "--quiet",
        ]
    )
    if args.include_board_tensor:
        cmd.append("--include-board-tensor")

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0:
        write_dataset_metadata(
            Path(args.output),
            teachers=args.teachers,
            num_games=args.num,
            command=" ".join(cmd),
        )
        print(f"Wrote {Path(args.output) / 'dataset_meta.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

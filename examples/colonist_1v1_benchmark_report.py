#!/usr/bin/env python3
"""
Produce a Colonist 1v1 strength report (JSON + stdout) for a trained checkpoint.

Example::

    python examples/colonist_1v1_benchmark_report.py --agent L:runs/colonist_1v1/colonist_maskable_ppo.zip \\
        --num-games 300 --gates --output runs/strength_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from catanatron.colonist_1v1_eval import (
    DEFAULT_BENCHMARK_GATES,
    print_report,
    run_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", required=True, help='e.g. "L:path.zip" or "T:path.pt"')
    p.add_argument("--num-games", type=int, default=300)
    p.add_argument("--output", type=Path, default=Path("colonist_1v1_strength_report.json"))
    p.add_argument(
        "--gates",
        action="store_true",
        help="Apply default win-rate gates and exit 1 if any fail.",
    )
    args = p.parse_args(argv)

    report = run_benchmark(
        args.agent,
        gates=DEFAULT_BENCHMARK_GATES if args.gates else None,
        num_games=args.num_games,
        quiet=True,
    )
    print_report(report)
    report.write_json(args.output)
    print(f"Wrote {args.output}")
    return 0 if report.all_gates_passed or not args.gates else 1


if __name__ == "__main__":
    sys.exit(main())

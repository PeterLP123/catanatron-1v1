#!/usr/bin/env python3
"""Audit existing corpora before authorizing an outcome-critic experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from catanatron.gym.outcome_target_audit import audit_outcome_targets


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
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--minimum-win-row-coverage", type=float)
    parser.add_argument("--minimum-margin-row-coverage", type=float)
    parser.add_argument("--minimum-split-groups", type=int)
    parser.add_argument("--minimum-minority-fraction", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_outcome_targets(
        args.corpus,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_shards=args.expected_shards,
        minimum_win_row_coverage=args.minimum_win_row_coverage,
        minimum_margin_row_coverage=args.minimum_margin_row_coverage,
        minimum_split_groups=args.minimum_split_groups,
        minimum_minority_fraction=args.minimum_minority_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        f"Outcome target audit: rows={report['dataset']['rows']} "
        f"win={report['combined']['win_target']['row_coverage']:.1%} "
        f"margin={report['combined']['vp_margin_target']['row_coverage']:.1%} "
        f"gates={'PASS' if report['all_gates_passed'] else 'FAIL'}"
    )
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

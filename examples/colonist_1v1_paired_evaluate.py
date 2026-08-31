#!/usr/bin/env python3
"""Evaluate a candidate and baseline on the exact same 1v1 schedules.

This is the promotion test for a new checkpoint.  It writes both complete
per-game reports plus a paired bootstrap comparison, so an apparent win-rate
gain cannot be caused by changing seats or seeds between models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from catanatron.colonist_1v1_eval import (
    DEFAULT_EVAL_SEED,
    EVAL_PROTOCOLS,
    EvalProtocol,
    compare_paired_reports,
    get_eval_protocol,
    resolve_eval_seed,
    run_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Candidate agent CLI spec.")
    parser.add_argument("--baseline", required=True, help="Baseline agent CLI spec.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=("F",),
        help="Opponent specs to pair (default: F).",
    )
    parser.add_argument(
        "--num-games", type=int, default=200, help="Games per opponent."
    )
    parser.add_argument("--protocol", choices=sorted(EVAL_PROTOCOLS), default="fast")
    parser.add_argument(
        "--seed-suite",
        choices=("manual", "dev", "promotion", "final"),
        default="promotion",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Explicit game seed; otherwise resolve the selected seed suite.",
    )
    parser.add_argument(
        "--seed-round",
        type=int,
        default=0,
        help="Disjoint repeat of the selected locked seed suite (default: 0).",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument(
        "--minimum-delta",
        type=float,
        default=0.0,
        help=(
            "Threshold the candidate-minus-baseline confidence lower bound must "
            "strictly exceed."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    seat_group = parser.add_mutually_exclusive_group()
    seat_group.add_argument(
        "--both-seats", dest="both_seats", action="store_true", default=True
    )
    seat_group.add_argument(
        "--first-seat-only", dest="both_seats", action="store_false"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_games <= 0:
        raise ValueError("num-games must be positive")
    if not 0 < args.confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if args.resamples <= 0:
        raise ValueError("resamples must be positive")
    if args.seed_round < 0:
        raise ValueError("seed-round must be non-negative")
    if args.seed is not None and args.seed_round != 0:
        raise ValueError("seed-round cannot be combined with an explicit seed")

    base_protocol = get_eval_protocol(args.protocol)
    game_seed = (
        args.seed
        if args.seed is not None
        else resolve_eval_seed(
            base_protocol.seed,
            suite=args.seed_suite,
            seed_round=args.seed_round,
        )
    )
    protocol = EvalProtocol(
        name=f"{base_protocol.name}-paired",
        opponents=tuple(args.opponents),
        num_games=args.num_games,
        description="Paired checkpoint promotion schedule",
        seed=base_protocol.seed,
    )
    command = ["colonist_1v1_paired_evaluate.py", *(argv or sys.argv[1:])]
    common = {
        "opponents": protocol.opponents,
        "num_games": protocol.num_games,
        "protocol": protocol,
        "both_seats": args.both_seats,
        "quiet": True,
        "eval_kind": "promotion",
        "seed": game_seed,
        "seed_suite": ("explicit" if args.seed is not None else args.seed_suite),
        "seed_round": args.seed_round,
        "gate_mode": "lower_bound",
        "command": command,
    }

    print(
        f"candidate={args.candidate} baseline={args.baseline} "
        f"opponents={','.join(args.opponents)} games={args.num_games} seed={game_seed}"
    )
    candidate = run_benchmark(args.candidate, **common)
    baseline = run_benchmark(args.baseline, **common)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_report.json"
    baseline_path = args.output_dir / "baseline_report.json"
    paired_path = args.output_dir / "paired_comparison.json"
    candidate.write_json(candidate_path)
    baseline.write_json(baseline_path)
    paired = compare_paired_reports(
        candidate,
        baseline,
        confidence=args.confidence,
        resamples=args.resamples,
        seed=args.bootstrap_seed,
        threshold=args.minimum_delta,
    )
    paired.write_json(paired_path)

    for item in paired.comparisons:
        print(
            f"{item.opponent}: candidate={item.candidate_win_rate:.1%} "
            f"baseline={item.baseline_win_rate:.1%} "
            f"paired_delta={item.score.mean_delta:+.1%} "
            f"CI=[{item.score.confidence_low:+.1%}, "
            f"{item.score.confidence_high:+.1%}] "
            f"gate={'PASS' if item.score.passed_gate else 'FAIL'}"
        )
    print(f"Wrote {candidate_path}, {baseline_path}, and {paired_path}")
    return 0 if paired.all_gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect deterministic student-visited states with F or MCTS teacher labels.

This is a data-generation scaffold for DAgger / expert iteration.  The student
controls its seat; the teacher labels each visited legal-action set.  Each
iteration creates immutable Parquet shards and updates a hash-indexed replay
manifest.  It does not launch a large training run.

Examples::

    python examples/colonist_1v1_distill.py --dry-run
    python examples/colonist_1v1_distill.py --student T:runs/bc.pt \
        --teacher M:50:False:base_fn --iteration 1 --games 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from catanatron.gym.distillation import (
    DistillationConfig,
    distillation_plan,
    run_distillation_iteration,
    verify_distillation_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student",
        default="W",
        help="Behavior player spec (for example W, L:model.zip, or T:model.pt).",
    )
    parser.add_argument(
        "--teacher",
        default="F",
        help="Teacher spec: F or fixed-simulation MCTS such as M:50:False:base_fn.",
    )
    parser.add_argument(
        "--opponent",
        default="W",
        help="Opponent player spec used to generate student-visited states.",
    )
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1701, help="Base seed namespace.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/colonist_1v1_distillation"),
    )
    parser.add_argument("--shard-games", type=int, default=10)
    parser.add_argument(
        "--feature-profile",
        choices=("raw", "public_derived"),
        default="raw",
    )
    parser.add_argument(
        "--human-visible-obs",
        action="store_true",
        help="Replace own actual VP with visible VP in the recorded observation.",
    )
    parser.add_argument(
        "--fixed-seat",
        action="store_true",
        help="Keep the student in seat 0 instead of alternating seats by game.",
    )
    parser.add_argument(
        "--include-forced",
        action="store_true",
        help="Also retain decisions with exactly one legal action.",
    )
    parser.add_argument(
        "--no-candidate-scores",
        action="store_true",
        help="Skip expensive chance-aware F candidate scoring (teacher action remains).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved hashes, schema, paths, and deterministic seeds only.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing output dataset's manifest and shard hashes, then exit.",
    )
    return parser


def _config(args: argparse.Namespace) -> DistillationConfig:
    return DistillationConfig(
        iteration=args.iteration,
        games=args.games,
        base_seed=args.seed,
        student_spec=args.student,
        teacher_spec=args.teacher,
        opponent_spec=args.opponent,
        feature_profile=args.feature_profile,
        human_visible_obs=args.human_visible_obs,
        alternate_seats=not args.fixed_seat,
        include_forced=args.include_forced,
        score_f_candidates=not args.no_candidate_scores,
        shard_games=args.shard_games,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        problems = verify_distillation_dataset(args.output)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        print(f"Verified {args.output / 'manifest.json'}")
        return 0

    config = _config(args)
    if args.dry_run:
        print(json.dumps(distillation_plan(config, output=args.output), indent=2))
        return 0

    manifest = run_distillation_iteration(config, output=args.output)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        f"Wrote iteration {config.iteration}: {payload['rows']} rows across "
        f"{len(payload['shards'])} shard(s) -> {manifest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Evaluate a Colonist 1v1 bot (SB3 zip via ``L:`` or classical code) against baselines.

Example::

    python examples/colonist_1v1_evaluate.py --agent L:colonist_maskable_ppo.zip --num-games 200
    python examples/colonist_1v1_evaluate.py --agent L:ckpt.zip --benchmark --report runs/eval.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from catanatron.colonist_1v1_eval import (
    DEFAULT_BENCHMARK_GATES,
    DEFAULT_BENCHMARK_OPPONENTS,
    EVAL_PROTOCOLS,
    EvaluationReport,
    append_model_registry,
    evaluate_matchup,
    format_matchup_line,
    get_eval_protocol,
    print_report,
    run_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--agent",
        required=True,
        help='Agent CLI spec (first seat), e.g. "L:runs/ppo.zip" or "F".',
    )
    p.add_argument(
        "--opponent",
        default=None,
        help="Single opponent code (default: run full --benchmark battery).",
    )
    p.add_argument("--num-games", type=int, default=200)
    p.add_argument(
        "--protocol",
        choices=sorted(EVAL_PROTOCOLS),
        default="full",
        help="Comparable eval protocol. Ignored for --opponent except metadata.",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help=f"Run full battery: {', '.join(DEFAULT_BENCHMARK_OPPONENTS)}",
    )
    p.add_argument(
        "--gates",
        action="store_true",
        help="Apply default win-rate gates (implies --benchmark if no --opponent).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSON evaluation report to this path.",
    )
    p.add_argument("--quiet", action="store_true", default=True)
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Append a compact model row to this JSONL registry.",
    )
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--checkpoint-label", default=None)
    p.add_argument("--training-timesteps", type=int, default=None)
    args = p.parse_args(argv)

    use_benchmark = args.benchmark or (args.opponent is None and args.gates)
    if args.opponent and use_benchmark:
        print("Specify either --opponent or --benchmark, not both.", file=sys.stderr)
        return 2

    if use_benchmark or args.opponent is None:
        proto = get_eval_protocol(args.protocol, num_games=args.num_games)
        report = run_benchmark(
            args.agent,
            opponents=DEFAULT_BENCHMARK_OPPONENTS if args.benchmark else proto.opponents,
            gates=DEFAULT_BENCHMARK_GATES if args.gates else None,
            num_games=proto.num_games,
            protocol=proto,
            quiet=args.quiet,
            run_dir=args.run_dir,
            checkpoint_label=args.checkpoint_label,
            training_timesteps=args.training_timesteps,
        )
        print_report(report)
        if args.report:
            report.write_json(args.report)
            print(f"Wrote {args.report}")
        if args.registry:
            append_model_registry(args.registry, report, report_path=args.report)
            print(f"Updated registry {args.registry}")
        return 0 if report.all_gates_passed or not args.gates else 1

    result = evaluate_matchup(
        args.agent,
        args.opponent,
        num_games=args.num_games,
        gate=DEFAULT_BENCHMARK_GATES.get(args.opponent) if args.gates else None,
        quiet=args.quiet,
    )
    print(format_matchup_line(result))
    if args.report:
        r = EvaluationReport(
            agent=args.agent,
            matchups=[result],
            summary={
                "gates_passed_count": 1 if result.passed_gate else 0,
                "gates_total": 1 if result.gate is not None else 0,
                "mean_win_rate": result.win_rate,
                "weighted_score": result.win_rate,
            },
        )
        r.write_json(args.report)
        if args.registry:
            append_model_registry(args.registry, r, report_path=args.report)
    return 0 if result.passed_gate is not False else 1


if __name__ == "__main__":
    sys.exit(main())

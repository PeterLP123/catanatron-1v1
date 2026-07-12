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
    EVAL_PROTOCOLS,
    EvalProtocol,
    EvaluationReport,
    append_model_registry,
    build_eval_meta,
    evaluate_matchup,
    format_matchup_line,
    get_eval_protocol,
    print_report,
    resolve_eval_seed,
    run_benchmark,
    summarize_report,
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
        help="Single opponent code (default: run the selected protocol's battery).",
    )
    p.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Games per opponent (default: the selected protocol's value).",
    )
    p.add_argument(
        "--protocol",
        choices=sorted(EVAL_PROTOCOLS),
        default="full",
        help="Comparable eval protocol. Ignored for --opponent except metadata.",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the selected protocol's opponent battery (also the default without --opponent).",
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
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Append a compact model row to this JSONL registry.",
    )
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--checkpoint-label", default=None)
    p.add_argument("--training-timesteps", type=int, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base game seed (default: the selected protocol's fixed seed).",
    )
    p.add_argument(
        "--eval-kind",
        choices=("manual", "dev", "promotion", "final", "final_benchmark"),
        default="manual",
        help="Evidence purpose; controls report provenance and default seed namespace.",
    )
    p.add_argument(
        "--seed-suite",
        choices=("manual", "dev", "promotion", "final"),
        default=None,
        help="Explicit deterministic seed namespace (normally inferred from --eval-kind).",
    )
    p.add_argument(
        "--gate-mode",
        choices=("point", "lower_bound"),
        default="point",
        help="Use the observed rate or Wilson lower bound for gates.",
    )
    seat_group = p.add_mutually_exclusive_group()
    seat_group.add_argument(
        "--both-seats",
        dest="both_seats",
        action="store_true",
        default=True,
        help="Split games between the agent moving first and second (default).",
    )
    seat_group.add_argument(
        "--first-seat-only",
        dest="both_seats",
        action="store_false",
        help="Only play the agent in the first seat (legacy, first-player-biased).",
    )
    args = p.parse_args(argv)

    use_benchmark = args.benchmark or (args.opponent is None and args.gates)
    if args.opponent and use_benchmark:
        print("Specify either --opponent or --benchmark, not both.", file=sys.stderr)
        return 2

    if use_benchmark or args.opponent is None:
        proto = get_eval_protocol(args.protocol, num_games=args.num_games)
        report = run_benchmark(
            args.agent,
            opponents=proto.opponents,
            gates=DEFAULT_BENCHMARK_GATES if args.gates else None,
            num_games=proto.num_games,
            protocol=proto,
            both_seats=args.both_seats,
            quiet=True,
            run_dir=args.run_dir,
            checkpoint_label=args.checkpoint_label,
            training_timesteps=args.training_timesteps,
            seed=args.seed,
            eval_kind=args.eval_kind,
            seed_suite=args.seed_suite,
            gate_mode=args.gate_mode,
        )
        print_report(report)
        if args.report:
            report.write_json(args.report)
            print(f"Wrote {args.report}")
        if args.registry:
            append_model_registry(args.registry, report, report_path=args.report)
            print(f"Updated registry {args.registry}")
        return 0 if report.all_gates_passed or not args.gates else 1

    base_seed = get_eval_protocol(args.protocol).seed
    inferred_suite = args.seed_suite
    if inferred_suite is None:
        inferred_suite = {
            "dev": "dev",
            "promotion": "promotion",
            "final": "final",
            "final_benchmark": "final",
        }.get(args.eval_kind, "manual")
    single_seed = (
        args.seed
        if args.seed is not None
        else resolve_eval_seed(base_seed, suite=inferred_suite)
    )
    result = evaluate_matchup(
        args.agent,
        args.opponent,
        num_games=args.num_games or 200,
        gate=DEFAULT_BENCHMARK_GATES.get(args.opponent) if args.gates else None,
        both_seats=args.both_seats,
        quiet=True,
        seed=single_seed,
        gate_mode=args.gate_mode,
    )
    print(format_matchup_line(result))
    if args.report:
        r = EvaluationReport(
            agent=args.agent,
            matchups=[result],
            all_gates_passed=result.passed_gate is not False,
            meta=build_eval_meta(
                agent_spec=args.agent,
                protocol=EvalProtocol(
                    name=f"{args.protocol}-single",
                    opponents=(args.opponent,),
                    num_games=args.num_games or 200,
                    seed=single_seed,
                ),
                run_dir=args.run_dir,
                checkpoint_label=args.checkpoint_label,
                training_timesteps=args.training_timesteps,
                both_seats=args.both_seats,
                eval_kind=args.eval_kind,
                seed_suite=("explicit" if args.seed is not None else inferred_suite),
                base_seed=base_seed,
                gate_mode=args.gate_mode,
                gates=(
                    {args.opponent: DEFAULT_BENCHMARK_GATES.get(args.opponent)}
                    if args.gates
                    else {}
                ),
            ),
            summary=summarize_report([result]),
        )
        r.write_json(args.report)
        if args.registry:
            append_model_registry(args.registry, r, report_path=args.report)
    return 0 if result.passed_gate is not False else 1


if __name__ == "__main__":
    sys.exit(main())

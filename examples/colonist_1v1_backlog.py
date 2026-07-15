#!/usr/bin/env python3
"""List, inspect, compare and launch the gated GPU experiment backlog."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from catanatron.gym.experiment_backlog import (
    EXPERIMENTS,
    backlog_statuses,
    check_backlog_markdown,
    evaluate_experiment,
    experiments_by_id,
    launch_argv,
    launch_command,
    launch_environment,
    load_final_metrics,
    render_backlog_markdown,
    write_launch_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _experiment(value: str):
    try:
        return experiments_by_id()[value]
    except KeyError as exc:
        raise SystemExit(
            f"Unknown experiment {value!r}. Use 'list' to see IDs."
        ) from exc


def _print_list(runs_root: Path) -> None:
    statuses = backlog_statuses(runs_root)
    try:
        from rich.console import Console
        from rich import box
        from rich.table import Table

        table = Table(
            title="Colonist 1v1 GPU Backlog",
            header_style="bold orange3",
            box=box.SIMPLE_HEAVY,
        )
        table.add_column("ID", style="bold cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Stage", no_wrap=True)
        table.add_column("GPU h", justify="right", no_wrap=True)
        table.add_column("Disk", justify="right", no_wrap=True)
        for experiment in EXPERIMENTS:
            lo, hi = experiment.gpu_hours
            table.add_row(
                experiment.id,
                statuses[experiment.id],
                experiment.stage,
                f"{lo:g}–{hi:g}",
                f"{experiment.storage_gib:g} GiB",
            )
        console = Console()
        console.print(table)
        console.print("[dim]Use: python examples/colonist_1v1_backlog.py show <ID>[/]")
    except ImportError:
        for experiment in EXPERIMENTS:
            lo, hi = experiment.gpu_hours
            print(
                f"{experiment.id:28} {statuses[experiment.id]:8} "
                f"{lo:g}-{hi:g}h {experiment.storage_gib:g}GiB  {experiment.title}"
            )


def _print_show(experiment, supplied: dict[str, str]) -> None:
    print(f"{experiment.id} · {experiment.title}")
    print(f"Stage: {experiment.stage}")
    print(f"Hypothesis: {experiment.hypothesis}")
    print(
        f"Expected GPU time: {experiment.gpu_hours[0]:g}–{experiment.gpu_hours[1]:g} hours"
    )
    print(f"Expected run storage: {experiment.storage_gib:g} GiB")
    if experiment.depends_on:
        print(f"Depends on: {', '.join(experiment.depends_on)}")
    if experiment.required_inputs:
        print(f"Required input: {', '.join(experiment.required_inputs)}")
    print(f"Decision rule: {experiment.success_rule}")
    try:
        print(f"Command: {launch_command(experiment, supplied)}")
    except ValueError as exc:
        print(f"Command: blocked — {exc}")


def _print_decision(experiment, runs_root: Path) -> None:
    status = backlog_statuses(runs_root)[experiment.id]
    print(f"Status: {status}")
    if status in {"accepted", "rejected", "inconclusive"}:
        decision = evaluate_experiment(experiment, runs_root)
        print(f"Evidence: {decision.reason}")
        if decision.evidence:
            print(f"Details: {dict(decision.evidence)}")


def _print_comparison(ids: list[str], runs_root: Path) -> int:
    headings = ("ID", "Score", "R", "W", "VP", "F", "Max seat gap", "Gates")
    print("  ".join(f"{heading:>12}" for heading in headings))
    missing = False
    for experiment_id in ids:
        metrics = load_final_metrics(runs_root / experiment_id)
        if metrics is None:
            print(f"{experiment_id:>12}  {'no final benchmark':>12}")
            missing = True
            continue
        rates = metrics["rates"]

        def pct(value):
            return "-" if value is None else f"{100 * value:.1f}%"

        score = metrics["weighted_score"]
        values = (
            experiment_id,
            "-" if score is None else f"{score:.3f}",
            pct(rates.get("R")),
            pct(rates.get("W")),
            pct(rates.get("VP")),
            pct(rates.get("F")),
            pct(metrics["max_seat_gap"]),
            "PASS" if metrics["all_gates_passed"] else "FAIL",
        )
        print("  ".join(f"{value:>12}" for value in values))
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show queue, status, time and storage estimates.")
    sub.add_parser("next", help="Show the first dependency-ready experiment.")
    show = sub.add_parser(
        "show", help="Explain one experiment and print its launch command."
    )
    show.add_argument("experiment_id")
    start = sub.add_parser(
        "start", help="Launch one experiment in the visual tmux command center."
    )
    start.add_argument("experiment_id")
    start.add_argument(
        "--force", action="store_true", help="Ignore incomplete dependencies."
    )
    for child in (show, start):
        child.add_argument("--bc-checkpoint")
        child.add_argument("--bc-baseline-checkpoint")
        child.add_argument("--resume-checkpoint")
    compare = sub.add_parser(
        "compare", help="Compare final benchmark reports for experiment IDs."
    )
    compare.add_argument("experiment_ids", nargs="+")
    render = sub.add_parser(
        "render", help="Render the authoritative Markdown queue table."
    )
    render.add_argument("--output", type=Path)
    check = sub.add_parser(
        "check", help="Fail when a generated Markdown queue table has drifted."
    )
    check.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    if args.command == "list":
        _print_list(args.runs_root)
        return 0
    if args.command == "next":
        statuses = backlog_statuses(args.runs_root)
        ready = next((e for e in EXPERIMENTS if statuses[e.id] == "ready"), None)
        if ready is None:
            print("No dependency-ready pending experiment.")
            return 1
        _print_show(ready, {})
        return 0
    if args.command == "compare":
        return _print_comparison(args.experiment_ids, args.runs_root)
    if args.command == "render":
        rendered = render_backlog_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(rendered, end="")
        return 0
    if args.command == "check":
        if check_backlog_markdown(args.path):
            print(f"Backlog table is current: {args.path}")
            return 0
        print(
            f"Backlog table has drifted: {args.path}\n"
            "Regenerate it with: python examples/colonist_1v1_backlog.py render "
            f"--output {args.path}",
            file=sys.stderr,
        )
        return 1

    experiment = _experiment(args.experiment_id)
    supplied = {
        "BC_CHECKPOINT": args.bc_checkpoint,
        "BC_BASELINE_CHECKPOINT": args.bc_baseline_checkpoint,
        "RESUME_CHECKPOINT": args.resume_checkpoint,
    }
    if args.command == "show":
        _print_show(experiment, supplied)
        _print_decision(experiment, args.runs_root)
        return 0

    statuses = backlog_statuses(args.runs_root)
    if statuses[experiment.id] in {"blocked", "skipped"} and not args.force:
        dependencies = ", ".join(experiment.depends_on)
        raise SystemExit(
            f"{experiment.id} is {statuses[experiment.id]} by its evidence gate "
            f"(dependencies: {dependencies or 'promotion evidence'}). "
            "Use --force only when those results exist elsewhere."
        )
    try:
        launch = launch_environment(experiment, supplied=supplied)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    env = dict(os.environ)
    env.update(launch)
    write_launch_evidence(experiment, args.runs_root, launch)
    os.chdir(ROOT)
    command = list(launch_argv(experiment))
    if command[0] == "python":
        command[0] = sys.executable
    os.execvpe(command[0], command, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())

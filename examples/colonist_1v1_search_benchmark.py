#!/usr/bin/env python3
"""Measure MCTS latency and honest two-seat strength on held-out seed suites."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

from catanatron import Color, Game
from catanatron.colonist_1v1_eval import (
    MatchupResult,
    current_git_commit,
    evaluate_matchup,
    format_matchup_line,
    utc_now_iso,
)
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer


DEFAULT_BUDGETS_MS = (10.0, 25.0, 50.0, 100.0)
DEFAULT_OPPONENTS = ("F", "AB:2")
DEFAULT_SEEDS = (20_260_711, 20_260_712, 20_260_713)

ProgressCallback = Callable[[dict[str, Any], str], None]


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not parsed or any(not math.isfinite(item) or item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("budgets must be finite positive numbers")
    return parsed


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return parsed


def _csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one opponent is required")
    return parsed


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a dependency-free, linearly interpolated percentile."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * min(max(percentile_value, 0.0), 100.0) / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _profile_once(budget_ms: float, *, value_fn: str, seed: int) -> dict[str, float]:
    """Run one deterministic mid-game decision and return raw MCTS counters."""
    random_state = random.getstate()
    try:
        random.seed(seed)
        game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
        for _ in range(400):
            if game.winning_color() is not None:
                break
            if len(game.playable_actions) > 1 and not game.state.is_initial_build_phase:
                break
            game.play_tick()
        player = MCTSPlayer(
            game.state.current_color(),
            max_time_ms=budget_ms,
            value_fn_name=value_fn,
        )
        player.decide(game, game.playable_actions)
        return dict(player.last_search_stats or {})
    finally:
        random.setstate(random_state)


def profile_budget(
    budget_ms: float,
    *,
    value_fn: str = "base_fn",
    samples: int = 20,
    seed: int = 20_260_700,
) -> dict[str, Any]:
    """Profile a budget repeatedly so tail latency is visible, not inferred."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    raw = [
        _profile_once(budget_ms, value_fn=value_fn, seed=seed + offset)
        for offset in range(samples)
    ]
    latencies_ms = [float(row.get("elapsed_s", 0.0)) * 1000.0 for row in raw]
    elapsed = sum(float(row.get("elapsed_s", 0.0)) for row in raw)
    simulations = sum(int(row.get("simulations", 0)) for row in raw)
    expansions = sum(int(row.get("expansions", 0)) for row in raw)
    hits = sum(int(row.get("leaf_cache_hits", 0)) for row in raw)
    misses = sum(int(row.get("leaf_cache_misses", 0)) for row in raw)
    return {
        "samples": samples,
        "budget_ms": float(budget_ms),
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "p50_latency_ms": percentile(latencies_ms, 50.0),
        "p95_latency_ms": percentile(latencies_ms, 95.0),
        "max_latency_ms": max(latencies_ms),
        "simulations_per_s": simulations / elapsed if elapsed else 0.0,
        "nodes_per_s": expansions / elapsed if elapsed else 0.0,
        "leaf_cache_hit_rate": hits / (hits + misses) if hits + misses else 0.0,
        "latencies_ms": latencies_ms,
    }


def _profile_key(budget_ms: float) -> str:
    return f"{float(budget_ms):g}"


def _matrix_cells(
    budgets_ms: Sequence[float],
    opponents: Sequence[str],
    seeds: Sequence[int],
    *,
    profile_only: bool,
) -> list[tuple[float, str | None, int | None]]:
    if profile_only:
        return [(float(budget), None, None) for budget in budgets_ms]
    return [
        (float(budget), str(opponent), int(seed))
        for budget in budgets_ms
        for opponent in opponents
        for seed in seeds
    ]


def _cell_key(
    row: Mapping[str, Any], *, profile_only: bool
) -> tuple[float, str | None, int | None]:
    budget = float(row["budget_ms"])
    if profile_only:
        return (budget, None, None)
    return (budget, str(row["opponent"]), int(row["seed"]))


def _new_strength_report(
    *,
    budgets_ms: Sequence[float],
    opponents: Sequence[str],
    seeds: Sequence[int],
    num_games: int,
    value_fn: str,
    profile_samples: int,
    profile_seed: int,
    profile_only: bool,
) -> dict[str, Any]:
    cells = _matrix_cells(budgets_ms, opponents, seeds, profile_only=profile_only)
    return {
        "schema_version": "1.0",
        "kind": "mcts_strength_sweep",
        "meta": {
            "created_at": utc_now_iso(),
            "git_commit": current_git_commit(),
            "budgets_ms": [float(value) for value in budgets_ms],
            "opponents": list(opponents),
            "seeds": [int(value) for value in seeds],
            "num_games_per_cell": int(num_games),
            "both_seats": True,
            "value_fn": value_fn,
            "profile_samples": int(profile_samples),
            "profile_seed": int(profile_seed),
            "profile_only": bool(profile_only),
            "agent_spec_template": f"M:1:False:{value_fn}:<budget_ms>",
        },
        "profiles": {},
        "progress": {
            "status": "running",
            "completed_cells": 0,
            "total_cells": len(cells),
            "percent_complete": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "current_cell": None,
            "resume_count": 0,
            "updated_at": utc_now_iso(),
        },
        "results": [],
    }


def _resume_config(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "git_commit": meta.get("git_commit"),
        "budgets_ms": [float(value) for value in meta.get("budgets_ms", ())],
        "opponents": list(meta.get("opponents", ())),
        "seeds": [int(value) for value in meta.get("seeds", ())],
        "num_games_per_cell": meta.get("num_games_per_cell"),
        "both_seats": meta.get("both_seats"),
        "value_fn": meta.get("value_fn"),
        "profile_samples": meta.get("profile_samples"),
        "profile_seed": meta.get("profile_seed"),
        "profile_only": meta.get("profile_only"),
        "agent_spec_template": meta.get("agent_spec_template"),
    }


def _validate_resume_report(
    report: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if (
        report.get("schema_version") != "1.0"
        or report.get("kind") != "mcts_strength_sweep"
    ):
        raise ValueError("existing report has an incompatible schema or kind")
    actual_meta = report.get("meta")
    expected_meta = expected.get("meta")
    if not isinstance(actual_meta, Mapping) or not isinstance(expected_meta, Mapping):
        raise ValueError("existing report is missing sweep metadata")
    if _resume_config(actual_meta) != _resume_config(expected_meta):
        raise ValueError(
            "existing report does not match this commit and sweep configuration"
        )

    profile_only = bool(expected_meta.get("profile_only"))
    expected_cells = set(
        _matrix_cells(
            expected_meta["budgets_ms"],
            expected_meta["opponents"],
            expected_meta["seeds"],
            profile_only=profile_only,
        )
    )
    seen: set[tuple[float, str | None, int | None]] = set()
    for row in report.get("results", ()):
        if not isinstance(row, Mapping):
            raise ValueError("existing report contains a malformed result cell")
        try:
            key = _cell_key(row, profile_only=profile_only)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "existing report contains a malformed result cell"
            ) from exc
        if key not in expected_cells or key in seen:
            raise ValueError("existing report contains unexpected or duplicate cells")
        if not isinstance(row.get("profile"), Mapping):
            raise ValueError("existing report contains a cell without latency profile")
        if not profile_only:
            matchup = row.get("matchup")
            if not isinstance(matchup, Mapping):
                raise ValueError(
                    "existing report contains a cell without matchup evidence"
                )
            requested = matchup.get("requested_games", matchup.get("games"))
            games = matchup.get("game_results")
            if (
                requested != expected_meta["num_games_per_cell"]
                or not isinstance(games, list)
                or len(games) != requested
            ):
                raise ValueError("existing report contains an incomplete matchup cell")
        seen.add(key)


def _format_duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or not math.isfinite(float(seconds)):
        return "-"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _current_cell_label(current: Any) -> str:
    if not isinstance(current, Mapping):
        return "-"
    phase = current.get("phase", "cell")
    budget = current.get("budget_ms")
    budget_label = f"{budget:g}ms" if isinstance(budget, (int, float)) else "-"
    if phase == "profile":
        return f"profile {budget_label}"
    opponent = current.get("opponent", "-")
    seed = current.get("seed", "-")
    if opponent is None and seed is None:
        return f"{budget_label}/profile-only"
    return f"{budget_label}/{opponent}/seed-{seed}"


def progress_status_line(report: Mapping[str, Any]) -> str:
    progress = report.get("progress", {})
    if not isinstance(progress, Mapping):
        progress = {}
    completed = int(progress.get("completed_cells", 0) or 0)
    total = int(progress.get("total_cells", 0) or 0)
    percent = float(progress.get("percent_complete", 0.0) or 0.0)
    return (
        f"05-mcts-strength-sweep {completed}/{total} cells ({percent:.1f}%) "
        f"status={progress.get('status', 'unknown')} "
        f"elapsed={_format_duration(progress.get('elapsed_seconds'))} "
        f"eta={_format_duration(progress.get('eta_seconds'))} "
        f"current={_current_cell_label(progress.get('current_cell'))}"
    )


def _print_progress(report: Mapping[str, Any], event: str, *, stream: TextIO) -> None:
    print(f"[{event}] {progress_status_line(report)}", file=stream, flush=True)


def _refresh_progress(
    report: dict[str, Any],
    *,
    completed_cells: int,
    total_cells: int,
    base_elapsed: float,
    active_started: float,
    status: str,
    current_cell: Mapping[str, Any] | None,
) -> None:
    elapsed = base_elapsed + max(0.0, time.monotonic() - active_started)
    remaining = max(0, total_cells - completed_cells)
    eta = (elapsed / completed_cells) * remaining if completed_cells > 0 else None
    report["progress"].update(
        {
            "status": status,
            "completed_cells": completed_cells,
            "total_cells": total_cells,
            "percent_complete": (
                100.0 * completed_cells / total_cells if total_cells else 100.0
            ),
            "elapsed_seconds": elapsed,
            "eta_seconds": 0.0 if status == "complete" else eta,
            "current_cell": dict(current_cell) if current_cell else None,
            "updated_at": utc_now_iso(),
        }
    )


def build_strength_report(
    *,
    budgets_ms: Sequence[float],
    opponents: Sequence[str],
    seeds: Sequence[int],
    num_games: int,
    value_fn: str,
    profile_samples: int,
    profile_seed: int,
    profile_only: bool = False,
    existing_report: Mapping[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run or resume the declared matrix and return its evidence report."""
    expected = _new_strength_report(
        budgets_ms=budgets_ms,
        opponents=opponents,
        seeds=seeds,
        num_games=num_games,
        value_fn=value_fn,
        profile_samples=profile_samples,
        profile_seed=profile_seed,
        profile_only=profile_only,
    )
    if existing_report is None:
        report = expected
    else:
        _validate_resume_report(existing_report, expected)
        # JSON round-trip gives us a plain mutable copy without sharing nested state.
        report = json.loads(json.dumps(existing_report))
        report.setdefault("profiles", {})
        report.setdefault("progress", {})
        report["progress"]["resume_count"] = (
            int(report["progress"].get("resume_count", 0) or 0) + 1
        )

    cells = _matrix_cells(budgets_ms, opponents, seeds, profile_only=profile_only)
    total_cells = len(cells)
    results = report.setdefault("results", [])
    completed = {_cell_key(row, profile_only=profile_only) for row in results}
    profiles = report.setdefault("profiles", {})
    for row in results:
        profiles.setdefault(_profile_key(row["budget_ms"]), row["profile"])

    progress = report.setdefault("progress", {})
    base_elapsed = float(progress.get("elapsed_seconds", 0.0) or 0.0)
    active_started = time.monotonic()
    if len(completed) == total_cells:
        _refresh_progress(
            report,
            completed_cells=len(completed),
            total_cells=total_cells,
            base_elapsed=base_elapsed,
            active_started=active_started,
            status="complete",
            current_cell=None,
        )
        if progress_callback:
            progress_callback(report, "already-complete")
        return report

    for budget, opponent, seed in cells:
        key = (budget, opponent, seed)
        if key in completed:
            continue
        profile_key = _profile_key(budget)
        if profile_key not in profiles:
            current = {"phase": "profile", "budget_ms": budget}
            _refresh_progress(
                report,
                completed_cells=len(completed),
                total_cells=total_cells,
                base_elapsed=base_elapsed,
                active_started=active_started,
                status="profiling",
                current_cell=current,
            )
            if progress_callback:
                progress_callback(report, "profiling")
            profiles[profile_key] = profile_budget(
                budget,
                value_fn=value_fn,
                samples=profile_samples,
                seed=profile_seed,
            )
            _refresh_progress(
                report,
                completed_cells=len(completed),
                total_cells=total_cells,
                base_elapsed=base_elapsed,
                active_started=active_started,
                status="running",
                current_cell=None,
            )
            if progress_callback:
                progress_callback(report, "profile-complete")

        current = {
            "phase": "cell",
            "budget_ms": budget,
            "opponent": opponent,
            "seed": seed,
        }
        _refresh_progress(
            report,
            completed_cells=len(completed),
            total_cells=total_cells,
            base_elapsed=base_elapsed,
            active_started=active_started,
            status="running",
            current_cell=current,
        )
        if progress_callback:
            progress_callback(report, "cell-start")

        if profile_only:
            row = {"budget_ms": budget, "profile": profiles[profile_key]}
        else:
            spec = f"M:1:False:{value_fn}:{budget:g}"
            matchup = evaluate_matchup(
                spec,
                opponent,
                num_games=num_games,
                both_seats=True,
                quiet=True,
                seed=int(seed),
            )
            row = {
                "budget_ms": budget,
                "opponent": opponent,
                "seed": int(seed),
                "profile": profiles[profile_key],
                "matchup": matchup.to_dict(),
            }
        results.append(row)
        completed.add(key)
        _refresh_progress(
            report,
            completed_cells=len(completed),
            total_cells=total_cells,
            base_elapsed=base_elapsed,
            active_started=active_started,
            status="running",
            current_cell=None,
        )
        if progress_callback:
            progress_callback(report, "cell-complete")

    order = {cell: index for index, cell in enumerate(cells)}
    results.sort(key=lambda row: order[_cell_key(row, profile_only=profile_only)])
    report["meta"]["completed_at"] = utc_now_iso()
    _refresh_progress(
        report,
        completed_cells=len(completed),
        total_cells=total_cells,
        base_elapsed=base_elapsed,
        active_started=active_started,
        status="complete",
        current_cell=None,
    )
    if progress_callback:
        progress_callback(report, "complete")
    return report


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_report(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read existing report {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"existing report {path} is not a JSON object")
    return loaded


def run_strength_report(
    *,
    budgets_ms: Sequence[float],
    opponents: Sequence[str],
    seeds: Sequence[int],
    num_games: int,
    value_fn: str,
    profile_samples: int,
    profile_seed: int,
    profile_only: bool = False,
    report_path: Path | None = None,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    """Run a sweep, atomically checkpointing and resuming when a path is given."""
    existing = (
        _read_report(report_path)
        if report_path is not None and report_path.exists()
        else None
    )
    stream = progress_stream if progress_stream is not None else sys.stdout

    def persist_progress(report: dict[str, Any], event: str) -> None:
        if report_path is not None:
            _write_report(report, report_path)
        _print_progress(report, event, stream=stream)

    report = build_strength_report(
        budgets_ms=budgets_ms,
        opponents=opponents,
        seeds=seeds,
        num_games=num_games,
        value_fn=value_fn,
        profile_samples=profile_samples,
        profile_seed=profile_seed,
        profile_only=profile_only,
        existing_report=existing,
        progress_callback=persist_progress,
    )
    if report_path is not None:
        _write_report(report, report_path)
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"MCTS(F-leaf, {report['meta']['value_fn']}) · "
        f"{report['meta']['num_games_per_cell']} games/cell"
    )
    print(
        f"{'budget':>8}  {'opponent':>8}  {'seed':>10}  {'p95 ms':>8}  "
        f"{'sims/s':>10}  {'win%':>7}  {'95% CI':>16}"
    )
    for row in report["results"]:
        profile = row["profile"]
        matchup = row.get("matchup")
        if matchup is None:
            print(
                f"{row['budget_ms']:>7.0f}m  {'-':>8}  {'-':>10}  "
                f"{profile['p95_latency_ms']:>8.1f}  "
                f"{profile['simulations_per_s']:>10.0f}"
            )
            continue
        ci = f"[{matchup['wilson_low']:.1%},{matchup['wilson_high']:.1%}]"
        print(
            f"{row['budget_ms']:>7.0f}m  {row['opponent']:>8}  {row['seed']:>10}  "
            f"{profile['p95_latency_ms']:>8.1f}  "
            f"{profile['simulations_per_s']:>10.0f}  "
            f"{matchup['win_rate']:>6.1%}  {ci:>16}"
        )
        # Rehydrate-free detail output keeps this script decoupled from report readers.
        print("           " + format_matchup_line(MatchupResult.from_dict(matchup)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        type=_csv_floats,
        default=DEFAULT_BUDGETS_MS,
        help="Comma-separated per-decision time budgets in milliseconds.",
    )
    parser.add_argument(
        "--num-games", type=int, default=20, help="Games per budget/opponent/seed cell."
    )
    parser.add_argument(
        "--opponents",
        type=_csv_strings,
        default=DEFAULT_OPPONENTS,
        help="Comma-separated opponent CLI codes.",
    )
    parser.add_argument(
        "--opponent",
        help="Deprecated single-opponent alias; overrides --opponents when supplied.",
    )
    parser.add_argument(
        "--seeds",
        type=_csv_ints,
        default=DEFAULT_SEEDS,
        help="Comma-separated held-out evaluation seeds.",
    )
    parser.add_argument(
        "--value-fn",
        default="base_fn",
        choices=("base_fn", "contender_fn"),
        help="MCTS leaf value function builder.",
    )
    parser.add_argument("--profile-samples", type=int, default=20)
    parser.add_argument("--profile-seed", type=int, default=20_260_700)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Only report throughput/tail latency; do not satisfy the backlog evidence gate.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Atomically write or resume the JSON evidence report.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print one progress/ETA line from --report without running work.",
    )
    args = parser.parse_args(argv)
    opponents = (args.opponent,) if args.opponent else tuple(args.opponents)

    if args.status:
        if args.report is None:
            parser.error("--status requires --report")
        if not args.report.exists():
            total_cells = len(
                _matrix_cells(
                    args.budgets,
                    opponents,
                    args.seeds,
                    profile_only=args.profile_only,
                )
            )
            print(
                f"05-mcts-strength-sweep 0/{total_cells} cells (0.0%) "
                "status=pending elapsed=00:00:00 eta=- current=-"
            )
            return 0
        print(progress_status_line(_read_report(args.report)))
        return 0
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.profile_samples <= 0:
        parser.error("--profile-samples must be positive")
    report = run_strength_report(
        budgets_ms=args.budgets,
        opponents=opponents,
        seeds=args.seeds,
        num_games=args.num_games,
        value_fn=args.value_fn,
        profile_samples=args.profile_samples,
        profile_seed=args.profile_seed,
        profile_only=args.profile_only,
        report_path=args.report,
    )
    _print_report(report)
    if args.report:
        print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

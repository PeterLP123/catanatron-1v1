#!/usr/bin/env python3
"""Benchmark practical search teachers against the full Colonist 1v1 battery.

Unlike the wall-clock MCTS sweep, this benchmark accepts arbitrary deterministic
CLI player specifications, profiles their decision latency, and writes every
completed candidate/opponent cell atomically so expensive runs can resume.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from catanatron import Color, Game
from catanatron.colonist_1v1_eval import (
    DEFAULT_BENCHMARK_OPPONENTS,
    MatchupResult,
    current_git_commit,
    evaluate_matchup,
    summarize_report,
    utc_now_iso,
)
from catanatron.gym.distillation import build_player
from catanatron.gym.provenance import collect_run_provenance
from catanatron.players.weighted_random import WeightedRandomPlayer


SCHEMA_VERSION = "1.0"
DEFAULT_CANDIDATES = ("AB:2", "M:200", "M:800", "M:2000")
DEFAULT_OPPONENTS = DEFAULT_BENCHMARK_OPPONENTS
DEFAULT_SEED = 20_260_717
COMMON_OPPONENTS = ("R", "W", "VP", "F", "G:25")
STRENGTH_OPPONENTS = ("F", "G:25", "M:200", "AB:2")
PROFILE_ACTION_TICKS = (25, 100, 200)


def _csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError(
            "at least one player specification is required"
        )
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("player specifications must be unique")
    return parsed


def percentile(values: Sequence[float], percentile_value: float) -> float:
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


def _profile_state(seed: int, *, target_action_tick: int) -> Game:
    random.seed(seed)
    game = Game(
        [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.BLUE),
        ],
        seed=seed,
        colonist_1v1=True,
        shuffle_players=False,
    )
    for _ in range(5_000):
        if game.winning_color() is not None:
            break
        if (
            len(game.state.action_records) >= target_action_tick
            and len(game.playable_actions) > 1
            and not game.state.is_initial_build_phase
        ):
            break
        game.play_tick()
    if game.winning_color() is not None or len(game.playable_actions) <= 1:
        raise RuntimeError(
            f"Could not build a profiled decision at action tick {target_action_tick}"
        )
    return game


def profile_candidate(
    candidate: str,
    *,
    samples: int = 3,
    seed: int = DEFAULT_SEED - 100,
) -> dict[str, Any]:
    """Measure deterministic representative-decision latency for one candidate."""

    if samples <= 0:
        raise ValueError("profile samples must be positive")
    random_state = random.getstate()
    rows: list[dict[str, Any]] = []
    try:
        for offset in range(samples):
            if samples == 1:
                target_action_tick = PROFILE_ACTION_TICKS[1]
            else:
                target_action_tick = PROFILE_ACTION_TICKS[
                    round(offset * (len(PROFILE_ACTION_TICKS) - 1) / (samples - 1))
                ]
            game = _profile_state(seed + offset, target_action_tick=target_action_tick)
            player = build_player(candidate, game.state.current_color())
            started = time.perf_counter()
            action = player.decide(game, game.playable_actions)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if action not in game.playable_actions:
                raise RuntimeError(f"{candidate} returned an illegal profiled action")
            stats = getattr(player, "last_search_stats", None) or {}
            rows.append(
                {
                    "seed": seed + offset,
                    "target_action_tick": target_action_tick,
                    "actual_action_tick": len(game.state.action_records),
                    "latency_ms": elapsed_ms,
                    "simulations": stats.get("simulations"),
                    "nodes_per_s": stats.get("nodes_per_s"),
                    "leaf_cache_hit_rate": stats.get("leaf_cache_hit_rate"),
                }
            )
    finally:
        random.setstate(random_state)

    latencies = [float(row["latency_ms"]) for row in rows]
    simulations = [
        int(row["simulations"])
        for row in rows
        if isinstance(row.get("simulations"), int)
    ]
    return {
        "candidate": candidate,
        "samples": samples,
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": percentile(latencies, 50.0),
        "p95_latency_ms": percentile(latencies, 95.0),
        "max_latency_ms": max(latencies),
        "mean_simulations": statistics.fmean(simulations) if simulations else None,
        "rows": rows,
    }


def _configuration(
    *,
    candidates: Sequence[str],
    opponents: Sequence[str],
    num_games: int,
    seed: int,
    profile_samples: int,
    profile_seed: int,
) -> dict[str, Any]:
    return {
        "git_commit": current_git_commit(),
        "candidates": list(candidates),
        "opponents": list(opponents),
        "num_games_per_cell": int(num_games),
        "both_seats": True,
        "seed": int(seed),
        "profile_samples": int(profile_samples),
        "profile_seed": int(profile_seed),
    }


def _atomic_write(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_event(path: Path, event_type: str, **data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": utc_now_iso(), "type": event_type, **data}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _summary_for(
    candidate: str,
    opponents: Sequence[str],
    cells: Sequence[dict[str, Any]],
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    matchups = [
        MatchupResult.from_dict(cell["matchup"])
        for cell in cells
        if cell["candidate"] == candidate
    ]
    by_opponent = {matchup.opponent: matchup for matchup in matchups}

    def selected(names: Sequence[str]) -> list[MatchupResult]:
        return [by_opponent[name] for name in names if name in by_opponent]

    common = selected(COMMON_OPPONENTS)
    strength = selected(STRENGTH_OPPONENTS)
    requested = len(opponents)
    return {
        "candidate": candidate,
        "cells_completed": len(matchups),
        "cells_expected": requested,
        "complete": len(matchups) == requested,
        "profile": profile,
        "all": summarize_report(matchups),
        "common_population": summarize_report(common),
        "strength_population": summarize_report(strength),
        "win_rates": {name: matchup.win_rate for name, matchup in by_opponent.items()},
        "wilson_low": {
            name: matchup.wilson_low for name, matchup in by_opponent.items()
        },
        "vp_diffs": {
            name: matchup.avg_vp_diff for name, matchup in by_opponent.items()
        },
    }


def _refresh_summaries(report: dict[str, Any]) -> None:
    config = report["configuration"]
    profiles = report.get("profiles", {})
    report["summaries"] = [
        _summary_for(
            candidate,
            config["opponents"],
            report["cells"],
            profiles.get(candidate),
        )
        for candidate in config["candidates"]
    ]
    expected = len(config["candidates"]) * len(config["opponents"])
    report["status"] = {
        "profiles_completed": len(profiles),
        "profiles_expected": len(config["candidates"]),
        "cells_completed": len(report["cells"]),
        "cells_expected": expected,
        "complete": len(profiles) == len(config["candidates"])
        and len(report["cells"]) == expected,
        "updated_at": utc_now_iso(),
    }


def run_teacher_benchmark(
    *,
    candidates: Sequence[str],
    opponents: Sequence[str],
    num_games: int,
    seed: int,
    profile_samples: int,
    profile_seed: int,
    report_path: Path,
    resume: bool = False,
    profile_only: bool = False,
    max_cells: int | None = None,
    quiet: bool = True,
) -> dict[str, Any]:
    """Profile candidates and evaluate each missing matrix cell atomically."""

    if num_games <= 0:
        raise ValueError("num_games must be positive")
    if profile_samples <= 0:
        raise ValueError("profile_samples must be positive")
    if max_cells is not None and max_cells <= 0:
        raise ValueError("max_cells must be positive")

    config = _configuration(
        candidates=candidates,
        opponents=opponents,
        num_games=num_games,
        seed=seed,
        profile_samples=profile_samples,
        profile_seed=profile_seed,
    )
    if report_path.exists():
        if not resume:
            raise FileExistsError(
                f"Report already exists: {report_path}; pass --resume to continue"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Existing teacher benchmark schema does not match")
        if report.get("configuration") != config:
            raise ValueError("Existing teacher benchmark configuration does not match")
    else:
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "teacher_population_benchmark",
            "created_at": utc_now_iso(),
            "configuration": config,
            "provenance": collect_run_provenance(),
            "profiles": {},
            "cells": [],
            "summaries": [],
            "status": {},
        }

    event_path = report_path.with_suffix(".events.jsonl")
    _append_event(event_path, "benchmark_start", resume=resume)
    for candidate in candidates:
        if candidate in report["profiles"]:
            continue
        _append_event(event_path, "profile_start", candidate=candidate)
        profile = profile_candidate(
            candidate, samples=profile_samples, seed=profile_seed
        )
        report["profiles"][candidate] = profile
        _refresh_summaries(report)
        _atomic_write(report, report_path)
        _append_event(
            event_path,
            "profile_complete",
            candidate=candidate,
            p95_latency_ms=profile["p95_latency_ms"],
        )

    completed = {(cell["candidate"], cell["opponent"]) for cell in report["cells"]}
    cells_run = 0
    if not profile_only:
        for candidate in candidates:
            for opponent in opponents:
                if (candidate, opponent) in completed:
                    continue
                if max_cells is not None and cells_run >= max_cells:
                    break
                _append_event(
                    event_path,
                    "cell_start",
                    candidate=candidate,
                    opponent=opponent,
                    num_games=num_games,
                )
                started = time.perf_counter()
                matchup = evaluate_matchup(
                    candidate,
                    opponent,
                    num_games=num_games,
                    both_seats=True,
                    quiet=quiet,
                    seed=seed,
                )
                report["cells"].append(
                    {
                        "candidate": candidate,
                        "opponent": opponent,
                        "duration_seconds": time.perf_counter() - started,
                        "matchup": matchup.to_dict(),
                    }
                )
                cells_run += 1
                _refresh_summaries(report)
                _atomic_write(report, report_path)
                _append_event(
                    event_path,
                    "cell_complete",
                    candidate=candidate,
                    opponent=opponent,
                    win_rate=matchup.win_rate,
                    requested_games=matchup.requested_games,
                    error_games=matchup.error_games,
                )
            if max_cells is not None and cells_run >= max_cells:
                break

    _refresh_summaries(report)
    _atomic_write(report, report_path)
    _append_event(
        event_path,
        "benchmark_complete" if report["status"]["complete"] else "benchmark_paused",
        **report["status"],
    )
    return report


def print_summary(report: dict[str, Any]) -> None:
    status = report["status"]
    print(
        "Teacher population benchmark · "
        f"{status['cells_completed']}/{status['cells_expected']} cells · "
        f"complete={status['complete']}"
    )
    print(
        f"{'candidate':>12}  {'p95 ms':>10}  {'cells':>7}  "
        f"{'common':>8}  {'strength':>8}  {'worst':>7}"
    )
    for row in report["summaries"]:
        profile = row.get("profile") or {}
        all_summary = row["all"]
        print(
            f"{row['candidate']:>12}  "
            f"{profile.get('p95_latency_ms', 0.0):>10.1f}  "
            f"{row['cells_completed']:>2}/{row['cells_expected']:<4}  "
            f"{row['common_population']['weighted_score']:>8.3f}  "
            f"{row['strength_population']['weighted_score']:>8.3f}  "
            f"{all_summary['worst_win_rate']:>6.1%}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=_csv_strings, default=DEFAULT_CANDIDATES)
    parser.add_argument("--opponents", type=_csv_strings, default=DEFAULT_OPPONENTS)
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile-samples", type=int, default=3)
    parser.add_argument("--profile-seed", type=int, default=DEFAULT_SEED - 100)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/teacher-population-benchmark/report.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Run at most this many missing cells, then leave a resumable report.",
    )
    args = parser.parse_args(argv)

    try:
        report = run_teacher_benchmark(
            candidates=args.candidates,
            opponents=args.opponents,
            num_games=args.num_games,
            seed=args.seed,
            profile_samples=args.profile_samples,
            profile_seed=args.profile_seed,
            report_path=args.report,
            resume=args.resume,
            profile_only=args.profile_only,
            max_cells=args.max_cells,
        )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print_summary(report)
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

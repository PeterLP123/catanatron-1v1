#!/usr/bin/env python3
"""Measure MCTS latency and honest two-seat strength on held-out seed suites."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

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
) -> dict[str, Any]:
    """Run the declared matrix and return a JSON-serializable evidence report."""
    profiles = {
        float(budget): profile_budget(
            float(budget),
            value_fn=value_fn,
            samples=profile_samples,
            seed=profile_seed,
        )
        for budget in budgets_ms
    }
    results: list[dict[str, Any]] = []
    if profile_only:
        for budget in budgets_ms:
            results.append(
                {"budget_ms": float(budget), "profile": profiles[float(budget)]}
            )
    else:
        for budget in budgets_ms:
            spec = f"M:1:False:{value_fn}:{float(budget):g}"
            for opponent in opponents:
                for seed in seeds:
                    matchup = evaluate_matchup(
                        spec,
                        opponent,
                        num_games=num_games,
                        both_seats=True,
                        quiet=True,
                        seed=int(seed),
                    )
                    results.append(
                        {
                            "budget_ms": float(budget),
                            "opponent": opponent,
                            "seed": int(seed),
                            "profile": profiles[float(budget)],
                            "matchup": matchup.to_dict(),
                        }
                    )
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
        "results": results,
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


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
        "--report", type=Path, help="Write the complete JSON evidence report."
    )
    args = parser.parse_args(argv)

    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.profile_samples <= 0:
        parser.error("--profile-samples must be positive")
    opponents = (args.opponent,) if args.opponent else tuple(args.opponents)
    report = build_strength_report(
        budgets_ms=args.budgets,
        opponents=opponents,
        seeds=args.seeds,
        num_games=args.num_games,
        value_fn=args.value_fn,
        profile_samples=args.profile_samples,
        profile_seed=args.profile_seed,
        profile_only=args.profile_only,
    )
    _print_report(report)
    if args.report:
        _write_report(report, args.report)
        print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

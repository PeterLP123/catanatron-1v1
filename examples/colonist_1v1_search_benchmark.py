#!/usr/bin/env python3
"""
Benchmark the repaired MCTS (F-leaf) search against a baseline at fixed latency
budgets.

This is the Phase 01 "search profiler" tool: it measures both *strength*
(honest two-seat win rate with a Wilson confidence interval, via the shared
evaluation harness) and *throughput* (simulations/sec and nodes/sec from the
MCTS profiler) at each wall-clock budget. Use it to find the latency at which
search stops underperforming the baseline.

Example::

    python examples/colonist_1v1_search_benchmark.py --budgets 10,25,50,100 \\
        --num-games 40 --opponent F

Note: this runs real games with a per-decision time budget, so larger budgets
and game counts are slow. Start small.
"""

from __future__ import annotations

import argparse
import random
import sys

from catanatron import Game, Color
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.colonist_1v1_eval import evaluate_matchup, format_matchup_line


def _profile_throughput(budget_ms: float, seed: int = 0) -> dict:
    """Run a single mid-game decision at the budget and return profiler stats."""
    random.seed(seed)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    # Advance to a non-trivial in-turn decision (more than one legal action).
    for _ in range(400):
        if game.winning_color() is not None:
            break
        if len(game.playable_actions) > 1 and not game.state.is_initial_build_phase:
            break
        game.play_tick()
    player = MCTSPlayer(game.state.current_color(), max_time_ms=budget_ms)
    player.decide(game, game.playable_actions)
    return player.last_search_stats or {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--budgets",
        default="10,25,50,100",
        help="Comma-separated per-decision time budgets in milliseconds.",
    )
    p.add_argument("--num-games", type=int, default=40, help="Games per budget.")
    p.add_argument(
        "--opponent",
        default="F",
        help="Opponent CLI code (default F, the hand-crafted value bot).",
    )
    p.add_argument(
        "--value-fn",
        default="base_fn",
        help="MCTS leaf value function builder (base_fn or contender_fn).",
    )
    p.add_argument(
        "--profile-only",
        action="store_true",
        help="Only report throughput (sims/sec), skip the win-rate games.",
    )
    args = p.parse_args(argv)

    budgets = [float(b) for b in args.budgets.split(",") if b.strip()]

    print(f"MCTS(F-leaf, {args.value_fn}) vs {args.opponent}")
    print(
        f"{'budget':>8}  {'sims/s':>10}  {'cache hit':>9}  "
        f"{'win%':>7}  {'95% CI':>16}  {'seat gap':>9}"
    )
    for budget in budgets:
        prof = _profile_throughput(budget)
        sims_s = prof.get("simulations_per_s", float("nan"))
        hit_rate = prof.get("leaf_cache_hit_rate", float("nan"))

        if args.profile_only:
            print(f"{budget:>7.0f}m  {sims_s:>10.0f}  {hit_rate:>8.1%}")
            continue

        spec = f"M:1:False:{args.value_fn}:{budget:g}"
        res = evaluate_matchup(
            spec,
            args.opponent,
            num_games=args.num_games,
            both_seats=True,
            quiet=True,
        )
        seat_gap = (
            abs(res.win_rate_seat0 - res.win_rate_seat1)
            if res.win_rate_seat0 is not None and res.win_rate_seat1 is not None
            else float("nan")
        )
        ci = f"[{res.wilson_low:.1%},{res.wilson_high:.1%}]"
        print(
            f"{budget:>7.0f}m  {sims_s:>10.0f}  {hit_rate:>8.1%}  "
            f"{res.win_rate:>6.1%}  {ci:>16}  {seat_gap:>8.1%}"
        )
        if not args.profile_only:
            print(f"           {format_matchup_line(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

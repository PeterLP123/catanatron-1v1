#!/usr/bin/env python3
"""Run a non-promotional operational gate for an outcome-critic reranker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from catanatron.colonist_1v1 import COLONIST_1V1_SETTINGS
from catanatron.game import Game
from catanatron.gym.distillation import derive_seed, isolated_random_seed
from catanatron.gym.provenance import sha256_file
from catanatron.models.map import build_map
from catanatron.models.player import Color
from catanatron.players.learned import OutcomeRerankerCheckpointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-manifest", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--minimum-choice-decisions", type=int, default=200)
    parser.add_argument("--minimum-rerank-rate", type=float, default=0.01)
    parser.add_argument("--maximum-rerank-rate", type=float, default=0.35)
    parser.add_argument("--maximum-p95-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games < 1 or args.minimum_choice_decisions < 1:
        raise ValueError("games and minimum-choice-decisions must be positive")
    if not 0 <= args.minimum_rerank_rate <= args.maximum_rerank_rate <= 1:
        raise ValueError("rerank-rate bounds must satisfy 0 <= minimum <= maximum <= 1")
    if args.maximum_p95_ms <= 0:
        raise ValueError("maximum-p95-ms must be positive")
    manifest = args.agent_manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing reranker manifest: {manifest}")

    game_rows = []
    all_latencies = []
    accepted_improvements = []
    totals = {
        "decisions": 0,
        "choice_decisions": 0,
        "reranked_decisions": 0,
        "fallback_decisions": 0,
        "candidate_actions_evaluated": 0,
    }
    for game_index in range(args.games):
        game_seed = derive_seed(
            "reranker-diagnostic-game",
            base_seed=args.seed,
            iteration=0,
            game_index=game_index,
        )
        student_seat = game_index % 2
        student_color = (Color.BLUE, Color.RED)[student_seat]
        student = OutcomeRerankerCheckpointPlayer(student_color, manifest)
        opponent_color = Color.RED if student_color == Color.BLUE else Color.BLUE
        opponent = ValueFunctionPlayer(opponent_color)
        players = [student, opponent] if student_seat == 0 else [opponent, student]
        try:
            with isolated_random_seed(game_seed):
                catan_map = build_map(
                    COLONIST_1V1_SETTINGS.map_type,
                    COLONIST_1V1_SETTINGS.number_placement,
                )
                game = Game(
                    players,
                    seed=game_seed,
                    catan_map=catan_map,
                    colonist_1v1=True,
                    shuffle_players=False,
                )
                winner = game.play()
            error = None
            student_vp = float(get_actual_victory_points(game.state, student_color))
            opponent_vp = float(get_actual_victory_points(game.state, opponent_color))
        except Exception as exc:  # preserve every requested diagnostic schedule
            winner = None
            error = f"{type(exc).__name__}: {exc}"
            student_vp = None
            opponent_vp = None
        stats = student.stats_summary()
        for key in totals:
            totals[key] += int(stats[key])
        all_latencies.extend(student.decision_stats["latencies_ms"])
        accepted_improvements.extend(student.decision_stats["accepted_improvements"])
        game_rows.append(
            {
                "game_index": game_index,
                "seed": game_seed,
                "student_seat": student_seat,
                "student_color": student_color.name,
                "winner": winner.name if winner is not None else None,
                "student_won": winner == student_color,
                "truncated": winner is None and error is None,
                "error": error,
                "student_vp": student_vp,
                "opponent_vp": opponent_vp,
                "stats": stats,
            }
        )

    latencies = np.asarray(all_latencies, dtype=float)
    improvements = np.asarray(accepted_improvements, dtype=float)
    choices = totals["choice_decisions"]
    rerank_rate = totals["reranked_decisions"] / choices if choices else 0.0
    p95 = float(np.percentile(latencies, 95)) if len(latencies) else None
    errors = sum(row["error"] is not None for row in game_rows)
    truncations = sum(row["truncated"] for row in game_rows)
    summary = {
        **totals,
        "rerank_rate": rerank_rate,
        "latency_mean_ms": float(latencies.mean()) if len(latencies) else None,
        "latency_p95_ms": p95,
        "latency_max_ms": float(latencies.max()) if len(latencies) else None,
        "accepted_improvement_mean": (
            float(improvements.mean()) if len(improvements) else None
        ),
        "student_wins": sum(row["student_won"] for row in game_rows),
        "errors": errors,
        "truncations": truncations,
    }
    gates = [
        {
            "name": "all_games_error_free",
            "passed": errors == 0,
            "actual": errors,
            "threshold": 0,
        },
        {
            "name": "all_games_completed",
            "passed": truncations == 0,
            "actual": truncations,
            "threshold": 0,
        },
        {
            "name": "minimum_choice_decisions",
            "passed": choices >= args.minimum_choice_decisions,
            "actual": choices,
            "threshold": args.minimum_choice_decisions,
        },
        {
            "name": "minimum_rerank_rate",
            "passed": rerank_rate >= args.minimum_rerank_rate,
            "actual": rerank_rate,
            "threshold": args.minimum_rerank_rate,
        },
        {
            "name": "maximum_rerank_rate",
            "passed": rerank_rate <= args.maximum_rerank_rate,
            "actual": rerank_rate,
            "threshold": args.maximum_rerank_rate,
        },
        {
            "name": "maximum_p95_ms",
            "passed": p95 is not None and p95 <= args.maximum_p95_ms,
            "actual": p95,
            "threshold": args.maximum_p95_ms,
        },
    ]
    report = {
        "schema_version": "1.0",
        "kind": "outcome_reranker_operational_diagnostic",
        "agent_manifest": str(manifest),
        "agent_manifest_sha256": sha256_file(manifest),
        "games_requested": args.games,
        "base_seed": args.seed,
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates),
        "game_results": game_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        f"reranker diagnostic: choices={choices} rerank_rate={rerank_rate:.1%} "
        f"p95_ms={p95 if p95 is not None else 'missing'} "
        f"gates={'PASS' if report['all_gates_passed'] else 'FAIL'}"
    )
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

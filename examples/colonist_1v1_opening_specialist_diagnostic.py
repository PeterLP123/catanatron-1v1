#!/usr/bin/env python3
"""Run the fixed operational gate for a setup-only opening specialist."""

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
from catanatron.players.learned import OpeningSpecialistCheckpointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-manifest", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maximum-p95-ms", type=float, default=250.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games < 1:
        raise ValueError("games must be positive")
    if args.maximum_p95_ms <= 0:
        raise ValueError("maximum-p95-ms must be positive")
    manifest = args.agent_manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing opening-specialist manifest: {manifest}")

    games = []
    latencies = []
    totals = {
        "decisions": 0,
        "choice_decisions": 0,
        "opening_decisions": 0,
        "opening_choice_decisions": 0,
        "policy_decisions": 0,
    }
    opening_prompt_counts = {
        prompt: 0 for prompt in OpeningSpecialistCheckpointPlayer.OPENING_PROMPTS
    }
    for game_index in range(args.games):
        game_seed = derive_seed(
            "opening-specialist-diagnostic-game",
            base_seed=args.seed,
            iteration=0,
            game_index=game_index,
        )
        student_seat = game_index % 2
        student_color = (Color.BLUE, Color.RED)[student_seat]
        student = OpeningSpecialistCheckpointPlayer(student_color, manifest)
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
        except Exception as exc:  # preserve every requested diagnostic game
            winner = None
            error = f"{type(exc).__name__}: {exc}"
            student_vp = None
            opponent_vp = None
        stats = student.stats_summary()
        for key in totals:
            totals[key] += int(stats[key])
        for prompt, count in stats["opening_prompt_counts"].items():
            opening_prompt_counts[prompt] += int(count)
        latencies.extend(student.decision_stats["opening_latencies_ms"])
        games.append(
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

    latency_array = np.asarray(latencies, dtype=float)
    p95 = float(np.percentile(latency_array, 95)) if len(latency_array) else None
    errors = sum(row["error"] is not None for row in games)
    truncations = sum(row["truncated"] for row in games)
    expected_total = 4 * args.games
    expected_per_prompt = 2 * args.games
    summary = {
        **totals,
        "opening_prompt_counts": opening_prompt_counts,
        "opening_latency_mean_ms": (
            float(latency_array.mean()) if len(latency_array) else None
        ),
        "opening_latency_p95_ms": p95,
        "opening_latency_max_ms": (
            float(latency_array.max()) if len(latency_array) else None
        ),
        "student_wins": sum(row["student_won"] for row in games),
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
            "name": "exact_opening_decisions",
            "passed": totals["opening_decisions"] == expected_total,
            "actual": totals["opening_decisions"],
            "threshold": expected_total,
        },
        {
            "name": "exact_initial_settlement_decisions",
            "passed": (
                opening_prompt_counts["BUILD_INITIAL_SETTLEMENT"] == expected_per_prompt
            ),
            "actual": opening_prompt_counts["BUILD_INITIAL_SETTLEMENT"],
            "threshold": expected_per_prompt,
        },
        {
            "name": "exact_initial_road_decisions",
            "passed": (
                opening_prompt_counts["BUILD_INITIAL_ROAD"] == expected_per_prompt
            ),
            "actual": opening_prompt_counts["BUILD_INITIAL_ROAD"],
            "threshold": expected_per_prompt,
        },
        {
            "name": "maximum_opening_p95_ms",
            "passed": p95 is not None and p95 <= args.maximum_p95_ms,
            "actual": p95,
            "threshold": args.maximum_p95_ms,
        },
    ]
    report = {
        "schema_version": "1.0",
        "kind": "opening_specialist_operational_diagnostic",
        "agent_manifest": str(manifest),
        "agent_manifest_sha256": sha256_file(manifest),
        "games_requested": args.games,
        "base_seed": args.seed,
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates),
        "game_results": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        "opening-specialist diagnostic: "
        f"opening={totals['opening_decisions']}/{expected_total} "
        f"settlements={opening_prompt_counts['BUILD_INITIAL_SETTLEMENT']} "
        f"roads={opening_prompt_counts['BUILD_INITIAL_ROAD']} "
        f"p95_ms={p95 if p95 is not None else 'missing'} "
        f"gates={'PASS' if report['all_gates_passed'] else 'FAIL'}"
    )
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

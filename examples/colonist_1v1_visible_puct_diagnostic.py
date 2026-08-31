#!/usr/bin/env python3
"""Run the operational and information-boundary gate for visible-state PUCT."""

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
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.visible_puct import VisibleSameTurnPuctPlayer
from catanatron.state_functions import get_actual_victory_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-manifest", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--minimum-search-decisions", type=int, default=200)
    parser.add_argument("--minimum-multi-ply-decisions", type=int, default=20)
    parser.add_argument("--minimum-change-rate", type=float, default=0.01)
    parser.add_argument("--maximum-change-rate", type=float, default=0.50)
    parser.add_argument("--maximum-p95-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games < 1 or args.minimum_search_decisions < 1:
        raise ValueError("games and minimum-search-decisions must be positive")
    if args.minimum_multi_ply_decisions < 1:
        raise ValueError("minimum-multi-ply-decisions must be positive")
    if not 0 <= args.minimum_change_rate <= args.maximum_change_rate <= 1:
        raise ValueError("change-rate bounds must satisfy 0 <= minimum <= maximum <= 1")
    if args.maximum_p95_ms <= 0:
        raise ValueError("maximum-p95-ms must be positive")
    manifest = args.agent_manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing visible PUCT manifest: {manifest}")

    scalar_keys = (
        "decisions",
        "choice_decisions",
        "search_decisions",
        "fallback_decisions",
        "changed_decisions",
        "multi_ply_decisions",
        "expanded_nodes",
        "expanded_actions",
        "forbidden_action_expansions",
        "opponent_turn_expansions",
        "value_evaluations",
        "critic_evaluations",
        "public_f_evaluations",
    )
    totals = {key: 0 for key in scalar_keys}
    fallback_action_types: dict[str, int] = {}
    all_latencies: list[float] = []
    game_rows = []
    for game_index in range(args.games):
        game_seed = derive_seed(
            "visible-puct-diagnostic-game",
            base_seed=args.seed,
            iteration=0,
            game_index=game_index,
        )
        student_seat = game_index % 2
        student_color = (Color.BLUE, Color.RED)[student_seat]
        student = VisibleSameTurnPuctPlayer(student_color, manifest)
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
        except Exception as exc:
            winner = None
            error = f"{type(exc).__name__}: {exc}"
            student_vp = None
            opponent_vp = None
        stats = student.stats_summary()
        for key in totals:
            totals[key] += int(stats[key])
        all_latencies.extend(student.decision_stats["latencies_ms"])
        for action_type, count in stats["fallback_action_types"].items():
            fallback_action_types[action_type] = fallback_action_types.get(
                action_type, 0
            ) + int(count)
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
    searches = totals["search_decisions"]
    change_rate = totals["changed_decisions"] / searches if searches else 0.0
    p95 = float(np.percentile(latencies, 95)) if len(latencies) else None
    errors = sum(row["error"] is not None for row in game_rows)
    truncations = sum(row["truncated"] for row in game_rows)
    summary = {
        **totals,
        "change_rate": change_rate,
        "latency_mean_ms": float(latencies.mean()) if len(latencies) else None,
        "latency_p95_ms": p95,
        "latency_max_ms": float(latencies.max()) if len(latencies) else None,
        "fallback_action_types": fallback_action_types,
        "student_wins": sum(row["student_won"] for row in game_rows),
        "errors": errors,
        "truncations": truncations,
    }
    gates = [
        ("all_games_error_free", errors == 0, errors, 0),
        ("all_games_completed", truncations == 0, truncations, 0),
        (
            "minimum_search_decisions",
            searches >= args.minimum_search_decisions,
            searches,
            args.minimum_search_decisions,
        ),
        (
            "minimum_multi_ply_decisions",
            totals["multi_ply_decisions"] >= args.minimum_multi_ply_decisions,
            totals["multi_ply_decisions"],
            args.minimum_multi_ply_decisions,
        ),
        (
            "minimum_change_rate",
            change_rate >= args.minimum_change_rate,
            change_rate,
            args.minimum_change_rate,
        ),
        (
            "maximum_change_rate",
            change_rate <= args.maximum_change_rate,
            change_rate,
            args.maximum_change_rate,
        ),
        (
            "maximum_p95_ms",
            p95 is not None and p95 <= args.maximum_p95_ms,
            p95,
            args.maximum_p95_ms,
        ),
        (
            "zero_forbidden_action_expansions",
            totals["forbidden_action_expansions"] == 0,
            totals["forbidden_action_expansions"],
            0,
        ),
        (
            "zero_opponent_turn_expansions",
            totals["opponent_turn_expansions"] == 0,
            totals["opponent_turn_expansions"],
            0,
        ),
    ]
    gate_rows = [
        {"name": name, "passed": passed, "actual": actual, "threshold": threshold}
        for name, passed, actual, threshold in gates
    ]
    report = {
        "schema_version": "1.0",
        "kind": "visible_same_turn_puct_operational_diagnostic",
        "agent_manifest": str(manifest),
        "agent_manifest_sha256": sha256_file(manifest),
        "games_requested": args.games,
        "base_seed": args.seed,
        "summary": summary,
        "gates": gate_rows,
        "all_gates_passed": all(gate[1] for gate in gates),
        "game_results": game_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        f"visible PUCT diagnostic: searches={searches} "
        f"multi_ply={totals['multi_ply_decisions']} change_rate={change_rate:.1%} "
        f"p95_ms={p95 if p95 is not None else 'missing'} "
        f"gates={'PASS' if report['all_gates_passed'] else 'FAIL'}"
    )
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

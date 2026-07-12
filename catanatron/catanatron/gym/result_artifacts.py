"""Validate run-local evaluations and emit compact, tracked evidence artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from catanatron.colonist_1v1_eval import (
    DEFAULT_BENCHMARK_GATES,
    EvaluationReport,
    confidence_gate_passed,
    summarize_report,
    wilson_score_interval,
)
from catanatron.gym.model_schema import canonical_hash
from catanatron.gym.provenance import sha256_file

TRACKED_RESULT_SCHEMA = "catanatron-result-1"


def validate_publishable_report(report: EvaluationReport) -> None:
    eval_kind = report.meta.get("eval_kind")
    if eval_kind not in {"promotion", "final", "final_benchmark"}:
        raise ValueError(
            f"Only locked promotion/final evidence is publishable, got {eval_kind!r}"
        )
    protocol = report.meta.get("protocol", {})
    if protocol.get("seed_suite") not in {"promotion", "final"}:
        raise ValueError(
            "Publishable evidence must use a locked promotion/final seed suite"
        )
    model_meta = report.meta.get("model", {})
    model_hash = model_meta.get("file_sha256")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise ValueError("Evaluation is missing the checkpoint file hash")
    try:
        int(model_hash, 16)
    except ValueError as exc:
        raise ValueError("Evaluation checkpoint hash is not hexadecimal") from exc
    if model_meta.get("agent_spec") != report.agent:
        raise ValueError("Evaluation agent disagrees with model metadata")
    if report.colonist_1v1 is not True:
        raise ValueError("Only Colonist 1v1 evidence is publishable")
    if report.meta.get("both_seats") is not True:
        raise ValueError("Publishable evidence must evaluate both player seats")
    if not report.matchups:
        raise ValueError("Evaluation has no matchup evidence")
    matchup_opponents = [matchup.opponent for matchup in report.matchups]
    protocol_opponents = protocol.get("opponents")
    if not isinstance(protocol_opponents, list) or not protocol_opponents:
        raise ValueError("Locked protocol is missing its opponent list")
    if len(set(protocol_opponents)) != len(protocol_opponents):
        raise ValueError("Locked protocol contains duplicate opponents")
    if protocol_opponents != matchup_opponents:
        raise ValueError("Matchup opponents do not exactly match the locked protocol")
    protocol_gates = protocol.get("gates")
    if not isinstance(protocol_gates, dict) or set(protocol_gates) != set(
        protocol_opponents
    ):
        raise ValueError(
            "predeclared gates must exactly cover the locked protocol opponents"
        )
    protocol_gate_mode = protocol.get("gate_mode")
    if protocol_gate_mode not in {"point", "lower_bound"}:
        raise ValueError("Locked protocol has an invalid gate mode")
    try:
        protocol_games = int(protocol["num_games_per_matchup"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Locked protocol has no valid game count") from exc
    if protocol_games <= 0:
        raise ValueError("Locked protocol game count must be positive")
    total_requested = 0
    for matchup in report.matchups:
        requested = int(matchup.requested_games or matchup.games)
        if requested <= 0:
            raise ValueError(f"{matchup.opponent} requested no games")
        if requested != protocol_games:
            raise ValueError(
                f"{matchup.opponent} game count disagrees with the locked protocol"
            )
        declared_gate = protocol_gates[matchup.opponent]
        canonical_gate = DEFAULT_BENCHMARK_GATES.get(matchup.opponent)
        try:
            declared_gate_value = float(declared_gate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{matchup.opponent} has a non-numeric locked gate"
            ) from exc
        if (
            canonical_gate is None
            or isinstance(declared_gate, bool)
            or not math.isfinite(declared_gate_value)
            or not math.isclose(
                declared_gate_value,
                canonical_gate,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"{matchup.opponent} does not use the canonical benchmark gate"
            )
        try:
            matchup_gate = float(matchup.gate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{matchup.opponent} has no numeric matchup gate") from exc
        if not math.isclose(
            matchup_gate,
            declared_gate_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{matchup.opponent} gate disagrees with the locked protocol"
            )
        if matchup.gate_mode != protocol_gate_mode:
            raise ValueError(
                f"{matchup.opponent} gate mode disagrees with the locked protocol"
            )
        total_requested += requested
        if matchup.agent_code != report.agent:
            raise ValueError(
                f"{matchup.opponent} agent code disagrees with the report agent"
            )
        if len(matchup.game_results) != requested:
            raise ValueError(
                f"{matchup.opponent} lacks per-game evidence for every requested game"
            )
        schedule_ids = [outcome.schedule_id for outcome in matchup.game_results]
        if len(set(schedule_ids)) != len(schedule_ids):
            raise ValueError(f"{matchup.opponent} contains duplicate game evidence")
        valid_status_results = {
            "completed": {"win", "loss"},
            "truncated": {"draw"},
            "error": {"error"},
        }
        for outcome in matchup.game_results:
            if (
                outcome.status not in valid_status_results
                or outcome.result not in valid_status_results[outcome.status]
            ):
                raise ValueError(
                    f"{matchup.opponent} has invalid status/result pair: "
                    f"{outcome.status}/{outcome.result}"
                )
            if outcome.agent_vp is not None and outcome.opponent_vp is not None:
                expected_vp_diff = outcome.agent_vp - outcome.opponent_vp
                if outcome.vp_diff is None or not math.isclose(
                    outcome.vp_diff, expected_vp_diff, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"{matchup.opponent} game vp_diff disagrees with final VP"
                    )
        expected_counts = {
            "wins": sum(row.result == "win" for row in matchup.game_results),
            "losses": sum(row.result == "loss" for row in matchup.game_results),
            "draws": sum(row.result == "draw" for row in matchup.game_results),
            "error_games": sum(row.status == "error" for row in matchup.game_results),
            "completed_games": sum(
                row.status == "completed" for row in matchup.game_results
            ),
            "truncated_games": sum(
                row.status == "truncated" for row in matchup.game_results
            ),
            "observed_games": sum(
                row.status != "error" for row in matchup.game_results
            ),
        }
        for name, expected in expected_counts.items():
            actual = getattr(matchup, name)
            if int(actual or 0) != expected:
                raise ValueError(
                    f"{matchup.opponent} aggregate {name}={actual} "
                    f"disagrees with per-game evidence={expected}"
                )
        if matchup.games != requested:
            raise ValueError(f"{matchup.opponent} games denominator is inconsistent")
        expected_win_rate = expected_counts["wins"] / requested
        expected_low, expected_high = wilson_score_interval(
            expected_counts["wins"], requested
        )
        observed_rows = [row for row in matchup.game_results if row.status != "error"]
        if not observed_rows:
            raise ValueError(f"{matchup.opponent} has no observed game evidence")
        expected_agent_vp = sum(row.agent_vp or 0.0 for row in observed_rows) / len(
            observed_rows
        )
        expected_opponent_vp = sum(
            row.opponent_vp or 0.0 for row in observed_rows
        ) / len(observed_rows)
        expected_turns = sum(row.turns or 0 for row in observed_rows) / len(
            observed_rows
        )
        expected_floats = {
            "win_rate": expected_win_rate,
            "wilson_low": expected_low,
            "wilson_high": expected_high,
            "avg_agent_vp": expected_agent_vp,
            "avg_opponent_vp": expected_opponent_vp,
            "avg_vp_diff": expected_agent_vp - expected_opponent_vp,
            "avg_turns": expected_turns,
        }
        for name, expected in expected_floats.items():
            if not math.isclose(float(getattr(matchup, name)), expected, abs_tol=1e-12):
                raise ValueError(
                    f"{matchup.opponent} aggregate {name} disagrees with per-game evidence"
                )
        expected_gate_value = (
            expected_low if matchup.gate_mode == "lower_bound" else expected_win_rate
        )
        expected_passed = confidence_gate_passed(
            estimate=expected_win_rate,
            threshold=matchup.gate,
            confidence_low=expected_low,
            mode=matchup.gate_mode,
        )
        if matchup.gate_value is None or not math.isclose(
            float(matchup.gate_value), expected_gate_value, abs_tol=1e-12
        ):
            raise ValueError(
                f"{matchup.opponent} gate_value disagrees with per-game evidence"
            )
        if matchup.passed_gate is not expected_passed:
            raise ValueError(
                f"{matchup.opponent} passed_gate disagrees with per-game evidence"
            )
    if total_requested <= 0:
        raise ValueError("Evaluation requested no games")

    recomputed = summarize_report(report.matchups)
    for key in (
        "requested_games",
        "accounted_games",
        "all_games_accounted",
        "observed_games",
        "completed_games",
        "truncated_games",
        "error_games",
        "gates_total",
        "gates_passed_count",
    ):
        if report.summary.get(key) != recomputed.get(key):
            raise ValueError(f"Evaluation summary {key} is internally inconsistent")
    for key in (
        "mean_win_rate",
        "weighted_score",
        "best_win_rate",
        "worst_win_rate",
    ):
        try:
            consistent = math.isclose(
                float(report.summary[key]),
                float(recomputed[key]),
                abs_tol=1e-12,
            )
        except (KeyError, TypeError, ValueError):
            consistent = False
        if not consistent:
            raise ValueError(f"Evaluation summary {key} is internally inconsistent")
    if not recomputed["all_games_accounted"]:
        raise ValueError("Evaluation omitted or double-counted requested games")
    if int(recomputed["error_games"]):
        raise ValueError("Evaluation contains errored games")
    if int(recomputed["gates_total"]) <= 0:
        raise ValueError("Publishable evidence must contain predeclared gates")
    computed_all_gates = all(
        matchup.gate is not None and matchup.passed_gate is True
        for matchup in report.matchups
    )
    if report.all_gates_passed != computed_all_gates:
        raise ValueError("all_gates_passed disagrees with matchup gate evidence")


def compact_result_artifact(
    report: EvaluationReport,
    *,
    source_path: str | Path,
) -> dict[str, Any]:
    validate_publishable_report(report)
    source = Path(source_path)
    game_evidence = [
        outcome.to_dict()
        for matchup in report.matchups
        for outcome in matchup.game_results
    ]
    matchups = []
    for matchup in report.matchups:
        aggregate = asdict(matchup)
        aggregate.pop("game_results", None)
        matchups.append(aggregate)
    artifact = {
        "schema_version": TRACKED_RESULT_SCHEMA,
        "status": "accepted" if report.all_gates_passed else "rejected",
        "agent": report.agent,
        "colonist_1v1": report.colonist_1v1,
        "source_report": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "model": report.meta["model"],
        "protocol": report.meta["protocol"],
        "summary": report.summary,
        "matchups": matchups,
        "game_evidence": {
            "rows": len(game_evidence),
            "sha256": canonical_hash(game_evidence),
        },
    }
    artifact["result_sha256"] = canonical_hash(artifact)
    return artifact


def publish_compact_result(
    report_path: str | Path,
    output_path: str | Path,
) -> Path:
    source = Path(report_path)
    report = EvaluationReport.read_json(source)
    artifact = compact_result_artifact(report, source_path=source)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output

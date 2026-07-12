from __future__ import annotations

import json

import pytest

from catanatron.colonist_1v1_eval import (
    EvaluationReport,
    GameOutcome,
    MatchupResult,
    summarize_report,
    wilson_score_interval,
)
from catanatron.gym.result_artifacts import publish_compact_result


def _report(
    tmp_path, *, eval_kind="final_benchmark", errors=0, with_gate=True, passed=False
):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"model")
    second_result = "win" if passed else "loss"
    outcomes = [
        GameOutcome(
            0,
            0,
            True,
            "completed",
            "win",
            seed=100,
            agent_vp=15,
            opponent_vp=5,
            vp_diff=10,
            turns=40,
        ),
        GameOutcome(
            1,
            1,
            False,
            "completed",
            second_result,
            seed=100,
            agent_vp=15 if passed else 5,
            opponent_vp=5 if passed else 15,
            vp_diff=10 if passed else -10,
            turns=40,
        ),
    ]
    wins = 2 if passed else 1
    losses = 0 if passed else 1
    low, high = wilson_score_interval(wins, 2)
    avg_agent_vp = sum(row.agent_vp for row in outcomes) / 2
    avg_opponent_vp = sum(row.opponent_vp for row in outcomes) / 2
    matchup = MatchupResult(
        opponent="F",
        agent_code="L:model.zip",
        games=2,
        wins=wins,
        losses=losses,
        draws=0,
        win_rate=wins / 2,
        wilson_low=low,
        wilson_high=high,
        avg_agent_vp=avg_agent_vp,
        avg_opponent_vp=avg_opponent_vp,
        avg_vp_diff=avg_agent_vp - avg_opponent_vp,
        avg_turns=40,
        requested_games=2,
        observed_games=2,
        completed_games=2,
        truncated_games=0,
        error_games=errors,
        game_results=outcomes,
        gate=0.52 if with_gate else None,
        passed_gate=passed if with_gate else None,
    )
    summary = summarize_report([matchup])
    return EvaluationReport(
        agent="L:model.zip",
        matchups=[matchup],
        all_gates_passed=bool(with_gate and passed),
        meta={
            "eval_kind": eval_kind,
            "both_seats": True,
            "protocol": {
                "seed_suite": "final",
                "opponents": ["F"],
                "num_games_per_matchup": 2,
                "gates": {"F": 0.52} if with_gate else {},
                "gate_mode": "point",
            },
            "model": {
                "agent_spec": "L:model.zip",
                "file_sha256": "a" * 64,
            },
        },
        summary=summary,
    )


def test_publish_compacts_per_game_rows_but_hashes_them(tmp_path):
    source = tmp_path / "full.json"
    _report(tmp_path).write_json(source)
    output = publish_compact_result(source, tmp_path / "tracked.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "rejected"
    assert "game_results" not in payload["matchups"][0]
    assert payload["game_evidence"]["rows"] == 2
    assert len(payload["game_evidence"]["sha256"]) == 64
    assert len(payload["result_sha256"]) == 64


def test_publish_rejects_dev_or_broken_accounting(tmp_path):
    for report in (_report(tmp_path, eval_kind="dev"), _report(tmp_path, errors=1)):
        source = (
            tmp_path
            / f"{report.meta['eval_kind']}-{report.summary['error_games']}.json"
        )
        report.write_json(source)
        with pytest.raises(ValueError):
            publish_compact_result(source, tmp_path / "nope.json")


def test_publish_marks_only_fully_gated_evidence_accepted(tmp_path):
    report = _report(tmp_path, passed=True)
    source = tmp_path / "accepted.json"
    report.write_json(source)

    output = publish_compact_result(source, tmp_path / "accepted-out.json")
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "accepted"


def test_publish_rejects_vacuous_gates_or_inconsistent_aggregates(tmp_path):
    vacuous = _report(tmp_path, with_gate=False)
    vacuous.all_gates_passed = True
    vacuous_source = tmp_path / "vacuous.json"
    vacuous.write_json(vacuous_source)
    with pytest.raises(ValueError, match="predeclared gates"):
        publish_compact_result(vacuous_source, tmp_path / "vacuous-out.json")

    inconsistent = _report(tmp_path)
    inconsistent.matchups[0].wins = 2
    inconsistent.matchups[0].losses = 2
    inconsistent.summary = summarize_report(inconsistent.matchups)
    inconsistent_source = tmp_path / "inconsistent.json"
    inconsistent.write_json(inconsistent_source)
    with pytest.raises(ValueError, match="disagrees with per-game evidence"):
        publish_compact_result(inconsistent_source, tmp_path / "bad-out.json")


def test_publish_rejects_forged_headlines_and_gate_drift(tmp_path):
    forged = _report(tmp_path, passed=True)
    forged.summary["weighted_score"] = 999
    forged.summary["mean_win_rate"] = -100
    source = tmp_path / "forged-headlines.json"
    forged.write_json(source)
    with pytest.raises(ValueError, match="summary mean_win_rate"):
        publish_compact_result(source, tmp_path / "forged-out.json")

    weakened = _report(tmp_path, passed=True)
    weakened.matchups[0].gate = 0.0
    weakened.meta["protocol"]["gates"] = {"F": 0.99}
    source = tmp_path / "weakened-gate.json"
    weakened.write_json(source)
    with pytest.raises(ValueError, match="canonical benchmark gate"):
        publish_compact_result(source, tmp_path / "weakened-out.json")


def test_publish_rejects_protocol_coverage_or_gate_mode_drift(tmp_path):
    missing = _report(tmp_path)
    missing.meta["protocol"]["opponents"] = ["F", "W"]
    missing.meta["protocol"]["gates"]["W"] = 0.70
    source = tmp_path / "missing-opponent.json"
    missing.write_json(source)
    with pytest.raises(ValueError, match="exactly match"):
        publish_compact_result(source, tmp_path / "missing-out.json")

    mode_drift = _report(tmp_path)
    mode_drift.meta["protocol"]["gate_mode"] = "lower_bound"
    source = tmp_path / "mode-drift.json"
    mode_drift.write_json(source)
    with pytest.raises(ValueError, match="gate mode disagrees"):
        publish_compact_result(source, tmp_path / "mode-out.json")

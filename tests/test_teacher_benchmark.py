from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from catanatron.colonist_1v1_eval import MatchupResult


MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "examples"
        / "colonist_1v1_teacher_benchmark.py"
    )
)


def _matchup(opponent: str, *, games: int, win_rate: float) -> MatchupResult:
    wins = round(games * win_rate)
    return MatchupResult(
        opponent=opponent,
        agent_code="candidate",
        games=games,
        wins=wins,
        losses=games - wins,
        draws=0,
        win_rate=wins / games,
        wilson_low=max(0.0, wins / games - 0.1),
        wilson_high=min(1.0, wins / games + 0.1),
        avg_agent_vp=10.0,
        avg_opponent_vp=8.0,
        avg_vp_diff=2.0,
        avg_turns=50.0,
    )


def test_teacher_benchmark_is_incremental_and_resumable(tmp_path, monkeypatch):
    run = MODULE["run_teacher_benchmark"]
    profile_calls = []
    eval_calls = []

    def fake_profile(candidate, *, samples, seed):
        profile_calls.append((candidate, samples, seed))
        return {
            "candidate": candidate,
            "samples": samples,
            "p95_latency_ms": 12.0,
        }

    def fake_evaluate(candidate, opponent, *, num_games, both_seats, seed, **_kwargs):
        assert both_seats is True
        eval_calls.append((candidate, opponent, seed))
        return _matchup(opponent, games=num_games, win_rate=0.75)

    monkeypatch.setitem(run.__globals__, "profile_candidate", fake_profile)
    monkeypatch.setitem(run.__globals__, "evaluate_matchup", fake_evaluate)
    report_path = tmp_path / "teacher" / "report.json"
    kwargs = {
        "candidates": ("AB:2", "M:200"),
        "opponents": ("R", "F"),
        "num_games": 4,
        "seed": 101,
        "profile_samples": 2,
        "profile_seed": 7,
        "report_path": report_path,
    }

    partial = run(**kwargs, max_cells=2)
    assert partial["status"]["cells_completed"] == 2
    assert partial["status"]["complete"] is False
    assert len(json.loads(report_path.read_text())["cells"]) == 2

    complete = run(**kwargs, resume=True)
    assert complete["status"]["complete"] is True
    assert complete["status"]["cells_completed"] == 4
    assert len(profile_calls) == 2
    assert len(eval_calls) == 4
    assert all(row["all"]["all_games_accounted"] for row in complete["summaries"])
    events = report_path.with_suffix(".events.jsonl").read_text().splitlines()
    assert json.loads(events[-1])["type"] == "benchmark_complete"


def test_resume_rejects_changed_matrix(tmp_path, monkeypatch):
    run = MODULE["run_teacher_benchmark"]
    monkeypatch.setitem(
        run.__globals__,
        "profile_candidate",
        lambda candidate, **_kwargs: {
            "candidate": candidate,
            "p95_latency_ms": 1.0,
        },
    )
    report_path = tmp_path / "report.json"
    base = {
        "candidates": ("M:200",),
        "opponents": ("R",),
        "num_games": 2,
        "seed": 101,
        "profile_samples": 1,
        "profile_seed": 7,
        "report_path": report_path,
        "profile_only": True,
    }
    run(**base)

    with pytest.raises(ValueError, match="configuration does not match"):
        run(**{**base, "opponents": ("W",), "resume": True})


def test_profile_only_never_starts_matchups(tmp_path, monkeypatch):
    run = MODULE["run_teacher_benchmark"]
    monkeypatch.setitem(
        run.__globals__,
        "profile_candidate",
        lambda candidate, **_kwargs: {
            "candidate": candidate,
            "p95_latency_ms": 1.0,
        },
    )

    def unexpected_evaluate(*_args, **_kwargs):
        raise AssertionError("profile-only benchmark started a matchup")

    monkeypatch.setitem(run.__globals__, "evaluate_matchup", unexpected_evaluate)
    report = run(
        candidates=("AB:2", "M:800"),
        opponents=("R", "F"),
        num_games=2,
        seed=101,
        profile_samples=1,
        profile_seed=7,
        report_path=tmp_path / "report.json",
        profile_only=True,
    )

    assert report["status"]["profiles_completed"] == 2
    assert report["status"]["cells_completed"] == 0


def test_default_matrix_covers_full_battery_and_larger_fixed_search():
    assert MODULE["DEFAULT_CANDIDATES"] == ("AB:2", "M:200", "M:800", "M:2000")
    assert MODULE["DEFAULT_OPPONENTS"] == (
        "R",
        "W",
        "VP",
        "F",
        "G:25",
        "M:200",
        "AB:2",
    )

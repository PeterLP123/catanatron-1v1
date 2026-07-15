import io
import json
import runpy
from pathlib import Path

import pytest


MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "examples"
        / "colonist_1v1_search_benchmark.py"
    )
)


def test_percentile_interpolates_tail_latency():
    percentile = MODULE["percentile"]
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0], 95) == pytest.approx(2.9)


def test_strength_report_covers_every_budget_opponent_and_seed(monkeypatch):
    build = MODULE["build_strength_report"]

    def fake_profile(budget, **_kwargs):
        return {
            "budget_ms": budget,
            "p95_latency_ms": budget + 1,
            "simulations_per_s": 100.0,
        }

    class FakeMatchup:
        def __init__(self, opponent, seed, games):
            self.opponent = opponent
            self.seed = seed
            self.games = games

        def to_dict(self):
            return {
                "opponent": self.opponent,
                "requested_games": self.games,
                "games": self.games,
                "game_results": [
                    {
                        "schedule_id": f"seat-{index % 2}:game-{self.seed + index}",
                        "outcome": "draw",
                    }
                    for index in range(self.games)
                ],
            }

    def fake_evaluate(_spec, opponent, *, num_games, seed, both_seats, **_kwargs):
        assert both_seats is True
        return FakeMatchup(opponent, seed, num_games)

    monkeypatch.setitem(build.__globals__, "profile_budget", fake_profile)
    monkeypatch.setitem(build.__globals__, "evaluate_matchup", fake_evaluate)
    report = build(
        budgets_ms=(10.0, 25.0),
        opponents=("F", "AB:2"),
        seeds=(101, 202),
        num_games=4,
        value_fn="base_fn",
        profile_samples=3,
        profile_seed=7,
    )

    assert report["meta"]["both_seats"] is True
    assert report["meta"]["seeds"] == [101, 202]
    assert len(report["results"]) == 8
    assert {
        (row["budget_ms"], row["opponent"], row["seed"]) for row in report["results"]
    } == {
        (budget, opponent, seed)
        for budget in (10.0, 25.0)
        for opponent in ("F", "AB:2")
        for seed in (101, 202)
    }
    assert all(len(row["matchup"]["game_results"]) == 4 for row in report["results"])
    assert report["progress"]["status"] == "complete"
    assert report["progress"]["completed_cells"] == 8
    assert report["progress"]["percent_complete"] == 100.0


def test_resumable_report_persists_each_cell_and_skips_completed(monkeypatch, tmp_path):
    run = MODULE["run_strength_report"]
    report_path = tmp_path / "mcts_strength_sweep.json"
    profile_calls = []
    matchup_calls = []

    def fake_profile(budget, **_kwargs):
        profile_calls.append(budget)
        return {
            "budget_ms": budget,
            "p95_latency_ms": budget + 1,
            "simulations_per_s": 100.0,
        }

    class FakeMatchup:
        def __init__(self, opponent, seed, games):
            self.opponent = opponent
            self.seed = seed
            self.games = games

        def to_dict(self):
            return {
                "opponent": self.opponent,
                "requested_games": self.games,
                "games": self.games,
                "game_results": [
                    {
                        "schedule_id": f"seat-{index % 2}:game-{self.seed + index}",
                        "agent_seat": index % 2,
                        "outcome": "draw",
                    }
                    for index in range(self.games)
                ],
            }

    def interrupt_second_cell(
        _spec, opponent, *, num_games, seed, both_seats, **_kwargs
    ):
        assert both_seats is True
        matchup_calls.append((opponent, seed))
        if opponent == "AB:2" and matchup_calls.count((opponent, seed)) == 1:
            raise KeyboardInterrupt
        return FakeMatchup(opponent, seed, num_games)

    monkeypatch.setitem(run.__globals__, "profile_budget", fake_profile)
    monkeypatch.setitem(run.__globals__, "evaluate_matchup", interrupt_second_cell)
    monkeypatch.setitem(run.__globals__, "current_git_commit", lambda: "test-commit")

    kwargs = {
        "budgets_ms": (10.0,),
        "opponents": ("F", "AB:2"),
        "seeds": (101,),
        "num_games": 2,
        "value_fn": "base_fn",
        "profile_samples": 3,
        "profile_seed": 7,
        "report_path": report_path,
        "progress_stream": io.StringIO(),
    }
    with pytest.raises(KeyboardInterrupt):
        run(**kwargs)

    partial = json.loads(report_path.read_text())
    assert partial["progress"]["completed_cells"] == 1
    assert partial["progress"]["total_cells"] == 2
    assert partial["progress"]["status"] == "running"
    assert partial["progress"]["current_cell"]["opponent"] == "AB:2"
    assert len(partial["results"]) == 1
    assert not report_path.with_suffix(".json.tmp").exists()

    completed = run(**kwargs)

    assert profile_calls == [10.0]
    assert matchup_calls == [("F", 101), ("AB:2", 101), ("AB:2", 101)]
    assert completed["progress"]["status"] == "complete"
    assert completed["progress"]["completed_cells"] == 2
    assert completed["progress"]["percent_complete"] == 100.0
    assert completed["progress"]["eta_seconds"] == 0.0
    assert completed["progress"]["resume_count"] == 1
    assert len(completed["results"]) == 2


def test_resume_rejects_a_different_commit(monkeypatch, tmp_path):
    run = MODULE["run_strength_report"]
    report_path = tmp_path / "mcts_strength_sweep.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "mcts_strength_sweep",
                "meta": {
                    "git_commit": "old-commit",
                    "budgets_ms": [10.0],
                    "opponents": ["F"],
                    "seeds": [101],
                    "num_games_per_cell": 2,
                    "both_seats": True,
                    "value_fn": "base_fn",
                    "profile_samples": 3,
                    "profile_seed": 7,
                    "profile_only": False,
                    "agent_spec_template": "M:1:False:base_fn:<budget_ms>",
                },
                "results": [],
            }
        )
    )
    monkeypatch.setitem(run.__globals__, "current_git_commit", lambda: "new-commit")

    with pytest.raises(ValueError, match="commit and sweep configuration"):
        run(
            budgets_ms=(10.0,),
            opponents=("F",),
            seeds=(101,),
            num_games=2,
            value_fn="base_fn",
            profile_samples=3,
            profile_seed=7,
            report_path=report_path,
            progress_stream=io.StringIO(),
        )


def test_status_mode_reports_pending_and_saved_progress(tmp_path, capsys):
    main = MODULE["main"]
    report_path = tmp_path / "mcts_strength_sweep.json"

    assert main(["--status", "--report", str(report_path)]) == 0
    assert "0/24 cells" in capsys.readouterr().out

    report_path.write_text(
        json.dumps(
            {
                "progress": {
                    "completed_cells": 7,
                    "total_cells": 24,
                    "percent_complete": 100 * 7 / 24,
                    "status": "running",
                    "elapsed_seconds": 120,
                    "eta_seconds": 300,
                    "current_cell": {
                        "phase": "cell",
                        "budget_ms": 25.0,
                        "opponent": "F",
                        "seed": 20260712,
                    },
                }
            }
        )
    )
    assert main(["--status", "--report", str(report_path)]) == 0
    output = capsys.readouterr().out
    assert "7/24 cells (29.2%)" in output
    assert "elapsed=00:02:00" in output
    assert "eta=00:05:00" in output
    assert "current=25ms/F/seed-20260712" in output

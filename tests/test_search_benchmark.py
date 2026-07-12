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

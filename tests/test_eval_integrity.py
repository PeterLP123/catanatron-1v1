import json
from types import SimpleNamespace

import pytest

from catanatron import Color
from catanatron.cli.accumulators import StatisticsAccumulator
from catanatron.colonist_1v1_eval import (
    DEFAULT_EVAL_SEED,
    EvaluationReport,
    GameOutcome,
    MatchupResult,
    compare_paired_matchups,
    confidence_gate_passed,
    resolve_eval_seed,
    run_benchmark,
)


def _game(*, winner, seed, game_id, turns, ticks=3):
    return SimpleNamespace(
        id=game_id,
        seed=seed,
        state=SimpleNamespace(
            colors=(Color.BLUE, Color.RED),
            num_turns=turns,
            action_records=[None] * ticks,
        ),
        winning_color=lambda: winner,
    )


def _matchup(*, game_results):
    wins = sum(result.outcome == "win" for result in game_results)
    losses = sum(result.outcome == "loss" for result in game_results)
    draws = sum(result.outcome in {"draw", "truncated"} for result in game_results)
    games = len(game_results)
    return MatchupResult(
        opponent="F",
        agent_code="L:model.zip",
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / games,
        wilson_low=0.0,
        wilson_high=1.0,
        avg_agent_vp=0.0,
        avg_opponent_vp=0.0,
        avg_vp_diff=0.0,
        avg_turns=0.0,
        game_results=list(game_results),
    )


def test_statistics_accumulator_keeps_truncated_game_and_final_vp(monkeypatch):
    game = _game(
        winner=None,
        seed=11,
        game_id="seed-11",
        turns=1000,
    )
    final_vp = {Color.BLUE: 8, Color.RED: 7}
    monkeypatch.setattr(
        "catanatron.cli.accumulators.get_actual_victory_points",
        lambda state, color: final_vp[color],
    )

    accumulator = StatisticsAccumulator()
    accumulator.before(game)
    accumulator.after(game)

    assert accumulator.games == [game]
    assert accumulator.draws == 1
    assert accumulator.truncations == 1
    assert accumulator.results_by_player == {
        Color.BLUE: [8],
        Color.RED: [7],
    }


def test_evaluate_matchup_accounts_for_truncation_with_per_game_vp(monkeypatch):
    def fake_play_batch(num_games, players, output_options, game_config, quiet):
        agent_color, opponent_color = (player.color for player in players)
        games = [
            _game(
                winner=agent_color,
                seed=game_config.seed,
                game_id=f"seed-{game_config.seed}",
                turns=70,
            ),
            _game(
                winner=None,
                seed=game_config.seed + 1,
                game_id=f"seed-{game_config.seed + 1}",
                turns=1000,
            ),
        ]
        return (
            {agent_color: 1, opponent_color: 0},
            {agent_color: [10, 8], opponent_color: [5, 7]},
            games,
        )

    monkeypatch.setattr(
        "catanatron.colonist_1v1_eval.play_batch",
        fake_play_batch,
    )
    from catanatron.colonist_1v1_eval import evaluate_matchup

    result = evaluate_matchup("F", "R", num_games=2, both_seats=False, seed=900)

    assert result.games == result.requested_games == 2
    assert result.observed_games == 2
    assert result.completed_games == 1
    assert result.truncated_games == result.draws == 1
    assert result.error_games == 0
    assert result.wins + result.losses + result.draws + result.error_games == 2
    assert [row.outcome for row in result.game_results] == ["win", "truncated"]
    truncated = result.game_results[1]
    assert truncated.schedule_id == "seat-0:game-901"
    assert truncated.agent_seat == 0
    assert truncated.truncated is True and truncated.errored is False
    assert (truncated.agent_vp, truncated.opponent_vp, truncated.vp_diff) == (
        8.0,
        7.0,
        1.0,
    )


def test_evaluate_matchup_marks_missing_batch_results_as_errors(monkeypatch):
    def fake_play_batch(num_games, players, output_options, game_config, quiet):
        agent_color, opponent_color = (player.color for player in players)
        game = _game(
            winner=agent_color,
            seed=game_config.seed,
            game_id=f"seed-{game_config.seed}",
            turns=50,
        )
        return (
            {agent_color: 1, opponent_color: 0},
            {agent_color: [10], opponent_color: [4]},
            [game],
        )

    monkeypatch.setattr(
        "catanatron.colonist_1v1_eval.play_batch",
        fake_play_batch,
    )
    from catanatron.colonist_1v1_eval import evaluate_matchup

    result = evaluate_matchup("F", "R", num_games=2, both_seats=False)

    assert result.requested_games == 2
    assert result.observed_games == 1
    assert result.error_games == 1
    assert result.game_results[-1].outcome == "error"
    assert result.game_results[-1].errored is True
    assert "returned 1 of 2" in result.game_results[-1].error


def test_seed_suites_are_distinct_deterministic_and_recorded():
    suites = {
        name: resolve_eval_seed(DEFAULT_EVAL_SEED, suite=name)
        for name in ("dev", "promotion", "final")
    }
    assert len(set(suites.values())) == 3
    assert suites["dev"] == resolve_eval_seed(DEFAULT_EVAL_SEED, suite="dev")

    dev = run_benchmark("F", opponents=(), protocol="fast", eval_kind="mid_training")
    final = run_benchmark(
        "F", opponents=(), protocol="fast", eval_kind="final_benchmark"
    )
    assert dev.meta["protocol"]["seed_suite"] == "dev"
    assert final.meta["protocol"]["seed_suite"] == "final"
    assert dev.meta["protocol"]["seed"] != final.meta["protocol"]["seed"]


def test_report_round_trip_and_legacy_reader_preserve_accounting(tmp_path):
    legacy = {
        "schema_version": "1.0",
        "agent": "F",
        "matchups": [
            {
                "opponent": "R",
                "agent_code": "F",
                "games": 10,
                "wins": 7,
                "losses": 2,
                "draws": 1,
                "win_rate": 0.7,
                "wilson_low": 0.4,
                "wilson_high": 0.9,
                "avg_agent_vp": 8.0,
                "avg_opponent_vp": 6.0,
                "avg_vp_diff": 2.0,
                "avg_turns": 100.0,
            }
        ],
    }
    report = EvaluationReport.from_dict(legacy)
    assert report.matchups[0].requested_games == 10
    assert report.matchups[0].completed_games == 9
    assert report.matchups[0].truncated_games == 1

    report_path = tmp_path / "report.json"
    report.write_json(report_path)
    payload = json.loads(report_path.read_text())
    assert payload["schema_version"] == "1.1"
    assert "game_results" in payload["matchups"][0]
    assert EvaluationReport.read_json(report_path).matchups[0].games == 10


def test_paired_bootstrap_and_confidence_gate_are_deterministic():
    candidate = _matchup(
        game_results=[
            GameOutcome(i, 0, True, "completed", "win", seed=100 + i) for i in range(8)
        ]
    )
    baseline = _matchup(
        game_results=[
            GameOutcome(i, 0, True, "completed", "loss", seed=100 + i) for i in range(8)
        ]
    )

    first = compare_paired_matchups(candidate, baseline, resamples=300, seed=7)
    second = compare_paired_matchups(candidate, baseline, resamples=300, seed=7)
    assert first == second
    assert first.matched_games == 8
    assert first.mean_delta == first.confidence_low == first.confidence_high == 1.0
    assert first.passed_gate is True
    assert confidence_gate_passed(estimate=0.7, threshold=0.6, mode="point")
    assert not confidence_gate_passed(
        estimate=0.7,
        confidence_low=0.55,
        threshold=0.6,
        mode="lower_bound",
    )

    unrelated = _matchup(
        game_results=[GameOutcome(0, 1, False, "completed", "win", seed=999)]
    )
    with pytest.raises(ValueError, match="exact same schedule"):
        compare_paired_matchups(candidate, unrelated, resamples=10)

    duplicate = _matchup(
        game_results=[
            GameOutcome(0, 0, True, "completed", "win", seed=100),
            GameOutcome(1, 0, True, "completed", "win", seed=100),
        ]
    )
    with pytest.raises(ValueError, match="Duplicate paired schedule_id"):
        compare_paired_matchups(duplicate, duplicate, resamples=10)


def test_evaluation_restores_python_numpy_and_torch_rng(monkeypatch):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    def fake_play_batch(num_games, players, output_options, game_config, quiet):
        random_value = __import__("random")
        random_value.random()
        np.random.random()
        torch.rand(3)
        agent_color, opponent_color = (player.color for player in players)
        game = _game(
            winner=agent_color,
            seed=game_config.seed,
            game_id=f"seed-{game_config.seed}",
            turns=20,
        )
        return (
            {agent_color: 1, opponent_color: 0},
            {agent_color: [15], opponent_color: [5]},
            [game],
        )

    monkeypatch.setattr("catanatron.colonist_1v1_eval.play_batch", fake_play_batch)
    import random

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    from catanatron.colonist_1v1_eval import evaluate_matchup

    evaluate_matchup("F", "R", num_games=1, both_seats=False, seed=99)

    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)

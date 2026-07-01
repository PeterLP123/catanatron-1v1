"""Colonist 1v1 gym, eval, league, and learned-player integration tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium
import numpy as np
import pytest

from catanatron import Color
from catanatron.colonist_1v1_eval import (
    DEFAULT_BENCHMARK_GATES,
    EvalProtocol,
    EvaluationReport,
    MatchupResult,
    append_model_registry,
    get_eval_protocol,
    summarize_report,
    wilson_score_interval,
    evaluate_matchup,
)
from catanatron.gym.colonist_rewards import colonist_shaped_reward
from catanatron.gym.colonist_training import (
    CheckpointLeague,
    TrainingRunTracker,
    curriculum_from_name,
    make_mixed_opponent_factory,
    resolve_teacher_parquet_paths,
    warmstart_bc_into_maskable_ppo,
    write_dataset_metadata,
)
from catanatron.gym.tui_data import (
    build_bc_command,
    build_data_commands,
    build_eval_command,
    detect_warnings,
    load_registry,
    summarize_run,
    update_manifest,
)
from catanatron.gym.tui_jobs import JobRunner
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.gym.wrappers.self_play import SelfPlayEnv
from catanatron.players.weighted_random import WeightedRandomPlayer


def test_wilson_interval_bounds():
    lo, hi = wilson_score_interval(50, 100)
    assert 0.0 <= lo <= hi <= 1.0


def test_colonist_1v1_gym_reset_and_masks():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "colonist_1v1": True,
            "enemies": [WeightedRandomPlayer(Color.RED)],
            "reward_function": colonist_shaped_reward,
        },
    )
    obs, info = env.reset(seed=42)
    assert obs.shape[0] > 600
    assert len(info["valid_actions"]) > 0
    mask = env.unwrapped.action_masks()
    assert len(mask) == env.action_space.n
    assert any(mask)
    assert env.unwrapped.game.vps_to_win == 15
    env.close()


def test_colonist_shaped_reward_on_env_step():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "colonist_1v1": True,
            "enemies": [WeightedRandomPlayer(Color.RED)],
            "reward_function": colonist_shaped_reward,
        },
    )
    _, info = env.reset(seed=1)
    action = info["valid_actions"][0]
    _, reward, terminated, truncated, _ = env.step(action)
    assert not terminated or reward in (-1.0, 1.0)
    assert isinstance(reward, float)
    env.close()


def test_self_play_env_swaps_opponent():
    base = CatanatronEnv(
        config={
            "colonist_1v1": True,
            "enemies": [WeightedRandomPlayer(Color.RED)],
        }
    )
    opponent = WeightedRandomPlayer(Color.RED)
    wrapped = SelfPlayEnv(base, opponent=opponent)
    wrapped.reset(seed=0)
    assert wrapped.env.unwrapped.enemies[0] is opponent


def test_seat_randomization_disabled_by_default():
    """p0 always moves first unless randomize_seats is explicitly enabled."""
    env = CatanatronEnv(
        config={"colonist_1v1": True, "enemies": [WeightedRandomPlayer(Color.RED)]}
    )
    env.reset(seed=0)
    for _ in range(20):
        env.reset()
        assert env.players[0] is env.p0


def test_seat_randomization_reaches_both_seats():
    env = CatanatronEnv(
        config={
            "colonist_1v1": True,
            "enemies": [WeightedRandomPlayer(Color.RED)],
            "randomize_seats": True,
        }
    )
    env.reset(seed=0)
    seats = {env.players.index(env.p0)}
    for _ in range(40):
        env.reset()
        seats.add(env.players.index(env.p0))
    assert seats == {0, 1}


def test_seat_randomization_does_not_change_observation_or_action_shape():
    """p0's color (and hence observation/action encoding) never changes -- only
    turn order does -- so shapes must match regardless of which seat is drawn."""
    make = lambda: CatanatronEnv(
        config={"colonist_1v1": True, "enemies": [WeightedRandomPlayer(Color.RED)]}
    )
    seat_first = make()
    seat_second = make()
    obs_first, _ = seat_first.reset(seed=0)

    # Force the second env's agent to seat 1 (moves second) without enabling
    # randomize_seats, so reset() won't redraw and clobber the manual choice.
    seat_second._p0_seat_index = 1
    obs_second, _ = seat_second.reset(seed=0)
    assert seat_second.players.index(seat_second.p0) == 1

    assert obs_first.shape == obs_second.shape
    assert seat_first.observation_space.shape == seat_second.observation_space.shape
    assert seat_first.action_space.n == seat_second.action_space.n
    assert len(seat_first.action_masks()) == len(seat_second.action_masks())


def test_self_play_env_respects_randomized_seat():
    base = CatanatronEnv(
        config={
            "colonist_1v1": True,
            "enemies": [WeightedRandomPlayer(Color.RED)],
            "randomize_seats": True,
        }
    )
    opponent = WeightedRandomPlayer(Color.RED)
    wrapped = SelfPlayEnv(base, opponent=opponent)
    seats = set()
    for i in range(40):
        wrapped.reset(seed=i)
        u = wrapped.env.unwrapped
        assert u.enemies[0] is opponent
        seats.add(u.players.index(u.p0))
    assert seats == {0, 1}


def test_checkpoint_league_register_and_prune(tmp_path):
    league = CheckpointLeague(tmp_path, max_checkpoints=2)
    f1 = tmp_path / "a.zip"
    f2 = tmp_path / "b.zip"
    f3 = tmp_path / "c.zip"
    for f in (f1, f2, f3):
        f.write_bytes(b"zip")
    league.register(f1, label="a")
    league.register(f2, label="b")
    league.register(f3, label="c")
    paths = league.paths()
    assert len(paths) == 2
    assert not (league.league_dir / "a.zip").exists() or "a.zip" not in paths


def test_resolve_teacher_parquet_paths_uses_newest_when_meta_present(tmp_path):
    import json

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(5):
        p = data_dir / f"old_{i}.parquet"
        p.write_bytes(b"x")
    for i in range(3):
        p = data_dir / f"new_{i}.parquet"
        p.write_bytes(b"x")
    (data_dir / "dataset_meta.json").write_text(
        json.dumps({"num_games": 3, "teachers": "F,F"}),
        encoding="utf-8",
    )
    paths = resolve_teacher_parquet_paths(data_dir)
    assert len(paths) == 3
    assert all("new_" in p.name for p in paths)


def test_write_dataset_metadata(tmp_path):
    write_dataset_metadata(
        tmp_path,
        teachers="F,F",
        num_games=10,
        command="test",
    )
    meta = tmp_path / "dataset_meta.json"
    assert meta.exists()
    assert "F,F" in meta.read_text()


def test_evaluate_matchup_mocked():
    with patch("catanatron.colonist_1v1_eval.play_batch") as pb:
        pb.return_value = (
            {Color.BLUE: 7, Color.RED: 3},
            {Color.BLUE: [10] * 10, Color.RED: [8] * 10},
            [MagicMock(state=MagicMock(num_turns=50)) for _ in range(10)],
        )
        with patch("catanatron.colonist_1v1_eval.parse_cli_string") as pcs:
            pcs.return_value = [
                MagicMock(color=Color.BLUE),
                MagicMock(color=Color.RED),
            ]
            r = evaluate_matchup("F", "R", num_games=10, gate=0.5, both_seats=False)
    assert r.wins == 7
    assert r.win_rate == 0.7
    assert r.passed_gate is True


def test_default_gates_order():
    assert DEFAULT_BENCHMARK_GATES["R"] > DEFAULT_BENCHMARK_GATES["AB:2"]


def test_eval_protocol_and_registry_schema(tmp_path):
    proto = get_eval_protocol("fast", num_games=12)
    assert proto.opponents == ("R", "W", "VP", "F")
    assert proto.num_games == 12

    report = EvaluationReport(
        agent="L:model.zip",
        meta={"model": {"checkpoint_path": "model.zip"}, "protocol": {"name": "fast"}},
        matchups=[
            MatchupResult(
                "R", "L:model.zip", 10, 8, 2, 0, 0.8, 0.5, 0.9, 10, 4, 6, 100
            ),
            MatchupResult(
                "F", "L:model.zip", 10, 1, 9, 0, 0.1, 0.02, 0.4, 4, 10, -6, 80
            ),
        ],
    )
    report.summary = summarize_report(report.matchups)
    registry = tmp_path / "models_index.jsonl"
    row = append_model_registry(registry, report, report_path=tmp_path / "report.json")
    assert registry.exists()
    assert row["summary"]["weighted_score"] < 0.8
    assert row["win_rates"]["F"] == 0.1


def test_evaluate_cli_uses_protocol_defaults():
    from examples import colonist_1v1_evaluate

    report = EvaluationReport(agent="F", all_gates_passed=True)
    with patch.object(
        colonist_1v1_evaluate, "run_benchmark", return_value=report
    ) as run:
        assert colonist_1v1_evaluate.main(["--agent", "F", "--protocol", "fast"]) == 0

    kwargs = run.call_args.kwargs
    assert kwargs["opponents"] == ("R", "W", "VP", "F")
    assert kwargs["num_games"] == 50


def test_evaluate_cli_allows_game_count_override():
    from examples import colonist_1v1_evaluate

    report = EvaluationReport(agent="F", all_gates_passed=True)
    with patch.object(
        colonist_1v1_evaluate, "run_benchmark", return_value=report
    ) as run:
        assert (
            colonist_1v1_evaluate.main(
                ["--agent", "F", "--protocol", "milestone", "--num-games", "7"]
            )
            == 0
        )

    kwargs = run.call_args.kwargs
    assert kwargs["opponents"] == ("R", "W", "VP", "F", "G:25")
    assert kwargs["num_games"] == 7


def test_curriculum_schedule_changes_weights():
    schedule = curriculum_from_name("strong")
    early = schedule.stage_for(0)
    late = schedule.stage_for(1_000_000)
    assert early.teacher_weight > early.league_weight
    assert late.league_weight > early.league_weight


def test_mixed_opponent_factory_uses_live_league_fallback():
    league = MagicMock()
    league.sample_path.return_value = None
    factory = make_mixed_opponent_factory(
        league=league,
        league_weight=1.0,
        teacher_weight=0.0,
        baseline_weight=0.0,
        baseline_code="W",
        rng=np.random.default_rng(0),
    )
    player = factory()
    assert player.color == Color.RED
    league.sample_path.assert_called()


def test_training_run_tracker_writes_manifest_and_events(tmp_path):
    tracker = TrainingRunTracker(tmp_path, run_id="test-run", preset="smoke")
    tracker.phase("ppo_training")
    tracker.event("ppo_progress", timesteps=123)
    assert (tmp_path / "run_manifest.json").exists()
    assert "ppo_progress" in (tmp_path / "training_events.jsonl").read_text()


def test_tui_dashboard_loads_empty_run(tmp_path):
    pytest.importorskip("rich")
    from examples.colonist_1v1_tui import build_dashboard

    panel = build_dashboard(tmp_path)
    assert panel is not None


def test_tui_run_summary_registry_and_warnings(tmp_path):
    update_manifest(
        tmp_path,
        run_id="run-a",
        phase="ppo_training",
        training={"timesteps": 1000},
    )
    (tmp_path / "training_events.jsonl").write_text(
        '{"type":"ppo_progress","timesteps":500}\n',
        encoding="utf-8",
    )
    (tmp_path / "models_index.jsonl").write_text(
        '{"checkpoint_label":"a","summary":{"weighted_score":0.1},"win_rates":{"F":0.0}}\n'
        '{"checkpoint_label":"b","summary":{"weighted_score":0.7},"win_rates":{"F":0.5}}\n',
        encoding="utf-8",
    )
    summary = summarize_run(tmp_path)
    rows = load_registry(tmp_path)
    assert summary.progress_ratio == 0.5
    assert rows[0]["checkpoint_label"] == "b"
    assert not detect_warnings(tmp_path, {"phase": "done"}, [], rows)


def test_tui_command_builders():
    data_cmds = build_data_commands(
        python="python",
        teacher_specs=["F,F", "G:25,F"],
        num_games=3,
        data_root=Path("data/run"),
    )
    assert data_cmds[0][-1] == "data/run/F_F"
    bc_cmd = build_bc_command(
        python="python",
        data_dirs=[Path("data/run/F_F"), Path("data/run/G_25_F")],
        epochs=2,
        run_dir=Path("runs/x"),
    )
    assert "--data-dir" in bc_cmd
    assert "runs/x/bc.pt" in bc_cmd
    eval_cmd = build_eval_command(
        python="python",
        run_dir=Path("runs/x"),
        agent="L:runs/x/model.zip",
        protocol="fast",
        num_games=20,
        label="model",
    )
    assert eval_cmd[1] == "examples/colonist_1v1_evaluate.py"
    assert "--benchmark" in eval_cmd
    assert "--report" in eval_cmd


def test_job_runner_streams_and_records_status(tmp_path):
    import sys
    import time

    lines = []
    runner = JobRunner(tmp_path, on_log=lines.append)
    job = runner.start(
        "short",
        [sys.executable, "-c", "print('hello from job')"],
    )
    deadline = time.time() + 5
    while job.status in {"pending", "running"} and time.time() < deadline:
        time.sleep(0.05)
    assert job.status == "succeeded"
    assert job.exit_code == 0
    assert "hello from job" in "\n".join(lines)
    assert "job_finished" in (tmp_path / "training_events.jsonl").read_text()


def test_textual_app_smoke(tmp_path):
    pytest.importorskip("textual")
    import asyncio
    from examples.colonist_1v1_tui import make_textual_app

    async def run_app():
        app = make_textual_app(tmp_path, tmp_path / "run", 0.1)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("4")
            await pilot.pause(0.1)

    asyncio.run(run_app())


def test_warmstart_bc_into_ppo_policy():
    pytest.importorskip("torch")
    from torch import nn

    obs_dim, n_actions, hidden = 32, 20, 64
    bc = nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )
    state = bc.state_dict()

    policy = MagicMock()
    policy.mlp_extractor.policy_net = nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )
    policy.action_net = nn.Linear(hidden, n_actions)

    n = warmstart_bc_into_maskable_ppo(policy, state)
    assert n >= 4


def test_colonist1v1_player_legal_action():
    pytest.importorskip("torch")
    from catanatron.gym.colonist_training import build_mlp_layers
    from catanatron.players.learned import Colonist1v1Player
    from catanatron.colonist_1v1 import create_colonist_1v1_game
    from catanatron.models.player import RandomPlayer

    players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
    game = create_colonist_1v1_game(players, seed=0)
    while game.winning_color() is None and game.state.current_color() != Color.BLUE:
        game.play_tick()

    if game.winning_color() is not None:
        pytest.skip("Game ended before BLUE turn")

    obs_dim = 614
    n_actions = 332
    net = build_mlp_layers(obs_dim, n_actions, (64, 64))
    player = Colonist1v1Player(
        Color.BLUE,
        torch_policy=net,
        map_type="BASE",
    )
    playable = game.playable_actions
    action = player.decide(game, playable)
    assert action in playable

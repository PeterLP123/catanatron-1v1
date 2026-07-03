import json
import multiprocessing
import sys
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from catanatron import Color
from catanatron.cli.cli_players import parse_cli_string
from catanatron.cli.play import GameConfigOptions, OutputOptions, play_batch
from catanatron.colonist_1v1_eval import (
    DEFAULT_EVAL_SEED,
    build_eval_meta,
    evaluate_matchup,
    get_eval_protocol,
)
from catanatron.gym.colonist_training import (
    CheckpointLeague,
    CurriculumSchedule,
    CurriculumStage,
    load_teacher_parquet,
    make_mixed_opponent_factory,
)
from catanatron.gym.tui_data import summarize_run
from examples import colonist_1v1_generate_data
from examples.colonist_1v1_bc import DEFAULT_BC_CHECKPOINT_PATH, build_parser
from examples.colonist_1v1_train import make_colonist_env


def _game_config(seed=0):
    return GameConfigOptions.from_cli(
        discard_limit=7,
        vps_to_win=10,
        map_type="BASE",
        number_placement="official_spiral",
        friendly_robber=False,
        colonist_1v1=True,
        seed=seed,
    )


def test_bc_default_checkpoint_stays_out_of_repo_root():
    args = build_parser().parse_args(["--data-dir", "data/c1"])
    assert args.out == DEFAULT_BC_CHECKPOINT_PATH
    assert args.out == Path("runs/colonist_bc_policy.pt")


def _play_summary(seed):
    wins, vps, games = play_batch(
        8,
        parse_cli_string("W,R"),
        OutputOptions(),
        _game_config(seed),
        quiet=True,
    )
    return wins, vps, [game.state.num_turns for game in games]


def test_seeded_play_batch_is_reproducible():
    assert _play_summary(1234) == _play_summary(1234)


def test_evaluation_reuses_seed_for_both_seats(monkeypatch):
    seen = []
    orders = []

    def fake_play_batch(num_games, players, output_options, game_config, quiet):
        seen.append(game_config.seed)
        orders.append([type(player).__name__ for player in players])
        assert game_config.shuffle_players is False
        colors = [player.color for player in players]
        games = [type("G", (), {"state": type("S", (), {"num_turns": 10})()})()]
        return {colors[0]: 1, colors[1]: 0}, {colors[0]: [15], colors[1]: [4]}, games

    monkeypatch.setattr("catanatron.colonist_1v1_eval.play_batch", fake_play_batch)
    evaluate_matchup("W", "R", num_games=2, both_seats=True)
    assert seen == [DEFAULT_EVAL_SEED, DEFAULT_EVAL_SEED]
    assert orders == [
        ["WeightedRandomPlayer", "RandomPlayer"],
        ["RandomPlayer", "WeightedRandomPlayer"],
    ]


def test_evaluation_metadata_records_fixed_seed():
    protocol = get_eval_protocol("fast", num_games=12)
    meta = build_eval_meta(agent_spec="F", protocol=protocol)
    assert meta["protocol"]["seed"] == DEFAULT_EVAL_SEED
    assert meta["protocol"]["num_games_per_matchup"] == 12


def _write_dataset(path: Path, *, shard_games: int, games: int = 4):
    path.mkdir()
    play_batch(
        games,
        parse_cli_string("W,W"),
        OutputOptions(
            output=str(path),
            output_format="parquet",
            include_board_tensor=False,
            score_candidates=True,
            parquet_shard_games=shard_games,
        ),
        _game_config(10),
        quiet=True,
    )


def test_sharded_dataset_matches_legacy_and_is_compact(tmp_path):
    legacy = tmp_path / "legacy"
    sharded = tmp_path / "sharded"
    _write_dataset(legacy, shard_games=1)
    _write_dataset(sharded, shard_games=4)

    legacy_df = load_teacher_parquet(legacy, progress=False).sort_values(
        ["GAME_ID", "SEAT", "ACTION"]
    )
    sharded_df = load_teacher_parquet(sharded, progress=False).sort_values(
        ["GAME_ID", "SEAT", "ACTION"]
    )
    pd.testing.assert_frame_equal(
        legacy_df.reset_index(drop=True), sharded_df.reset_index(drop=True)
    )
    legacy_size = sum(path.stat().st_size for path in legacy.glob("*.parquet"))
    sharded_size = sum(path.stat().st_size for path in sharded.glob("*.parquet"))
    # Even this tiny four-game fixture removes most per-file schema overhead;
    # the 20-game acceptance probe exercises the stricter 80% target.
    assert sharded_size < legacy_size * 0.50
    assert {p.name for p in sharded.glob("*.parquet")} == {"shard-00000.parquet"}


def test_generator_resumes_without_duplicate_games(tmp_path):
    output = tmp_path / "generated"
    args = [
        "--num",
        "4",
        "--teachers",
        "W,W",
        "--seed",
        "50",
        "--shard-games",
        "2",
        "--output",
        str(output),
    ]
    assert colonist_1v1_generate_data.main(args) == 0
    shards = sorted(output.glob("*.parquet"))
    assert len(shards) == 2

    # Simulate interruption after the first atomic shard was committed.
    shards[1].unlink()
    first = pd.read_parquet(shards[0])
    meta_path = output / "dataset_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(
        status="in_progress",
        completed_games=2,
        next_seed=52,
        rows=len(first),
        parquet_files=1,
    )
    meta_path.write_text(json.dumps(meta))

    assert colonist_1v1_generate_data.main([*args, "--resume"]) == 0
    frames = [
        pd.read_parquet(path, columns=["GAME_ID"]) for path in output.glob("*.parquet")
    ]
    game_ids = pd.concat(frames)["GAME_ID"].unique().tolist()
    assert sorted(game_ids) == ["seed-50", "seed-51", "seed-52", "seed-53"]
    final_meta = json.loads(meta_path.read_text())
    assert final_meta["status"] == "complete"
    assert final_meta["completed_games"] == 4


def test_generator_rejects_resume_configuration_mismatch(tmp_path):
    output = tmp_path / "generated"
    output.mkdir()
    (output / "dataset_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "teachers": "W,W",
                "num_games": 4,
                "requested_games": 4,
                "seed": 1,
                "shard_games": 2,
                "choices_only": False,
                "score_candidates": False,
                "include_board_tensor": False,
                "colonist_1v1": True,
                "status": "in_progress",
            }
        )
    )
    with pytest.raises(SystemExit):
        colonist_1v1_generate_data.main(
            [
                "--num",
                "4",
                "--teachers",
                "W,W",
                "--seed",
                "2",
                "--shard-games",
                "2",
                "--output",
                str(output),
                "--resume",
            ]
        )


def test_generator_command_uses_active_interpreter(tmp_path, monkeypatch):
    captured = []

    def fake_call(command):
        captured.append(command)
        meta_path = Path(command[command.index("--dataset-meta") + 1])
        meta = json.loads(meta_path.read_text())
        meta.update(completed_games=1, rows=1, parquet_files=1, next_seed=1)
        meta_path.write_text(json.dumps(meta))
        return 0

    monkeypatch.setattr(colonist_1v1_generate_data.subprocess, "call", fake_call)
    assert (
        colonist_1v1_generate_data.main(
            ["--num", "1", "--teachers", "W,W", "--output", str(tmp_path / "d")]
        )
        == 0
    )
    assert captured[0][:3] == [sys.executable, "-m", "catanatron.cli.play"]


def test_checkpoint_league_reloads_index_from_disk(tmp_path):
    writer = CheckpointLeague(tmp_path, max_checkpoints=2)
    reader = CheckpointLeague(tmp_path, max_checkpoints=2)
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"model")
    writer.register(checkpoint, label="new")
    assert any(path.endswith("new.zip") for path in reader.paths())


def _curriculum_worker(step_state, run_dir, queue):
    league = CheckpointLeague(run_dir)
    schedule = CurriculumSchedule(
        stages=(
            CurriculumStage(0, 0.0, 1.0, 0.0, teacher_codes=("R",)),
            CurriculumStage(10, 0.0, 1.0, 0.0, teacher_codes=("W",)),
        )
    )
    factory = make_mixed_opponent_factory(
        league=league,
        curriculum=schedule,
        step_getter=lambda: step_state["timesteps"],
        rng=np.random.default_rng(0),
    )
    queue.put(type(factory()).__name__)
    step_state["timesteps"] = 10
    queue.put(type(factory()).__name__)


def test_shared_curriculum_state_is_visible_in_worker(tmp_path):
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        step_state = manager.dict(timesteps=0)
        queue = context.Queue()
        process = context.Process(
            target=_curriculum_worker, args=(step_state, tmp_path, queue)
        )
        process.start()
        process.join(timeout=15)
        assert process.exitcode == 0
        assert queue.get(timeout=2) != queue.get(timeout=2)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="fork unavailable"
)
def test_dummy_and_subproc_env_shapes_and_masks_match():
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = [partial(make_colonist_env, seed=90 + i) for i in range(2)]
    dummy = DummyVecEnv(factories)
    subproc = SubprocVecEnv(factories, start_method="fork")
    try:
        dummy.seed(90)
        subproc.seed(90)
        dummy_obs = dummy.reset()
        subproc_obs = subproc.reset()
        assert dummy_obs.shape == subproc_obs.shape
        assert get_action_masks(dummy).shape == get_action_masks(subproc).shape
        assert np.array_equal(get_action_masks(dummy), get_action_masks(subproc))
    finally:
        dummy.close()
        subproc.close()


def test_run_summary_rate_eta_and_staleness(tmp_path):
    now = datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc)
    manifest = {
        "run_id": "rate-test",
        "phase": "ppo_training",
        "created_at": (now - timedelta(hours=1)).isoformat(),
        "updated_at": now.isoformat(),
        "training": {
            "timesteps": 20_000,
            "seed": 7,
            "vec_env": "subproc",
            "n_envs": 4,
        },
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    events = [
        {
            "type": "ppo_progress",
            "timesteps": 5_000,
            "time": (now - timedelta(minutes=30)).isoformat(),
        },
        {
            "type": "ppo_progress",
            "timesteps": 10_000,
            "time": (now - timedelta(minutes=20)).isoformat(),
        },
    ]
    (tmp_path / "training_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    summary = summarize_run(tmp_path, now=now)
    assert summary.steps_per_second == pytest.approx(5000 / 600)
    assert summary.eta_seconds == pytest.approx(1200)
    assert summary.elapsed_seconds == pytest.approx(3600)
    assert summary.stale_seconds == pytest.approx(1200)
    assert any("No training progress" in warning for warning in summary.warnings)

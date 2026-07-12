"""Focused integrity tests for the student-visited distillation replay path."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from catanatron.game import Game
from catanatron.gym import accumulators as accumulator_module
from catanatron.gym.accumulators import (
    ParquetDataAccumulator,
    ReinforcementLearningAccumulator,
)
from catanatron.gym.distillation import (
    AgentIdentity,
    DistillationConfig,
    DistillationDatasetWriter,
    DistillationDecisionRecorder,
    derive_seed,
    isolated_random_seed,
    run_distillation_iteration,
    validate_teacher_spec,
    verify_distillation_dataset,
)
from catanatron.gym.model_schema import build_model_schema
from catanatron.models.player import Color, RandomPlayer, SimplePlayer


def _metadata(schema_hash: str = "schema-a") -> dict:
    return {
        "schema": {
            "schema_hash": schema_hash,
            "feature_hash": "features",
            "action_hash": "actions",
            "rules_hash": "rules",
        }
    }


def test_seed_derivation_is_stable_and_iteration_namespaced():
    first = derive_seed(
        "game", base_seed=123, iteration=0, game_index=4, decision_index=0
    )
    assert first == derive_seed(
        "game", base_seed=123, iteration=0, game_index=4, decision_index=0
    )
    assert first != derive_seed(
        "game", base_seed=123, iteration=1, game_index=4, decision_index=0
    )


def test_isolated_teacher_rng_does_not_perturb_behavior_stream():
    random.seed(91)
    np.random.seed(91)
    expected_python = random.random()
    expected_numpy = np.random.random()

    random.seed(91)
    np.random.seed(91)
    with isolated_random_seed(808):
        _ = [random.random() for _ in range(10)]
        _ = np.random.random(10)

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy


@pytest.mark.parametrize("spec", ["F", "F:C", "M", "M:25", "M:25:False:base_fn"])
def test_teacher_validation_accepts_f_and_fixed_simulation_mcts(spec):
    validate_teacher_spec(spec)


@pytest.mark.parametrize("spec", ["W", "AB:2", "M:0", "M:1:False:base_fn:25"])
def test_teacher_validation_rejects_untrusted_or_nondeterministic_specs(spec):
    with pytest.raises(ValueError):
        validate_teacher_spec(spec)


def test_recorder_keeps_behavior_rng_and_writes_aligned_teacher_target():
    behavior = RandomPlayer(Color.BLUE)
    teacher = SimplePlayer(Color.BLUE)
    opponent = SimplePlayer(Color.RED)
    game = Game(
        [behavior, opponent],
        seed=17,
        colonist_1v1=True,
        shuffle_players=False,
    )
    config = DistillationConfig(
        iteration=2,
        games=1,
        base_seed=444,
        student_spec="W",
        teacher_spec="F",
        opponent_spec="W",
        score_f_candidates=False,
    )
    schema = build_model_schema(player_colors=(Color.BLUE, Color.RED))
    recorder = DistillationDecisionRecorder(
        config=config,
        model_schema=schema,
        student=AgentIdentity.from_spec("W"),
        teacher=AgentIdentity.from_spec("F"),
    )

    rng_state = random.getstate()
    expected_behavior = random.choice(game.playable_actions)
    random.setstate(rng_state)
    action, row = recorder.decide_and_label(
        behavior=behavior,
        teacher=teacher,
        game=game,
        playable_actions=game.playable_actions,
        game_index=0,
        game_seed=17,
        decision_index=0,
        seat=0,
    )

    assert action == expected_behavior
    assert row is not None
    assert row["NUM_LEGAL"] == len(row["LEGAL_ACTIONS"])
    assert row["NUM_LEGAL"] == len(row["TEACHER_DISTRIBUTION"])
    assert row["NUM_LEGAL"] == len(row["CANDIDATE_SCORES"])
    assert sum(row["TEACHER_DISTRIBUTION"]) == 1.0
    assert row["TEACHER_ACTION"] == row["LEGAL_ACTIONS"][0]
    assert not row["CANDIDATE_SCORES_AVAILABLE"]
    assert all(np.isnan(value) for value in row["CANDIDATE_SCORES"])
    assert row["STUDENT_HASH"] == AgentIdentity.from_spec("W").agent_hash
    assert row["SCHEMA_HASH"] == schema["schema_hash"]
    assert any(name.startswith("F_") for name in row)


def test_rl_accumulator_threads_feature_profile_into_samples(monkeypatch):
    game = Game(
        [SimplePlayer(Color.BLUE), SimplePlayer(Color.RED)],
        seed=7,
        colonist_1v1=True,
        shuffle_players=False,
    )
    seen = []

    def fake_sample(_game, _color, feature_profile="raw"):
        seen.append(feature_profile)
        return {"ONLY": 1.0}

    monkeypatch.setattr(accumulator_module, "create_sample", fake_sample)
    accumulator = ReinforcementLearningAccumulator(
        (Color.BLUE, Color.RED),
        include_board_tensor=False,
        feature_profile="public_derived",
    )
    monkeypatch.setattr(
        accumulator, "_legal_and_candidates", lambda _game, _color: ([0], [])
    )
    accumulator.before(game)
    accumulator.step(game, game.playable_actions[0])
    assert seen == ["public_derived"]


def test_parquet_progress_records_and_guards_feature_profile(tmp_path):
    metadata = tmp_path / "dataset_meta.json"
    metadata.write_text("{}\n", encoding="utf-8")
    accumulator = ParquetDataAccumulator(
        (Color.BLUE, Color.RED),
        "BASE",
        tmp_path,
        include_board_tensor=False,
        dataset_meta=metadata,
        feature_profile="public_derived",
    )
    accumulator._update_progress(games=1, rows=2, files=1)
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved["feature_profile"] == "public_derived"

    incompatible = ParquetDataAccumulator(
        (Color.BLUE, Color.RED),
        "BASE",
        tmp_path,
        include_board_tensor=False,
        dataset_meta=metadata,
        feature_profile="raw",
    )
    with pytest.raises(ValueError, match="feature profile changed"):
        incompatible._update_progress(games=1, rows=2, files=1)


def test_writer_aggregates_immutable_iterations_and_verifies_hashes(tmp_path):
    pytest.importorskip("pyarrow")
    root = tmp_path / "replay"
    writer0 = DistillationDatasetWriter(
        root, iteration=0, shard_games=1, metadata=_metadata()
    )
    writer0.add_game(
        [{"ITERATION": 0, "LEGAL_ACTIONS": [1, 2], "VALUE": 3.0}],
        game_index=0,
        game_seed=10,
        student_color=Color.BLUE,
        winner=Color.BLUE,
        truncated=False,
    )
    writer0.finalize()

    writer1 = DistillationDatasetWriter(
        root, iteration=1, shard_games=1, metadata=_metadata()
    )
    writer1.add_game(
        [{"ITERATION": 1, "LEGAL_ACTIONS": [2, 3], "VALUE": 4.0}],
        game_index=0,
        game_seed=11,
        student_color=Color.RED,
        winner=None,
        truncated=True,
    )
    writer1.finalize()

    aggregate = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert aggregate["rows"] == 2
    assert aggregate["games"] == 2
    assert [item["iteration"] for item in aggregate["iterations"]] == [0, 1]
    assert verify_distillation_dataset(root) == []

    with pytest.raises(FileExistsError, match="immutable"):
        DistillationDatasetWriter(
            root, iteration=1, shard_games=1, metadata=_metadata()
        )
    with pytest.raises(ValueError, match="different model schemas"):
        DistillationDatasetWriter(
            root, iteration=2, shard_games=1, metadata=_metadata("schema-b")
        )

    shard = root / aggregate["iterations"][0]["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"tampered")
    assert any(
        "shard hash mismatch" in item for item in verify_distillation_dataset(root)
    )


def test_distillation_cli_dry_run_is_non_mutating(tmp_path):
    output = tmp_path / "dry-run"
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.fspath(repo / "catanatron")
    result = subprocess.run(
        [
            sys.executable,
            os.fspath(repo / "examples" / "colonist_1v1_distill.py"),
            "--dry-run",
            "--games",
            "2",
            "--output",
            os.fspath(output),
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert len(plan["game_seeds"]) == 2
    assert plan["metadata"]["schema"]["schema_hash"]
    assert not output.exists()


def test_distillation_game_seed_controls_map_and_trajectory(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    config = DistillationConfig(
        iteration=0,
        games=1,
        base_seed=71,
        student_spec="W",
        teacher_spec="F",
        opponent_spec="W",
        score_f_candidates=False,
        shard_games=1,
    )

    run_distillation_iteration(config, output=tmp_path / "first")
    run_distillation_iteration(config, output=tmp_path / "second")
    columns = [
        "STATE_HASH",
        "BEHAVIOR_ACTION",
        "TEACHER_ACTION",
        "GAME_SEED",
        "DECISION_SEED",
    ]
    first = pd.read_parquet(
        next((tmp_path / "first").glob("iteration-*/*.parquet")), columns=columns
    )
    second = pd.read_parquet(
        next((tmp_path / "second").glob("iteration-*/*.parquet")), columns=columns
    )

    pd.testing.assert_frame_equal(first, second)

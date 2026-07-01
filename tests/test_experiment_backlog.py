import json

import pytest

from catanatron.gym.experiment_backlog import (
    EXPERIMENTS,
    backlog_statuses,
    experiments_by_id,
    launch_command,
    launch_environment,
    load_final_metrics,
    validate_backlog,
)


def test_backlog_is_unique_acyclic_and_dependency_complete():
    validate_backlog()
    assert len(EXPERIMENTS) == len({experiment.id for experiment in EXPERIMENTS})


def test_backlog_status_respects_dependencies(tmp_path):
    statuses = backlog_statuses(tmp_path)
    assert statuses["00-gpu-smoke"] == "ready"
    assert statuses["10-balanced-actual-s101"] == "blocked"

    smoke = tmp_path / "00-gpu-smoke"
    smoke.mkdir()
    (smoke / "run_manifest.json").write_text(json.dumps({"phase": "done"}))
    statuses = backlog_statuses(tmp_path)
    assert statuses["00-gpu-smoke"] == "done"
    assert statuses["10-balanced-actual-s101"] == "ready"


def test_launch_command_contains_reproducible_profile():
    experiment = experiments_by_id()["11-balanced-visible-s101"]
    command = launch_command(experiment)
    assert "TRAIN_PRESET=standard" in command
    assert "SEED=101" in command
    assert "VISIBLE_VP_REWARD=1" in command
    assert "RUN_NAME=11-balanced-visible-s101" in command


def test_checkpoint_dependent_experiment_requires_input():
    experiment = experiments_by_id()["30-strong-promoted"]
    with pytest.raises(ValueError, match="RESUME_CHECKPOINT"):
        launch_environment(experiment, inherited={})
    env = launch_environment(
        experiment,
        supplied={"RESUME_CHECKPOINT": "/tmp/model.zip"},
        inherited={},
    )
    assert env["RESUME_CHECKPOINT"] == "/tmp/model.zip"


def test_load_final_metrics_extracts_comparison_fields(tmp_path):
    report = {
        "summary": {"weighted_score": 0.42},
        "all_gates_passed": False,
        "matchups": [
            {
                "opponent": "F",
                "win_rate": 0.1,
                "win_rate_seat0": 0.05,
                "win_rate_seat1": 0.15,
            }
        ],
    }
    (tmp_path / "final_benchmark.json").write_text(json.dumps(report))
    metrics = load_final_metrics(tmp_path)
    assert metrics["weighted_score"] == 0.42
    assert metrics["rates"]["F"] == 0.1
    assert metrics["max_seat_gap"] == pytest.approx(0.1)

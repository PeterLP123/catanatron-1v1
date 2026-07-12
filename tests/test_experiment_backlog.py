import json

import pytest

from catanatron.gym.experiment_backlog import (
    EXPERIMENTS,
    ExperimentOutcome,
    backlog_statuses,
    check_backlog_markdown,
    evaluate_experiment,
    experiments_by_id,
    launch_command,
    launch_environment,
    load_final_metrics,
    paired_outcome_delta,
    render_backlog_markdown,
    reports_comparable,
    validate_backlog,
    write_launch_evidence,
)


def _game_results(outcomes, opponent):
    return [
        {
            "game_index": index,
            "seed": 9_000 + index,
            "schedule_id": f"{opponent}:seat-{index % 2}:game-{9_000 + index}",
            "agent_seat": index % 2,
            "outcome": outcome,
        }
        for index, outcome in enumerate(outcomes)
    ]


def _report(*, outcomes=("win", "loss"), rates=None, commit="abc123"):
    rates = rates or {"R": 0.95, "W": 0.80, "VP": 0.70, "F": 0.15}
    requested = len(outcomes)
    matchups = []
    for opponent, win_rate in rates.items():
        matchups.append(
            {
                "opponent": opponent,
                "requested_games": requested,
                "games": requested,
                "win_rate": win_rate,
                "avg_vp_diff": 1.0 if opponent == "F" else 2.0,
                "win_rate_seat0": win_rate,
                "win_rate_seat1": win_rate,
                "game_results": _game_results(outcomes, opponent),
            }
        )
    return {
        "schema_version": "1.1",
        "colonist_1v1": True,
        "all_gates_passed": False,
        "meta": {
            "both_seats": True,
            "git_commit": commit,
            "protocol": {
                "name": "promotion",
                "opponents": list(rates),
                "num_games_per_matchup": requested,
                "seed": 123,
                "seed_suite": "final",
                "gate_mode": "point",
            },
        },
        "summary": {"weighted_score": 0.42},
        "matchups": matchups,
    }


def _complete_run(runs_root, experiment_id, report):
    run_dir = runs_root / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"phase": "done"}))
    (run_dir / "final_benchmark.json").write_text(json.dumps(report))
    return run_dir


def _complete_smoke(runs_root):
    run_dir = runs_root / "00-gpu-smoke"
    (run_dir / "eval_reports").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "phase": "done",
                "training": {"timesteps": 20_000},
                "provenance": {"hardware": {"cuda_available": True}},
            }
        )
    )
    (run_dir / "colonist_maskable_ppo.zip").write_bytes(b"model")
    (run_dir / "eval_reports" / "dev_step_20000.json").write_text(
        json.dumps({"meta": {"eval_kind": "dev"}})
    )


def _complete_search_sweep(runs_root):
    run_dir = runs_root / "05-mcts-strength-sweep"
    run_dir.mkdir(parents=True)
    results = []
    for budget in (10.0, 25.0, 50.0, 100.0):
        for opponent in ("F", "AB:2"):
            for seed in (20_260_711, 20_260_712, 20_260_713):
                results.append(
                    {
                        "budget_ms": budget,
                        "opponent": opponent,
                        "seed": seed,
                        "profile": {"p95_latency_ms": budget + 1.0},
                        "matchup": {
                            "opponent": opponent,
                            "requested_games": 2,
                            "games": 2,
                            "game_results": _game_results(("win", "loss"), opponent),
                        },
                    }
                )
    (run_dir / "mcts_strength_sweep.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "mcts_strength_sweep",
                "meta": {
                    "git_commit": "abc123",
                    "budgets_ms": [10.0, 25.0, 50.0, 100.0],
                    "opponents": ["F", "AB:2"],
                    "seeds": [20_260_711, 20_260_712, 20_260_713],
                    "both_seats": True,
                    "num_games_per_cell": 2,
                    "profile_only": False,
                },
                "results": results,
            }
        )
    )


def test_backlog_is_unique_acyclic_and_dependency_complete():
    validate_backlog()
    assert len(EXPERIMENTS) == len({experiment.id for experiment in EXPERIMENTS})
    ids = [experiment.id for experiment in EXPERIMENTS]
    assert ids.index("05-mcts-strength-sweep") < ids.index("10-balanced-actual-s101")


def test_backlog_requires_accepted_evidence_not_phase_done(tmp_path):
    statuses = backlog_statuses(tmp_path)
    assert statuses["00-gpu-smoke"] == "ready"
    assert statuses["05-mcts-strength-sweep"] == "ready"
    assert statuses["10-balanced-actual-s101"] == "blocked"

    smoke = tmp_path / "00-gpu-smoke"
    smoke.mkdir()
    (smoke / "run_manifest.json").write_text(
        json.dumps(
            {
                "phase": "done",
                "training": {"timesteps": 20_000},
                "provenance": {"hardware": {"cuda_available": True}},
            }
        )
    )
    statuses = backlog_statuses(tmp_path)
    assert statuses["00-gpu-smoke"] == "rejected"
    assert statuses["10-balanced-actual-s101"] == "blocked"

    (smoke / "colonist_maskable_ppo.zip").write_bytes(b"model")
    (smoke / "eval_reports").mkdir()
    (smoke / "eval_reports" / "dev_step_20000.json").write_text(
        json.dumps({"meta": {"eval_kind": "dev"}})
    )
    _complete_search_sweep(tmp_path)
    statuses = backlog_statuses(tmp_path)
    assert statuses["00-gpu-smoke"] == "accepted"
    assert statuses["05-mcts-strength-sweep"] == "accepted"
    assert statuses["10-balanced-actual-s101"] == "ready"


def test_reward_reports_require_comparable_paired_games():
    control = _report(outcomes=("loss", "loss"))
    treatment = _report(outcomes=("win", "win"))
    assert reports_comparable(treatment, control)
    assert paired_outcome_delta(treatment, control) == 1.0

    treatment["meta"]["protocol"]["seed"] = 999
    assert not reports_comparable(treatment, control)


def test_reward_replication_branches_are_mutually_exclusive(tmp_path):
    _complete_smoke(tmp_path)
    _complete_search_sweep(tmp_path)
    _complete_run(
        tmp_path, "10-balanced-actual-s101", _report(outcomes=("loss", "loss"))
    )
    visible = _complete_run(
        tmp_path, "11-balanced-visible-s101", _report(outcomes=("win", "win"))
    )

    decision = evaluate_experiment(
        experiments_by_id()["11-balanced-visible-s101"], tmp_path
    )
    assert decision.outcome is ExperimentOutcome.ACCEPTED
    statuses = backlog_statuses(tmp_path)
    assert statuses["12-balanced-actual-s202"] == "skipped"
    assert statuses["13-balanced-visible-s202"] == "ready"

    (visible / "final_benchmark.json").write_text(
        json.dumps(_report(outcomes=("loss", "loss")))
    )
    decision = evaluate_experiment(
        experiments_by_id()["11-balanced-visible-s101"], tmp_path
    )
    assert decision.outcome is ExperimentOutcome.INCONCLUSIVE
    statuses = backlog_statuses(tmp_path)
    assert statuses["12-balanced-actual-s202"] == "blocked"
    assert statuses["13-balanced-visible-s202"] == "blocked"

    control = tmp_path / "10-balanced-actual-s101" / "final_benchmark.json"
    control.write_text(json.dumps(_report(outcomes=("win", "win"))))
    decision = evaluate_experiment(
        experiments_by_id()["11-balanced-visible-s101"], tmp_path
    )
    assert decision.outcome is ExperimentOutcome.REJECTED
    statuses = backlog_statuses(tmp_path)
    assert statuses["12-balanced-actual-s202"] == "ready"
    assert statuses["13-balanced-visible-s202"] == "skipped"


def test_launch_command_contains_reproducible_profile():
    experiment = experiments_by_id()["11-balanced-visible-s101"]
    command = launch_command(experiment)
    assert "TRAIN_PRESET=standard" in command
    assert "SEED=101" in command
    assert "VISIBLE_VP_REWARD=1" in command
    assert "RUN_NAME=11-balanced-visible-s101" in command

    search = launch_command(experiments_by_id()["05-mcts-strength-sweep"])
    assert "--budgets 10,25,50,100" in search
    assert "--opponents F,AB:2" in search


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


def test_bc_launch_requires_measured_regret_improvement(tmp_path):
    checkpoint = tmp_path / "bc.pt"
    baseline_checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"model")
    baseline_checkpoint.write_bytes(b"baseline")
    experiment = experiments_by_id()["20-hard-bc-actual-s101"]
    metadata = checkpoint.with_suffix(".meta.json")
    baseline_metadata = baseline_checkpoint.with_suffix(".meta.json")
    metadata.write_text(json.dumps({"val_metrics": {"mean_regret": 0.4}}))
    baseline_metadata.write_text(json.dumps({"val_metrics": {"mean_regret": 0.3}}))
    with pytest.raises(ValueError, match="regret gate failed"):
        launch_environment(
            experiment,
            supplied={
                "BC_CHECKPOINT": str(checkpoint),
                "BC_BASELINE_CHECKPOINT": str(baseline_checkpoint),
            },
            inherited={},
        )

    metadata.write_text(json.dumps({"val_metrics": {"mean_regret": 0.2}}))
    env = launch_environment(
        experiment,
        supplied={
            "BC_CHECKPOINT": str(checkpoint),
            "BC_BASELINE_CHECKPOINT": str(baseline_checkpoint),
        },
        inherited={},
    )
    assert env["BC_CHECKPOINT"] == str(checkpoint)
    evidence_path = write_launch_evidence(experiment, tmp_path / "runs", env)
    evidence = json.loads(evidence_path.read_text())
    assert evidence["bc"]["candidate"]["mean_regret"] == 0.2
    assert evidence["bc"]["baseline"]["mean_regret"] == 0.3


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


def test_backlog_markdown_renderer_is_deterministic_and_checkable(tmp_path):
    first = render_backlog_markdown()
    assert first == render_backlog_markdown()
    assert "`05-mcts-strength-sweep`" in first
    output = tmp_path / "backlog.md"
    output.write_text(first)
    assert check_backlog_markdown(output)
    output.write_text(f"# Backlog\n\n{first}\nMore prose.\n")
    assert check_backlog_markdown(output)
    output.write_text(first + "drift\n")
    assert check_backlog_markdown(output)
    output.write_text(first.replace("0.1–0.3 h", "9–9 h"))
    assert not check_backlog_markdown(output)

from __future__ import annotations

from dataclasses import dataclass, field
import json
import pytest
from unittest.mock import patch

from catanatron.colonist_1v1_eval import EvaluationReport, MatchupResult
from examples.colonist_1v1_train import ColonistTrainCallback
from examples.colonist_1v1_train import materialize_final_candidate
from examples.colonist_1v1_train import main as train_main
from catanatron.gym.model_schema import checkpoint_schema_path, write_model_schema


@dataclass
class _League:
    run_dir: object
    registered: list[tuple[str, str]] = field(default_factory=list)

    def register(self, path, *, label=None, metrics=None):
        self.registered.append((str(path), label))
        return str(path)

    def paths(self):
        return [path for path, _ in self.registered]

    def entries(self):
        return []


def _report(*, passed: bool = False, score: float = 0.25):
    return EvaluationReport(
        agent="L:test.zip",
        all_gates_passed=passed,
        summary={"weighted_score": score, "all_games_accounted": True},
    )


def _retention_report(win_rates: dict[str, float]):
    matchups = [
        MatchupResult(
            opponent=opponent,
            agent_code="L:test.zip",
            games=10,
            wins=round(rate * 10),
            losses=10 - round(rate * 10),
            draws=0,
            win_rate=rate,
            wilson_low=0.0,
            wilson_high=1.0,
            avg_agent_vp=0.0,
            avg_opponent_vp=0.0,
            avg_vp_diff=0.0,
            avg_turns=0.0,
        )
        for opponent, rate in win_rates.items()
    ]
    return EvaluationReport(
        agent="L:test.zip",
        matchups=matchups,
        summary={"weighted_score": 0.5, "all_games_accounted": True},
    )


def test_eval_frequency_is_independent_of_checkpoint_frequency(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "ppo_colonist_6_steps.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    league = _League(tmp_path)
    callback = ColonistTrainCallback(
        league,
        checkpoint.parent,
        save_freq=10,
        eval_freq=6,
        report_dir=tmp_path / "eval_reports",
    )
    callback.n_calls = 6
    callback.num_timesteps = 6

    with (
        patch.object(callback, "_current_checkpoint", return_value=checkpoint),
        patch(
            "examples.colonist_1v1_train.run_benchmark", return_value=_report()
        ) as run,
    ):
        assert callback._on_step()

    run.assert_called_once()
    assert run.call_args.kwargs["eval_kind"] == "dev"
    assert run.call_args.kwargs["gate_mode"] == "point"
    assert (tmp_path / "eval_reports" / "dev_step_6.json").exists()


def test_dev_eval_stops_training_when_f_retention_is_lost(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "ppo_colonist_10_steps.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    callback = ColonistTrainCallback(
        _League(tmp_path),
        checkpoint.parent,
        save_freq=100,
        eval_freq=10,
        report_dir=tmp_path / "eval_reports",
        retention_min_f_win_rate=0.1,
        retention_require_weak_gates=True,
    )
    callback.n_calls = 10
    callback.num_timesteps = 10
    report = _retention_report({"R": 1.0, "W": 0.8, "VP": 0.7, "F": 0.0})

    with (
        patch.object(callback, "_current_checkpoint", return_value=checkpoint),
        patch("examples.colonist_1v1_train.run_benchmark", return_value=report),
    ):
        assert not callback._on_step()

    assert callback.retention_stop_reason == {
        "timesteps": 10,
        "failures": [{"opponent": "F", "win_rate": 0.0, "minimum": 0.1}],
    }


def test_dev_eval_continues_when_all_retention_gates_pass(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "ppo_colonist_10_steps.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    callback = ColonistTrainCallback(
        _League(tmp_path),
        checkpoint.parent,
        save_freq=100,
        eval_freq=10,
        report_dir=tmp_path / "eval_reports",
        retention_min_f_win_rate=0.1,
        retention_require_weak_gates=True,
    )
    callback.n_calls = 10
    callback.num_timesteps = 10
    report = _retention_report({"R": 1.0, "W": 0.8, "VP": 0.7, "F": 0.2})

    with (
        patch.object(callback, "_current_checkpoint", return_value=checkpoint),
        patch("examples.colonist_1v1_train.run_benchmark", return_value=report),
    ):
        assert callback._on_step()

    assert callback.retention_stop_reason is None


def test_locked_promotion_uses_lower_bound_gates(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "ppo_colonist_12_steps.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    league = _League(tmp_path)
    callback = ColonistTrainCallback(
        league,
        checkpoint.parent,
        save_freq=10,
        promotion_eval_freq=12,
        promotion_eval_protocol="milestone",
        report_dir=tmp_path / "eval_reports",
    )
    callback.n_calls = 12
    callback.num_timesteps = 12

    with (
        patch.object(callback, "_current_checkpoint", return_value=checkpoint),
        patch(
            "examples.colonist_1v1_train.run_benchmark",
            return_value=_report(passed=True, score=0.8),
        ) as run,
    ):
        assert callback._on_step()

    assert run.call_args.kwargs["eval_kind"] == "promotion"
    assert run.call_args.kwargs["gate_mode"] == "lower_bound"
    assert (tmp_path / "promoted" / "best_promotion.zip").exists()


def test_resume_never_downgrades_legacy_promoted_champion(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "ppo_colonist_12_steps.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"candidate")
    promoted = tmp_path / "promoted" / "best_promotion.zip"
    promoted.parent.mkdir()
    promoted.write_bytes(b"existing champion")
    (tmp_path / "training_events.jsonl").write_text(
        json.dumps(
            {
                "label": "best_promotion",
                "path": str(promoted),
                "weighted_score": 1.0,
                "timesteps": 6,
            }
        )
        + "\n"
    )
    callback = ColonistTrainCallback(
        _League(tmp_path),
        checkpoint.parent,
        save_freq=10,
        promotion_eval_freq=12,
        report_dir=tmp_path / "eval_reports",
        restore_selection_state=True,
    )
    callback.n_calls = 12
    callback.num_timesteps = 12

    with (
        patch.object(callback, "_current_checkpoint", return_value=checkpoint),
        patch(
            "examples.colonist_1v1_train.run_benchmark",
            return_value=_report(passed=True, score=0.8),
        ),
    ):
        assert callback._on_step()

    assert callback.best_promotion_path == promoted
    assert promoted.read_bytes() == b"existing champion"


def test_final_candidate_uses_locked_promotion_and_keeps_last_state(tmp_path):
    class _Model:
        num_timesteps = 100

        def save(self, path):
            path = str(path)
            if not path.endswith(".zip"):
                path += ".zip"
            from pathlib import Path

            Path(path).write_bytes(b"last optimizer state")

    promoted = tmp_path / "promoted" / "best_promotion.zip"
    promoted.parent.mkdir()
    promoted.write_bytes(b"locked champion")
    schema = {
        "schema_version": 1,
        "feature_hash": "f",
        "action_hash": "a",
        "rules_hash": "r",
    }
    write_model_schema(checkpoint_schema_path(promoted), schema)

    final, last, source, selected_timesteps = materialize_final_candidate(
        _Model(),
        tmp_path,
        model_schema=schema,
        promoted_checkpoint=promoted,
        promoted_timesteps=50,
    )

    assert source == promoted
    assert last.read_bytes() == b"last optimizer state"
    assert final.read_bytes() == b"locked champion"
    assert selected_timesteps == 50


def test_final_candidate_rejects_incompatible_promotion_schema(tmp_path):
    class _Model:
        num_timesteps = 100

        def save(self, path):
            from pathlib import Path

            Path(str(path)).write_bytes(b"last")

    expected = {
        "schema_version": 1,
        "feature_hash": "expected",
        "action_hash": "a",
        "rules_hash": "r",
    }
    promoted = tmp_path / "promoted.zip"
    promoted.write_bytes(b"old")
    write_model_schema(
        checkpoint_schema_path(promoted),
        {**expected, "feature_hash": "stale"},
    )

    with pytest.raises(ValueError, match="selected promotion schema mismatch"):
        materialize_final_candidate(
            _Model(),
            tmp_path,
            model_schema=expected,
            promoted_checkpoint=promoted,
            promoted_timesteps=50,
        )


def test_fresh_callback_does_not_restore_directory_champion(tmp_path):
    promoted = tmp_path / "promoted" / "best_promotion.zip"
    promoted.parent.mkdir()
    promoted.write_bytes(b"stale")

    callback = ColonistTrainCallback(
        _League(tmp_path),
        tmp_path / "checkpoints",
        save_freq=10,
    )

    assert callback.best_promotion_path is None
    assert callback.best_promotion_timesteps is None


def test_resume_manifest_records_checkpoint_effective_hyperparameters(tmp_path):
    first = tmp_path / "first"
    train_main(
        [
            "--timesteps",
            "8",
            "--n-steps",
            "8",
            "--batch-size",
            "8",
            "--n-epochs",
            "1",
            "--hidden",
            "16",
            "16",
            "--save-freq",
            "1000",
            "--curriculum",
            "none",
            "--run-dir",
            str(first),
            "--skip-final-eval",
        ]
    )
    resumed = tmp_path / "resumed"
    train_main(
        [
            "--timesteps",
            "8",
            "--resume-checkpoint",
            str(first / "colonist_maskable_ppo.zip"),
            "--n-steps",
            "4",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.01",
            "--hidden",
            "32",
            "32",
            "--save-freq",
            "1000",
            "--curriculum",
            "none",
            "--run-dir",
            str(resumed),
            "--skip-final-eval",
        ]
    )
    manifest = json.loads((resumed / "run_manifest.json").read_text())
    training = manifest["training"]

    assert training["configuration_source"] == "resume_checkpoint"
    assert training["hidden"] == [16, 16]
    assert training["hidden_requested"] == [32, 32]
    assert training["ppo"]["n_steps"] == 8
    assert training["ppo_requested"]["n_steps"] == 4
    assert training["ppo"]["learning_rate"] == 3e-4
    assert training["ppo_requested"]["learning_rate"] == 0.01


def test_resume_and_bc_warmstart_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        train_main(
            [
                "--resume-checkpoint",
                "resume.zip",
                "--bc-checkpoint",
                "bc.pt",
            ]
        )


def test_bc_anchor_requires_bc_checkpoint():
    with pytest.raises(SystemExit):
        train_main(["--bc-anchor-coef", "0.1"])


def test_retention_gate_requires_development_evaluation():
    with pytest.raises(SystemExit):
        train_main(["--retention-min-f-win-rate", "0.1"])

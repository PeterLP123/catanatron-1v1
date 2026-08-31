from __future__ import annotations

import pytest

from catanatron.gym.distillation import AgentIdentity, DistillationDatasetWriter
from catanatron.gym.envs.action_space import get_action_array
from catanatron.gym.model_schema import build_model_schema
from catanatron.models.enums import ActionType
from catanatron.models.player import Color
from examples.colonist_1v1_teacher_label_compare import compare_teacher_labels


def _metadata(*, teacher: str, teacher_seed_round: int, p95_ms: float) -> dict:
    return {
        "config": {
            "iteration": 0,
            "games": 1,
            "base_seed": 99,
            "student_spec": "W",
            "teacher_spec": teacher,
            "opponent_spec": "F",
            "feature_profile": "raw",
            "human_visible_obs": False,
            "alternate_seats": True,
            "include_forced": False,
            "record_legal_action_types": ["BUILD_ROAD"],
            "teacher_seed_round": teacher_seed_round,
        },
        "schema": build_model_schema(),
        "student": AgentIdentity.from_spec("W").__dict__,
        "teacher": AgentIdentity.from_spec(teacher).__dict__,
        "opponent": AgentIdentity.from_spec("F").__dict__,
        "collection_stats": {
            "teacher_latency": {
                "count": 2,
                "mean_ms": p95_ms,
                "p50_ms": p95_ms,
                "p95_ms": p95_ms,
                "max_ms": p95_ms,
            }
        },
    }


def _write_iteration(root, rows, *, teacher, teacher_seed_round, p95_ms):
    writer = DistillationDatasetWriter(
        root,
        iteration=0,
        shard_games=1,
        metadata=_metadata(
            teacher=teacher,
            teacher_seed_round=teacher_seed_round,
            p95_ms=p95_ms,
        ),
    )
    writer.add_game(
        rows,
        game_index=0,
        game_seed=7,
        student_color=Color.BLUE,
        winner=Color.RED,
        truncated=False,
    )
    writer.finalize()
    return root / "iteration-0000"


def test_teacher_label_comparison_requires_exact_behavior_trajectory(tmp_path):
    pytest.importorskip("pyarrow")
    codec = get_action_array((Color.BLUE, Color.RED), "BASE")
    roads = [
        index
        for index, (action_type, _) in enumerate(codec)
        if action_type == ActionType.BUILD_ROAD
    ]
    end_turn = next(
        index
        for index, (action_type, _) in enumerate(codec)
        if action_type == ActionType.END_TURN
    )
    common = [
        {
            "GAME_INDEX": 0,
            "DECISION_INDEX": 1,
            "STATE_HASH": "state-1",
            "LEGAL_ACTIONS": [roads[0], roads[1], end_turn],
            "BEHAVIOR_ACTION": end_turn,
            "TEACHER_ACTION": roads[0],
        },
        {
            "GAME_INDEX": 0,
            "DECISION_INDEX": 2,
            "STATE_HASH": "state-2",
            "LEGAL_ACTIONS": [roads[1], roads[2], end_turn],
            "BEHAVIOR_ACTION": roads[1],
            "TEACHER_ACTION": end_turn,
        },
    ]
    candidate_rows = [dict(row) for row in common]
    candidate_rows[1]["TEACHER_ACTION"] = roads[2]
    reference = _write_iteration(
        tmp_path / "reference",
        common,
        teacher="F",
        teacher_seed_round=0,
        p95_ms=100.0,
    )
    candidate = _write_iteration(
        tmp_path / "candidate",
        candidate_rows,
        teacher="M:25",
        teacher_seed_round=1,
        p95_ms=150.0,
    )

    report = compare_teacher_labels(
        reference,
        candidate,
        minimum_rows=2,
        minimum_agreement=0.5,
        minimum_disagreement=0.5,
        minimum_both_road_agreement=1.0,
        maximum_p95_ms=200.0,
    )

    assert report["all_gates_passed"]
    assert report["metrics"]["agreement_rate"] == 0.5
    assert report["metrics"]["disagreement_rate"] == 0.5
    assert report["metrics"]["both_choose_road_rows"] == 1
    assert report["metrics"]["both_choose_road_exact_agreement_rate"] == 1.0
    assert report["metrics"]["maximum_observed_p95_ms"] == 150.0

    drifted_rows = [dict(row) for row in candidate_rows]
    drifted_rows[0]["STATE_HASH"] = "different-state"
    drifted = _write_iteration(
        tmp_path / "drifted",
        drifted_rows,
        teacher="M:25",
        teacher_seed_round=1,
        p95_ms=150.0,
    )
    with pytest.raises(ValueError, match="trajectory drift in STATE_HASH"):
        compare_teacher_labels(reference, drifted)

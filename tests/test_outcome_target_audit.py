from __future__ import annotations

import math

import pytest

from catanatron.gym.bc_training import hash_parquet_shards
from catanatron.gym.distillation import DistillationDatasetWriter
from catanatron.gym.outcome_target_audit import audit_outcome_targets
from catanatron.models.player import Color


def test_outcome_audit_derives_margin_and_scores_public_baseline(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = tmp_path / "teacher"
    root.mkdir()
    path = root / "shard-00000.parquet"
    pd.DataFrame(
        {
            "GAME_ID": ["g0", "g0", "g1", "g1"],
            "SEAT": [0, 1, 0, 1],
            "RETURN": [1.0, -1.0, -1.0, 1.0],
            "VICTORY_POINTS_RETURN": [15.0, 9.0, 8.0, 15.0],
            "F_P0_PUBLIC_VPS": [12.0, 8.0, 7.0, 13.0],
            "F_P1_PUBLIC_VPS": [8.0, 12.0, 13.0, 7.0],
        }
    ).to_parquet(path, index=False)
    _, expected_hash = hash_parquet_shards([path], progress=False)

    report = audit_outcome_targets(
        [[root]],
        expected_dataset_sha256=expected_hash,
        expected_shards=1,
        minimum_win_row_coverage=1.0,
        minimum_margin_row_coverage=1.0,
        minimum_split_groups=2,
        minimum_minority_fraction=0.5,
    )

    assert report["all_gates_passed"]
    assert report["combined"]["vp_margin_target"]["sources"] == {"paired_vp_return": 4}
    assert report["combined"]["baselines"]["public_vp_win"]["auc"] == 1.0
    assert report["combined"]["baselines"]["public_vp_margin"]["mae"] == 1.5


def test_outcome_audit_recovers_legacy_distillation_winner(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = tmp_path / "distill"
    writer = DistillationDatasetWriter(
        root,
        iteration=0,
        shard_games=1,
        metadata={"schema": {"schema_hash": "test"}},
    )
    writer.add_game(
        [
            {
                "GAME_INDEX": 0,
                "GAME_ID": "g0",
                "ITERATION": 0,
                "COLOR": "BLUE",
                "SEAT": 0,
                "F_P0_PUBLIC_VPS": 4.0,
                "F_P1_PUBLIC_VPS": 6.0,
            }
        ],
        game_index=0,
        game_seed=17,
        student_color=Color.BLUE,
        winner=Color.RED,
        truncated=False,
    )
    writer.finalize()

    report = audit_outcome_targets(
        [[root / "iteration-0000"]], minimum_win_row_coverage=1.0
    )

    assert report["combined"]["win_target"]["sources"] == {"manifest": 1}
    assert report["combined"]["win_target"]["value_counts"] == {"-1.0": 1}
    assert math.isclose(report["combined"]["vp_margin_target"]["row_coverage"], 0.0)
    assert not report["all_gates_passed"]

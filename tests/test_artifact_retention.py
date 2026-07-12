from __future__ import annotations

import pytest

from catanatron.gym.artifact_retention import (
    archive_retention_plan,
    build_retention_plan,
)
from catanatron.gym.model_schema import checkpoint_schema_path


def test_retention_is_hash_first_and_reversible(tmp_path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    promoted = run_dir / "promoted"
    checkpoints.mkdir(parents=True)
    promoted.mkdir()
    old = checkpoints / "ppo_colonist_100_steps.zip"
    latest = checkpoints / "ppo_colonist_200_steps.zip"
    champion = promoted / "best.zip"
    for path, payload in ((old, b"old"), (latest, b"new"), (champion, b"best")):
        path.write_bytes(payload)
    checkpoint_schema_path(old).write_text("{}\n", encoding="utf-8")

    plan = build_retention_plan(run_dir, keep_latest=1)

    assert [item.path for item in plan.archive_candidates] == [str(old)]
    assert all(len(item.sha256) == 64 for item in plan.artifacts)
    assert old.exists(), "planning must be a dry run"
    moved = archive_retention_plan(plan, archive_dir=run_dir / "archive")
    assert moved == [run_dir / "archive" / "checkpoints" / old.name]
    assert not old.exists()
    assert moved[0].read_bytes() == b"old"
    assert checkpoint_schema_path(moved[0]).exists()
    assert latest.exists() and champion.exists()


def test_retention_apply_rejects_artifacts_changed_after_plan(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "ppo_colonist_1_steps.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"planned")
    plan = build_retention_plan(tmp_path / "run", keep_latest=0)
    checkpoint.write_bytes(b"changed after planning")

    with pytest.raises(ValueError, match="changed after retention plan"):
        archive_retention_plan(plan, archive_dir=tmp_path / "archive")
    assert checkpoint.exists()


def test_retention_apply_never_overwrites_existing_archive(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "ppo_colonist_1_steps.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"candidate")
    plan = build_retention_plan(run_dir, keep_latest=0)
    archive = tmp_path / "archive"
    existing = archive / "checkpoints" / checkpoint.name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep me")

    moved = archive_retention_plan(plan, archive_dir=archive)

    assert existing.read_bytes() == b"keep me"
    assert moved == [archive / "checkpoints" / "ppo_colonist_1_steps-1.zip"]
    assert moved[0].read_bytes() == b"candidate"

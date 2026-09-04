import shutil

import pytest

from catanatron.players.checkpoint_manifest import checkpoint_fields, verify_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model" / "policy.pt"
    path.parent.mkdir()
    for suffix in (".pt", ".meta.json", ".schema.json"):
        path.with_suffix(suffix).write_text(suffix)
    return path


def test_checkpoint_references_survive_moving_a_bundle(tmp_path, checkpoint):
    manifest = tmp_path / "manifests" / "player.json"
    fields = checkpoint_fields(checkpoint, "policy", manifest.parent)
    assert fields["policy_checkpoint"] == "../model/policy.pt"
    moved = tmp_path / "moved"
    shutil.copytree(checkpoint.parent, moved / "model")
    assert verify_checkpoint(fields, moved / "manifests" / "player.json", "policy") == (
        moved / "model" / "policy.pt"
    )


@pytest.mark.parametrize(
    "suffix, hash_key, failure",
    [
        (".pt", "checkpoint", "missing_file"),
        (".meta.json", "metadata", "missing_hash"),
        (".schema.json", "schema", "changed_file"),
    ],
)
def test_checkpoint_rejects_incomplete_or_changed_artifacts(
    tmp_path, checkpoint, suffix, hash_key, failure
):
    manifest = tmp_path / "player.json"
    fields = checkpoint_fields(checkpoint, "policy", manifest.parent)
    artifact = checkpoint.with_suffix(suffix)
    if failure == "missing_file":
        artifact.unlink()
        error, message = FileNotFoundError, "Missing"
    elif failure == "missing_hash":
        del fields[f"policy_{hash_key}_sha256"]
        error, message = ValueError, "missing"
    else:
        artifact.write_text("changed")
        error, message = ValueError, "hash mismatch"
    with pytest.raises(error, match=message):
        verify_checkpoint(fields, manifest, "policy")

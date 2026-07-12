from __future__ import annotations

import json

from catanatron.gym.provenance import (
    collect_run_provenance,
    sha256_file,
    write_artifact_manifest,
)


def test_artifact_manifest_hashes_files_without_deleting_them(tmp_path):
    artifact = tmp_path / "checkpoint.zip"
    artifact.write_bytes(b"model")
    output = write_artifact_manifest(
        tmp_path / "artifacts.json",
        artifact_paths=[artifact],
        metadata_values={"champion": True},
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert artifact.exists()
    assert payload["metadata"] == {"champion": True}
    assert payload["artifacts"][0]["sha256"] == sha256_file(artifact)


def test_run_provenance_records_python_hardware_and_git():
    provenance = collect_run_provenance()

    assert provenance["python"]["executable"]
    assert provenance["hardware"]["cpu_count"]
    assert "dirty" in provenance["git"]
    assert len(provenance["packages_hash"]) == 64

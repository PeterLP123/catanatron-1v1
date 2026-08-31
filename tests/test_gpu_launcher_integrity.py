from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    REPO_ROOT / "scripts/gpu/run_dagger_f_iteration.sh",
    REPO_ROOT / "scripts/gpu/run_factored_dagger_comparison.sh",
    REPO_ROOT / "scripts/gpu/run_paired_confirmation.sh",
)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_provenance_includes_index_and_untracked_sources(launcher):
    source = launcher.read_text(encoding="utf-8")

    assert "git diff HEAD --binary" in source
    assert "source_status_sha256" in source
    assert "source_untracked_sha256" in source
    assert "run_identity.sha256" in source


def _checkpoint(path: Path) -> Path:
    path.write_bytes(b"checkpoint")
    path.with_suffix(".meta.json").write_text("{}\n", encoding="utf-8")
    path.with_suffix(".schema.json").write_text("{}\n", encoding="utf-8")
    return path


def _stale_output(path: Path) -> Path:
    path.mkdir()
    (path / "old-evidence.json").write_text("{}\n", encoding="utf-8")
    return path


def test_paired_launcher_blocks_unidentified_existing_output(tmp_path):
    candidate = _checkpoint(tmp_path / "candidate.pt")
    baseline = _checkpoint(tmp_path / "baseline.pt")
    output = _stale_output(tmp_path / "paired-output")

    completed = subprocess.run(
        [str(LAUNCHERS[2]), str(candidate), str(baseline), str(output)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "lacks an immutable run identity" in completed.stderr


def test_dagger_launcher_blocks_unidentified_existing_output(tmp_path):
    student = _checkpoint(tmp_path / "student.pt")
    base_f = tmp_path / "base-f"
    base_vp = tmp_path / "base-vp"
    base_f.mkdir()
    base_vp.mkdir()
    (base_f / "data.parquet").write_bytes(b"f")
    (base_vp / "data.parquet").write_bytes(b"vp")
    prior = tmp_path / "iteration-0000"
    prior.mkdir()
    (prior / "manifest.json").write_text("{}\n", encoding="utf-8")
    output = _stale_output(tmp_path / "dagger-output")

    completed = subprocess.run(
        [
            str(LAUNCHERS[0]),
            str(student),
            str(base_f),
            str(base_vp),
            str(output),
            str(prior),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "lacks an immutable run identity" in completed.stderr


def test_factored_launcher_blocks_unidentified_existing_output(tmp_path):
    base_f = tmp_path / "base-f"
    base_vp = tmp_path / "base-vp"
    dagger = tmp_path / "dagger"
    for directory in (base_f, base_vp, dagger):
        directory.mkdir()
    (base_f / "data.parquet").write_bytes(b"f")
    (base_vp / "data.parquet").write_bytes(b"vp")
    (dagger / "manifest.json").write_text("{}\n", encoding="utf-8")
    control = _checkpoint(tmp_path / "control.pt")
    output = _stale_output(tmp_path / "factored-output")

    completed = subprocess.run(
        [
            str(LAUNCHERS[1]),
            str(base_f),
            str(base_vp),
            str(dagger),
            str(control),
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "lacks an immutable run identity" in completed.stderr

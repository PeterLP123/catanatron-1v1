"""Reproducibility metadata for datasets, training runs, and model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_provenance(repo: str | Path = ".") -> dict[str, Any]:
    root = Path(repo).resolve()
    top = _git_output(root, "rev-parse", "--show-toplevel")
    if top:
        root = Path(top)
    status = _git_output(root, "status", "--porcelain")
    return {
        "root": os.fspath(root),
        "commit": _git_output(root, "rev-parse", "HEAD"),
        "branch": _git_output(root, "branch", "--show-current"),
        "dirty": bool(status),
    }


def installed_packages() -> list[str]:
    rows = {
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    }
    return sorted(rows, key=str.casefold)


def hardware_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name(0)
        mps = getattr(torch.backends, "mps", None)
        result["mps_available"] = bool(mps and mps.is_available())
    except ImportError:
        result["torch"] = None
    return result


def collect_run_provenance(repo: str | Path = ".") -> dict[str, Any]:
    return {
        "git": git_provenance(repo),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "hardware": hardware_provenance(),
        "packages_hash": hashlib.sha256(
            "\n".join(installed_packages()).encode("utf-8")
        ).hexdigest(),
    }


def write_environment_snapshot(run_dir: str | Path) -> Path:
    output = Path(run_dir) / "environment.lock.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(installed_packages()) + "\n", encoding="utf-8")
    return output


def write_artifact_manifest(
    path: str | Path,
    *,
    artifact_paths: list[str | Path],
    metadata_values: dict[str, Any] | None = None,
) -> Path:
    """Write a compact hash manifest before artifacts are archived or pruned."""

    rows = []
    for item in artifact_paths:
        source = Path(item)
        rows.append(
            {
                "path": os.fspath(source),
                "exists": source.exists(),
                "bytes": source.stat().st_size if source.exists() else None,
                "sha256": sha256_file(source) if source.is_file() else None,
            }
        )
    payload = {"artifacts": rows, "metadata": metadata_values or {}}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output

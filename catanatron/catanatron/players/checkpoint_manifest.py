"""Portable checkpoint references shared by composite-player manifests."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from catanatron.gym.provenance import sha256_file


def relative_path(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def _checkpoint_files(checkpoint: Path, label: str):
    yield checkpoint, f"{label}_checkpoint_sha256", f"{label} checkpoint"
    yield checkpoint.with_suffix(
        ".meta.json"
    ), f"{label}_metadata_sha256", f"{label} sidecar"
    yield checkpoint.with_suffix(
        ".schema.json"
    ), f"{label}_schema_sha256", f"{label} sidecar"


def checkpoint_fields(checkpoint: Path, label: str, parent: Path) -> dict[str, str]:
    """Record a checkpoint and both required sidecars without absolute paths."""
    checkpoint = checkpoint.resolve()
    fields = {f"{label}_checkpoint": relative_path(checkpoint, parent)}
    for path, key, description in _checkpoint_files(checkpoint, label):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
        fields[key] = sha256_file(path)
    return fields


def verify_checkpoint(payload: Mapping[str, Any], manifest: Path, label: str) -> Path:
    """Resolve and verify a checkpoint, metadata, and schema before loading."""
    raw = Path(payload[f"{label}_checkpoint"])
    checkpoint = (manifest.parent / raw).resolve()
    for path, key, description in _checkpoint_files(checkpoint, label):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
        expected = payload.get(key)
        if not expected:
            raise ValueError(f"Manifest {manifest} is missing {key}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{description} hash mismatch for {path}: {actual} != {expected}"
            )
    return checkpoint

"""Atomic writes for file-backed training state and model manifests."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    overwrite: bool = True,
    validate: Callable[[Path], object] | None = None,
) -> None:
    """Publish complete JSON, optionally validating it before publication.

    Validation runs beside the destination so relative artifact paths work.
    With ``overwrite=False``, even a concurrent writer cannot replace an
    existing file. Temporary files are removed on both success and failure.
    """
    path = Path(path)
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    contents = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    output = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(output.name)
    try:
        with output:
            output.write(contents)
        if validate is not None:
            validate(temporary)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Linking on the same filesystem publishes without a
            # check-then-replace race or exposing a partially written file.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

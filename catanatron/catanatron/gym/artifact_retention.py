"""Hash-first, reversible retention for large training artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from catanatron.gym.model_schema import checkpoint_schema_path
from catanatron.gym.provenance import sha256_file


@dataclass(frozen=True)
class ArtifactDecision:
    path: str
    bytes: int
    sha256: str
    decision: str
    reason: str


@dataclass(frozen=True)
class RetentionPlan:
    run_dir: str
    keep_latest: int
    artifacts: tuple[ArtifactDecision, ...]

    @property
    def archive_candidates(self) -> tuple[ArtifactDecision, ...]:
        return tuple(item for item in self.artifacts if item.decision == "archive")

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-2])
    except (IndexError, ValueError):
        return -1


def build_retention_plan(
    run_dir: str | Path,
    *,
    keep_latest: int = 3,
    pins: Iterable[str | Path] = (),
) -> RetentionPlan:
    root = Path(run_dir).resolve()
    pinned = {Path(path).resolve() for path in pins}
    checkpoint_files = sorted(
        (root / "checkpoints").glob("*.zip"), key=_checkpoint_step, reverse=True
    )
    latest = set(checkpoint_files[: max(0, int(keep_latest))])
    all_artifacts = sorted(root.rglob("*.zip"))
    decisions = []
    for path in all_artifacts:
        resolved = path.resolve()
        if resolved in pinned:
            decision, reason = "keep", "explicitly pinned"
        elif "promoted" in path.parts or "league" in path.parts:
            decision, reason = "keep", "champion or active league artifact"
        elif path.parent == root:
            decision, reason = "keep", "top-level final artifact"
        elif path in latest:
            decision, reason = "keep", f"one of latest {keep_latest} checkpoints"
        elif path.parent == root / "checkpoints":
            decision, reason = "archive", "superseded checkpoint"
        else:
            decision, reason = "keep", "unclassified artifact; conservative default"
        decisions.append(
            ArtifactDecision(
                path=str(path),
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
                decision=decision,
                reason=reason,
            )
        )
    return RetentionPlan(str(root), int(keep_latest), tuple(decisions))


def archive_retention_plan(
    plan: RetentionPlan,
    *,
    archive_dir: str | Path | None = None,
) -> list[Path]:
    root = Path(plan.run_dir)
    if archive_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_root = root / "archive" / "checkpoints" / stamp
    else:
        archive_root = Path(archive_dir)
    verified: list[tuple[ArtifactDecision, Path]] = []
    for item in plan.archive_candidates:
        source = Path(item.path)
        if not source.exists():
            raise FileNotFoundError(
                f"Planned artifact disappeared before apply: {source}"
            )
        current_bytes = source.stat().st_size
        current_hash = sha256_file(source)
        if current_bytes != item.bytes or current_hash != item.sha256:
            raise ValueError(
                f"Artifact changed after retention plan: {source}; "
                "re-run the dry-run plan before applying"
            )
        verified.append((item, source))

    moved = []
    for _, source in verified:
        relative = source.relative_to(root)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix_index = 0
        while destination.exists() or checkpoint_schema_path(destination).exists():
            suffix_index += 1
            destination = destination.with_name(
                f"{source.stem}-{suffix_index}{source.suffix}"
            )
        shutil.move(str(source), str(destination))
        schema_source = checkpoint_schema_path(source)
        if schema_source.exists():
            schema_destination = checkpoint_schema_path(destination)
            schema_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(schema_source), str(schema_destination))
        moved.append(destination)
    return moved

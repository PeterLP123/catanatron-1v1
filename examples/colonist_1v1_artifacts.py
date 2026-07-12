#!/usr/bin/env python3
"""Plan or apply reversible checkpoint retention after hashing every artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from catanatron.gym.artifact_retention import (
    archive_retention_plan,
    build_retention_plan,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--keep-latest", type=int, default=3)
    parser.add_argument("--pin", type=Path, action="append", default=[])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move archive candidates into run_dir/archive; never deletes them.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output plan JSON (default: <run-dir>/artifact_retention_plan.json).",
    )
    args = parser.parse_args(argv)

    plan = build_retention_plan(
        args.run_dir, keep_latest=args.keep_latest, pins=args.pin
    )
    manifest = args.manifest or args.run_dir / "artifact_retention_plan.json"
    plan.write(manifest)
    print(f"Retention plan: {manifest}")
    print(
        f"Artifacts: {len(plan.artifacts)} total, "
        f"{len(plan.archive_candidates)} archive candidates"
    )
    for item in plan.archive_candidates:
        print(f"  ARCHIVE {item.path} ({item.reason}, sha256={item.sha256[:12]})")

    if args.apply:
        moved = archive_retention_plan(plan)
        print(f"Moved {len(moved)} artifacts into the reversible archive.")
    else:
        print(
            "Dry run only. Re-run with --apply to move candidates; nothing was deleted."
        )


if __name__ == "__main__":
    main()

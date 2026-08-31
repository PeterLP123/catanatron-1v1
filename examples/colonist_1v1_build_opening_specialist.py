#!/usr/bin/env python3
"""Build and validate a portable run-32-plus-opening-specialist manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from catanatron.gym.provenance import sha256_file
from catanatron.models.player import Color
from catanatron.players.learned import OpeningSpecialistCheckpointPlayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    policy = args.policy.resolve()
    if not policy.is_file():
        raise FileNotFoundError(f"Missing policy checkpoint: {policy}")
    for suffix in (".meta.json", ".schema.json"):
        sidecar = policy.with_suffix(suffix)
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing policy sidecar: {sidecar}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "kind": "opening_specialist",
        "policy_checkpoint": _relative(policy, args.output.parent),
        "policy_checkpoint_sha256": sha256_file(policy),
        "policy_metadata_sha256": sha256_file(policy.with_suffix(".meta.json")),
        "policy_schema_sha256": sha256_file(policy.with_suffix(".schema.json")),
        "policy_frozen": True,
        "opening_evaluator": "value_function_default",
        "opening_prompts": list(OpeningSpecialistCheckpointPlayer.OPENING_PROMPTS),
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    OpeningSpecialistCheckpointPlayer(Color.BLUE, temporary)
    os.replace(temporary, args.output)
    print(
        "Built opening-specialist manifest: "
        f"{args.output} sha256={sha256_file(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

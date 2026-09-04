#!/usr/bin/env python3
"""Build and validate a portable run-32-plus-opening-specialist manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from catanatron.file_utils import write_json_atomic
from catanatron.gym.provenance import sha256_file
from catanatron.players.checkpoint_manifest import checkpoint_fields
from catanatron.models.player import Color
from catanatron.players.learned import OpeningSpecialistCheckpointPlayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    policy_fields = checkpoint_fields(args.policy, "policy", args.output.parent)
    payload = {
        "schema_version": "1.0",
        "kind": "opening_specialist",
        **policy_fields,
        "policy_frozen": True,
        "opening_evaluator": "value_function_default",
        "opening_prompts": list(OpeningSpecialistCheckpointPlayer.OPENING_PROMPTS),
    }
    write_json_atomic(
        args.output,
        payload,
        overwrite=False,
        validate=lambda path: OpeningSpecialistCheckpointPlayer(Color.BLUE, path),
    )
    print(
        "Built opening-specialist manifest: "
        f"{args.output} sha256={sha256_file(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

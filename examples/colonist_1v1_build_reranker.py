#!/usr/bin/env python3
"""Build and validate a portable policy-plus-outcome-critic manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from catanatron.gym.provenance import sha256_file
from catanatron.models.player import Color
from catanatron.players.learned import OutcomeRerankerCheckpointPlayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--minimum-win-probability-improvement", type=float, default=0.05
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    policy = args.policy.resolve()
    critic = args.critic.resolve()
    for label, path in (("policy", policy), ("critic", critic)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} checkpoint: {path}")
        for suffix in (".meta.json", ".schema.json"):
            sidecar = path.with_suffix(suffix)
            if not sidecar.is_file():
                raise FileNotFoundError(f"Missing {label} sidecar: {sidecar}")
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    if not 0 <= args.minimum_win_probability_improvement <= 1:
        raise ValueError("minimum win-probability improvement must be in [0, 1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "kind": "outcome_critic_reranker",
        "policy_checkpoint": _relative(policy, args.output.parent),
        "policy_checkpoint_sha256": sha256_file(policy),
        "policy_metadata_sha256": sha256_file(policy.with_suffix(".meta.json")),
        "policy_schema_sha256": sha256_file(policy.with_suffix(".schema.json")),
        "critic_checkpoint": _relative(critic, args.output.parent),
        "critic_checkpoint_sha256": sha256_file(critic),
        "critic_metadata_sha256": sha256_file(critic.with_suffix(".meta.json")),
        "critic_schema_sha256": sha256_file(critic.with_suffix(".schema.json")),
        "top_k": args.top_k,
        "minimum_win_probability_improvement": (
            args.minimum_win_probability_improvement
        ),
        "policy_frozen": True,
        "critic_frozen": True,
        "chance_handling": "public_only_spectrum_with_policy_fallback",
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    OutcomeRerankerCheckpointPlayer(Color.BLUE, temporary)
    os.replace(temporary, args.output)
    print(f"Built reranker manifest: {args.output} sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

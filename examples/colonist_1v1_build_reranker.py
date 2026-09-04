#!/usr/bin/env python3
"""Build and validate a portable policy-plus-outcome-critic manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from catanatron.file_utils import write_json_atomic
from catanatron.gym.provenance import sha256_file
from catanatron.players.checkpoint_manifest import checkpoint_fields
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    policy_fields = checkpoint_fields(args.policy, "policy", args.output.parent)
    critic_fields = checkpoint_fields(args.critic, "critic", args.output.parent)
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    if not 0 <= args.minimum_win_probability_improvement <= 1:
        raise ValueError("minimum win-probability improvement must be in [0, 1]")
    payload = {
        "schema_version": "1.0",
        "kind": "outcome_critic_reranker",
        **policy_fields,
        **critic_fields,
        "top_k": args.top_k,
        "minimum_win_probability_improvement": (
            args.minimum_win_probability_improvement
        ),
        "policy_frozen": True,
        "critic_frozen": True,
        "chance_handling": "public_only_spectrum_with_policy_fallback",
    }
    write_json_atomic(
        args.output,
        payload,
        overwrite=False,
        validate=lambda path: OutcomeRerankerCheckpointPlayer(Color.BLUE, path),
    )
    print(f"Built reranker manifest: {args.output} sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

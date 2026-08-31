#!/usr/bin/env python3
"""Build and validate a portable visible-state same-turn PUCT manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from catanatron.gym.provenance import sha256_file
from catanatron.models.player import Color
from catanatron.players.visible_puct import (
    FORBIDDEN_SEARCH_ACTIONS,
    VISIBLE_DETERMINISTIC_ACTIONS,
    VisibleSameTurnPuctPlayer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--num-simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=2**0.5)
    parser.add_argument(
        "--leaf-evaluator",
        choices=("outcome_critic", "public_f"),
        default="outcome_critic",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = args.policy.resolve()
    critic = args.critic.resolve()
    for label, path in (("policy", policy), ("critic", critic)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} checkpoint: {path}")
        for suffix in (".meta.json", ".schema.json"):
            sidecar = path.with_suffix(suffix)
            if not sidecar.is_file():
                raise FileNotFoundError(f"Missing {label} sidecar: {sidecar}")
    if args.num_simulations < 1:
        raise ValueError("num-simulations must be positive")
    if args.c_puct <= 0:
        raise ValueError("c-puct must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "kind": "visible_same_turn_puct",
        "policy_checkpoint": _relative(policy, args.output.parent),
        "policy_checkpoint_sha256": sha256_file(policy),
        "policy_metadata_sha256": sha256_file(policy.with_suffix(".meta.json")),
        "policy_schema_sha256": sha256_file(policy.with_suffix(".schema.json")),
        "critic_checkpoint": _relative(critic, args.output.parent),
        "critic_checkpoint_sha256": sha256_file(critic),
        "critic_metadata_sha256": sha256_file(critic.with_suffix(".meta.json")),
        "critic_schema_sha256": sha256_file(critic.with_suffix(".schema.json")),
        "num_simulations": args.num_simulations,
        "c_puct": args.c_puct,
        "leaf_evaluator": args.leaf_evaluator,
        "policy_frozen": True,
        "critic_frozen": True,
        "search_scope": "same_player_turn_visible_deterministic_only",
        "visible_action_types": sorted(
            action_type.name for action_type in VISIBLE_DETERMINISTIC_ACTIONS
        ),
        "forbidden_action_types": sorted(
            action_type.name for action_type in FORBIDDEN_SEARCH_ACTIONS
        ),
        "chance_spectrum_usage": "forbidden",
        "opponent_turn_expansion": "forbidden",
        "final_move_rule": "visits_then_q_then_prior_then_action_id",
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    VisibleSameTurnPuctPlayer(Color.BLUE, temporary)
    os.replace(temporary, args.output)
    print(
        f"Built visible PUCT manifest: {args.output} sha256={sha256_file(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build and validate a portable visible-state same-turn PUCT manifest."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from catanatron.file_utils import write_json_atomic
from catanatron.gym.provenance import sha256_file
from catanatron.players.checkpoint_manifest import checkpoint_fields
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
        choices=("outcome_critic", "public_f", "public_f_own_hand_v1"),
        default="outcome_critic",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy_fields = checkpoint_fields(args.policy, "policy", args.output.parent)
    critic_fields = checkpoint_fields(args.critic, "critic", args.output.parent)
    if args.num_simulations < 1:
        raise ValueError("num-simulations must be positive")
    if not math.isfinite(args.c_puct) or args.c_puct <= 0:
        raise ValueError("c-puct must be finite and positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    payload = {
        "schema_version": "1.0",
        "kind": "visible_same_turn_puct",
        **policy_fields,
        **critic_fields,
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
    write_json_atomic(
        args.output,
        payload,
        overwrite=False,
        validate=lambda path: VisibleSameTurnPuctPlayer(Color.BLUE, path),
    )
    print(
        f"Built visible PUCT manifest: {args.output} sha256={sha256_file(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

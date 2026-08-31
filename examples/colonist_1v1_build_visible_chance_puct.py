#!/usr/bin/env python3
"""Build a run-55-lineage manifest with public dice/dev-card chance nodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from catanatron.gym.provenance import sha256_file
from catanatron.models.player import Color
from catanatron.players.visible_chance_puct import (
    PUBLIC_CHANCE_ACTIONS,
    PUBLIC_CHANCE_FORBIDDEN_ACTIONS,
    VisibleChancePuctPlayer,
)
from catanatron.players.visible_puct import VISIBLE_DETERMINISTIC_ACTIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve_from_manifest(manifest: Path, raw: str) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parent_manifest = args.parent_manifest.resolve()
    if not parent_manifest.is_file():
        raise FileNotFoundError(f"Missing parent manifest: {parent_manifest}")
    parent_hash = sha256_file(parent_manifest)
    if parent_hash != args.expected_parent_sha256:
        raise ValueError(
            f"Parent manifest hash mismatch: {parent_hash} != "
            f"{args.expected_parent_sha256}"
        )
    parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if parent.get("kind") != "visible_same_turn_puct":
        raise ValueError("Public chance PUCT parent must be visible_same_turn_puct")
    if parent.get("leaf_evaluator") != "public_f":
        raise ValueError("Public chance PUCT parent must use the public_f leaf")
    if parent.get("chance_spectrum_usage") != "forbidden":
        raise ValueError("Parent search must forbid chance spectra")

    policy = _resolve_from_manifest(parent_manifest, parent["policy_checkpoint"])
    critic = _resolve_from_manifest(parent_manifest, parent["critic_checkpoint"])
    for label, path in (("policy", policy), ("critic", critic)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing parent {label} checkpoint: {path}")
        if sha256_file(path) != parent[f"{label}_checkpoint_sha256"]:
            raise ValueError(f"Parent {label} checkpoint hash mismatch")
        for suffix, key in (
            (".meta.json", f"{label}_metadata_sha256"),
            (".schema.json", f"{label}_schema_sha256"),
        ):
            sidecar = path.with_suffix(suffix)
            if not sidecar.is_file() or sha256_file(sidecar) != parent[key]:
                raise ValueError(f"Parent {label} sidecar hash mismatch: {sidecar}")

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "kind": "visible_public_chance_puct",
        "parent_manifest": _relative(parent_manifest, args.output.parent),
        "parent_manifest_sha256": parent_hash,
        "policy_checkpoint": _relative(policy, args.output.parent),
        "critic_checkpoint": _relative(critic, args.output.parent),
        "num_simulations": parent["num_simulations"],
        "c_puct": parent["c_puct"],
        "leaf_evaluator": parent["leaf_evaluator"],
        "policy_frozen": True,
        "critic_frozen": True,
        "search_scope": "same_player_turn_public_chance",
        "visible_action_types": sorted(
            action_type.name for action_type in VISIBLE_DETERMINISTIC_ACTIONS
        ),
        "chance_action_types": sorted(
            action_type.name for action_type in PUBLIC_CHANCE_ACTIONS
        ),
        "forbidden_action_types": sorted(
            action_type.name for action_type in PUBLIC_CHANCE_FORBIDDEN_ACTIONS
        ),
        "chance_allocation": "deterministic_probability_deficit",
        "chance_spectrum_usage": "custom_public_only",
        "opponent_turn_expansion": "forbidden",
        "final_move_rule": parent["final_move_rule"],
    }
    for key in (
        "policy_checkpoint_sha256",
        "policy_metadata_sha256",
        "policy_schema_sha256",
        "critic_checkpoint_sha256",
        "critic_metadata_sha256",
        "critic_schema_sha256",
    ):
        payload[key] = parent[key]

    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    VisibleChancePuctPlayer(Color.BLUE, temporary)
    os.replace(temporary, args.output)
    print(
        f"Built visible chance PUCT manifest: {args.output} "
        f"sha256={sha256_file(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

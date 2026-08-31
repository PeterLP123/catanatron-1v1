#!/usr/bin/env python3
"""Compare two teachers on exactly matched student-visited decision states.

Use separate distillation roots with the same student, opponent, game seed,
iteration, and legal-action-family filter. Teacher work is RNG-isolated, so the
behavior trajectory must remain identical. This command rejects any state,
legal-set, or behavior drift before reporting label agreement and latency.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from catanatron.gym.envs.action_space import get_action_array
from catanatron.gym.provenance import sha256_file
from catanatron.models.player import Color

MATCH_COLUMNS = (
    "GAME_INDEX",
    "DECISION_INDEX",
    "STATE_HASH",
    "LEGAL_ACTIONS",
    "BEHAVIOR_ACTION",
    "TEACHER_ACTION",
)
MATCHED_CONFIG_FIELDS = (
    "iteration",
    "games",
    "base_seed",
    "student_spec",
    "opponent_spec",
    "feature_profile",
    "human_visible_obs",
    "alternate_seats",
    "include_forced",
    "record_legal_action_types",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=None)
    parser.add_argument("--minimum-agreement", type=float, default=None)
    parser.add_argument("--minimum-disagreement", type=float, default=None)
    parser.add_argument("--minimum-both-road-agreement", type=float, default=None)
    parser.add_argument("--maximum-p95-ms", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_verified_iteration(iteration_dir: Path):
    import pandas as pd

    iteration_dir = iteration_dir.resolve()
    manifest_path = iteration_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing iteration manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = iteration_dir.parent
    frames = []
    for shard in manifest.get("shards", []):
        path = root / shard["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing distillation shard: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != shard.get("sha256"):
            raise ValueError(
                f"Distillation shard hash mismatch: {path} "
                f"({actual_hash} != {shard.get('sha256')})"
            )
        frames.append(pd.read_parquet(path, columns=list(MATCH_COLUMNS)))
    if not frames:
        raise ValueError(f"Iteration has no verified shards: {iteration_dir}")
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != int(manifest.get("rows", -1)):
        raise ValueError(
            f"Manifest row count mismatch: {len(frame)} != {manifest.get('rows')}"
        )
    keys = ["GAME_INDEX", "DECISION_INDEX"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Iteration has duplicate decision keys: {iteration_dir}")
    frame["LEGAL_ACTIONS"] = frame["LEGAL_ACTIONS"].map(
        lambda values: tuple(int(value) for value in values)
    )
    return frame.set_index(keys).sort_index(), manifest, manifest_path


def _teacher_latency(manifest: dict[str, Any]) -> dict[str, Any]:
    return (
        manifest.get("metadata", {})
        .get("collection_stats", {})
        .get("teacher_latency", {})
    )


def _action_family(action_index: int) -> str:
    codec = get_action_array((Color.BLUE, Color.RED), "BASE")
    if action_index < 0 or action_index >= len(codec):
        raise ValueError(
            f"Teacher action index is outside the BASE codec: {action_index}"
        )
    return codec[action_index][0].name


def _validate_probability(name: str, value: float | None) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def compare_teacher_labels(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    minimum_rows: int | None = None,
    minimum_agreement: float | None = None,
    minimum_disagreement: float | None = None,
    minimum_both_road_agreement: float | None = None,
    maximum_p95_ms: float | None = None,
) -> dict[str, Any]:
    _validate_probability("minimum_agreement", minimum_agreement)
    _validate_probability("minimum_disagreement", minimum_disagreement)
    _validate_probability("minimum_both_road_agreement", minimum_both_road_agreement)
    if minimum_rows is not None and minimum_rows <= 0:
        raise ValueError("minimum_rows must be positive")
    if maximum_p95_ms is not None and maximum_p95_ms <= 0:
        raise ValueError("maximum_p95_ms must be positive")

    reference, reference_manifest, reference_manifest_path = _load_verified_iteration(
        reference_dir
    )
    candidate, candidate_manifest, candidate_manifest_path = _load_verified_iteration(
        candidate_dir
    )
    reference_config = reference_manifest.get("metadata", {}).get("config", {})
    candidate_config = candidate_manifest.get("metadata", {}).get("config", {})
    config_mismatches = {
        field: {
            "reference": reference_config.get(field),
            "candidate": candidate_config.get(field),
        }
        for field in MATCHED_CONFIG_FIELDS
        if reference_config.get(field) != candidate_config.get(field)
    }
    if config_mismatches:
        raise ValueError(f"Behavior-trajectory config mismatch: {config_mismatches}")

    for identity in ("schema", "student", "opponent"):
        reference_identity = reference_manifest.get("metadata", {}).get(identity, {})
        candidate_identity = candidate_manifest.get("metadata", {}).get(identity, {})
        identity_key = "schema_hash" if identity == "schema" else "agent_hash"
        if reference_identity.get(identity_key) != candidate_identity.get(identity_key):
            raise ValueError(f"{identity} identity mismatch between teacher datasets")

    if not reference.index.equals(candidate.index):
        missing = len(reference.index.difference(candidate.index))
        extra = len(candidate.index.difference(reference.index))
        raise ValueError(
            f"Matched decision keys differ: missing={missing} extra={extra}"
        )
    for column in ("STATE_HASH", "LEGAL_ACTIONS", "BEHAVIOR_ACTION"):
        unequal = reference[column] != candidate[column]
        if bool(unequal.any()):
            raise ValueError(
                f"Behavior trajectory drift in {column}: {int(unequal.sum())} rows"
            )

    reference_actions = [int(value) for value in reference["TEACHER_ACTION"]]
    candidate_actions = [int(value) for value in candidate["TEACHER_ACTION"]]
    rows = len(reference_actions)
    agreements = sum(
        left == right for left, right in zip(reference_actions, candidate_actions)
    )
    agreement_rate = agreements / rows if rows else float("nan")
    disagreement_rate = 1.0 - agreement_rate
    reference_families = [_action_family(value) for value in reference_actions]
    candidate_families = [_action_family(value) for value in candidate_actions]
    both_road = sum(
        left == right == "BUILD_ROAD"
        for left, right in zip(reference_families, candidate_families)
    )
    both_road_exact = sum(
        left_family == right_family == "BUILD_ROAD" and left == right
        for left_family, right_family, left, right in zip(
            reference_families,
            candidate_families,
            reference_actions,
            candidate_actions,
        )
    )
    pair_counts = Counter(zip(reference_families, candidate_families))

    reference_latency = _teacher_latency(reference_manifest)
    candidate_latency = _teacher_latency(candidate_manifest)
    observed_p95 = [
        float(value)
        for value in (
            reference_latency.get("p95_ms"),
            candidate_latency.get("p95_ms"),
        )
        if value is not None and math.isfinite(float(value))
    ]
    maximum_observed_p95_ms = max(observed_p95) if observed_p95 else None
    gates = []

    def add_gate(name: str, passed: bool, actual: Any, threshold: Any) -> None:
        gates.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": actual,
                "threshold": threshold,
            }
        )

    if minimum_rows is not None:
        add_gate("minimum_rows", rows >= minimum_rows, rows, minimum_rows)
    if minimum_agreement is not None:
        add_gate(
            "minimum_agreement",
            agreement_rate >= minimum_agreement,
            agreement_rate,
            minimum_agreement,
        )
    if minimum_disagreement is not None:
        add_gate(
            "minimum_disagreement",
            disagreement_rate >= minimum_disagreement,
            disagreement_rate,
            minimum_disagreement,
        )
    both_road_exact_agreement_rate = both_road_exact / both_road if both_road else None
    if minimum_both_road_agreement is not None:
        add_gate(
            "minimum_both_road_agreement",
            both_road_exact_agreement_rate is not None
            and both_road_exact_agreement_rate >= minimum_both_road_agreement,
            both_road_exact_agreement_rate,
            minimum_both_road_agreement,
        )
    if maximum_p95_ms is not None:
        add_gate(
            "maximum_p95_ms",
            maximum_observed_p95_ms is not None
            and maximum_observed_p95_ms <= maximum_p95_ms,
            maximum_observed_p95_ms,
            maximum_p95_ms,
        )

    return {
        "schema_version": "1.0",
        "kind": "matched_teacher_label_comparison",
        "protocol": {
            "matched_key": ["GAME_INDEX", "DECISION_INDEX"],
            "required_equal": [
                "STATE_HASH",
                "LEGAL_ACTIONS",
                "BEHAVIOR_ACTION",
            ],
            "gate_operator": "inclusive",
        },
        "datasets": {
            "reference": {
                "iteration_dir": str(reference_manifest_path.parent),
                "manifest_sha256": sha256_file(reference_manifest_path),
                "teacher": reference_manifest.get("metadata", {}).get("teacher"),
                "teacher_seed_round": reference_config.get("teacher_seed_round", 0),
                "latency": reference_latency,
            },
            "candidate": {
                "iteration_dir": str(candidate_manifest_path.parent),
                "manifest_sha256": sha256_file(candidate_manifest_path),
                "teacher": candidate_manifest.get("metadata", {}).get("teacher"),
                "teacher_seed_round": candidate_config.get("teacher_seed_round", 0),
                "latency": candidate_latency,
            },
        },
        "metrics": {
            "rows": rows,
            "agreements": agreements,
            "disagreements": rows - agreements,
            "agreement_rate": agreement_rate,
            "disagreement_rate": disagreement_rate,
            "reference_action_families": dict(
                sorted(Counter(reference_families).items())
            ),
            "candidate_action_families": dict(
                sorted(Counter(candidate_families).items())
            ),
            "family_pair_counts": {
                f"{left}->{right}": count
                for (left, right), count in sorted(pair_counts.items())
            },
            "both_choose_road_rows": both_road,
            "both_choose_road_exact_agreements": both_road_exact,
            "both_choose_road_exact_agreement_rate": both_road_exact_agreement_rate,
            "maximum_observed_p95_ms": maximum_observed_p95_ms,
        },
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates) if gates else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_teacher_labels(
        args.reference,
        args.candidate,
        minimum_rows=args.minimum_rows,
        minimum_agreement=args.minimum_agreement,
        minimum_disagreement=args.minimum_disagreement,
        minimum_both_road_agreement=args.minimum_both_road_agreement,
        maximum_p95_ms=args.maximum_p95_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    print(
        f"rows={metrics['rows']} agreement={metrics['agreement_rate']:.1%} "
        f"disagreement={metrics['disagreement_rate']:.1%} "
        f"max_p95_ms={metrics['maximum_observed_p95_ms']}"
    )
    print(f"Wrote {args.output}")
    return 0 if report["all_gates_passed"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())

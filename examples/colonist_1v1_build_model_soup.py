#!/usr/bin/env python3
"""Build a verified equal-weight soup from one MLP parent and its direct child."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catanatron.gym.colonist_training import (
    BcCheckpointMeta,
    load_bc_checkpoint_meta,
)
from catanatron.gym.model_schema import (
    checkpoint_schema_path,
    read_model_schema,
    validate_model_schema,
    write_model_schema,
)
from catanatron.gym.provenance import sha256_file
from catanatron.models.player import Color
from catanatron.players.learned import TorchBcCheckpointPlayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _checkpoint_record(path: Path) -> dict[str, str]:
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "metadata": str(path.with_suffix(".meta.json")),
        "metadata_sha256": sha256_file(path.with_suffix(".meta.json")),
        "schema": str(checkpoint_schema_path(path)),
        "schema_sha256": sha256_file(checkpoint_schema_path(path)),
    }


def main(argv: list[str] | None = None) -> int:
    import torch

    args = build_parser().parse_args(argv)
    parent = args.parent.resolve()
    child = args.child.resolve()
    output = args.output.resolve()
    if parent == child:
        raise ValueError("Model-soup parent and child must differ")
    for label, path in (("parent", parent), ("child", child)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model-soup {label}: {path}")
        for sidecar in (path.with_suffix(".meta.json"), checkpoint_schema_path(path)):
            if not sidecar.is_file():
                raise FileNotFoundError(
                    f"Missing model-soup {label} sidecar: {sidecar}"
                )

    parent_meta = load_bc_checkpoint_meta(parent.with_suffix(".meta.json"))
    child_meta = load_bc_checkpoint_meta(child.with_suffix(".meta.json"))
    if parent_meta is None or child_meta is None:
        raise ValueError("Model-soup checkpoint metadata could not be loaded")
    parent_hash = sha256_file(parent)
    child_hash = sha256_file(child)
    if child_meta.init_checkpoint_sha256 != parent_hash:
        raise ValueError(
            "Model-soup child is not a direct descendant of the declared parent: "
            f"{child_meta.init_checkpoint_sha256} != {parent_hash}"
        )
    contract_fields = (
        "architecture",
        "obs_dim",
        "n_actions",
        "hidden_sizes",
        "embedding_dim",
        "parameter_count",
    )
    contract_drift = {
        field: (getattr(parent_meta, field), getattr(child_meta, field))
        for field in contract_fields
        if getattr(parent_meta, field) != getattr(child_meta, field)
    }
    if contract_drift:
        raise ValueError(f"Model-soup architecture contract differs: {contract_drift}")
    if parent_meta.architecture != "mlp":
        raise ValueError("Equal-weight model soup currently requires aligned MLPs")

    parent_schema = read_model_schema(checkpoint_schema_path(parent))
    child_schema = read_model_schema(checkpoint_schema_path(child))
    if parent_schema is None or child_schema is None:
        raise ValueError("Model-soup schema sidecar could not be read")
    validate_model_schema(parent_schema, child_schema, context="model-soup child")

    parent_state = torch.load(parent, map_location="cpu", weights_only=True)
    child_state = torch.load(child, map_location="cpu", weights_only=True)
    if tuple(parent_state) != tuple(child_state):
        raise ValueError("Model-soup state dictionaries have different parameter keys")
    soup_state = {}
    for key, parent_tensor in parent_state.items():
        child_tensor = child_state[key]
        if parent_tensor.shape != child_tensor.shape:
            raise ValueError(
                f"Model-soup tensor shape differs for {key}: "
                f"{tuple(parent_tensor.shape)} != {tuple(child_tensor.shape)}"
            )
        if parent_tensor.dtype != child_tensor.dtype:
            raise ValueError(
                f"Model-soup tensor dtype differs for {key}: "
                f"{parent_tensor.dtype} != {child_tensor.dtype}"
            )
        if parent_tensor.is_floating_point():
            soup_state[key] = torch.lerp(parent_tensor, child_tensor, 0.5)
        else:
            if not torch.equal(parent_tensor, child_tensor):
                raise ValueError(f"Non-floating model-soup tensor differs for {key}")
            soup_state[key] = parent_tensor.clone()

    output_meta = BcCheckpointMeta(
        obs_dim=parent_meta.obs_dim,
        n_actions=parent_meta.n_actions,
        hidden_sizes=list(parent_meta.hidden_sizes),
        epochs=0,
        architecture=parent_meta.architecture,
        embedding_dim=parent_meta.embedding_dim,
        parameter_count=parent_meta.parameter_count,
        trainable_parameter_count=parent_meta.parameter_count,
        initialization_mode="equal_weight_parent_child_soup",
        init_checkpoint=str(parent),
        init_checkpoint_sha256=parent_hash,
        loss_name="equal_weight_model_soup",
        seed=parent_meta.seed,
        device="cpu",
        best_epoch=0,
        selection_metric="predeclared_equal_weight_model_soup",
        model_schema=parent_schema,
    )
    manifest_path = output.with_suffix(".soup.json")
    targets = (
        output,
        output.with_suffix(".meta.json"),
        checkpoint_schema_path(output),
        manifest_path,
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite model-soup artifacts: {existing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup_state, output)
    output_meta.save(output.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(output), parent_schema)
    TorchBcCheckpointPlayer(Color.BLUE, output)

    manifest = {
        "schema_version": "1.0",
        "kind": "equal_weight_parent_child_model_soup",
        "weights": {"parent": 0.5, "child": 0.5},
        "lineage_verified": True,
        "parent": _checkpoint_record(parent),
        "child": _checkpoint_record(child),
        "output": _checkpoint_record(output),
        "source_hashes": {"parent": parent_hash, "child": child_hash},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Built equal-weight model soup: {output} "
        f"sha256={sha256_file(output)} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

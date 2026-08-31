from __future__ import annotations

import json

import pytest

from catanatron.gym.colonist_training import BcCheckpointMeta
from catanatron.gym.model_architectures import build_bc_policy
from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    write_model_schema,
)
from catanatron.gym.provenance import sha256_file


def _write_checkpoint(path, *, fill, parent_hash=None):
    torch = pytest.importorskip("torch")
    schema = build_model_schema()
    features = schema["observation"]["features"]
    n_actions = len(schema["actions"])
    model = build_bc_policy("mlp", features, n_actions, hidden_sizes=(4, 4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(fill)
    torch.save(model.state_dict(), path)
    BcCheckpointMeta(
        obs_dim=len(features),
        n_actions=n_actions,
        hidden_sizes=[4, 4],
        epochs=1,
        architecture="mlp",
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        init_checkpoint_sha256=parent_hash,
        model_schema=schema,
    ).save(path.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(path), schema)


def test_model_soup_builder_writes_exact_midpoint_and_playable_artifact(tmp_path):
    torch = pytest.importorskip("torch")
    from catanatron.models.player import Color
    from catanatron.players.learned import TorchBcCheckpointPlayer
    from examples.colonist_1v1_build_model_soup import main

    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    output = tmp_path / "soup" / "bc.pt"
    _write_checkpoint(parent, fill=0.0)
    _write_checkpoint(child, fill=2.0, parent_hash=sha256_file(parent))

    assert (
        main(["--parent", str(parent), "--child", str(child), "--output", str(output)])
        == 0
    )

    state = torch.load(output, map_location="cpu", weights_only=True)
    assert all(
        torch.equal(tensor, torch.ones_like(tensor)) for tensor in state.values()
    )
    player = TorchBcCheckpointPlayer(Color.BLUE, output)
    assert player._inner.torch_policy is not None
    manifest = json.loads(output.with_suffix(".soup.json").read_text(encoding="utf-8"))
    assert manifest["weights"] == {"parent": 0.5, "child": 0.5}
    assert manifest["output"]["checkpoint_sha256"] == sha256_file(output)


def test_model_soup_builder_rejects_unrelated_child(tmp_path):
    from examples.colonist_1v1_build_model_soup import main

    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    _write_checkpoint(parent, fill=0.0)
    _write_checkpoint(child, fill=2.0, parent_hash="0" * 64)

    with pytest.raises(ValueError, match="not a direct descendant"):
        main(
            [
                "--parent",
                str(parent),
                "--child",
                str(child),
                "--output",
                str(tmp_path / "soup.pt"),
            ]
        )

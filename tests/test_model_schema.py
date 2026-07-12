from __future__ import annotations

import random

import numpy as np
import pytest

from catanatron.gym.model_schema import (
    build_model_schema,
    canonical_hash,
    checkpoint_schema_path,
    validate_model_schema,
    write_model_schema,
)


def test_schema_hashes_are_stable_and_profile_sensitive():
    first = build_model_schema(feature_profile="raw")
    second = build_model_schema(feature_profile="raw")
    derived = build_model_schema(feature_profile="public_derived")

    assert first == second
    assert first["schema_hash"] != derived["schema_hash"]
    assert first["feature_hash"] != derived["feature_hash"]
    assert first["action_hash"] == derived["action_hash"]


def test_validate_model_schema_rejects_semantic_drift():
    expected = build_model_schema(feature_profile="raw")
    actual = dict(expected)
    actual["action_hash"] = canonical_hash(["reordered"])

    with pytest.raises(ValueError, match="action_hash"):
        validate_model_schema(expected, actual, context="warm-start")


def test_validate_model_schema_rejects_legacy_shape_only_metadata():
    expected = build_model_schema()

    with pytest.raises(ValueError, match="missing required fields"):
        validate_model_schema(expected, {"obs_dim": 1}, context="checkpoint")


def test_torch_checkpoint_inference_uses_stored_feature_profile(tmp_path):
    torch = pytest.importorskip("torch")
    from catanatron.gym.colonist_training import BcCheckpointMeta, build_mlp_layers
    from catanatron.models.player import Color
    from catanatron.players.learned import TorchBcCheckpointPlayer

    schema = build_model_schema(feature_profile="public_derived")
    obs_dim = len(schema["observation"]["features"])
    n_actions = len(schema["actions"])
    checkpoint = tmp_path / "bc.pt"
    torch.save(build_mlp_layers(obs_dim, n_actions, (4,)).state_dict(), checkpoint)
    BcCheckpointMeta(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_sizes=[4],
        epochs=1,
        model_schema=schema,
    ).save(checkpoint.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(checkpoint), schema)

    player = TorchBcCheckpointPlayer(Color.BLUE, checkpoint)

    assert player._inner.feature_profile == "public_derived"
    assert len(player._inner.features) == obs_dim


def _assert_numpy_rng_state_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_inference_checkpoint_loaders_preserve_process_rng(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from sb3_contrib import MaskablePPO

    from catanatron.gym.colonist_training import build_mlp_layers
    from catanatron.models.player import Color
    from catanatron.players.learned import load_sb3_player, load_torch_bc_player

    schema = build_model_schema()

    class _FakeModel:
        catanatron_model_schema = schema

    def perturbing_load(*_args, **_kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        return _FakeModel()

    monkeypatch.setattr(MaskablePPO, "load", perturbing_load)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    load_sb3_player(tmp_path / "fake.zip", Color.BLUE)

    assert random.getstate() == python_state
    _assert_numpy_rng_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    obs_dim = len(schema["observation"]["features"])
    n_actions = len(schema["actions"])
    checkpoint = tmp_path / "bc.pt"
    torch.save(build_mlp_layers(obs_dim, n_actions, (4,)).state_dict(), checkpoint)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    load_torch_bc_player(
        checkpoint,
        Color.BLUE,
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_sizes=(4,),
    )

    assert random.getstate() == python_state
    _assert_numpy_rng_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)

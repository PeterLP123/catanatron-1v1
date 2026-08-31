from __future__ import annotations

import random
from types import SimpleNamespace

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


def test_torch_checkpoint_inference_loads_factored_policy_value(tmp_path):
    torch = pytest.importorskip("torch")
    from catanatron.gym.colonist_training import BcCheckpointMeta
    from catanatron.gym.model_architectures import FactoredPolicyValueNet
    from catanatron.models.player import Color
    from catanatron.players.learned import TorchBcCheckpointPlayer

    schema = build_model_schema()
    features = schema["observation"]["features"]
    n_actions = len(schema["actions"])
    checkpoint = tmp_path / "factored.pt"
    model = FactoredPolicyValueNet(features, n_actions, embedding_dim=16)
    torch.save(model.state_dict(), checkpoint)
    BcCheckpointMeta(
        obs_dim=len(features),
        n_actions=n_actions,
        hidden_sizes=[512, 512],
        epochs=1,
        architecture="factored_policy_value",
        embedding_dim=16,
        model_schema=schema,
    ).save(checkpoint.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(checkpoint), schema)

    player = TorchBcCheckpointPlayer(Color.BLUE, checkpoint)

    assert isinstance(player._inner.torch_policy, FactoredPolicyValueNet)
    logits = player._inner.torch_policy(torch.zeros(1, len(features)))
    assert logits.shape == (1, n_actions)


def test_torch_checkpoint_inference_loads_action_conditioned_policy(tmp_path):
    torch = pytest.importorskip("torch")
    from catanatron.gym.colonist_training import BcCheckpointMeta
    from catanatron.gym.model_architectures import ActionConditionedScorer
    from catanatron.models.player import Color
    from catanatron.players.learned import TorchBcCheckpointPlayer

    schema = build_model_schema()
    features = schema["observation"]["features"]
    n_actions = len(schema["actions"])
    checkpoint = tmp_path / "action-conditioned.pt"
    model = ActionConditionedScorer(
        len(features), n_actions, hidden_sizes=(8, 8), embedding_dim=16
    )
    torch.save(model.state_dict(), checkpoint)
    BcCheckpointMeta(
        obs_dim=len(features),
        n_actions=n_actions,
        hidden_sizes=[8, 8],
        epochs=1,
        architecture="action_conditioned",
        embedding_dim=16,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        model_schema=schema,
    ).save(checkpoint.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(checkpoint), schema)

    player = TorchBcCheckpointPlayer(Color.BLUE, checkpoint)

    assert isinstance(player._inner.torch_policy, ActionConditionedScorer)
    logits = player._inner.torch_policy(torch.zeros(1, len(features)))
    assert logits.shape == (1, n_actions)


def test_outcome_reranker_manifest_loads_and_hash_checks(tmp_path):
    import json

    torch = pytest.importorskip("torch")
    from catanatron.gym.colonist_training import BcCheckpointMeta
    from catanatron.gym.model_architectures import (
        FactoredOutcomeCritic,
        build_bc_policy,
    )
    from catanatron.gym.provenance import sha256_file
    from catanatron.models.player import Color
    from catanatron.players.learned import OutcomeRerankerCheckpointPlayer

    schema = build_model_schema()
    sample_features = schema["observation"]["features"]
    critic_features = [f"F_{name}" for name in sample_features]
    n_actions = len(schema["actions"])
    policy_path = tmp_path / "policy.pt"
    policy = build_bc_policy("mlp", sample_features, n_actions, hidden_sizes=(8, 8))
    torch.save(policy.state_dict(), policy_path)
    BcCheckpointMeta(
        obs_dim=len(sample_features),
        n_actions=n_actions,
        hidden_sizes=[8, 8],
        epochs=1,
        model_schema=schema,
    ).save(policy_path.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(policy_path), schema)

    critic_path = tmp_path / "critic.pt"
    critic = FactoredOutcomeCritic(critic_features, embedding_dim=16)
    torch.save(critic.state_dict(), critic_path)
    critic_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "kind": "factored_outcome_critic",
                "feature_columns": critic_features,
                "embedding_dim": 16,
            }
        ),
        encoding="utf-8",
    )
    write_model_schema(checkpoint_schema_path(critic_path), schema)
    manifest = tmp_path / "reranker.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "outcome_critic_reranker",
                "policy_checkpoint": policy_path.name,
                "policy_checkpoint_sha256": sha256_file(policy_path),
                "policy_metadata_sha256": sha256_file(
                    policy_path.with_suffix(".meta.json")
                ),
                "policy_schema_sha256": sha256_file(
                    policy_path.with_suffix(".schema.json")
                ),
                "critic_checkpoint": critic_path.name,
                "critic_checkpoint_sha256": sha256_file(critic_path),
                "critic_metadata_sha256": sha256_file(
                    critic_path.with_suffix(".meta.json")
                ),
                "critic_schema_sha256": sha256_file(
                    critic_path.with_suffix(".schema.json")
                ),
                "top_k": 3,
                "minimum_win_probability_improvement": 0.05,
                "chance_handling": "public_only_spectrum_with_policy_fallback",
            }
        ),
        encoding="utf-8",
    )

    player = OutcomeRerankerCheckpointPlayer(Color.BLUE, manifest)
    assert player.top_k == 3
    assert player.minimum_win_probability_improvement == 0.05
    assert player.stats_summary()["choice_decisions"] == 0

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["chance_handling"] = "exact_execute_spectrum_expectation"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hidden-safe chance handling"):
        OutcomeRerankerCheckpointPlayer(Color.BLUE, manifest)

    manifest_payload["chance_handling"] = "public_only_spectrum_with_policy_fallback"
    manifest_payload["critic_checkpoint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        OutcomeRerankerCheckpointPlayer(Color.BLUE, manifest)

    manifest_payload["critic_checkpoint_sha256"] = sha256_file(critic_path)
    critic_path.with_suffix(".meta.json").write_text("{}\n", encoding="utf-8")
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        OutcomeRerankerCheckpointPlayer(Color.BLUE, manifest)


def test_outcome_reranker_falls_back_outside_public_chance_boundary():
    from catanatron.models.enums import Action, ActionType
    from catanatron.models.player import Color
    from catanatron.players.learned import OutcomeRerankerCheckpointPlayer

    fallback = Action(Color.BLUE, ActionType.END_TURN, None)
    player = object.__new__(OutcomeRerankerCheckpointPlayer)
    player.color = Color.BLUE
    player.policy = SimpleNamespace(decide=lambda game, actions: fallback)
    player.decision_stats = {
        "decisions": 0,
        "choice_decisions": 0,
        "reranked_decisions": 0,
        "fallback_decisions": 0,
        "candidate_actions_evaluated": 0,
        "latencies_ms": [],
        "accepted_improvements": [],
    }
    player.last_decision_stats = None
    actions = [fallback, Action(Color.BLUE, ActionType.MOVE_ROBBER, (0, 0, 0))]

    assert player.decide(SimpleNamespace(), actions) == fallback
    assert player.stats_summary()["fallback_decisions"] == 1
    assert player.last_decision_stats == {
        "reranked": False,
        "fallback": "outside_public_chance_boundary",
    }


def test_outcome_reranker_uses_only_public_successor_spectrum(monkeypatch):
    from catanatron.models.enums import Action, ActionType
    from catanatron.models.player import Color
    from catanatron.players.learned import OutcomeRerankerCheckpointPlayer

    player = object.__new__(OutcomeRerankerCheckpointPlayer)
    player.color = Color.BLUE
    action = Action(Color.BLUE, ActionType.END_TURN, None)
    outcome = SimpleNamespace(winning_color=lambda: Color.BLUE)
    calls = []

    def public_spectrum(game, selected):
        calls.append((game, selected))
        return ((outcome, 1.0),)

    monkeypatch.setattr(
        "catanatron.players.visible_chance_puct.public_action_spectrum",
        public_spectrum,
    )
    monkeypatch.setattr(
        "catanatron.players.tree_search_utils.execute_spectrum",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("omniscient spectrum must not run")
        ),
    )
    game = SimpleNamespace()

    assert player._critic_action_value(game, action) == 1.0
    assert calls == [(game, action)]


@pytest.mark.parametrize(
    "builder",
    [
        "examples.colonist_1v1_build_reranker",
        "examples.colonist_1v1_build_opening_specialist",
    ],
)
def test_manifest_builders_refuse_existing_outputs(tmp_path, builder):
    import importlib

    output = tmp_path / "existing.json"
    output.write_text("keep\n", encoding="utf-8")
    main = importlib.import_module(builder).main
    argv = ["--policy", str(tmp_path / "missing.pt"), "--output", str(output)]
    if builder.endswith("build_reranker"):
        argv[2:2] = ["--critic", str(tmp_path / "missing-critic.pt")]

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(argv)
    assert output.read_text(encoding="utf-8") == "keep\n"


def test_torch_checkpoint_inference_loads_spatial_edge_residual(tmp_path):
    torch = pytest.importorskip("torch")
    from catanatron.gym.colonist_training import BcCheckpointMeta
    from catanatron.gym.model_architectures import SpatialEdgeResidualPolicy
    from catanatron.models.player import Color
    from catanatron.players.learned import TorchBcCheckpointPlayer

    schema = build_model_schema()
    features = schema["observation"]["features"]
    n_actions = len(schema["actions"])
    checkpoint = tmp_path / "spatial-road.pt"
    model = SpatialEdgeResidualPolicy(
        features, n_actions, hidden_sizes=(8, 8), embedding_dim=16
    )
    model.freeze_base_policy()
    torch.save(model.state_dict(), checkpoint)
    BcCheckpointMeta(
        obs_dim=len(features),
        n_actions=n_actions,
        hidden_sizes=[8, 8],
        epochs=1,
        architecture="spatial_edge_residual",
        embedding_dim=16,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameter_count=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        base_policy_frozen=True,
        initialization_mode="mlp_base_policy",
        model_schema=schema,
    ).save(checkpoint.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(checkpoint), schema)

    player = TorchBcCheckpointPlayer(Color.BLUE, checkpoint)

    assert isinstance(player._inner.torch_policy, SpatialEdgeResidualPolicy)
    logits = player._inner.torch_policy(torch.zeros(1, len(features)))
    assert logits.shape == (1, n_actions)


def test_torch_checkpoint_inference_loads_spatial_robber_residual(tmp_path):
    torch = pytest.importorskip("torch")
    from catanatron.gym.colonist_training import BcCheckpointMeta
    from catanatron.gym.model_architectures import SpatialRobberResidualPolicy
    from catanatron.models.player import Color
    from catanatron.players.learned import TorchBcCheckpointPlayer

    schema = build_model_schema()
    features = schema["observation"]["features"]
    n_actions = len(schema["actions"])
    checkpoint = tmp_path / "spatial-robber.pt"
    model = SpatialRobberResidualPolicy(
        features, n_actions, hidden_sizes=(8, 8), embedding_dim=16
    )
    model.freeze_base_policy()
    torch.save(model.state_dict(), checkpoint)
    BcCheckpointMeta(
        obs_dim=len(features),
        n_actions=n_actions,
        hidden_sizes=[8, 8],
        epochs=1,
        architecture="spatial_robber_residual",
        embedding_dim=16,
        base_policy_frozen=True,
        initialization_mode="mlp_base_policy",
        model_schema=schema,
    ).save(checkpoint.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(checkpoint), schema)

    player = TorchBcCheckpointPlayer(Color.BLUE, checkpoint)

    assert isinstance(player._inner.torch_policy, SpatialRobberResidualPolicy)
    logits = player._inner.torch_policy(torch.zeros(1, len(features)))
    assert logits.shape == (1, n_actions)


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

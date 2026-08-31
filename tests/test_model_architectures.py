from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catanatron.gym.model_architectures import (  # noqa: E402
    ActionConditionedScorer,
    BoardTensorEncoder,
    FactoredPolicyValueNet,
    SpatialEdgeResidualPolicy,
    SpatialRobberResidualPolicy,
    build_bc_policy,
    factored_feature_groups,
)
from catanatron.gym.model_schema import build_model_schema  # noqa: E402


def test_action_conditioned_scorer_supports_full_and_legal_only_scores():
    model = ActionConditionedScorer(12, 20, hidden_sizes=(16,), embedding_dim=8)
    observations = torch.randn(3, 12)
    legal = torch.tensor([[1, 4, 9], [2, 3, 5], [0, 8, 19]])

    full = model(observations)
    candidates = model(observations, legal)

    assert full.shape == (3, 20)
    assert candidates.shape == (3, 3)
    torch.testing.assert_close(candidates, torch.gather(full, 1, legal))


def test_action_conditioned_builder_matches_run52_capacity_contract():
    schema = build_model_schema()
    features = schema["observation"]["features"]
    n_actions = len(schema["actions"])

    model = build_bc_policy(
        "action_conditioned",
        features,
        n_actions,
        hidden_sizes=(512, 512),
        embedding_dim=128,
    )

    assert isinstance(model, ActionConditionedScorer)
    assert sum(parameter.numel() for parameter in model.parameters()) == 686_028
    observations = torch.randn(2, len(features))
    legal = torch.tensor([[1, 4, 9], [2, 3, 5]])
    torch.testing.assert_close(
        model(observations, legal), torch.gather(model(observations), 1, legal)
    )


def test_board_tensor_encoder_fuses_board_and_numeric_state():
    encoder = BoardTensorEncoder(7, 15, output_dim=32)
    encoded = encoder(torch.randn(4, 7, 21, 11), torch.randn(4, 15))

    assert encoded.shape == (4, 32)


def test_factored_policy_value_uses_schema_groups_and_legal_action_scores():
    features = build_model_schema()["observation"]["features"]
    groups = factored_feature_groups(features)
    assert set(groups) == {"edges", "nodes", "tiles", "ports", "global"}
    assert sum(map(len, groups.values())) == len(features)

    model = FactoredPolicyValueNet(features, 332, embedding_dim=32)
    observations = torch.randn(2, len(features))
    legal = torch.tensor([[1, 4, 9], [2, 3, 5]])
    full = model.policy_value(observations)
    candidates = model(observations, legal)

    assert full.policy_logits.shape == (2, 332)
    assert full.win_value.shape == full.vp_margin.shape == (2,)
    torch.testing.assert_close(candidates, torch.gather(full.policy_logits, 1, legal))
    (
        full.policy_logits.mean() + full.win_value.mean() + full.vp_margin.mean()
    ).backward()
    assert all(
        any(parameter.grad is not None for parameter in encoder.parameters())
        for encoder in model.encoders.values()
    )


def test_spatial_edge_residual_starts_exactly_at_base_and_only_changes_roads():
    schema = build_model_schema()
    features = schema["observation"]["features"]
    n_actions = len(schema["actions"])
    base = build_bc_policy("mlp", features, n_actions, hidden_sizes=(8, 8))
    model = SpatialEdgeResidualPolicy(
        features, n_actions, hidden_sizes=(8, 8), embedding_dim=16
    )
    model.load_base_policy_state_dict(base.state_dict())
    observations = torch.randn(3, len(features))

    torch.testing.assert_close(model(observations), base(observations), rtol=0, atol=0)
    assert len(model.road_action_indices) == 72
    assert model.edge_feature_indices.shape == (72, 2)
    assert model.endpoint_node_feature_indices.shape == (72, 8)

    with torch.no_grad():
        model.delta_head.bias.fill_(1.0)
    residual_logits = model(observations)
    base_logits = base(observations)
    non_road = torch.ones(n_actions, dtype=torch.bool)
    non_road[model.road_action_indices] = False
    torch.testing.assert_close(
        residual_logits[:, non_road], base_logits[:, non_road], rtol=0, atol=0
    )
    torch.testing.assert_close(
        residual_logits[:, model.road_action_indices],
        base_logits[:, model.road_action_indices] + 1.0,
    )

    model.freeze_base_policy()
    assert not any(
        parameter.requires_grad for parameter in model.base_policy.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.delta_head.parameters())


def test_spatial_robber_residual_starts_exactly_at_base_and_only_changes_robber():
    schema = build_model_schema()
    features = schema["observation"]["features"]
    n_actions = len(schema["actions"])
    base = build_bc_policy("mlp", features, n_actions, hidden_sizes=(8, 8))
    model = SpatialRobberResidualPolicy(
        features, n_actions, hidden_sizes=(8, 8), embedding_dim=16
    )
    model.load_base_policy_state_dict(base.state_dict())
    observations = torch.randn(3, len(features))

    torch.testing.assert_close(model(observations), base(observations), rtol=0, atol=0)
    assert len(model.robber_action_indices) == 57
    assert model.robber_tile_feature_indices.shape == (57, 8)
    assert torch.bincount(model.robber_tile_positions).tolist() == [3] * 19
    assert float(model.robber_victim_present.sum()) == 38.0

    with torch.no_grad():
        model.delta_head.bias.fill_(1.0)
    residual_logits = model(observations)
    base_logits = base(observations)
    non_robber = torch.ones(n_actions, dtype=torch.bool)
    non_robber[model.robber_action_indices] = False
    torch.testing.assert_close(
        residual_logits[:, non_robber], base_logits[:, non_robber], rtol=0, atol=0
    )
    torch.testing.assert_close(
        residual_logits[:, model.robber_action_indices],
        base_logits[:, model.robber_action_indices] + 1.0,
    )

    model.freeze_base_policy()
    assert not any(
        parameter.requires_grad for parameter in model.base_policy.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.delta_head.parameters())

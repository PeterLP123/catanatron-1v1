from types import SimpleNamespace
from unittest.mock import MagicMock
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catanatron.gym.bc_training import (
    DecisionMetricAccumulator,
    ParquetDecisionBatches,
    candidate_listwise_loss,
    hash_parquet_shards,
    inspect_parquet_dataset,
    legal_masked_cross_entropy,
    padded_decision_columns,
)
from catanatron.gym.colonist_training import warmstart_bc_into_maskable_ppo
from catanatron.gym.model_schema import build_model_schema, write_model_schema
from examples.colonist_1v1_bc import _batch_loss, _resolve_dataset_paths


def test_legal_masked_cross_entropy_ignores_illegal_logits():
    logits = torch.tensor([[3.0, 1.0, 100.0]], requires_grad=True)
    target = torch.tensor([0])
    legal, mask, _, _ = padded_decision_columns([[0, 1]])

    loss = legal_masked_cross_entropy(logits, target, legal, mask)
    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[3.0, 1.0]]), torch.tensor([0])
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[0, 2] == 0


def test_parquet_shard_hash_binds_exact_bytes_but_not_directory(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first = left / "a.parquet"
    second = right / "b.parquet"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    rows, combined = hash_parquet_shards([first, second], progress=False)
    _, reordered = hash_parquet_shards([second, first], progress=False)
    assert combined == reordered
    assert {row["path"] for row in rows} == {str(first), str(second)}
    assert all(len(row["sha256"]) == 64 for row in rows)

    second.write_bytes(b"changed")
    _, changed = hash_parquet_shards([first, second], progress=False)
    assert changed != combined


def test_legal_masked_cross_entropy_rejects_corrupt_target():
    legal, mask, _, _ = padded_decision_columns([[0, 1]])
    with pytest.raises(ValueError, match="Target action"):
        legal_masked_cross_entropy(torch.zeros((1, 3)), torch.tensor([2]), legal, mask)


def test_listwise_loss_normalizes_value_scale_and_preserves_ties():
    legal, mask, values_a, value_mask = padded_decision_columns(
        [[0, 1, 2]], [[1.0, 2.0, 2.0]]
    )
    _, _, values_b, _ = padded_decision_columns([[0, 1, 2]], [[101.0, 201.0, 201.0]])
    logits = torch.tensor([[0.0, 1.0, 1.0]])
    loss_a, valid_a = candidate_listwise_loss(
        logits, legal, mask, values_a, value_mask, temperature=0.5
    )
    loss_b, valid_b = candidate_listwise_loss(
        logits, legal, mask, values_b, value_mask, temperature=0.5
    )
    assert valid_a.tolist() == [True]
    assert valid_b.tolist() == [True]
    assert torch.allclose(loss_a, loss_b)

    # Equal candidate values produce a uniform target, so equal predicted
    # logits beat an arbitrary preference for one tied action.
    _, _, tied_values, tied_mask = padded_decision_columns([[0, 1]], [[5.0, 5.0]])
    equal_loss, _ = candidate_listwise_loss(
        torch.tensor([[0.0, 0.0]]),
        legal[:, :2],
        mask[:, :2],
        tied_values,
        tied_mask,
    )
    skewed_loss, _ = candidate_listwise_loss(
        torch.tensor([[5.0, -5.0]]),
        legal[:, :2],
        mask[:, :2],
        tied_values,
        tied_mask,
    )
    assert equal_loss < skewed_loss


def test_hybrid_loss_adds_weighted_listwise_regularizer():
    legal, legal_mask, values, value_mask = padded_decision_columns(
        [[0, 1]], [[0.0, 1.0]]
    )
    batch = {
        "features": torch.tensor([[2.0, 0.0]]),
        "targets": torch.tensor([0]),
        "legal_indices": legal,
        "legal_mask": legal_mask,
        "candidate_values": values,
        "candidate_mask": value_mask,
        "sample_weights": torch.ones(1),
    }
    net = torch.nn.Identity()
    args = SimpleNamespace(
        listwise_temperature=0.05,
        tie_tolerance=1e-6,
        hybrid_listwise_weight=0.0,
    )

    legal_loss, _, legal_rows = _batch_loss(net, batch, "legal_ce", "cpu", args)
    zero_weight_loss, _, hybrid_rows = _batch_loss(net, batch, "hybrid", "cpu", args)
    assert legal_rows == hybrid_rows == 1
    assert torch.allclose(legal_loss, zero_weight_loss)

    args.hybrid_listwise_weight = 0.1
    hybrid_loss, _, _ = _batch_loss(net, batch, "hybrid", "cpu", args)
    assert hybrid_loss > legal_loss


def test_parquet_batches_split_whole_games_and_stream_batches(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    paths = []
    for game in range(4):
        path = tmp_path / f"game-{game}.parquet"
        pd.DataFrame(
            {
                "F_A": [float(game), float(game + 1)],
                "F_B": [1.0, 2.0],
                "ACTION": [0, 1],
                "ACTION_TYPE": [0, 1],
                "GAME_ID": [f"g{game}", f"g{game}"],
                "NUM_LEGAL": [2, 2],
                "LEGAL_ACTIONS": [[0, 1], [0, 1]],
                "CANDIDATE_VALUES": [[0.1, 0.9], [0.8, 0.2]],
            }
        ).to_parquet(path)
        paths.append(path)

    plan = inspect_parquet_dataset(paths, val_fraction=0.25, test_fraction=0.25, seed=7)
    assert plan.train_groups.isdisjoint(plan.val_groups)
    assert plan.train_groups.isdisjoint(plan.test_groups)
    assert plan.val_groups.isdisjoint(plan.test_groups)
    assert plan.rows_for("train") == 4
    assert plan.rows_for("val") == 2
    assert plan.rows_for("test") == 2

    dataset = ParquetDecisionBatches(plan, "train", batch_size=1, seed=3, shuffle=True)
    batches = list(dataset.loader())
    assert len(batches) == 4
    assert all(batch["features"].shape == (1, 2) for batch in batches)
    assert all(batch["has_decision_metadata"] for batch in batches)


def test_small_dataset_allocates_validation_or_requires_explicit_opt_out(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    paths = []
    for game_id in ("g0", "g1"):
        path = tmp_path / f"{game_id}.parquet"
        pd.DataFrame({"F_A": [1.0], "ACTION": [0], "GAME_ID": [game_id]}).to_parquet(
            path
        )
        paths.append(path)

    plan = inspect_parquet_dataset(paths, val_fraction=0.1, seed=2)
    assert len(plan.train_groups) == 1
    assert len(plan.val_groups) == 1

    with pytest.raises(ValueError, match="explicitly set"):
        inspect_parquet_dataset(paths[:1], val_fraction=0.1, seed=2)
    no_holdout = inspect_parquet_dataset(paths[:1], val_fraction=0.0, seed=2)
    assert len(no_holdout.train_groups) == 1
    assert not no_holdout.val_groups


def test_online_decision_metrics_do_not_need_full_logit_matrix():
    accumulator = DecisionMetricAccumulator(topk=(1, 2))
    accumulator.update(
        np.array([[0.0, 9.0, 20.0], [4.0, 0.0, 1.0]]),
        np.array([1, 2]),
        num_legal=np.array([2, 2]),
        legal_actions=[[0, 1], [0, 2]],
        candidate_values=[[0.2, 0.8], [0.1, 0.9]],
    )
    metrics = accumulator.compute()
    assert metrics["rows"] == 2
    assert metrics["legal_choice_accuracy"] == 0.5
    assert metrics["mean_regret"] == 0.5


def _policy(obs_dim=4, hidden=3, actions=2):
    from torch import nn

    policy = MagicMock()
    policy.mlp_extractor.policy_net = nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
    )
    policy.action_net = nn.Linear(hidden, actions)
    return policy


def test_bc_warmstart_is_full_and_atomic_on_failure():
    from torch import nn

    source = nn.Sequential(
        nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 2)
    ).state_dict()
    policy = _policy()
    assert warmstart_bc_into_maskable_ppo(policy, source) == 6

    broken = dict(source)
    broken.pop("4.bias")
    policy = _policy()
    before = policy.mlp_extractor.policy_net[0].weight.detach().clone()
    with pytest.raises(ValueError, match="missing required tensor"):
        warmstart_bc_into_maskable_ppo(policy, broken)
    assert torch.equal(before, policy.mlp_extractor.policy_net[0].weight)


def test_bc_warmstart_rejects_schema_mismatch_before_copy():
    from torch import nn

    source = nn.Sequential(
        nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 2)
    ).state_dict()
    schema = {
        "schema_version": 1,
        "feature_hash": "features-a",
        "action_hash": "actions",
        "rules_hash": "rules",
    }
    mismatch = {**schema, "feature_hash": "features-b"}
    with pytest.raises(ValueError, match="schema mismatch"):
        warmstart_bc_into_maskable_ppo(
            _policy(),
            source,
            checkpoint_schema=mismatch,
            expected_schema=schema,
        )


@pytest.mark.parametrize("hash_name", ["action_hash", "rules_hash"])
def test_bc_rejects_dataset_action_or_rules_schema_drift(tmp_path, hash_name):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "game.parquet").write_bytes(b"fixture")
    (dataset / "dataset_meta.json").write_text(
        json.dumps({"status": "complete", "num_games": 1}), encoding="utf-8"
    )
    expected = build_model_schema()
    incompatible = dict(expected)
    incompatible[hash_name] = f"wrong-{hash_name}"
    write_model_schema(dataset / "dataset_schema.json", incompatible)

    with pytest.raises(ValueError, match=hash_name):
        _resolve_dataset_paths([dataset], expected_schema=expected)

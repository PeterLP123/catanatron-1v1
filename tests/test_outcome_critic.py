from __future__ import annotations

import pytest

from catanatron.gym.model_architectures import FactoredOutcomeCritic
from catanatron.gym.outcome_critic import (
    OutcomeCriticBatches,
    OutcomeMetricAccumulator,
    build_outcome_dataset_plan,
    critic_gate_deltas,
    outcome_critic_loss,
)


FEATURES = (
    "F_EDGE(0, 1)_P0_ROAD",
    "F_NODE0_P0_SETTLEMENT",
    "F_TILE0_IS_WOOD",
    "F_PORT0_IS_WOOD",
    "F_P0_PUBLIC_VPS",
    "F_P1_PUBLIC_VPS",
)


def test_factored_outcome_critic_shapes_and_loss():
    torch = pytest.importorskip("torch")
    model = FactoredOutcomeCritic(FEATURES, embedding_dim=16)
    features = torch.randn(3, len(FEATURES))
    output = model(features)
    assert output.win_logit.shape == (3,)
    assert output.vp_margin.shape == (3,)
    batch = {
        "win_targets": torch.tensor([1.0, 0.0, 1.0]),
        "win_mask": torch.tensor([True, True, True]),
        "margin_targets": torch.tensor([3.0, -2.0, 0.0]),
        "margin_mask": torch.tensor([True, True, False]),
    }
    loss, win_loss, margin_loss = outcome_critic_loss(output, batch, margin_weight=0.05)
    assert loss.item() > 0
    assert win_loss.item() > 0
    assert margin_loss.item() > 0
    loss.backward()
    assert model.win_head.weight.grad is not None
    assert model.vp_margin_head.weight.grad is not None


def test_outcome_batches_preserve_whole_game_splits_and_targets(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = tmp_path / "corpus"
    root.mkdir()
    rows = []
    for game in range(10):
        won = game % 2 == 0
        row = {
            "GAME_ID": f"g{game}",
            "SEAT": game % 2,
            "ACTION": 0,
            "RETURN": 1.0 if won else -1.0,
            "VICTORY_POINT_MARGIN_RETURN": 4.0 if won else -3.0,
        }
        row.update({feature: 0.0 for feature in FEATURES})
        row["F_P0_PUBLIC_VPS"] = 8.0 if won else 4.0
        row["F_P1_PUBLIC_VPS"] = 4.0 if won else 8.0
        rows.append(row)
    pd.DataFrame(rows).to_parquet(root / "shard.parquet", index=False)

    plan, targets = build_outcome_dataset_plan(
        [[root]], val_fraction=0.2, test_fraction=0.2, seed=7
    )
    assert plan.train_groups.isdisjoint(plan.val_groups)
    assert plan.train_groups.isdisjoint(plan.test_groups)
    assert plan.val_groups.isdisjoint(plan.test_groups)
    batches = list(
        OutcomeCriticBatches(plan, targets, "test", batch_size=4, seed=7).loader()
    )
    assert sum(len(batch["features"]) for batch in batches) == plan.rows_for("test")
    assert all(bool(batch["win_mask"].all()) for batch in batches)
    assert all(bool(batch["margin_mask"].all()) for batch in batches)


def test_outcome_metrics_report_critic_minus_public_deltas():
    torch = pytest.importorskip("torch")
    accumulator = OutcomeMetricAccumulator()
    output = type(
        "Output",
        (),
        {
            "win_logit": torch.tensor([4.0, -4.0, 3.0, -3.0]),
            "vp_margin": torch.tensor([5.0, -5.0, 4.0, -4.0]),
        },
    )()
    batch = {
        "features": torch.zeros(4, 1),
        "win_targets": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "win_mask": torch.tensor([True, True, True, True]),
        "margin_targets": torch.tensor([5.0, -5.0, 4.0, -4.0]),
        "margin_mask": torch.tensor([True, True, True, True]),
        "public_vp_diff": torch.tensor([-1.0, 1.0, -0.5, 0.5]),
    }
    accumulator.update(output, batch)
    metrics = accumulator.finalize()
    deltas = critic_gate_deltas(metrics)
    assert metrics["win"]["auc"] == 1.0
    assert deltas["win_auc"] > 0
    assert deltas["win_brier"] < 0
    assert deltas["vp_margin_mae"] < 0
    assert deltas["vp_margin_rmse"] < 0

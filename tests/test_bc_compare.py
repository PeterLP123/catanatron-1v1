import pytest

torch = pytest.importorskip("torch")

from catanatron.gym.colonist_training import BcCheckpointMeta
from catanatron.gym.model_architectures import build_bc_policy
from catanatron.gym.model_schema import build_model_schema, write_model_schema
from examples.colonist_1v1_bc_compare import (
    _load_checkpoint,
    _numeric_metric_deltas,
)


def test_numeric_metric_deltas_skip_nested_non_finite_and_boolean_values():
    candidate = {
        "mean_regret": 0.2,
        "rows": 10,
        "passed": True,
        "nested": {"x": 1},
        "non_finite": float("nan"),
    }
    baseline = {
        "mean_regret": 0.3,
        "rows": 8,
        "passed": False,
        "nested": {"x": 2},
        "non_finite": 1.0,
    }

    assert _numeric_metric_deltas(candidate, baseline) == pytest.approx(
        {"mean_regret": -0.1, "rows": 2.0}
    )


def test_load_checkpoint_verifies_schema_shape_and_parameter_count(tmp_path):
    schema = build_model_schema()
    feature_columns = tuple(f"F_{name}" for name in schema["observation"]["features"])
    checkpoint = tmp_path / "bc.pt"
    net = build_bc_policy(
        "mlp", feature_columns, len(schema["actions"]), hidden_sizes=(8, 8)
    )
    torch.save(net.state_dict(), checkpoint)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    BcCheckpointMeta(
        obs_dim=len(feature_columns),
        n_actions=len(schema["actions"]),
        hidden_sizes=[8, 8],
        epochs=1,
        parameter_count=parameter_count,
        model_schema=schema,
    ).save(checkpoint.with_suffix(".meta.json"))
    write_model_schema(checkpoint.with_suffix(".schema.json"), schema)

    loaded, record = _load_checkpoint(
        checkpoint,
        expected_schema=schema,
        feature_columns=feature_columns,
        device=torch.device("cpu"),
    )

    assert record["parameter_count"] == parameter_count
    assert record["sha256"]
    assert all(
        torch.equal(loaded.state_dict()[name], value)
        for name, value in net.state_dict().items()
    )

    metadata = checkpoint.with_suffix(".meta.json")
    bad_meta = BcCheckpointMeta(
        obs_dim=len(feature_columns) - 1,
        n_actions=len(schema["actions"]),
        hidden_sizes=[8, 8],
        epochs=1,
        parameter_count=parameter_count,
        model_schema=schema,
    )
    bad_meta.save(metadata)
    with pytest.raises(ValueError, match="obs_dim"):
        _load_checkpoint(
            checkpoint,
            expected_schema=schema,
            feature_columns=feature_columns,
            device=torch.device("cpu"),
        )

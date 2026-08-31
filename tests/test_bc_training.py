import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catanatron.gym.bc_training import (
    DecisionMetricAccumulator,
    ParquetDecisionBatches,
    action_type_indices_from_full_actions,
    candidate_listwise_loss,
    combine_parquet_dataset_plans,
    hash_parquet_shards,
    inspect_parquet_corpora,
    inspect_parquet_dataset,
    legal_masked_cross_entropy,
    padded_decision_columns,
    seed_everything,
)
from catanatron.gym.colonist_training import (
    BcCheckpointMeta,
    hard_state_sample_weights,
    outcome_deficit_sample_weights,
    warmstart_bc_into_maskable_ppo,
)
from catanatron.gym.distillation import DistillationDatasetWriter
from catanatron.gym.envs.action_space import get_action_array, get_action_type_array
from catanatron.gym.model_schema import build_model_schema, write_model_schema
from catanatron.gym.model_architectures import PolicyValueOutput, build_bc_policy
from catanatron.models.enums import ActionType
from catanatron.models.player import Color
from examples.colonist_1v1_bc import (
    _batch_loss,
    _load_initial_checkpoint,
    _resolve_dataset_paths,
    _validate_dataset_contract,
    build_parser,
)


def test_deterministic_seed_configures_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    seed_everything(17)

    assert __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_initial_checkpoint_requires_exact_model_contract(tmp_path):
    schema = build_model_schema()
    features = tuple(f"F_{name}" for name in schema["observation"]["features"])
    checkpoint = tmp_path / "parent.pt"
    net = torch.nn.Sequential(
        torch.nn.Linear(len(features), 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, len(schema["actions"])),
    )
    torch.save(net.state_dict(), checkpoint)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    BcCheckpointMeta(
        obs_dim=len(features),
        n_actions=len(schema["actions"]),
        hidden_sizes=[8],
        epochs=1,
        parameter_count=parameter_count,
        model_schema=schema,
    ).save(checkpoint.with_suffix(".meta.json"))
    write_model_schema(checkpoint.with_suffix(".schema.json"), schema)

    child = torch.nn.Sequential(
        torch.nn.Linear(len(features), 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, len(schema["actions"])),
    )
    record = _load_initial_checkpoint(
        checkpoint,
        child,
        expected_schema=schema,
        architecture="mlp",
        hidden_sizes=(8,),
        embedding_dim=128,
        obs_dim=len(features),
        n_actions=len(schema["actions"]),
        device=torch.device("cpu"),
    )
    assert record["sha256"]
    assert all(
        torch.equal(child.state_dict()[name], value)
        for name, value in net.state_dict().items()
    )

    with pytest.raises(ValueError, match="architecture contract mismatch"):
        _load_initial_checkpoint(
            checkpoint,
            child,
            expected_schema=schema,
            architecture="mlp",
            hidden_sizes=(16,),
            embedding_dim=128,
            obs_dim=len(features),
            n_actions=len(schema["actions"]),
            device=torch.device("cpu"),
        )

    spatial = build_bc_policy(
        "spatial_edge_residual",
        features,
        len(schema["actions"]),
        hidden_sizes=(8,),
        embedding_dim=16,
    )
    spatial_record = _load_initial_checkpoint(
        checkpoint,
        spatial,
        expected_schema=schema,
        architecture="spatial_edge_residual",
        hidden_sizes=(8,),
        embedding_dim=16,
        obs_dim=len(features),
        n_actions=len(schema["actions"]),
        device=torch.device("cpu"),
    )
    observations = torch.randn(2, len(features))
    torch.testing.assert_close(spatial(observations), net(observations), rtol=0, atol=0)
    assert spatial_record["initialization_mode"] == "mlp_base_policy"

    robber = build_bc_policy(
        "spatial_robber_residual",
        features,
        len(schema["actions"]),
        hidden_sizes=(8,),
        embedding_dim=16,
    )
    robber_record = _load_initial_checkpoint(
        checkpoint,
        robber,
        expected_schema=schema,
        architecture="spatial_robber_residual",
        hidden_sizes=(8,),
        embedding_dim=16,
        obs_dim=len(features),
        n_actions=len(schema["actions"]),
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(robber(observations), net(observations), rtol=0, atol=0)
    assert robber_record["initialization_mode"] == "mlp_base_policy"


def test_bc_parser_accumulates_repeated_corpus_options():
    args = build_parser().parse_args(
        [
            "--data-dir",
            "base-a",
            "--data-dir",
            "base-b",
            "base-c",
            "--augmentation-data-dir",
            "dagger-a",
            "--augmentation-data-dir",
            "dagger-b",
            "--outcome-weighted-augmentation-data-dir",
            "outcome-a",
            "--outcome-weighted-augmentation-data-dir",
            "outcome-b",
        ]
    )

    assert args.data_dir == [Path("base-a"), Path("base-b"), Path("base-c")]
    assert args.augmentation_data_dir == [Path("dagger-a"), Path("dagger-b")]
    assert args.outcome_weighted_augmentation_data_dir == [
        Path("outcome-a"),
        Path("outcome-b"),
    ]


def test_dataset_contract_rejects_corpus_or_split_drift():
    contract = {
        "dataset_sha256": "frozen-sha",
        "shard_count": 70,
        "train_rows": 497_532,
        "val_rows": 62_417,
        "test_rows": 62_990,
        "expected_dataset_sha256": "frozen-sha",
        "expected_shards": 70,
        "expected_train_rows": 497_532,
        "expected_val_rows": 62_417,
        "expected_test_rows": 62_990,
    }
    _validate_dataset_contract(**contract)

    contract["shard_count"] = 30
    contract["test_rows"] = 0
    with pytest.raises(ValueError, match="shards.*30.*test_rows.*0"):
        _validate_dataset_contract(**contract)


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


def test_policy_value_loss_trains_supervised_heads():
    class TinyPolicyValue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.win = torch.nn.Parameter(torch.tensor(0.0))
            self.margin = torch.nn.Parameter(torch.tensor(0.0))

        def policy_value(self, features):
            return PolicyValueOutput(
                policy_logits=features,
                win_value=self.win.expand(len(features)),
                vp_margin=self.margin.expand(len(features)),
            )

    legal, legal_mask, values, value_mask = padded_decision_columns([[0, 1]])
    batch = {
        "features": torch.tensor([[2.0, 0.0]]),
        "targets": torch.tensor([0]),
        "legal_indices": legal,
        "legal_mask": legal_mask,
        "candidate_values": values,
        "candidate_mask": value_mask,
        "sample_weights": torch.ones(1),
        "win_value_targets": torch.tensor([1.0]),
        "win_value_mask": torch.tensor([True]),
        "vp_margin_targets": torch.tensor([3.0]),
        "vp_margin_mask": torch.tensor([True]),
    }
    args = SimpleNamespace(
        listwise_temperature=0.25,
        tie_tolerance=1e-6,
        hybrid_listwise_weight=0.0,
        win_value_weight=0.25,
        vp_margin_weight=0.05,
    )
    net = TinyPolicyValue()

    loss, _, used_rows = _batch_loss(net, batch, "legal_ce", "cpu", args)
    loss.backward()

    assert used_rows == 1
    assert net.win.grad is not None and net.win.grad < 0
    assert net.margin.grad is not None and net.margin.grad < 0


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
                "RETURN": [1.0, -1.0],
                "VICTORY_POINT_MARGIN_RETURN": [3.0, -2.0],
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
    assert all(bool(batch["win_value_mask"].all()) for batch in batches)
    assert all(bool(batch["vp_margin_mask"].all()) for batch in batches)


def test_parquet_batches_accept_and_mix_distillation_teacher_targets(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    teacher = tmp_path / "teacher.parquet"
    distillation = tmp_path / "distillation.parquet"
    common = {
        "F_A": [1.0],
        "F_B": [2.0],
        "NUM_LEGAL": [2],
        "LEGAL_ACTIONS": [[0, 1]],
    }
    pd.DataFrame(
        {
            **common,
            "GAME_ID": ["teacher-game"],
            "ACTION": [0],
            "CANDIDATE_VALUES": [[0.8, 0.2]],
        }
    ).to_parquet(teacher)
    pd.DataFrame(
        {
            **common,
            "GAME_ID": ["student-visited-game"],
            "TEACHER_ACTION": [1],
            "CANDIDATE_SCORES": [[0.1, 0.9]],
        }
    ).to_parquet(distillation)

    plan = inspect_parquet_dataset(
        [teacher, distillation], val_fraction=0.0, test_fraction=0.0, seed=7
    )
    assert plan.path_target_columns[teacher] == "ACTION"
    assert plan.path_target_columns[distillation] == "TEACHER_ACTION"
    assert "CANDIDATE_VALUES" in plan.available_columns

    batches = list(ParquetDecisionBatches(plan, "train", batch_size=1).loader())
    assert {int(batch["targets"].item()) for batch in batches} == {0, 1}
    assert all(bool(batch["candidate_mask"].all()) for batch in batches)


def test_augmentation_plan_preserves_frozen_base_split(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    def write_games(directory, prefix, *, target_column):
        directory.mkdir()
        paths = []
        for index in range(5):
            path = directory / f"{prefix}-{index}.parquet"
            pd.DataFrame(
                {
                    "F_A": [float(index)],
                    target_column: [index % 2],
                    "GAME_ID": [f"{prefix}-game-{index}"],
                    "NUM_LEGAL": [2],
                    "LEGAL_ACTIONS": [[0, 1]],
                }
            ).to_parquet(path)
            paths.append(path)
        return paths

    base_paths = write_games(tmp_path / "base", "base", target_column="ACTION")
    augmentation_paths = write_games(
        tmp_path / "augmentation", "dagger", target_column="TEACHER_ACTION"
    )
    base = inspect_parquet_dataset(
        base_paths, val_fraction=0.2, test_fraction=0.2, seed=101
    )
    augmentation = inspect_parquet_dataset(
        augmentation_paths, val_fraction=0.2, test_fraction=0.2, seed=101
    )
    combined = combine_parquet_dataset_plans((base, augmentation))

    assert combined.train_groups & set(base.rows_by_group) == base.train_groups
    assert combined.val_groups & set(base.rows_by_group) == base.val_groups
    assert combined.test_groups & set(base.rows_by_group) == base.test_groups
    assert len(combined.paths) == 10


def test_augmentation_corpora_freeze_each_iteration_split(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    def write_iteration(name):
        directory = tmp_path / name
        directory.mkdir()
        paths = []
        for index in range(10):
            path = directory / f"game-{index}.parquet"
            pd.DataFrame(
                {
                    "F_A": [float(index)],
                    "TEACHER_ACTION": [index % 2],
                    "GAME_ID": [f"{name}-game-{index}"],
                    "NUM_LEGAL": [2],
                    "LEGAL_ACTIONS": [[0, 1]],
                }
            ).to_parquet(path)
            paths.append(path)
        return paths

    iteration_0_paths = write_iteration("iteration-0000")
    iteration_1_paths = write_iteration("iteration-0001")
    iteration_0 = inspect_parquet_dataset(
        iteration_0_paths, val_fraction=0.2, test_fraction=0.2, seed=101
    )
    iteration_1 = inspect_parquet_dataset(
        iteration_1_paths, val_fraction=0.2, test_fraction=0.2, seed=101
    )

    combined = inspect_parquet_corpora(
        (iteration_0_paths, iteration_1_paths),
        val_fraction=0.2,
        test_fraction=0.2,
        seed=101,
    )

    for frozen in (iteration_0, iteration_1):
        groups = set(frozen.rows_by_group)
        assert combined.train_groups & groups == frozen.train_groups
        assert combined.val_groups & groups == frozen.val_groups
        assert combined.test_groups & groups == frozen.test_groups
    assert len(combined.paths) == 20


def test_resolve_dataset_paths_accepts_verified_distillation_iteration(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = tmp_path / "distillation"
    schema = build_model_schema()
    writer = DistillationDatasetWriter(
        root,
        iteration=0,
        shard_games=1,
        metadata={"schema": schema},
    )
    writer.add_game(
        [{"F_A": 1.0, "TEACHER_ACTION": 0}],
        game_index=0,
        game_seed=17,
        student_color=Color.BLUE,
        winner=Color.BLUE,
        truncated=False,
    )
    writer.finalize()

    paths = _resolve_dataset_paths([root / "iteration-0000"], expected_schema=schema)

    assert paths == [root / "iteration-0000" / "shard-00000.parquet"]

    paths[0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity checks"):
        _resolve_dataset_paths([root / "iteration-0000"], expected_schema=schema)


def test_training_path_weight_applies_only_to_selected_shard(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    paths = []
    for index in range(2):
        path = tmp_path / f"game-{index}.parquet"
        pd.DataFrame(
            {
                "F_A": [float(index)],
                "ACTION": [index],
                "GAME_ID": [f"game-{index}"],
            }
        ).to_parquet(path)
        paths.append(path)
    plan = inspect_parquet_dataset(paths, val_fraction=0.0, test_fraction=0.0, seed=1)
    batches = list(
        ParquetDecisionBatches(
            plan,
            "train",
            batch_size=1,
            path_weights={paths[1]: 4.0},
        ).loader()
    )
    weights_by_target = {
        int(batch["targets"].item()): float(batch["sample_weights"].item())
        for batch in batches
    }
    assert weights_by_target == {0: 1.0, 1: 4.0}


def test_outcome_deficit_weights_are_bounded_and_reject_missing_targets():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "RETURN": [1.0, -1.0, 0.0],
            "VICTORY_POINT_MARGIN_RETURN": [4.0, -5.0, -20.0],
        }
    )

    weights = outcome_deficit_sample_weights(frame)

    np.testing.assert_allclose(weights, [1.0, 2.25, 1.5])
    with pytest.raises(ValueError, match="native terminal targets"):
        outcome_deficit_sample_weights(frame.drop(columns="RETURN"))
    frame.loc[0, "RETURN"] = np.nan
    with pytest.raises(ValueError, match="finite native terminal targets"):
        outcome_deficit_sample_weights(frame)


def test_outcome_deficit_weighting_applies_only_to_selected_paths(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    base = tmp_path / "base.parquet"
    outcome = tmp_path / "outcome.parquet"
    pd.DataFrame({"F_A": [0.0], "ACTION": [0], "GAME_ID": ["base-game"]}).to_parquet(
        base
    )
    pd.DataFrame(
        {
            "F_A": [1.0],
            "ACTION": [1],
            "GAME_ID": ["outcome-game"],
            "RETURN": [-1.0],
            "VICTORY_POINT_MARGIN_RETURN": [-5.0],
        }
    ).to_parquet(outcome)
    plan = inspect_parquet_dataset(
        [base, outcome], val_fraction=0.0, test_fraction=0.0, seed=1
    )

    batches = list(
        ParquetDecisionBatches(
            plan,
            "train",
            batch_size=1,
            path_weights={outcome: 4.0},
            path_sample_weight_fns={outcome: outcome_deficit_sample_weights},
        ).loader()
    )

    weights_by_target = {
        int(batch["targets"].item()): float(batch["sample_weights"].item())
        for batch in batches
    }
    assert weights_by_target == {0: 1.0, 1: 9.0}


def test_training_hard_state_weights_distillation_target_without_action_type(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    codec = get_action_array((Color.BLUE, Color.RED), "BASE")
    end_turn = next(
        index
        for index, (family, _) in enumerate(codec)
        if family == ActionType.END_TURN
    )
    path = tmp_path / "distillation.parquet"
    pd.DataFrame(
        {
            "F_A": [1.0],
            "TEACHER_ACTION": [end_turn],
            "GAME_ID": ["game-0"],
            "NUM_LEGAL": [3],
            "PHASE": ["PLAY_TURN"],
        }
    ).to_parquet(path)
    plan = inspect_parquet_dataset([path], val_fraction=0.0, test_fraction=0.0, seed=1)

    batch = next(
        iter(
            ParquetDecisionBatches(
                plan,
                "train",
                batch_size=1,
                sample_weight_fn=hard_state_sample_weights,
            ).loader()
        )
    )

    assert float(batch["sample_weights"].item()) == 0.25
    assert (
        int(batch["action_types"][0])
        == get_action_type_array((Color.BLUE, Color.RED), "BASE")[end_turn]
    )


def test_action_family_derivation_rejects_out_of_schema_indices():
    codec = get_action_array((Color.BLUE, Color.RED), "BASE")

    assert action_type_indices_from_full_actions([0]).shape == (1,)
    with pytest.raises(ValueError, match="outside"):
        action_type_indices_from_full_actions([-1])
    with pytest.raises(ValueError, match="outside"):
        action_type_indices_from_full_actions([len(codec)])


def test_non_finite_distillation_scores_are_not_listwise_targets():
    legal, legal_mask, values, value_mask = padded_decision_columns(
        [[0, 1]], [[float("nan"), float("nan")]]
    )
    assert legal.tolist() == [[0, 1]]
    assert legal_mask.tolist() == [[True, True]]
    assert values.tolist() == [[0.0, 0.0]]
    assert value_mask.tolist() == [[False, False]]


def test_bc_resolver_accepts_verified_distillation_manifest(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    schema = build_model_schema()
    root = tmp_path / "distillation"
    writer = DistillationDatasetWriter(
        root,
        iteration=0,
        shard_games=1,
        metadata={"schema": schema},
    )
    writer.add_game(
        [{"F_A": 1.0, "TEACHER_ACTION": 0}],
        game_index=0,
        game_seed=10,
        student_color=Color.BLUE,
        winner=Color.BLUE,
        truncated=False,
    )
    writer.finalize()

    paths = _resolve_dataset_paths([root], expected_schema=schema)
    assert len(paths) == 1
    assert paths[0].name == "shard-00000.parquet"


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
        action_types=np.array([1, 1]),
        num_legal=np.array([2, 2]),
        legal_actions=[[0, 1], [0, 2]],
        candidate_values=[[0.2, 0.8], [0.1, 0.9]],
    )
    metrics = accumulator.compute()
    assert metrics["rows"] == 2
    assert metrics["legal_choice_accuracy"] == 0.5
    assert metrics["mean_regret"] == 0.5
    assert metrics["per_action_family"]["1"] == {
        "name": "MOVE_ROBBER",
        "rows": 2,
        "accuracy": 0.5,
        "regret_rows": 2,
        "mean_regret": 0.5,
        "total_regret": 1.0,
    }


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

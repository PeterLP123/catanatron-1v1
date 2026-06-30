import random

import numpy as np

from catanatron import Game, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.gym.accumulators import ParquetDataAccumulator
from catanatron.gym.colonist_training import (
    CANDIDATE_VALUES_COLUMN,
    GAME_ID_COLUMN,
    LEGAL_ACTIONS_COLUMN,
    NUM_LEGAL_COLUMN,
    PHASE_COLUMN,
    SEAT_COLUMN,
    decision_metrics,
    grouped_split_masks,
)


# --- dataset v2 parquet schema -------------------------------------------------


def test_parquet_records_v2_decision_metadata(tmp_path):
    import pandas as pd

    random.seed(0)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    acc = ParquetDataAccumulator(
        player_colors=game.state.colors,
        map_type="BASE",
        output=str(tmp_path),
        include_board_tensor=False,
    )
    game.play(accumulators=[acc])
    assert game.winning_color() is not None

    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])

    for col in (
        GAME_ID_COLUMN,
        SEAT_COLUMN,
        PHASE_COLUMN,
        NUM_LEGAL_COLUMN,
        LEGAL_ACTIONS_COLUMN,
        "ACTION",
        "ACTION_TYPE",
    ):
        assert col in df.columns

    # GAME_ID column matches the per-game filename and is constant within a game.
    assert df[GAME_ID_COLUMN].nunique() == 1
    assert df[GAME_ID_COLUMN].iloc[0] == files[0].stem

    # Seats are valid indices into player_colors.
    assert set(df[SEAT_COLUMN].unique()).issubset({0, 1})

    # The legal-action set is consistent with the chosen action and its count.
    for _, row in df.iterrows():
        legal = list(row[LEGAL_ACTIONS_COLUMN])
        assert len(legal) == int(row[NUM_LEGAL_COLUMN])
        assert int(row["ACTION"]) in legal


def test_parquet_records_candidate_values_when_scoring(tmp_path):
    import pandas as pd

    random.seed(0)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    acc = ParquetDataAccumulator(
        player_colors=game.state.colors,
        map_type="BASE",
        output=str(tmp_path),
        include_board_tensor=False,
        score_candidates=True,
    )
    game.play(accumulators=[acc])
    df = pd.read_parquet(list(tmp_path.glob("*.parquet"))[0])

    assert CANDIDATE_VALUES_COLUMN in df.columns

    # Genuine choices are scored and aligned with the legal set.
    import math

    choices = df[df[NUM_LEGAL_COLUMN] > 1]
    assert len(choices) > 0
    for _, row in choices.iterrows():
        cand = list(row[CANDIDATE_VALUES_COLUMN])
        assert len(cand) == len(list(row[LEGAL_ACTIONS_COLUMN]))
        assert all(math.isfinite(v) for v in cand)
    # At least one decision discriminates between its candidates (raw F values
    # preserve the sub-VP signal that the bounded proxy would erase).
    assert any(
        len(set(row[CANDIDATE_VALUES_COLUMN])) > 1 for _, row in choices.iterrows()
    )

    # Forced decisions are not scored.
    forced = df[df[NUM_LEGAL_COLUMN] == 1]
    assert all(len(list(c)) == 0 for c in forced[CANDIDATE_VALUES_COLUMN])


def test_parquet_candidate_column_empty_without_scoring(tmp_path):
    import pandas as pd

    random.seed(0)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    acc = ParquetDataAccumulator(
        player_colors=game.state.colors,
        map_type="BASE",
        output=str(tmp_path),
        include_board_tensor=False,
    )
    game.play(accumulators=[acc])
    df = pd.read_parquet(list(tmp_path.glob("*.parquet"))[0])
    # Column exists for a stable schema, but holds only empty lists.
    assert CANDIDATE_VALUES_COLUMN in df.columns
    assert all(len(list(c)) == 0 for c in df[CANDIDATE_VALUES_COLUMN])


# --- grouped (by-game) split ---------------------------------------------------


def test_grouped_split_keeps_games_disjoint():
    # 10 games, ~12 rows each.
    game_ids = np.repeat([f"g{i}" for i in range(10)], 12)
    train, val, test = grouped_split_masks(
        game_ids, val_fraction=0.2, test_fraction=0.2, seed=1
    )

    # Partition: every row belongs to exactly one split.
    assert np.array_equal(train | val | test, np.ones(len(game_ids), dtype=bool))
    assert not np.any(train & val)
    assert not np.any(val & test)
    assert not np.any(train & test)

    # No game id appears in more than one split.
    def games_in(mask):
        return set(np.asarray(game_ids)[mask].tolist())

    assert games_in(train).isdisjoint(games_in(val))
    assert games_in(train).isdisjoint(games_in(test))
    assert games_in(val).isdisjoint(games_in(test))

    # Roughly the requested fraction of *games* (2 of 10) land in val/test.
    assert len(games_in(val)) == 2
    assert len(games_in(test)) == 2


def test_grouped_split_is_deterministic_for_seed():
    game_ids = np.repeat([f"g{i}" for i in range(8)], 5)
    a = grouped_split_masks(game_ids, 0.25, 0.25, seed=7)
    b = grouped_split_masks(game_ids, 0.25, 0.25, seed=7)
    for m1, m2 in zip(a, b):
        assert np.array_equal(m1, m2)


# --- decision metrics ----------------------------------------------------------


def test_decision_metrics_filters_forced_and_masks_to_legal():
    # 3 rows, action space size 5.
    #  row0: forced (only action 4 legal), prediction irrelevant to choice acc.
    #  row1: choice between {0,1}; logits prefer 0 but illegal-2 is highest -> masked picks legal best.
    #  row2: choice between {1,3}; true is 3.
    logits = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 9.0],  # forced row
            [5.0, 1.0, 9.0, 0.0, 0.0],  # illegal action 2 has the max logit
            [0.0, 1.0, 0.0, 8.0, 0.0],
        ]
    )
    y_true = np.array([4, 0, 3])
    action_types = np.array([9, 1, 1])
    num_legal = np.array([1, 2, 2])
    legal_actions = [[4], [0, 1], [1, 3]]

    m = decision_metrics(
        logits,
        y_true,
        action_types=action_types,
        num_legal=num_legal,
        legal_actions=legal_actions,
        topk=(1, 2),
    )

    assert m["rows"] == 3
    assert m["choice_rows"] == 2
    assert abs(m["forced_fraction"] - (1 / 3)) < 1e-9

    # row1: legal-masked argmax over {0,1} -> 0 (correct). row2: over {1,3} -> 3 (correct).
    assert m["legal_choice_accuracy"] == 1.0
    assert m["legal_top1_accuracy"] == 1.0
    assert m["legal_top2_accuracy"] == 1.0
    # Both choice rows are action family 1.
    assert m["per_action_family_accuracy"]["1"] == 1.0


def test_decision_metrics_without_v2_columns_is_plain_accuracy():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 0])
    m = decision_metrics(logits, y_true)
    assert m["accuracy"] == 0.5
    assert "legal_choice_accuracy" not in m


def test_decision_metrics_reports_regret():
    # 2 choice rows, action space size 5; candidate values aligned with legal sets.
    logits = np.array(
        [
            [0.0, 9.0, 0.0, 1.0, 0.0],  # legal {1,3}; model picks 1
            [5.0, 0.0, 0.0, 2.0, 0.0],  # legal {0,3}; model picks 0
        ]
    )
    y_true = np.array([1, 3])
    num_legal = np.array([2, 2])
    legal_actions = [[1, 3], [0, 3]]
    candidate_values = [[0.8, 0.2], [0.4, 0.9]]

    m = decision_metrics(
        logits,
        y_true,
        num_legal=num_legal,
        legal_actions=legal_actions,
        candidate_values=candidate_values,
    )
    # Normalized per row: row0 picks the best (0.8 of [0.8,0.2]) -> regret 0;
    # row1 picks the worst (0.4 of [0.4,0.9]) -> regret 1.0. Mean = 0.5.
    assert m["regret_rows"] == 2
    assert abs(m["mean_regret"] - 0.5) < 1e-9


def test_decision_metrics_skips_regret_for_empty_candidates():
    logits = np.array([[0.0, 9.0, 0.0, 1.0, 0.0]])
    y_true = np.array([1])
    m = decision_metrics(
        logits,
        y_true,
        num_legal=np.array([2]),
        legal_actions=[[1, 3]],
        candidate_values=[[]],  # not scored
    )
    assert "mean_regret" not in m

import os
import time
from collections import defaultdict
from typing import Tuple, Literal

import numpy as np
import pandas as pd

from catanatron import Action, Color, Game
from catanatron.features import create_sample
from catanatron.game import GameAccumulator
from catanatron.gym.board_tensor_features import create_board_tensor
from catanatron.gym.envs.action_space import (
    get_action_array,
    to_action_space,
    to_action_type_space,
)
from catanatron.gym.utils import (
    DISCOUNT_FACTOR,
    get_tournament_total_return,
    get_victory_points_total_return,
    populate_matrices,
    simple_total_return,
)
from catanatron.utils import format_secs


class ReinforcementLearningAccumulator(GameAccumulator):
    def __init__(
        self,
        player_colors: Tuple[Color],
        map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
        include_board_tensor=True,
        total_return_fns={
            "RETURN": simple_total_return,
            "TOURNAMENT_RETURN": get_tournament_total_return,
            "VICTORY_POINTS_RETURN": get_victory_points_total_return,
        },
        score_candidates=False,
        value_fn_name="base_fn",
    ):
        self.player_colors = player_colors
        self.map_type = map_type
        self.include_board_tensor = include_board_tensor
        # TODO: Generalize to "rewards_fn" that can yield intermediary rewards
        #   while still rewarding big on terminal states.
        self.total_return_fns = total_return_fns
        # When set, label each genuine decision's legal actions with their F
        # candidate values (Phase 02 "train on real choices"). Expensive, so off
        # by default. Stored as plain config so the accumulator stays picklable.
        self.score_candidates = bool(score_candidates)
        self.value_fn_name = value_fn_name
        # Lazily-built {(action_type, value): action_index} map, used to record
        # the legal-action set per decision (dataset v2 "honest measurement").
        self._action_index = None

    def _ensure_action_index(self):
        if self._action_index is None:
            actions_array = get_action_array(self.player_colors, self.map_type)
            self._action_index = {pair: i for i, pair in enumerate(actions_array)}

    def _legal_and_candidates(self, game, color):
        """Legal action indices and (optionally) their F candidate values.

        Both lists are aligned with ``game.playable_actions`` (skipping any
        action absent from the action map), so ``CANDIDATE_VALUES[i]`` is the
        value of ``LEGAL_ACTIONS[i]``. Candidate values are only computed for
        genuine choices (more than one legal action) when scoring is enabled.
        """
        self._ensure_action_index()
        playable = game.playable_actions
        score = self.score_candidates and len(playable) > 1
        value_fn = None
        if score:
            from catanatron.players.leaf_evaluation import (
                action_value,
                make_f_value_fn,
            )

            value_fn = make_f_value_fn(self.value_fn_name)

        legal_indices = []
        cand_values = []
        for action in playable:
            idx = self._action_index.get((action.action_type, action.value))
            if idx is None:
                continue
            legal_indices.append(idx)
            if score:
                cand_values.append(action_value(game, action, color, value_fn))
        return legal_indices, cand_values

    def before(self, game):
        self.data = {
            # e.g. {RED: [1,5]} if RED acted at tick 1 and 5
            "color_action_indices": defaultdict(list),
            "acting_color": [],
            "samples": [],
            "actions": [],
            # Dataset v2 decision metadata (one entry per recorded decision).
            "game_ids": [],
            "seats": [],
            "phases": [],
            "num_legal": [],
            "legal_actions": [],
            "candidate_values": [],
        }
        if self.include_board_tensor:
            self.data["board_tensors"] = []

    def step(self, game_before_action: Game, action: Action):
        self.data["color_action_indices"][action.color].append(
            len(self.data["samples"])
        )
        self.data["acting_color"].append(action.color)
        self.data["samples"].append(create_sample(game_before_action, action.color))
        self.data["actions"].append(
            [
                to_action_space(action, self.player_colors, self.map_type),
                to_action_type_space(action.action_type),
            ]
        )

        # Dataset v2: record who decided, in which phase, and over how many legal
        # actions, so downstream training can split by game, filter forced
        # decisions, and score against the legal candidate set.
        legal_indices, cand_values = self._legal_and_candidates(
            game_before_action, action.color
        )
        prompt = game_before_action.state.current_prompt
        self.data["game_ids"].append(game_before_action.id)
        self.data["seats"].append(self.player_colors.index(action.color))
        self.data["phases"].append(getattr(prompt, "name", str(prompt)))
        self.data["num_legal"].append(len(game_before_action.playable_actions))
        self.data["candidate_values"].append(cand_values)
        self.data["legal_actions"].append(legal_indices)

        if self.include_board_tensor:
            board_tensor = create_board_tensor(game_before_action, action.color)
            flattened_tensor = board_tensor.reshape(-1)
            self.data["board_tensors"].append(flattened_tensor)

    def after(self, game):
        if game.winning_color() is None:
            return None  # drop game

        t1 = time.time()

        # Now that the game is over, we can calculate the returns
        # for each sample (so trajectories that lost still contribute data).
        returns = {
            name: np.zeros(len(self.data["samples"]), dtype=np.float64)
            for name in self.total_return_fns.keys()
        }
        for color, action_indices in self.data["color_action_indices"].items():
            # Set total return for the return of the perspective of this player
            player_returns = {
                name: np.full_like(
                    action_indices, total_return_fn(game, color), dtype=np.float64
                )
                for name, total_return_fn in self.total_return_fns.items()
            }

            # For each column, modify the indexes of this player
            for column_name, step_returns in player_returns.items():
                returns[column_name][action_indices] = step_returns

        T = len(self.data["samples"])
        discounts = DISCOUNT_FACTOR ** np.arange(T)[::-1]
        discount_columns = dict()
        for name, step_returns in returns.items():
            discount_columns["DISCOUNTED_" + name] = step_returns * discounts

        # Build Q-learning Design Matrix
        samples = self.data["samples"]
        actions = self.data["actions"]
        samples_df = (
            pd.DataFrame.from_records(samples, columns=sorted(samples[0].keys()))
            .astype("float64")
            .add_prefix("F_")
        )
        actions_df = pd.DataFrame(actions, columns=["ACTION", "ACTION_TYPE"]).astype(
            "int"
        )
        returns_df = pd.DataFrame({**returns, **discount_columns}).astype("float64")

        # Dataset v2 per-decision metadata (kept out of the CSV main_df; the
        # parquet writer attaches it). LEGAL_ACTIONS is a variable-length list
        # column, so it is built separately from the typed columns.
        meta_df = pd.DataFrame(
            {
                "GAME_ID": self.data["game_ids"],
                "SEAT": np.asarray(self.data["seats"], dtype="int64"),
                "PHASE": self.data["phases"],
                "NUM_LEGAL": np.asarray(self.data["num_legal"], dtype="int64"),
            }
        )
        meta_df["LEGAL_ACTIONS"] = self.data["legal_actions"]
        # Aligned with LEGAL_ACTIONS; empty lists when candidate scoring is off.
        meta_df["CANDIDATE_VALUES"] = self.data["candidate_values"]

        results = {
            "samples_df": samples_df,
            "actions_df": actions_df,
            "returns_df": returns_df,
            "meta_df": meta_df,
        }
        if self.include_board_tensor:
            board_tensors = self.data["board_tensors"]
            board_tensors_df = (
                pd.DataFrame(board_tensors).astype("float64").add_prefix("BT_")
            )
            main_df = pd.concat(
                [samples_df, board_tensors_df, actions_df, returns_df], axis=1
            )
            results["board_tensors_df"] = board_tensors_df
            results["main_df"] = main_df
        else:
            main_df = pd.concat([samples_df, actions_df, returns_df], axis=1)
            results["main_df"] = main_df
        print(
            "Building matrices at took",
            format_secs(time.time() - t1),
        )
        return results


class CsvDataAccumulator(ReinforcementLearningAccumulator):
    def __init__(
        self,
        player_colors: Tuple[Color],
        map_type: Literal["BASE", "TOURNAMENT", "MINI"],
        output,
        include_board_tensor=True,
    ):
        super().__init__(player_colors, map_type, include_board_tensor)
        self.output = output

    def after(self, game):
        data = super().after(game)
        if data is None:
            return

        t1 = time.time()
        main_df = data["main_df"]
        samples_df = data["samples_df"]
        board_tensors_df = (
            None if not self.include_board_tensor else data["board_tensors_df"]
        )
        actions_df = data["actions_df"]
        returns_df = data["returns_df"]
        populate_matrices(
            samples_df,
            board_tensors_df,
            actions_df,
            returns_df,
            main_df,
            self.output,
        )
        print(
            f"Saved matrices to {self.output}{' (including board tensors)' if self.include_board_tensor else ''} with shapes: "
            f"main={main_df.shape}, samples={samples_df.shape}, actions={actions_df.shape}, "
            f"rewards={returns_df.shape} in {format_secs(time.time() - t1)}"
        )
        return samples_df, board_tensors_df, actions_df, returns_df


class ParquetDataAccumulator(ReinforcementLearningAccumulator):
    def __init__(
        self,
        player_colors: Tuple[Color],
        map_type: Literal["BASE", "TOURNAMENT", "MINI"],
        output,
        include_board_tensor=True,
        score_candidates=False,
        value_fn_name="base_fn",
    ):
        super().__init__(
            player_colors,
            map_type,
            include_board_tensor,
            score_candidates=score_candidates,
            value_fn_name=value_fn_name,
        )
        self.output = output

    def after(self, game):
        data = super().after(game)
        if data is None:
            return

        t1 = time.time()
        # Lead with the dataset v2 metadata columns (GAME_ID, SEAT, PHASE,
        # NUM_LEGAL, LEGAL_ACTIONS) so grouped splits and decision metrics work.
        main_df = pd.concat([data["meta_df"], data["main_df"]], axis=1)
        filepath = os.path.join(self.output, f"{game.id}.parquet")
        main_df.to_parquet(filepath, index=False)
        print(
            f"Saved main_df to {self.output} with shapes {main_df.shape} in {format_secs(time.time() - t1)}"
        )
        return main_df

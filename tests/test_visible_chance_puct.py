from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    write_model_schema,
)
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    KNIGHT,
    YEAR_OF_PLENTY,
    Action,
    ActionPrompt,
    ActionType,
)


def _write_inputs(tmp_path):
    torch = pytest.importorskip("torch")
    from catanatron.gym.colonist_training import BcCheckpointMeta
    from catanatron.gym.model_architectures import (
        FactoredOutcomeCritic,
        build_bc_policy,
    )

    schema = build_model_schema()
    features = schema["observation"]["features"]
    policy_path = tmp_path / "policy.pt"
    policy = build_bc_policy("mlp", features, len(schema["actions"]), hidden_sizes=(4,))
    torch.save(policy.state_dict(), policy_path)
    BcCheckpointMeta(
        obs_dim=len(features),
        n_actions=len(schema["actions"]),
        hidden_sizes=[4],
        epochs=1,
        model_schema=schema,
    ).save(policy_path.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(policy_path), schema)

    critic_path = tmp_path / "critic.pt"
    critic_features = [f"F_{name}" for name in features]
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
    return policy_path, critic_path


def _write_manifest(tmp_path, *, simulations=4):
    from catanatron.gym.provenance import sha256_file
    from examples.colonist_1v1_build_visible_chance_puct import main as build_chance
    from examples.colonist_1v1_build_visible_puct import main as build_parent

    policy, critic = _write_inputs(tmp_path)
    parent = tmp_path / "parent.json"
    assert (
        build_parent(
            [
                "--policy",
                str(policy),
                "--critic",
                str(critic),
                "--leaf-evaluator",
                "public_f",
                "--num-simulations",
                str(simulations),
                "--output",
                str(parent),
            ]
        )
        == 0
    )
    manifest = tmp_path / "chance.json"
    assert (
        build_chance(
            [
                "--parent-manifest",
                str(parent),
                "--expected-parent-sha256",
                sha256_file(parent),
                "--output",
                str(manifest),
            ]
        )
        == 0
    )
    return manifest


def _game(player):
    from catanatron.game import Game
    from catanatron.models.player import Color
    from catanatron.players.value import ValueFunctionPlayer

    return Game(
        [player, ValueFunctionPlayer(Color.RED)],
        seed=56,
        colonist_1v1=True,
        shuffle_players=False,
    )


def _make_buyable_play_state(game):
    from catanatron.models.actions import generate_playable_actions
    from catanatron.models.player import Color
    from catanatron.state_functions import player_key

    game.state.is_initial_build_phase = False
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    game.state.current_player_index = game.state.color_to_index[Color.BLUE]
    key = player_key(game.state, Color.BLUE)
    game.state.player_state[f"{key}_HAS_ROLLED"] = True
    game.state.player_state[f"{key}_SHEEP_IN_HAND"] = 1
    game.state.player_state[f"{key}_WHEAT_IN_HAND"] = 1
    game.state.player_state[f"{key}_ORE_IN_HAND"] = 1
    game.playable_actions = generate_playable_actions(game.state)


def test_visible_chance_manifest_loads_registers_and_resolves(tmp_path):
    from catanatron.cli.cli_players import CLI_PLAYERS
    from catanatron.colonist_1v1_eval import checkpoint_path_from_agent
    from catanatron.models.player import Color
    from catanatron.players.visible_chance_puct import VisibleChancePuctPlayer

    manifest = _write_manifest(tmp_path)
    player = VisibleChancePuctPlayer(Color.BLUE, manifest)
    assert player.num_simulations == 4
    assert next(row for row in CLI_PLAYERS if row.code == "Q").import_fn is (
        VisibleChancePuctPlayer
    )
    assert checkpoint_path_from_agent("Q:runs/chance.json") == Path("runs/chance.json")


def test_unseen_dev_distribution_and_public_successors_ignore_hidden_partition(
    tmp_path,
):
    from catanatron.models.player import Color
    from catanatron.players.visible_chance_puct import (
        VisibleChancePuctPlayer,
        public_buy_development_spectrum,
        public_unseen_development_distribution,
    )
    from catanatron.players.visible_puct import public_f_leaf_value
    from catanatron.state_functions import player_key, player_num_dev_cards

    player = VisibleChancePuctPlayer(Color.BLUE, _write_manifest(tmp_path))
    base = _game(player)
    _make_buyable_play_state(base)
    opponent_key = player_key(base.state, Color.RED)

    first = base.copy()
    first.state.development_listdeck.remove(YEAR_OF_PLENTY)
    first.state.player_state[f"{opponent_key}_{YEAR_OF_PLENTY}_IN_HAND"] = 1
    second = base.copy()
    second.state.development_listdeck.remove(KNIGHT)
    second.state.player_state[f"{opponent_key}_{KNIGHT}_IN_HAND"] = 1

    assert public_unseen_development_distribution(
        first, Color.BLUE
    ) == public_unseen_development_distribution(second, Color.BLUE)
    action = Action(Color.BLUE, ActionType.BUY_DEVELOPMENT_CARD, None)

    def projection(game):
        rows = []
        for outcome, probability in public_buy_development_spectrum(game, action):
            blue_key = player_key(outcome.state, Color.BLUE)
            rows.append(
                (
                    probability,
                    tuple(
                        outcome.state.player_state[f"{blue_key}_{card}_IN_HAND"]
                        for card in DEVELOPMENT_CARDS
                    ),
                    len(outcome.state.development_listdeck),
                    player_num_dev_cards(outcome.state, Color.RED),
                    outcome.state.player_state[f"{opponent_key}_ACTUAL_VICTORY_POINTS"],
                    public_f_leaf_value(outcome, Color.BLUE),
                )
            )
        return rows

    assert projection(first) == projection(second)
    assert sum(
        probability for _, probability in public_buy_development_spectrum(first, action)
    ) == pytest.approx(1.0)


def test_visible_chance_search_uses_no_omniscient_spectrum(tmp_path, monkeypatch):
    from catanatron.models.player import Color
    from catanatron.players.visible_chance_puct import VisibleChancePuctPlayer

    player = VisibleChancePuctPlayer(Color.BLUE, _write_manifest(tmp_path))
    game = _game(player)
    _make_buyable_play_state(game)
    assert {action.action_type for action in game.playable_actions} == {
        ActionType.END_TURN,
        ActionType.BUY_DEVELOPMENT_CARD,
    }

    def forbidden_call(*args, **kwargs):
        raise AssertionError("visible chance PUCT must not call execute_spectrum")

    monkeypatch.setattr(
        "catanatron.players.tree_search_utils.execute_spectrum", forbidden_call
    )
    selected = player.decide(game, list(game.playable_actions))
    assert selected in game.playable_actions
    stats = player.stats_summary()
    assert stats["search_decisions"] == 1
    assert stats["chance_actions_expanded"] >= 1
    assert stats["chance_outcomes_expanded"] >= 1
    assert stats["probability_sum_violations"] == 0


def test_visible_chance_falls_back_for_robber_and_monopoly(tmp_path):
    from catanatron.models.player import Color
    from catanatron.players.visible_chance_puct import VisibleChancePuctPlayer

    player = VisibleChancePuctPlayer(Color.BLUE, _write_manifest(tmp_path))
    fallback = Action(Color.BLUE, ActionType.END_TURN, None)
    player.policy.decide = lambda game, actions: fallback
    for forbidden in (ActionType.MOVE_ROBBER, ActionType.PLAY_MONOPOLY):
        actions = [fallback, Action(Color.BLUE, forbidden, None)]
        assert player.decide(SimpleNamespace(), actions) == fallback
    stats = player.stats_summary()
    assert stats["search_decisions"] == 0
    assert stats["fallback_decisions"] == 2

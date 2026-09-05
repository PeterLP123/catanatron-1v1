from __future__ import annotations

import json
from pathlib import Path

import pytest

from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    write_model_schema,
)
from catanatron.models.enums import Action, ActionType


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


def _write_manifest(tmp_path, *, simulations=4, leaf_evaluator="outcome_critic"):
    from examples.colonist_1v1_build_visible_puct import main

    policy, critic = _write_inputs(tmp_path)
    manifest = tmp_path / "visible-puct.json"
    assert (
        main(
            [
                "--policy",
                str(policy),
                "--critic",
                str(critic),
                "--num-simulations",
                str(simulations),
                "--leaf-evaluator",
                leaf_evaluator,
                "--output",
                str(manifest),
            ]
        )
        == 0
    )
    return manifest


def test_visible_puct_manifest_loads_and_is_registered(tmp_path):
    from catanatron.colonist_1v1_eval import checkpoint_path_from_agent
    from catanatron.cli.cli_players import CLI_PLAYERS
    from catanatron.models.player import Color
    from catanatron.players.visible_puct import VisibleSameTurnPuctPlayer

    manifest = _write_manifest(tmp_path)
    player = VisibleSameTurnPuctPlayer(Color.BLUE, manifest)
    assert player.num_simulations == 4
    assert next(row for row in CLI_PLAYERS if row.code == "N").import_fn is (
        VisibleSameTurnPuctPlayer
    )
    assert checkpoint_path_from_agent("N:runs/visible-puct.json") == Path(
        "runs/visible-puct.json"
    )


def test_visible_puct_falls_back_for_every_forbidden_action(tmp_path):
    from types import SimpleNamespace

    from catanatron.models.player import Color
    from catanatron.players.visible_puct import (
        FORBIDDEN_SEARCH_ACTIONS,
        VisibleSameTurnPuctPlayer,
    )

    player = VisibleSameTurnPuctPlayer(Color.BLUE, _write_manifest(tmp_path))
    fallback = Action(Color.BLUE, ActionType.END_TURN, None)
    player.policy.decide = lambda game, actions: fallback
    game = SimpleNamespace()
    for forbidden in FORBIDDEN_SEARCH_ACTIONS:
        actions = [fallback, Action(Color.BLUE, forbidden, None)]
        assert player.decide(game, actions) == fallback
    stats = player.stats_summary()
    assert stats["search_decisions"] == 0
    assert stats["fallback_decisions"] == len(FORBIDDEN_SEARCH_ACTIONS)
    assert stats["forbidden_action_expansions"] == 0


def test_visible_puct_search_never_uses_hidden_spectrum(tmp_path, monkeypatch):
    from catanatron.colonist_1v1 import create_colonist_1v1_game
    from catanatron.models.player import Color
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.visible_puct import VisibleSameTurnPuctPlayer

    player = VisibleSameTurnPuctPlayer(Color.BLUE, _write_manifest(tmp_path))
    opponent = ValueFunctionPlayer(Color.RED)
    game = create_colonist_1v1_game([player, opponent], seed=54)

    def forbidden_call(*args, **kwargs):
        raise AssertionError("visible PUCT must not call execute_spectrum")

    monkeypatch.setattr(
        "catanatron.players.tree_search_utils.execute_spectrum", forbidden_call
    )
    selected = player.decide(game, list(game.playable_actions))
    assert selected in game.playable_actions
    stats = player.stats_summary()
    assert stats["search_decisions"] == 1
    assert stats["forbidden_action_expansions"] == 0
    assert stats["opponent_turn_expansions"] == 0


def test_public_f_leaf_is_invariant_to_opponent_hidden_resource_mix(tmp_path):
    from catanatron.colonist_1v1 import create_colonist_1v1_game
    from catanatron.models.enums import RESOURCES
    from catanatron.models.player import Color
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.players.visible_puct import (
        VisibleSameTurnPuctPlayer,
        public_f_leaf_value,
    )
    from catanatron.state_functions import player_key

    player = VisibleSameTurnPuctPlayer(
        Color.BLUE, _write_manifest(tmp_path, leaf_evaluator="public_f")
    )
    game = create_colonist_1v1_game([player, ValueFunctionPlayer(Color.RED)], seed=55)
    opponent_key = player_key(game.state, Color.RED)
    for resource in RESOURCES:
        game.state.player_state[f"{opponent_key}_{resource}_IN_HAND"] = 0
    game.state.player_state[f"{opponent_key}_{RESOURCES[0]}_IN_HAND"] = 8
    first = public_f_leaf_value(game, Color.BLUE)
    game.state.player_state[f"{opponent_key}_{RESOURCES[0]}_IN_HAND"] = 0
    game.state.player_state[f"{opponent_key}_{RESOURCES[-1]}_IN_HAND"] = 8
    second = public_f_leaf_value(game, Color.BLUE)
    assert first == second


@pytest.fixture
def own_hand_game():
    from catanatron.game import Game
    from catanatron.models.player import Color, RandomPlayer

    game = Game(
        [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)],
        seed=57,
        colonist_1v1=True,
        shuffle_players=False,
    )
    game.execute(game.playable_actions[0])  # One real settlement to upgrade.
    game.state.is_initial_build_phase = False
    return game


@pytest.mark.parametrize(
    "ports, ore, expected", [((), 7, 0.8), ((None,), 6, 0.8), (("ORE",), 7, 1.0)]
)
def test_build_readiness_reserves_cost_before_port_trades(
    own_hand_game, monkeypatch, ports, ore, expected
):
    from catanatron.models.player import Color
    from catanatron.players.visible_puct import own_hand_build_readiness
    from catanatron.state_functions import player_key

    game = own_hand_game
    key = player_key(game.state, Color.BLUE)
    game.state.player_state[f"{key}_ORE_IN_HAND"] = ore
    monkeypatch.setattr(
        game.state.board, "get_player_port_resources", lambda color: set(ports)
    )
    assert own_hand_build_readiness(game, Color.BLUE) == pytest.approx(expected)


def test_build_readiness_excludes_unavailable_builds_and_initial_placement(
    own_hand_game, monkeypatch
):
    from catanatron.models.player import Color
    from catanatron.models.enums import RESOURCES
    from catanatron.players.visible_puct import own_hand_build_readiness
    from catanatron.state_functions import player_key

    game = own_hand_game
    key = player_key(game.state, Color.BLUE)
    for resource in RESOURCES:
        game.state.player_state[f"{key}_{resource}_IN_HAND"] = 4
    game.state.player_state[f"{key}_CITIES_AVAILABLE"] = 0
    assert own_hand_build_readiness(game, Color.BLUE) == 0.0
    monkeypatch.setattr(game.state.board, "buildable_node_ids", lambda color: [42])
    assert own_hand_build_readiness(game, Color.BLUE) == 1.0
    game.state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] = 0
    assert own_hand_build_readiness(game, Color.BLUE) == 0.0
    game.state.player_state[f"{key}_CITIES_AVAILABLE"] = 4
    game.state.is_initial_build_phase = True
    assert own_hand_build_readiness(game, Color.BLUE) == 0.0


def test_own_hand_leaf_responds_only_to_our_composition_and_preserves_terminal_values(
    own_hand_game, tmp_path, monkeypatch
):
    from catanatron.models.enums import RESOURCES
    from catanatron.models.player import Color
    from catanatron.players.visible_puct import (
        VisibleSameTurnPuctPlayer,
        own_hand_f_leaf_value,
        public_f_leaf_value,
    )
    from catanatron.state_functions import player_key

    game = own_hand_game
    monkeypatch.setattr(
        game.state.board, "get_player_port_resources", lambda color: set()
    )
    player = VisibleSameTurnPuctPlayer(
        Color.BLUE, _write_manifest(tmp_path, leaf_evaluator="public_f_own_hand_v1")
    )
    key = player_key(game.state, Color.BLUE)
    assert player._leaf_value(game) == public_f_leaf_value(game, Color.BLUE)
    game.state.player_state[f"{key}_WOOD_IN_HAND"] = 5
    poorly_matched = player._leaf_value(game)
    public_before = public_f_leaf_value(game, Color.BLUE)
    game.state.player_state[f"{key}_WOOD_IN_HAND"] = 0
    game.state.player_state[f"{key}_ORE_IN_HAND"] = 3
    game.state.player_state[f"{key}_WHEAT_IN_HAND"] = 2
    ready = player._leaf_value(game)
    assert public_f_leaf_value(game, Color.BLUE) == public_before
    assert ready > poorly_matched > public_before

    enemy_key = player_key(game.state, Color.RED)
    hidden_values = []
    decisions = []
    # Same public card count, two different hidden compositions. Search must
    # give the same leaf value and action with its new evaluator active.
    from catanatron.models.actions import generate_playable_actions
    from catanatron.models.enums import ActionPrompt

    game.state.current_prompt = ActionPrompt.PLAY_TURN
    game.state.current_player_index = game.state.color_to_index[Color.BLUE]
    game.state.player_state[f"{key}_HAS_ROLLED"] = True
    for hidden_resource in (RESOURCES[0], RESOURCES[-1]):
        for resource in RESOURCES:
            game.state.player_state[f"{enemy_key}_{resource}_IN_HAND"] = (
                8 if resource == hidden_resource else 0
            )
        game.playable_actions = generate_playable_actions(game.state)
        hidden_values.append(player._leaf_value(game))
        decisions.append(player.decide(game, game.playable_actions))
    assert hidden_values[0] == hidden_values[1]
    assert decisions[0] == decisions[1]
    assert player.stats_summary()["search_decisions"] == 2
    for winner, expected in ((Color.BLUE, 1.0), (Color.RED, 0.0)):
        monkeypatch.setattr(game, "winning_color", lambda: winner)
        assert own_hand_f_leaf_value(game, Color.BLUE) == expected

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

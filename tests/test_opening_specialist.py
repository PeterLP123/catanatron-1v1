from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from catanatron.colonist_1v1_eval import checkpoint_path_from_agent
from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    write_model_schema,
)
from catanatron.models.enums import ActionPrompt


def _write_policy(tmp_path):
    torch = pytest.importorskip("torch")
    from catanatron.gym.colonist_training import BcCheckpointMeta, build_mlp_layers

    schema = build_model_schema()
    features = schema["observation"]["features"]
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        build_mlp_layers(len(features), len(schema["actions"]), (4,)).state_dict(),
        checkpoint,
    )
    BcCheckpointMeta(
        obs_dim=len(features),
        n_actions=len(schema["actions"]),
        hidden_sizes=[4],
        epochs=1,
        model_schema=schema,
    ).save(checkpoint.with_suffix(".meta.json"))
    write_model_schema(checkpoint_schema_path(checkpoint), schema)
    return checkpoint


def _write_manifest(tmp_path, checkpoint):
    from catanatron.gym.provenance import sha256_file

    manifest = tmp_path / "opening-specialist.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "opening_specialist",
                "policy_checkpoint": checkpoint.name,
                "policy_checkpoint_sha256": sha256_file(checkpoint),
                "policy_metadata_sha256": sha256_file(
                    checkpoint.with_suffix(".meta.json")
                ),
                "policy_schema_sha256": sha256_file(
                    checkpoint.with_suffix(".schema.json")
                ),
                "policy_frozen": True,
                "opening_evaluator": "value_function_default",
                "opening_prompts": [
                    "BUILD_INITIAL_SETTLEMENT",
                    "BUILD_INITIAL_ROAD",
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_opening_specialist_routes_only_initial_build_prompts(tmp_path):
    from catanatron.models.player import Color
    from catanatron.players.learned import OpeningSpecialistCheckpointPlayer

    manifest = _write_manifest(tmp_path, _write_policy(tmp_path))
    player = OpeningSpecialistCheckpointPlayer(Color.BLUE, manifest)
    opening_action = object()
    policy_action = object()
    player.opening.decide = lambda game, actions: opening_action
    player.policy.decide = lambda game, actions: policy_action
    game = SimpleNamespace(
        state=SimpleNamespace(current_prompt=ActionPrompt.BUILD_INITIAL_SETTLEMENT)
    )

    assert player.decide(game, [opening_action, policy_action]) is opening_action
    game.state.current_prompt = ActionPrompt.BUILD_INITIAL_ROAD
    assert player.decide(game, [opening_action, policy_action]) is opening_action
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    assert player.decide(game, [opening_action, policy_action]) is policy_action

    stats = player.stats_summary()
    assert stats["opening_decisions"] == 2
    assert stats["policy_decisions"] == 1
    assert stats["opening_prompt_counts"] == {
        "BUILD_INITIAL_SETTLEMENT": 1,
        "BUILD_INITIAL_ROAD": 1,
    }


def test_opening_specialist_rejects_route_or_sidecar_drift(tmp_path):
    from catanatron.models.player import Color
    from catanatron.players.learned import OpeningSpecialistCheckpointPlayer

    checkpoint = _write_policy(tmp_path)
    manifest = _write_manifest(tmp_path, checkpoint)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["opening_prompts"].append("PLAY_TURN")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prompts must be exactly"):
        OpeningSpecialistCheckpointPlayer(Color.BLUE, manifest)

    manifest = _write_manifest(tmp_path, checkpoint)
    checkpoint.with_suffix(".meta.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        OpeningSpecialistCheckpointPlayer(Color.BLUE, manifest)


def test_opening_specialist_is_registered_and_publishable_path_is_detected():
    from catanatron.cli.cli_players import CLI_PLAYERS

    entry = next(player for player in CLI_PLAYERS if player.code == "O")
    assert entry.name == "OpeningSpecialistCheckpointPlayer"
    assert checkpoint_path_from_agent("O:runs/opening-specialist.json") == (
        Path("runs/opening-specialist.json")
    )


def test_opening_builder_cleans_up_failed_validation_and_can_retry(tmp_path):
    from examples.colonist_1v1_build_opening_specialist import main

    checkpoint = _write_policy(tmp_path)
    output = tmp_path / "manifests" / "opening.json"
    arguments = ["--policy", str(checkpoint), "--output", str(output)]
    metadata = checkpoint.with_suffix(".meta.json")
    original_metadata = metadata.read_bytes()
    metadata.write_text("{}")
    with pytest.raises(TypeError, match="required positional arguments"):
        main(arguments)
    assert list(output.parent.iterdir()) == []
    metadata.write_bytes(original_metadata)
    assert main(arguments) == 0
    assert output.is_file()

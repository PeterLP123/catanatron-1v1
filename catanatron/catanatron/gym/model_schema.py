"""Stable observation/action/rules identities for learned-model artifacts.

Tensor shapes alone do not prove that two checkpoints speak the same feature
and action language.  This module records the ordered schema and compact hashes
so training, warm-start, and inference can fail loudly on semantic drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from catanatron.colonist_1v1 import COLONIST_1V1_SETTINGS
from catanatron.features import get_feature_ordering
from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.player import Color

MODEL_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def default_rules(*, map_type: str = "BASE") -> dict[str, Any]:
    settings = COLONIST_1V1_SETTINGS
    return {
        "mode": "colonist_1v1",
        "num_players": settings.num_players,
        "vps_to_win": settings.vps_to_win,
        "dice_mode": settings.dice_mode,
        "friendly_robber": settings.friendly_robber,
        "friendly_robber_vp_threshold": settings.friendly_robber_vp_threshold,
        "friendly_robber_use_visible_vp": settings.friendly_robber_use_visible_vp,
        "discard_limit": settings.discard_limit,
        "map_type": map_type,
        "number_placement": settings.number_placement,
    }


def build_model_schema(
    *,
    num_players: int = 2,
    map_type: str = "BASE",
    player_colors: Sequence[Color] = (Color.BLUE, Color.RED),
    feature_profile: str = "raw",
    human_visible_obs: bool = False,
    rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    features = list(
        get_feature_ordering(
            num_players=num_players,
            map_type=map_type,
            feature_profile=feature_profile,
        )
    )
    actions = [
        {"type": action_type.name, "value": _jsonable(value)}
        for action_type, value in get_action_array(tuple(player_colors), map_type)
    ]
    resolved_rules = dict(default_rules(map_type=map_type))
    if rules:
        resolved_rules.update(rules)
    observation = {
        "feature_profile": feature_profile,
        "human_visible_obs": bool(human_visible_obs),
        "features": features,
    }
    schema = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "observation": observation,
        "actions": actions,
        "rules": _jsonable(resolved_rules),
        "feature_hash": canonical_hash(observation),
        "action_hash": canonical_hash(actions),
        "rules_hash": canonical_hash(resolved_rules),
    }
    schema["schema_hash"] = canonical_hash(
        {
            "schema_version": schema["schema_version"],
            "feature_hash": schema["feature_hash"],
            "action_hash": schema["action_hash"],
            "rules_hash": schema["rules_hash"],
        }
    )
    return schema


def validate_model_schema(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str = "model",
) -> None:
    """Raise ``ValueError`` when two model schemas are not semantically equal."""

    required = ("schema_version", "feature_hash", "action_hash", "rules_hash")
    missing = [key for key in required if key not in actual]
    if missing:
        raise ValueError(f"{context} schema is missing required fields: {missing}")
    mismatches = [key for key in required if expected.get(key) != actual.get(key)]
    if mismatches:
        details = ", ".join(
            f"{key} expected={expected.get(key)!r} actual={actual.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(f"{context} schema mismatch: {details}")


def checkpoint_schema_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint).with_suffix(".schema.json")


def write_model_schema(path: str | Path, schema: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(schema), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def read_model_schema(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Model schema must be a JSON object: {source}")
    return value

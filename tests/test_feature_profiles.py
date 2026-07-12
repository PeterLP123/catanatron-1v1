from __future__ import annotations

import random

import numpy as np

from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.action_space import get_action_array, to_action_space
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.models.actions import Action
from catanatron.models.player import Color, RandomPlayer


def _env(*, seed: int, feature_profile: str = "raw") -> CatanatronEnv:
    return CatanatronEnv(
        config={
            "colonist_1v1": True,
            "enemies": [RandomPlayer(Color.RED)],
            "seed": seed,
            "feature_profile": feature_profile,
        }
    )


def test_public_derived_profile_is_end_to_end_and_ordered():
    env = _env(seed=11, feature_profile="public_derived")
    observation, _ = env.reset(seed=11)
    ordering = get_feature_ordering(2, "BASE", "public_derived")
    sample = create_sample(env.game, env.p0.color, "public_derived")

    assert len(observation) == len(ordering)
    assert any(name.endswith("_PRODUCTION") for name in ordering)
    assert any("ROAD_REACHABLE" in name for name in ordering)
    np.testing.assert_array_equal(
        observation,
        np.asarray([sample[name] for name in ordering], dtype=np.float32),
    )


def test_env_seed_does_not_mutate_process_global_random_state():
    env = _env(seed=19)
    random.seed(1234)
    expected = random.random()

    random.seed(1234)
    env.reset(seed=99)

    assert random.random() == expected


def test_first_cache_miss_env_construction_does_not_reseed_global_random():
    get_feature_ordering.cache_clear()
    get_action_array.cache_clear()
    random.seed(1234)
    expected = random.random()

    random.seed(1234)
    _env(seed=19)

    assert random.random() == expected


def test_action_codec_reverse_lookup_matches_ordered_array():
    colors = (Color.BLUE, Color.RED)
    for expected, (action_type, value) in enumerate(get_action_array(colors, "BASE")):
        action = Action(Color.BLUE, action_type, value)
        assert to_action_space(action, colors, "BASE") == expected

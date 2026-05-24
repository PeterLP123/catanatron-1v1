from gymnasium.envs.registration import register

register(
    id="catanatron/Catanatron-v0",
    entry_point="catanatron.gym.envs:CatanatronEnv",
)

from catanatron.gym.colonist_rewards import (  # noqa: E402
    COLONIST_SHAPED_VP_SCALE,
    colonist_shaped_reward,
    make_colonist_shaped_reward,
)
from catanatron.gym.utils import infer_vps_cap  # noqa: E402

# Note: do not import SelfPlayEnv here — it pulls in players.learned which imports
# gym.envs.action_space while this package is still initializing (circular import).

__all__ = [
    "COLONIST_SHAPED_VP_SCALE",
    "colonist_shaped_reward",
    "infer_vps_cap",
    "make_colonist_shaped_reward",
]

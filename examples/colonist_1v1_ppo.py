#!/usr/bin/env python3
"""
MaskablePPO on Colonist 1v1 (thin wrapper — prefer ``colonist_1v1_train.py`` for full runs).

Example::

    python examples/colonist_1v1_ppo.py --timesteps 50_000 --save colonist_maskable_ppo.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO  # type: ignore[import-untyped]
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import catanatron.gym  # noqa: F401
from catanatron import Color
from catanatron.colonist_1v1 import Colonist1v1TrainConfig
from catanatron.gym.colonist_rewards import colonist_shaped_reward
from catanatron.gym.colonist_training import warmstart_bc_into_maskable_ppo
from catanatron.gym.wrappers.self_play import SelfPlayEnv
from catanatron.players.weighted_random import WeightedRandomPlayer


def mask_fn(env: gym.Env) -> np.ndarray:
    u = env.unwrapped
    valid = set(u.get_valid_actions())
    n = env.action_space.n
    return np.array([i in valid for i in range(n)], dtype=bool)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=Path, default=Path("colonist_maskable_ppo.zip"))
    p.add_argument("--hidden", type=int, nargs=2, default=(512, 512))
    p.add_argument("--opponent-checkpoints", type=Path, nargs="*", default=[])
    p.add_argument("--bc-checkpoint", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = Colonist1v1TrainConfig(seed=args.seed)
    base = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "colonist_1v1": True,
            "map_type": cfg.map_type,
            "number_placement": cfg.number_placement,
            "vps_to_win": cfg.vps_to_win,
            "representation": "vector",
            "enemies": [WeightedRandomPlayer(Color.RED)],
            "reward_function": colonist_shaped_reward,
        },
    )

    if args.opponent_checkpoints:
        base = SelfPlayEnv(
            base,
            opponent_checkpoints=[str(x) for x in args.opponent_checkpoints],
        )

    env: gym.Env = ActionMasker(base, mask_fn)
    hidden = list(args.hidden)
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        seed=args.seed,
        policy_kwargs=dict(net_arch=dict(pi=hidden, vf=hidden)),
    )

    if args.bc_checkpoint is not None:
        import torch

        state = torch.load(str(args.bc_checkpoint), map_location="cpu")
        n = warmstart_bc_into_maskable_ppo(model.policy, state)
        print(f"BC warm-start: loaded {n} tensors from {args.bc_checkpoint}")

    model.learn(total_timesteps=args.timesteps)
    model.save(str(args.save))
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()

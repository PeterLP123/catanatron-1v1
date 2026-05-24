#!/usr/bin/env python3
"""
Orchestrated Colonist 1v1 MaskablePPO training with checkpoints, league self-play, and eval.

**Dependencies**::

    pip install gymnasium numpy stable-baselines3 sb3-contrib torch

Smoke run::

    python examples/colonist_1v1_train.py --timesteps 20000 --n-envs 2 --eval-freq 10000

Full run (after BC data + optional --bc-checkpoint)::

    python examples/colonist_1v1_train.py --timesteps 1000000 --n-envs 4 \\
        --bc-checkpoint colonist_bc_policy.pt --league-size 8 --eval-freq 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO  # type: ignore[import-untyped]
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

import catanatron.gym  # noqa: F401
from catanatron import Color
from catanatron.colonist_1v1 import Colonist1v1TrainConfig
from catanatron.colonist_1v1_eval import run_benchmark
from catanatron.gym.colonist_rewards import colonist_shaped_reward, make_colonist_shaped_reward
from catanatron.gym.colonist_training import (
    CheckpointLeague,
    load_bc_checkpoint_meta,
    make_mixed_opponent_factory,
    warmstart_bc_into_maskable_ppo,
)
from catanatron.gym.wrappers.self_play import SelfPlayEnv
from catanatron.players.weighted_random import WeightedRandomPlayer


def mask_fn(env: gym.Env) -> np.ndarray:
    u = env.unwrapped
    valid = set(u.get_valid_actions())
    n = env.action_space.n
    return np.array([i in valid for i in range(n)], dtype=bool)


def make_colonist_env(
    *,
    seed: int = 0,
    reward_fn: Callable = colonist_shaped_reward,
    opponent: Optional[Any] = None,
    opponent_factory: Optional[Callable] = None,
    league_paths: Optional[list[str]] = None,
) -> gym.Env:
    cfg = Colonist1v1TrainConfig(seed=seed)
    enemies = [opponent or WeightedRandomPlayer(Color.RED)]
    base = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "colonist_1v1": True,
            "map_type": cfg.map_type,
            "number_placement": cfg.number_placement,
            "vps_to_win": cfg.vps_to_win,
            "representation": "vector",
            "enemies": enemies,
            "reward_function": reward_fn,
        },
    )
    if league_paths:
        base = SelfPlayEnv(base, opponent_checkpoints=league_paths)
    elif opponent_factory is not None:
        base = SelfPlayEnv(
            base,
            opponent_factory=opponent_factory,
            sample_each_reset=True,
        )
    return ActionMasker(base, mask_fn)


class ColonistTrainCallback(BaseCallback):
    """Register checkpoints to league and optionally run eval."""

    def __init__(
        self,
        league: CheckpointLeague,
        ckpt_dir: Path,
        save_freq: int,
        *,
        eval_freq: int = 0,
        report_dir: Optional[Path] = None,
        eval_games: int = 50,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league = league
        self.ckpt_dir = ckpt_dir
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.report_dir = report_dir
        self.eval_games = eval_games
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)

    def _latest_ckpt(self) -> Optional[Path]:
        files = list(self.ckpt_dir.glob("ppo_colonist_*_steps.zip"))

        def _step(p: Path) -> int:
            stem = p.stem  # ppo_colonist_500000_steps
            try:
                return int(stem.split("_")[-2])
            except (ValueError, IndexError):
                return -1

        files.sort(key=_step)
        return files[-1] if files else None

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        latest = self._latest_ckpt()
        if latest is not None:
            self.league.register(latest)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and latest:
            report = run_benchmark(
                f"L:{latest}",
                opponents=("R", "W", "VP"),
                gates=None,
                num_games=self.eval_games,
                quiet=True,
            )
            if self.report_dir:
                out = self.report_dir / f"eval_step_{self.num_timesteps}.json"
                report.write_json(out)
                if self.verbose:
                    print(f"[ColonistEval] saved {out}")
        return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--run-dir", type=Path, default=Path("runs/colonist_1v1"))
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=0, help="0 disables mid-run eval.")
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip post-training benchmark (faster smoke runs).",
    )
    p.add_argument("--bc-checkpoint", type=Path, default=None)
    p.add_argument("--hidden", type=int, nargs=2, default=(512, 512))
    p.add_argument(
        "--visible-vp-reward",
        action="store_true",
        help="Use public VP for shaping instead of actual VP.",
    )
    p.add_argument("--league-size", type=int, default=8)
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (requires tensorboard package).",
    )
    p.add_argument(
        "--mixed-league",
        action="store_true",
        help="Sample league + teacher + baseline opponents each reset.",
    )
    p.add_argument(
        "--league-checkpoints",
        type=Path,
        nargs="*",
        default=[],
        help="Initial league checkpoint paths.",
    )
    args = p.parse_args(argv)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    league = CheckpointLeague(args.run_dir, max_checkpoints=args.league_size)
    for ckpt in args.league_checkpoints:
        league.register(ckpt)

    reward_fn = (
        make_colonist_shaped_reward(use_visible_vp=True)
        if args.visible_vp_reward
        else colonist_shaped_reward
    )

    hidden = list(args.hidden)
    policy_kwargs = dict(net_arch=dict(pi=hidden, vf=hidden))

    def env_fn():
        if args.mixed_league and league.paths():
            factory = make_mixed_opponent_factory(league=league)
            return make_colonist_env(
                seed=args.seed,
                reward_fn=reward_fn,
                opponent_factory=factory,
            )
        if league.paths():
            return make_colonist_env(
                seed=args.seed,
                reward_fn=reward_fn,
                league_paths=league.paths(),
            )
        return make_colonist_env(seed=args.seed, reward_fn=reward_fn)

    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([env_fn for _ in range(args.n_envs)])
    else:
        env = env_fn()

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(args.run_dir / "tb") if args.tensorboard else None,
    )

    if args.bc_checkpoint is not None:
        import torch

        state = torch.load(str(args.bc_checkpoint), map_location="cpu")
        n = warmstart_bc_into_maskable_ppo(model.policy, state)
        meta = load_bc_checkpoint_meta(args.bc_checkpoint.with_suffix(".meta.json"))
        print(f"BC warm-start: loaded {n} tensors from {args.bc_checkpoint}")
        if meta:
            print(f"  BC meta: val_accuracy={meta.val_accuracy}")

    callbacks: list[BaseCallback] = []
    ckpt_dir = args.run_dir / "checkpoints"
    save_freq = max(args.save_freq // max(args.n_envs, 1), 1)
    callbacks.append(
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(ckpt_dir),
            name_prefix="ppo_colonist",
        )
    )
    callbacks.append(
        ColonistTrainCallback(
            league,
            ckpt_dir,
            save_freq,
            eval_freq=args.eval_freq,
            report_dir=args.run_dir / "eval_reports" if args.eval_freq else None,
            eval_games=args.eval_games,
        )
    )

    final_path = args.run_dir / "colonist_maskable_ppo.zip"

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(str(final_path))
    league.register(final_path, label="final")
    print(f"Saved {final_path}")

    if not args.skip_final_eval:
        report = run_benchmark(
            f"L:{final_path}",
            num_games=args.eval_games,
            gates=None,
            quiet=True,
        )
        report_path = args.run_dir / "final_benchmark.json"
        report.write_json(report_path)
        print(f"Final benchmark -> {report_path}")


if __name__ == "__main__":
    main()

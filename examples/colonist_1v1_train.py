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
import sys
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
from catanatron.colonist_1v1_eval import (
    DEFAULT_BENCHMARK_GATES,
    EVAL_PROTOCOLS,
    append_model_registry,
    run_benchmark,
)
from catanatron.gym.colonist_rewards import (
    colonist_shaped_reward,
    make_colonist_shaped_reward,
)
from catanatron.gym.colonist_training import (
    CheckpointLeague,
    MODEL_REGISTRY_NAME,
    TRAINING_PRESETS,
    TrainingRunTracker,
    curriculum_from_name,
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
        eval_protocol: str = "fast",
        tracker: Optional[TrainingRunTracker] = None,
        registry_path: Optional[Path] = None,
        progress_freq: int = 5_000,
        step_state: Optional[dict[str, int]] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league = league
        self.ckpt_dir = ckpt_dir
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.report_dir = report_dir
        self.eval_games = eval_games
        self.eval_protocol = eval_protocol
        self.tracker = tracker
        self.registry_path = registry_path
        self.progress_freq = max(progress_freq, 1)
        self.step_state = step_state
        self.best_weighted_score = -1.0
        self.best_f_win_rate = -1.0
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

    def _promote(self, checkpoint: Path, label: str) -> str:
        import shutil

        promoted_dir = self.league.run_dir / "promoted"
        promoted_dir.mkdir(parents=True, exist_ok=True)
        dest = promoted_dir / f"{label}{checkpoint.suffix}"
        shutil.copy2(checkpoint, dest)
        return str(dest)

    def _on_step(self) -> bool:
        if self.step_state is not None:
            self.step_state["timesteps"] = int(self.num_timesteps)
        if self.tracker and self.num_timesteps % self.progress_freq == 0:
            self.tracker.event(
                "ppo_progress",
                timesteps=int(self.num_timesteps),
                n_calls=int(self.n_calls),
            )
        if self.n_calls % self.save_freq != 0:
            return True
        latest = self._latest_ckpt()
        if latest is not None:
            self.league.register(latest, label=latest.stem)
            if self.tracker:
                self.tracker.event(
                    "checkpoint",
                    path=str(latest),
                    timesteps=int(self.num_timesteps),
                    league_size=len(self.league.paths()),
                )
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and latest:
            report = run_benchmark(
                f"L:{latest}",
                protocol=self.eval_protocol,
                gates=DEFAULT_BENCHMARK_GATES,
                num_games=self.eval_games,
                quiet=True,
                eval_kind="mid_training",
                run_dir=self.league.run_dir,
                checkpoint_path=latest,
                checkpoint_label=latest.stem,
                training_timesteps=int(self.num_timesteps),
            )
            score = float(report.summary.get("weighted_score", 0.0))
            f_rate = next(
                (m.win_rate for m in report.matchups if m.opponent == "F"),
                -1.0,
            )
            if self.report_dir:
                out = self.report_dir / f"eval_step_{self.num_timesteps}.json"
                report.write_json(out)
                if self.registry_path:
                    append_model_registry(self.registry_path, report, report_path=out)
                if score > self.best_weighted_score:
                    self.best_weighted_score = score
                    best_path = self._promote(latest, "best_fast")
                    self.league.register(
                        best_path, label="best_fast", metrics=report.summary
                    )
                    if self.tracker:
                        self.tracker.event(
                            "promotion",
                            label="best_fast",
                            path=best_path,
                            weighted_score=score,
                            timesteps=int(self.num_timesteps),
                        )
                if f_rate > self.best_f_win_rate:
                    self.best_f_win_rate = f_rate
                    best_path = self._promote(latest, "best_f")
                    self.league.register(
                        best_path, label="best_f", metrics={"f_win_rate": f_rate}
                    )
                    if self.tracker:
                        self.tracker.event(
                            "promotion",
                            label="best_f",
                            path=best_path,
                            f_win_rate=f_rate,
                            timesteps=int(self.num_timesteps),
                        )
                if self.tracker:
                    self.tracker.event(
                        "evaluation",
                        path=str(out),
                        protocol=self.eval_protocol,
                        weighted_score=score,
                        all_gates_passed=report.all_gates_passed,
                        timesteps=int(self.num_timesteps),
                    )
                if self.verbose:
                    print(f"[ColonistEval] saved {out}")
        return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=["custom", *sorted(TRAINING_PRESETS)],
        default="custom",
        help="Training preset; custom keeps explicit CLI values.",
    )
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--run-dir", type=Path, default=Path("runs/colonist_1v1"))
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=0, help="0 disables mid-run eval.")
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument(
        "--eval-protocol",
        choices=sorted(EVAL_PROTOCOLS),
        default="fast",
        help="Protocol used for mid-training eval.",
    )
    p.add_argument(
        "--final-eval-protocol",
        choices=sorted(EVAL_PROTOCOLS),
        default="fast",
        help="Protocol used for final eval; use full only for expensive milestone runs.",
    )
    p.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip post-training benchmark (faster smoke runs).",
    )
    p.add_argument("--bc-checkpoint", type=Path, default=None)
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--hidden", type=int, nargs=2, default=(512, 512))
    p.add_argument(
        "--visible-vp-reward",
        action="store_true",
        help="Use public VP for shaping instead of actual VP.",
    )
    p.add_argument("--league-size", type=int, default=8)
    p.add_argument(
        "--curriculum",
        choices=["none", *sorted(["balanced", "strong", "self_play"])],
        default="balanced",
    )
    p.add_argument(
        "--teacher-codes",
        nargs="*",
        default=None,
        help="Override curriculum teacher opponent codes, e.g. F VP G:25.",
    )
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
        "--registry",
        type=Path,
        default=None,
        help="Model registry JSONL path (default: <run-dir>/models_index.jsonl).",
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--progress-freq", type=int, default=5_000)
    p.add_argument(
        "--league-checkpoints",
        type=Path,
        nargs="*",
        default=[],
        help="Initial league checkpoint paths.",
    )
    args = p.parse_args(argv)

    if args.preset != "custom":
        preset = TRAINING_PRESETS[args.preset]
        args.timesteps = preset.timesteps
        args.save_freq = preset.save_freq
        args.eval_freq = preset.eval_freq
        args.eval_games = preset.eval_games
        args.n_envs = preset.n_envs
        args.curriculum = preset.curriculum
        args.mixed_league = True

    args.run_dir.mkdir(parents=True, exist_ok=True)
    tracker = TrainingRunTracker(
        args.run_dir,
        run_id=args.run_id,
        preset=args.preset,
        command=sys.argv if argv is None else ["colonist_1v1_train.py", *argv],
        extra={
            "training": {
                "timesteps": args.timesteps,
                "n_envs": args.n_envs,
                "save_freq": args.save_freq,
                "eval_freq": args.eval_freq,
                "eval_games": args.eval_games,
                "eval_protocol": args.eval_protocol,
                "final_eval_protocol": args.final_eval_protocol,
                "mixed_league": args.mixed_league,
                "curriculum": args.curriculum,
            }
        },
    )
    tracker.phase("initializing")
    registry_path = args.registry or (args.run_dir / MODEL_REGISTRY_NAME)
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
    curriculum = (
        None if args.curriculum == "none" else curriculum_from_name(args.curriculum)
    )
    if curriculum and args.teacher_codes:
        from catanatron.gym.colonist_training import CurriculumSchedule, CurriculumStage

        curriculum = CurriculumSchedule(
            stages=tuple(
                CurriculumStage(
                    start_step=s.start_step,
                    league_weight=s.league_weight,
                    teacher_weight=s.teacher_weight,
                    baseline_weight=s.baseline_weight,
                    teacher_codes=tuple(args.teacher_codes),
                    baseline_code=s.baseline_code,
                )
                for s in curriculum.stages
            )
        )
    step_state = {"timesteps": 0}

    def make_env_fn(rank: int = 0):
        def env_fn():
            env_seed = args.seed + rank
            if args.mixed_league:
                factory = make_mixed_opponent_factory(
                    league=league,
                    curriculum=curriculum,
                    step_getter=lambda: step_state["timesteps"],
                    teacher_codes=(
                        tuple(args.teacher_codes)
                        if args.teacher_codes
                        else ("F", "VP", "W")
                    ),
                    telemetry=tracker if rank == 0 else None,
                )
                return make_colonist_env(
                    seed=env_seed,
                    reward_fn=reward_fn,
                    opponent_factory=factory,
                )
            if league.paths():
                return make_colonist_env(
                    seed=env_seed,
                    reward_fn=reward_fn,
                    league_paths=league.paths(),
                )
            return make_colonist_env(seed=env_seed, reward_fn=reward_fn)

        return env_fn

    def env_fn():
        # Kept for single-env use and backward compatibility with tests/importers.
        return make_env_fn(0)()

    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([make_env_fn(i) for i in range(args.n_envs)])
    else:
        env = env_fn()

    if args.resume_checkpoint is not None:
        tracker.phase("loading_resume", checkpoint=str(args.resume_checkpoint))
        model = MaskablePPO.load(str(args.resume_checkpoint), env=env, seed=args.seed)
    else:
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

        tracker.phase("bc_warmstart", checkpoint=str(args.bc_checkpoint))
        state = torch.load(str(args.bc_checkpoint), map_location="cpu")
        n = warmstart_bc_into_maskable_ppo(model.policy, state)
        meta = load_bc_checkpoint_meta(args.bc_checkpoint.with_suffix(".meta.json"))
        print(f"BC warm-start: loaded {n} tensors from {args.bc_checkpoint}")
        if meta:
            print(f"  BC meta: val_accuracy={meta.val_accuracy}")
            tracker.update_manifest(bc_meta=meta.__dict__)

    callbacks: list[BaseCallback] = []
    ckpt_dir = args.run_dir / "checkpoints"
    save_freq = max(args.save_freq // max(args.n_envs, 1), 1)
    eval_freq = max(args.eval_freq // max(args.n_envs, 1), 1) if args.eval_freq else 0
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
            eval_freq=eval_freq,
            report_dir=args.run_dir / "eval_reports" if args.eval_freq else None,
            eval_games=args.eval_games,
            eval_protocol=args.eval_protocol,
            tracker=tracker,
            registry_path=registry_path,
            progress_freq=args.progress_freq,
            step_state=step_state,
        )
    )

    final_path = args.run_dir / "colonist_maskable_ppo.zip"

    tracker.phase("ppo_training")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=args.resume_checkpoint is None,
    )
    step_state["timesteps"] = int(model.num_timesteps)
    model.save(str(final_path))
    league.register(final_path, label="final")
    tracker.event(
        "checkpoint",
        path=str(final_path),
        label="final",
        timesteps=int(model.num_timesteps),
    )
    tracker.update_manifest(final_model=str(final_path), phase="training_complete")
    print(f"Saved {final_path}")

    if not args.skip_final_eval:
        tracker.phase("final_eval", protocol=args.final_eval_protocol)
        report = run_benchmark(
            f"L:{final_path}",
            num_games=args.eval_games,
            gates=DEFAULT_BENCHMARK_GATES,
            protocol=args.final_eval_protocol,
            quiet=True,
            eval_kind="final_benchmark",
            run_dir=args.run_dir,
            checkpoint_path=final_path,
            checkpoint_label="final",
            training_timesteps=int(model.num_timesteps),
        )
        report_path = args.run_dir / "final_benchmark.json"
        report.write_json(report_path)
        append_model_registry(registry_path, report, report_path=report_path)
        tracker.event(
            "evaluation",
            path=str(report_path),
            protocol=args.final_eval_protocol,
            weighted_score=report.summary.get("weighted_score"),
            all_gates_passed=report.all_gates_passed,
            timesteps=int(model.num_timesteps),
        )
        print(f"Final benchmark -> {report_path}")
    tracker.phase("done")


if __name__ == "__main__":
    main()

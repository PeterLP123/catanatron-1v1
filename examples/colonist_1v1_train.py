#!/usr/bin/env python3
"""
Orchestrated Colonist 1v1 MaskablePPO training with checkpoints, league self-play, and eval.

**Dependencies**::

    pip install gymnasium numpy stable-baselines3 sb3-contrib torch

Smoke run::

    python examples/colonist_1v1_train.py --timesteps 20000 --n-envs 2 --eval-freq 10000

Full run (after BC data + optional --bc-checkpoint)::

    python examples/colonist_1v1_train.py --timesteps 1000000 --n-envs 4 \\
        --bc-checkpoint runs/colonist_bc_policy.pt --league-size 8 --eval-freq 50000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import sys
import zipfile
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
from catanatron.gym.model_schema import (
    build_model_schema,
    checkpoint_schema_path,
    read_model_schema,
    validate_model_schema,
    write_model_schema,
)
from catanatron.gym.provenance import (
    collect_run_provenance,
    sha256_file,
    write_environment_snapshot,
)
from catanatron.gym.wrappers.self_play import SelfPlayEnv
from catanatron.players.weighted_random import WeightedRandomPlayer


FRESH_RUN_EVIDENCE_FILES = frozenset({"experiment_evidence.json"})


def _fresh_run_blockers(run_dir: Path) -> list[Path]:
    """Return files that make ``run_dir`` unsafe for a fresh training run.

    Evidence-gated backlog launches write ``experiment_evidence.json`` before
    invoking the trainer. That audited input must survive startup, while every
    other pre-existing file still blocks an accidental overwrite.
    """
    if not run_dir.exists():
        return []
    return [
        path for path in run_dir.iterdir() if path.name not in FRESH_RUN_EVIDENCE_FILES
    ]


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
    randomize_seats: bool = True,
    feature_profile: str = "raw",
    human_visible_obs: bool = False,
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
            "feature_profile": feature_profile,
            "human_visible_obs": human_visible_obs,
            "seed": seed,
            "enemies": enemies,
            "reward_function": reward_fn,
            "randomize_seats": randomize_seats,
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


def _schedule_value(value: Any) -> Any:
    if callable(value):
        try:
            return float(value(1.0))
        except (TypeError, ValueError):
            return repr(value)
    return float(value) if isinstance(value, (int, float)) else value


def effective_ppo_config(model: Any) -> dict[str, Any]:
    """Read the values SB3 will actually use, including resumed checkpoints."""

    from torch import nn

    policy_layers = [
        module.out_features
        for module in model.policy.mlp_extractor.policy_net
        if isinstance(module, nn.Linear)
    ]
    return {
        "learning_rate": _schedule_value(
            getattr(model, "lr_schedule", model.learning_rate)
        ),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "ent_coef": float(model.ent_coef),
        "clip_range": _schedule_value(model.clip_range),
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "hidden": policy_layers,
    }


def materialize_final_candidate(
    model: Any,
    run_dir: Path,
    *,
    model_schema: dict[str, Any],
    promoted_checkpoint: Optional[Path] = None,
    promoted_timesteps: Optional[int] = None,
    allow_legacy_schema: bool = False,
) -> tuple[Path, Path, Path, int]:
    """Save optimizer state and choose one pre-final-eval candidate canonically."""

    last_path = run_dir / "last_maskable_ppo.zip"
    model.save(str(last_path))
    write_model_schema(checkpoint_schema_path(last_path), model_schema)
    source = promoted_checkpoint if promoted_checkpoint is not None else last_path
    if not source.exists():
        raise FileNotFoundError(f"Selected final candidate does not exist: {source}")
    if promoted_checkpoint is not None:
        stored_schema = read_model_schema(checkpoint_schema_path(source))
        if stored_schema is None and not allow_legacy_schema:
            raise ValueError(
                f"Selected promotion {source} has no model schema; refusing to "
                "publish an unverifiable checkpoint"
            )
        if stored_schema is not None:
            validate_model_schema(
                model_schema, stored_schema, context="selected promotion"
            )
        if promoted_timesteps is None:
            promoted_timesteps = _checkpoint_num_timesteps(source)
        selected_timesteps = promoted_timesteps
    else:
        selected_timesteps = int(model.num_timesteps)
    final_path = run_dir / "colonist_maskable_ppo.zip"
    shutil.copy2(source, final_path)
    source_schema = checkpoint_schema_path(source)
    if source_schema.exists():
        shutil.copy2(source_schema, checkpoint_schema_path(final_path))
    else:
        write_model_schema(checkpoint_schema_path(final_path), model_schema)
    return final_path, last_path, source, int(selected_timesteps)


def _checkpoint_num_timesteps(checkpoint: Path) -> int:
    """Read SB3 training age without constructing a model or touching RNG state."""

    try:
        with zipfile.ZipFile(checkpoint) as archive:
            data = json.loads(archive.read("data"))
        return int(data["num_timesteps"])
    except (KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Cannot determine training timesteps for promoted checkpoint {checkpoint}"
        ) from exc


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
        promotion_eval_freq: int = 0,
        promotion_eval_games: Optional[int] = None,
        promotion_eval_protocol: str = "milestone",
        tracker: Optional[TrainingRunTracker] = None,
        registry_path: Optional[Path] = None,
        progress_freq: int = 5_000,
        step_state: Optional[dict[str, int]] = None,
        model_schema: Optional[dict[str, Any]] = None,
        restore_selection_state: bool = False,
        allow_legacy_schema: bool = False,
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
        self.promotion_eval_freq = promotion_eval_freq
        self.promotion_eval_games = promotion_eval_games
        self.promotion_eval_protocol = promotion_eval_protocol
        self.tracker = tracker
        self.registry_path = registry_path
        self.progress_freq = max(progress_freq, 1)
        self.step_state = step_state
        self.model_schema = model_schema
        self.allow_legacy_schema = allow_legacy_schema
        self.best_dev_weighted_score = -1.0
        self.best_dev_f_win_rate = -1.0
        self.best_promotion_weighted_score = -1.0
        self.best_promotion_path: Optional[Path] = None
        self.best_promotion_timesteps: Optional[int] = None
        if restore_selection_state:
            self._restore_selection_state()
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)

    def _restore_selection_state(self) -> None:
        """Resume model selection without downgrading an existing champion."""

        def restore_promotion(
            path: Path, *, score: float, timesteps: Any = None
        ) -> None:
            if not path.exists() or score < self.best_promotion_weighted_score:
                return
            if self.model_schema is not None:
                stored_schema = read_model_schema(checkpoint_schema_path(path))
                if stored_schema is None and not self.allow_legacy_schema:
                    raise ValueError(f"Resumed promotion {path} has no model schema")
                if stored_schema is not None:
                    validate_model_schema(
                        self.model_schema,
                        stored_schema,
                        context="resumed promotion",
                    )
            self.best_promotion_weighted_score = score
            self.best_promotion_path = path
            self.best_promotion_timesteps = (
                int(timesteps)
                if timesteps is not None
                else _checkpoint_num_timesteps(path)
            )

        for entry in self.league.entries():
            label = entry.get("label")
            metrics = entry.get("metrics") or {}
            if label == "best_dev_score":
                self.best_dev_weighted_score = max(
                    self.best_dev_weighted_score,
                    float(metrics.get("weighted_score", -1.0)),
                )
            elif label == "best_dev_f":
                self.best_dev_f_win_rate = max(
                    self.best_dev_f_win_rate,
                    float(metrics.get("f_win_rate", -1.0)),
                )
            elif label == "best_promotion":
                score = float(metrics.get("weighted_score", -1.0))
                restore_promotion(
                    Path(entry["path"]),
                    score=score,
                    timesteps=metrics.get("training_timesteps"),
                )

        events_path = self.league.run_dir / "training_events.jsonl"
        try:
            event_lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            event_lines = []
        for line in event_lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = event.get("label")
            path = event.get("path")
            if label == "best_dev_score":
                self.best_dev_weighted_score = max(
                    self.best_dev_weighted_score,
                    float(event.get("weighted_score", -1.0)),
                )
            elif label == "best_dev_f":
                self.best_dev_f_win_rate = max(
                    self.best_dev_f_win_rate,
                    float(event.get("f_win_rate", -1.0)),
                )
            elif label == "best_promotion" and path:
                score = float(event.get("weighted_score", -1.0))
                restore_promotion(
                    Path(path), score=score, timesteps=event.get("timesteps")
                )

        promoted = self.league.run_dir / "promoted" / "best_promotion.zip"
        if promoted.exists() and self.best_promotion_path is None:
            # Legacy runs may have the artifact without a persisted score. Keep
            # it rather than allowing an arbitrary first resumed eval to win.
            self.best_promotion_weighted_score = float("inf")
            restore_promotion(promoted, score=float("inf"))

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
        source_schema = checkpoint_schema_path(checkpoint)
        if source_schema.exists():
            shutil.copy2(source_schema, checkpoint_schema_path(dest))
        return str(dest)

    def _write_schema(self, checkpoint: Path) -> None:
        if self.model_schema is not None:
            write_model_schema(checkpoint_schema_path(checkpoint), self.model_schema)

    @staticmethod
    def _checkpoint_step(path: Optional[Path]) -> int:
        if path is None:
            return -1
        try:
            return int(path.stem.split("_")[-2])
        except (ValueError, IndexError):
            return -1

    def _current_checkpoint(self) -> Path:
        latest = self._latest_ckpt()
        if self._checkpoint_step(latest) != int(self.num_timesteps):
            latest = self.ckpt_dir / f"ppo_colonist_{self.num_timesteps}_steps.zip"
            self.model.save(str(latest))
        self._write_schema(latest)
        return latest

    def _record_evaluation(
        self,
        checkpoint: Path,
        *,
        eval_kind: str,
        protocol: str,
        num_games: Optional[int],
        gate_mode: str,
    ):
        kwargs = {}
        if num_games is not None:
            kwargs["num_games"] = num_games
        report = run_benchmark(
            f"L:{checkpoint}",
            protocol=protocol,
            gates=DEFAULT_BENCHMARK_GATES,
            quiet=True,
            eval_kind=eval_kind,
            run_dir=self.league.run_dir,
            checkpoint_path=checkpoint,
            checkpoint_label=checkpoint.stem,
            training_timesteps=int(self.num_timesteps),
            gate_mode=gate_mode,
            **kwargs,
        )
        if self.report_dir is None:
            return report, None
        out = self.report_dir / f"{eval_kind}_step_{self.num_timesteps}.json"
        report.write_json(out)
        if self.registry_path:
            append_model_registry(self.registry_path, report, report_path=out)
        if self.tracker:
            self.tracker.event(
                "evaluation",
                eval_kind=eval_kind,
                path=str(out),
                protocol=protocol,
                gate_mode=gate_mode,
                weighted_score=report.summary.get("weighted_score"),
                all_gates_passed=report.all_gates_passed,
                all_games_accounted=report.summary.get("all_games_accounted"),
                timesteps=int(self.num_timesteps),
            )
        if self.verbose:
            print(f"[ColonistEval] saved {out}")
        return report, out

    def _on_step(self) -> bool:
        if self.step_state is not None:
            self.step_state["timesteps"] = int(self.num_timesteps)
        if self.tracker and self.num_timesteps % self.progress_freq == 0:
            self.tracker.event(
                "ppo_progress",
                timesteps=int(self.num_timesteps),
                n_calls=int(self.n_calls),
            )
        save_due = self.n_calls % self.save_freq == 0
        eval_due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        promotion_due = (
            self.promotion_eval_freq > 0
            and self.n_calls % self.promotion_eval_freq == 0
        )
        if not save_due and not eval_due and not promotion_due:
            return True
        latest = self._current_checkpoint()
        if save_due or eval_due or promotion_due:
            self.league.register(latest, label=latest.stem)
            if self.tracker:
                self.tracker.event(
                    "checkpoint",
                    path=str(latest),
                    timesteps=int(self.num_timesteps),
                    league_size=len(self.league.paths()),
                )
        if eval_due:
            report, _ = self._record_evaluation(
                latest,
                eval_kind="dev",
                protocol=self.eval_protocol,
                num_games=self.eval_games,
                gate_mode="point",
            )
            score = float(report.summary.get("weighted_score", 0.0))
            f_rate = next(
                (m.win_rate for m in report.matchups if m.opponent == "F"),
                -1.0,
            )
            if score > self.best_dev_weighted_score:
                self.best_dev_weighted_score = score
                best_path = self._promote(latest, "best_dev_score")
                self.league.register(
                    best_path, label="best_dev_score", metrics=report.summary
                )
                if self.tracker:
                    self.tracker.event(
                        "model_selection",
                        label="best_dev_score",
                        path=best_path,
                        weighted_score=score,
                        evidence="dev_only",
                        timesteps=int(self.num_timesteps),
                    )
            if f_rate > self.best_dev_f_win_rate:
                self.best_dev_f_win_rate = f_rate
                best_path = self._promote(latest, "best_dev_f")
                self.league.register(
                    best_path, label="best_dev_f", metrics={"f_win_rate": f_rate}
                )
                if self.tracker:
                    self.tracker.event(
                        "model_selection",
                        label="best_dev_f",
                        path=best_path,
                        f_win_rate=f_rate,
                        evidence="dev_only",
                        timesteps=int(self.num_timesteps),
                    )
        if promotion_due:
            report, _ = self._record_evaluation(
                latest,
                eval_kind="promotion",
                protocol=self.promotion_eval_protocol,
                num_games=self.promotion_eval_games,
                gate_mode="lower_bound",
            )
            score = float(report.summary.get("weighted_score", 0.0))
            if report.all_gates_passed and score > self.best_promotion_weighted_score:
                self.best_promotion_weighted_score = score
                best_path = self._promote(latest, "best_promotion")
                self.best_promotion_path = Path(best_path)
                self.best_promotion_timesteps = int(self.num_timesteps)
                self.league.register(
                    best_path,
                    label="best_promotion",
                    metrics={
                        **report.summary,
                        "training_timesteps": int(self.num_timesteps),
                    },
                )
                if self.tracker:
                    self.tracker.event(
                        "promotion",
                        label="best_promotion",
                        path=best_path,
                        weighted_score=score,
                        gate_mode="lower_bound",
                        timesteps=int(self.num_timesteps),
                    )
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
        "--promotion-eval-freq",
        type=int,
        default=0,
        help="Locked promotion-suite frequency; 0 disables promotion during training.",
    )
    p.add_argument(
        "--promotion-eval-games",
        type=int,
        default=None,
        help="Override promotion protocol game count.",
    )
    p.add_argument(
        "--promotion-eval-protocol",
        choices=sorted(EVAL_PROTOCOLS),
        default="milestone",
    )
    p.add_argument(
        "--final-eval-games",
        type=int,
        default=None,
        help="Override the final protocol's game count; default respects the protocol.",
    )
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
        "--final-gate-mode",
        choices=("point", "lower_bound"),
        default="lower_bound",
        help="Final evidence defaults to confidence-lower-bound gates.",
    )
    p.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip post-training benchmark (faster smoke runs).",
    )
    checkpoint_group = p.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--bc-checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--hidden", type=int, nargs=2, default=(512, 512))
    p.add_argument(
        "--feature-profile",
        choices=("raw", "public_derived"),
        default="raw",
        help="Observation schema: raw baseline or raw plus public production/reachability.",
    )
    p.add_argument(
        "--human-visible-obs",
        action="store_true",
        help="Hide the acting player's private VP cards in observations.",
    )
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument(
        "--visible-vp-reward",
        action="store_true",
        help="Use public VP for shaping instead of actual VP.",
    )
    p.add_argument(
        "--no-randomize-seats",
        dest="randomize_seats",
        action="store_false",
        default=True,
        help="Disable seat randomization (legacy: the agent always moves first).",
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
    p.add_argument(
        "--allow-legacy-schema",
        action="store_true",
        help="Allow resume/warm-start artifacts without schema hashes (unsafe compatibility mode).",
    )
    p.add_argument("--progress-freq", type=int, default=5_000)
    p.add_argument(
        "--vec-env",
        choices=("auto", "dummy", "subproc"),
        default="auto",
        help="Vector environment backend; auto uses subprocesses when n-envs > 1.",
    )
    p.add_argument(
        "--vec-start-method",
        choices=("auto", "spawn", "forkserver", "fork"),
        default="auto",
        help="Multiprocessing start method for SubprocVecEnv.",
    )
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

    same_run_resume = bool(
        args.resume_checkpoint is not None
        and args.run_dir.resolve() in args.resume_checkpoint.resolve().parents
    )
    existing_run_files = _fresh_run_blockers(args.run_dir)
    if existing_run_files and not same_run_resume:
        raise ValueError(
            f"Run directory {args.run_dir} is not empty. Use a new directory for a "
            "fresh run, or resume a checkpoint from this run directory."
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    training_manifest = {
        "timesteps": args.timesteps,
        "requested_timesteps": args.timesteps,
        "starting_timesteps": 0,
        "n_envs": args.n_envs,
        "save_freq": args.save_freq,
        "eval_freq": args.eval_freq,
        "promotion_eval_freq": args.promotion_eval_freq,
        "promotion_eval_games": args.promotion_eval_games,
        "promotion_eval_protocol": args.promotion_eval_protocol,
        "eval_games": args.eval_games,
        "final_eval_games": args.final_eval_games,
        "eval_protocol": args.eval_protocol,
        "final_eval_protocol": args.final_eval_protocol,
        "final_gate_mode": args.final_gate_mode,
        "mixed_league": args.mixed_league,
        "curriculum": args.curriculum,
        "seed": args.seed,
        "visible_vp_reward": args.visible_vp_reward,
        "randomize_seats": args.randomize_seats,
        "teacher_codes": args.teacher_codes,
        "hidden_requested": list(args.hidden),
        "hidden": None,
        "feature_profile": args.feature_profile,
        "human_visible_obs": args.human_visible_obs,
        "ppo_requested": {
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "ent_coef": args.ent_coef,
            "clip_range": args.clip_range,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
        },
        "ppo": None,
        "vec_env_requested": args.vec_env,
        "vec_start_method_requested": args.vec_start_method,
    }
    tracker = TrainingRunTracker(
        args.run_dir,
        run_id=args.run_id,
        preset=args.preset,
        command=sys.argv if argv is None else ["colonist_1v1_train.py", *argv],
        extra={"training": training_manifest},
    )
    tracker.phase("initializing")
    model_schema = build_model_schema(
        feature_profile=args.feature_profile,
        human_visible_obs=args.human_visible_obs,
    )
    schema_path = write_model_schema(args.run_dir / "model_schema.json", model_schema)
    environment_path = write_environment_snapshot(args.run_dir)
    training_manifest["schema"] = {
        "path": str(schema_path),
        "schema_hash": model_schema["schema_hash"],
        "feature_hash": model_schema["feature_hash"],
        "action_hash": model_schema["action_hash"],
        "rules_hash": model_schema["rules_hash"],
        "num_features": len(model_schema["observation"]["features"]),
        "num_actions": len(model_schema["actions"]),
    }
    tracker.update_manifest(
        training=training_manifest,
        provenance=collect_run_provenance(),
        environment_lock=str(environment_path),
    )
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
    requested_vec_env = args.vec_env
    resolved_vec_env = (
        "subproc"
        if requested_vec_env == "auto" and args.n_envs > 1
        else requested_vec_env
    )
    if resolved_vec_env == "auto":
        resolved_vec_env = "dummy"
    available_methods = multiprocessing.get_all_start_methods()
    if args.vec_start_method == "auto":
        resolved_start_method = (
            "forkserver" if "forkserver" in available_methods else "spawn"
        )
    else:
        resolved_start_method = args.vec_start_method

    manager = None
    if resolved_vec_env == "subproc":
        manager = multiprocessing.Manager()
        step_state = manager.dict(timesteps=0)
    else:
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
                    rng=np.random.default_rng(env_seed),
                )
                return make_colonist_env(
                    seed=env_seed,
                    reward_fn=reward_fn,
                    opponent_factory=factory,
                    randomize_seats=args.randomize_seats,
                    feature_profile=args.feature_profile,
                    human_visible_obs=args.human_visible_obs,
                )
            if league.paths():
                return make_colonist_env(
                    seed=env_seed,
                    reward_fn=reward_fn,
                    league_paths=league.paths(),
                    randomize_seats=args.randomize_seats,
                    feature_profile=args.feature_profile,
                    human_visible_obs=args.human_visible_obs,
                )
            return make_colonist_env(
                seed=env_seed,
                reward_fn=reward_fn,
                randomize_seats=args.randomize_seats,
                feature_profile=args.feature_profile,
                human_visible_obs=args.human_visible_obs,
            )

        return env_fn

    def env_fn():
        # Kept for single-env use and backward compatibility with tests/importers.
        return make_env_fn(0)()

    vec_fallback_reason = None
    if resolved_vec_env == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv

        try:
            env = SubprocVecEnv(
                [make_env_fn(i) for i in range(args.n_envs)],
                start_method=resolved_start_method,
            )
        except Exception as exc:
            vec_fallback_reason = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING: SubprocVecEnv failed ({vec_fallback_reason}); falling back to DummyVecEnv.",
                file=sys.stderr,
            )
            tracker.event("vec_env_fallback", reason=vec_fallback_reason)
            if manager is not None:
                manager.shutdown()
                manager = None
            step_state = {"timesteps": 0}
            from stable_baselines3.common.vec_env import DummyVecEnv

            env = DummyVecEnv([make_env_fn(i) for i in range(args.n_envs)])
            resolved_vec_env = "dummy"
            resolved_start_method = None
    elif args.n_envs > 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([make_env_fn(i) for i in range(args.n_envs)])
        resolved_start_method = None
    else:
        env = env_fn()
        resolved_start_method = None

    training_manifest.update(
        {
            "vec_env": resolved_vec_env,
            "vec_start_method": resolved_start_method,
            "vec_fallback_reason": vec_fallback_reason,
        }
    )
    tracker.update_manifest(training=training_manifest)

    if args.resume_checkpoint is not None:
        tracker.phase("loading_resume", checkpoint=str(args.resume_checkpoint))
        model = MaskablePPO.load(str(args.resume_checkpoint), env=env, seed=args.seed)
        resume_schema = getattr(model, "catanatron_model_schema", None)
        if resume_schema is None:
            resume_schema = read_model_schema(
                checkpoint_schema_path(args.resume_checkpoint)
            )
        if resume_schema is None and not args.allow_legacy_schema:
            raise ValueError(
                "Resume checkpoint has no Catanatron model schema. "
                "Use --allow-legacy-schema only after manually confirming compatibility."
            )
        if resume_schema is not None:
            validate_model_schema(model_schema, resume_schema, context="resume")
        training_manifest["starting_timesteps"] = int(model.num_timesteps)
        training_manifest["timesteps"] = int(model.num_timesteps) + args.timesteps
        tracker.update_manifest(training=training_manifest)
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=str(args.run_dir / "tb") if args.tensorboard else None,
        )
    model.catanatron_model_schema = model_schema
    effective_config = effective_ppo_config(model)
    training_manifest["hidden"] = effective_config.pop("hidden")
    training_manifest["ppo"] = effective_config
    training_manifest["configuration_source"] = (
        "resume_checkpoint" if args.resume_checkpoint is not None else "cli"
    )
    tracker.update_manifest(training=training_manifest)

    if args.bc_checkpoint is not None:
        import torch

        tracker.phase("bc_warmstart", checkpoint=str(args.bc_checkpoint))
        state = torch.load(str(args.bc_checkpoint), map_location="cpu")
        meta = load_bc_checkpoint_meta(args.bc_checkpoint.with_suffix(".meta.json"))
        bc_schema = read_model_schema(checkpoint_schema_path(args.bc_checkpoint))
        if bc_schema is None and meta is not None and meta.model_schema:
            bc_schema = meta.model_schema
        if bc_schema is None and not args.allow_legacy_schema:
            raise ValueError(
                "BC checkpoint has no feature/action/rules schema. "
                "Use --allow-legacy-schema only after manually confirming compatibility."
            )
        warmstart_kwargs = {}
        if bc_schema is not None:
            warmstart_kwargs = {
                "checkpoint_schema": bc_schema,
                "expected_schema": model_schema,
            }
        n = warmstart_bc_into_maskable_ppo(model.policy, state, **warmstart_kwargs)
        print(f"BC warm-start: loaded {n} tensors from {args.bc_checkpoint}")
        if meta:
            print(f"  BC meta: val_accuracy={meta.val_accuracy}")
            tracker.update_manifest(bc_meta=meta.__dict__)

    callbacks: list[BaseCallback] = []
    ckpt_dir = args.run_dir / "checkpoints"
    save_freq = max(args.save_freq // max(args.n_envs, 1), 1)
    eval_freq = max(args.eval_freq // max(args.n_envs, 1), 1) if args.eval_freq else 0
    promotion_eval_freq = (
        max(args.promotion_eval_freq // max(args.n_envs, 1), 1)
        if args.promotion_eval_freq
        else 0
    )
    callbacks.append(
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(ckpt_dir),
            name_prefix="ppo_colonist",
        )
    )
    train_callback = ColonistTrainCallback(
        league,
        ckpt_dir,
        save_freq,
        eval_freq=eval_freq,
        report_dir=(
            args.run_dir / "eval_reports"
            if args.eval_freq or args.promotion_eval_freq
            else None
        ),
        eval_games=args.eval_games,
        eval_protocol=args.eval_protocol,
        promotion_eval_freq=promotion_eval_freq,
        promotion_eval_games=args.promotion_eval_games,
        promotion_eval_protocol=args.promotion_eval_protocol,
        tracker=tracker,
        registry_path=registry_path,
        progress_freq=args.progress_freq,
        step_state=step_state,
        model_schema=model_schema,
        restore_selection_state=same_run_resume,
        allow_legacy_schema=args.allow_legacy_schema,
    )
    callbacks.append(train_callback)

    tracker.phase("ppo_training")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=args.resume_checkpoint is None,
    )
    step_state["timesteps"] = int(model.num_timesteps)
    final_path, last_path, final_source, final_timesteps = materialize_final_candidate(
        model,
        args.run_dir,
        model_schema=model_schema,
        promoted_checkpoint=train_callback.best_promotion_path,
        promoted_timesteps=train_callback.best_promotion_timesteps,
        allow_legacy_schema=args.allow_legacy_schema,
    )
    league.register(final_path, label="final")
    tracker.event(
        "checkpoint",
        path=str(final_path),
        label="final",
        timesteps=final_timesteps,
    )
    tracker.update_manifest(
        final_model=str(final_path),
        final_model_sha256=sha256_file(final_path),
        final_model_schema=str(checkpoint_schema_path(final_path)),
        last_model=str(last_path),
        last_model_sha256=sha256_file(last_path),
        final_candidate_source=str(final_source),
        final_candidate_selection=(
            "locked_promotion" if train_callback.best_promotion_path else "last_state"
        ),
        final_candidate_timesteps=final_timesteps,
        last_model_timesteps=int(model.num_timesteps),
        phase="training_complete",
    )
    print(f"Saved {final_path}")

    if not args.skip_final_eval:
        tracker.phase("final_eval", protocol=args.final_eval_protocol)
        final_eval_kwargs = {}
        if args.final_eval_games is not None:
            final_eval_kwargs["num_games"] = args.final_eval_games
        report = run_benchmark(
            f"L:{final_path}",
            gates=DEFAULT_BENCHMARK_GATES,
            protocol=args.final_eval_protocol,
            quiet=True,
            eval_kind="final_benchmark",
            gate_mode=args.final_gate_mode,
            run_dir=args.run_dir,
            checkpoint_path=final_path,
            checkpoint_label="final",
            training_timesteps=final_timesteps,
            **final_eval_kwargs,
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
            timesteps=final_timesteps,
        )
        print(f"Final benchmark -> {report_path}")
    tracker.phase("done")
    env.close()
    if manager is not None:
        manager.shutdown()


if __name__ == "__main__":
    main()

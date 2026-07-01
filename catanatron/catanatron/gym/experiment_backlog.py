"""Definitions and status helpers for the Colonist 1v1 GPU experiment backlog."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class Experiment:
    id: str
    stage: str
    title: str
    hypothesis: str
    gpu_hours: tuple[float, float]
    storage_gib: float
    env: tuple[tuple[str, str], ...]
    success_rule: str
    depends_on: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()

    def launch_env(self) -> dict[str, str]:
        return dict(self.env)


# Keep matched comparisons adjacent. The 5M and self-play runs are deliberately
# gated: the existing evidence says scaling PPO without an early F signal is wasteful.
EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        id="00-gpu-smoke",
        stage="Validate",
        title="CUDA + dashboard smoke test",
        hypothesis="The UCL host can train, checkpoint, evaluate and feed the TUI.",
        gpu_hours=(0.1, 0.3),
        storage_gib=0.15,
        env=(("TRAIN_PRESET", "smoke"), ("SEED", "7"), ("SKIP_FINAL_EVAL", "1")),
        success_rule="20k steps finish; checkpoints and a mid-run evaluation appear; no CUDA error.",
    ),
    Experiment(
        id="10-balanced-actual-s101",
        stage="Reward A/B",
        title="Balanced PPO · actual VP reward · seed 101",
        hypothesis="The current actual-VP shaping is the stronger reward baseline.",
        gpu_hours=(0.3, 1.0),
        storage_gib=0.4,
        env=(("TRAIN_PRESET", "standard"), ("SEED", "101")),
        success_rule="Record fast two-seat scorecard; this is the matched control for experiment 11.",
        depends_on=("00-gpu-smoke",),
    ),
    Experiment(
        id="11-balanced-visible-s101",
        stage="Reward A/B",
        title="Balanced PPO · visible VP reward · seed 101",
        hypothesis="Public-score-only shaping improves robustness despite providing a weaker learning signal.",
        gpu_hours=(0.3, 1.0),
        storage_gib=0.4,
        env=(
            ("TRAIN_PRESET", "standard"),
            ("SEED", "101"),
            ("VISIBLE_VP_REWARD", "1"),
        ),
        success_rule="Beat experiment 10 by ≥0.03 weighted score without regressing R/W/VP gates.",
        depends_on=("00-gpu-smoke",),
    ),
    Experiment(
        id="12-balanced-actual-s202",
        stage="Replication",
        title="Balanced PPO · actual VP reward · seed 202",
        hypothesis="Any apparent actual-VP advantage survives a second seed.",
        gpu_hours=(0.3, 1.0),
        storage_gib=0.4,
        env=(("TRAIN_PRESET", "standard"), ("SEED", "202")),
        success_rule="Use only if actual VP wins seed 101; direction of improvement must repeat.",
        depends_on=("10-balanced-actual-s101", "11-balanced-visible-s101"),
    ),
    Experiment(
        id="13-balanced-visible-s202",
        stage="Replication",
        title="Balanced PPO · visible VP reward · seed 202",
        hypothesis="Any apparent visible-VP advantage survives a second seed.",
        gpu_hours=(0.3, 1.0),
        storage_gib=0.4,
        env=(
            ("TRAIN_PRESET", "standard"),
            ("SEED", "202"),
            ("VISIBLE_VP_REWARD", "1"),
        ),
        success_rule="Use only if visible VP wins seed 101; direction of improvement must repeat.",
        depends_on=("10-balanced-actual-s101", "11-balanced-visible-s101"),
    ),
    Experiment(
        id="20-hard-bc-actual-s101",
        stage="Hard states",
        title="Hard-state BC warm-start · actual VP",
        hypothesis="Choice-focused BC gives PPO a better decision-margin initialization.",
        gpu_hours=(0.4, 1.2),
        storage_gib=0.4,
        env=(("TRAIN_PRESET", "standard"), ("SEED", "101")),
        required_inputs=("BC_CHECKPOINT",),
        success_rule="Lower held-out decision regret, then improve F win rate or VP margin over experiment 10.",
        depends_on=("00-gpu-smoke",),
    ),
    Experiment(
        id="21-hard-bc-visible-s101",
        stage="Hard states",
        title="Hard-state BC warm-start · visible VP",
        hypothesis="The hard-state initialization combines best with public-score-only shaping.",
        gpu_hours=(0.4, 1.2),
        storage_gib=0.4,
        env=(
            ("TRAIN_PRESET", "standard"),
            ("SEED", "101"),
            ("VISIBLE_VP_REWARD", "1"),
        ),
        required_inputs=("BC_CHECKPOINT",),
        success_rule="Run only if visible VP wins the reward A/B; compare against experiment 11.",
        depends_on=("10-balanced-actual-s101", "11-balanced-visible-s101"),
    ),
    Experiment(
        id="30-strong-promoted",
        stage="Scale",
        title="Promote the best checkpoint to strong curriculum",
        hypothesis="A model with an early F signal benefits from 5M additional strong-curriculum steps.",
        gpu_hours=(3.0, 10.0),
        storage_gib=3.5,
        env=(("TRAIN_PRESET", "strong"), ("SEED", "303")),
        required_inputs=("RESUME_CHECKPOINT",),
        success_rule="Do not start unless a 500k run reaches ≥10% vs F and keeps all weak gates.",
        depends_on=("10-balanced-actual-s101", "11-balanced-visible-s101"),
    ),
    Experiment(
        id="40-selfplay-polish",
        stage="Polish",
        title="Short anchored self-play polish",
        hypothesis="Self-play refines an already-strong policy without collapsing against heuristics.",
        gpu_hours=(0.4, 1.5),
        storage_gib=0.5,
        env=(
            ("TRAIN_PRESET", "custom"),
            ("TIMESTEPS", "500000"),
            ("SAVE_FREQ", "50000"),
            ("EVAL_FREQ", "50000"),
            ("EVAL_GAMES", "50"),
            ("N_ENVS", "4"),
            ("CURRICULUM", "self_play"),
            ("SEED", "404"),
        ),
        required_inputs=("RESUME_CHECKPOINT",),
        success_rule="Keep only if F/search strength rises and R/W/VP remain within 2 points of the parent.",
        depends_on=("30-strong-promoted",),
    ),
)


def experiments_by_id() -> dict[str, Experiment]:
    return {experiment.id: experiment for experiment in EXPERIMENTS}


def validate_backlog(experiments: Sequence[Experiment] = EXPERIMENTS) -> None:
    ids = [experiment.id for experiment in experiments]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique")
    known = set(ids)
    for experiment in experiments:
        missing = set(experiment.depends_on) - known
        if missing:
            raise ValueError(
                f"{experiment.id} has unknown dependencies: {sorted(missing)}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()
    lookup = {experiment.id: experiment for experiment in experiments}

    def visit(experiment_id: str) -> None:
        if experiment_id in visiting:
            raise ValueError(f"Dependency cycle at {experiment_id}")
        if experiment_id in visited:
            return
        visiting.add(experiment_id)
        for dependency in lookup[experiment_id].depends_on:
            visit(dependency)
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in ids:
        visit(experiment_id)


def read_manifest(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def direct_status(experiment: Experiment, runs_root: Path) -> str:
    run_dir = runs_root / experiment.id
    manifest = read_manifest(run_dir)
    phase = manifest.get("phase")
    if phase == "done":
        return "done"
    if phase in {
        "ppo_training",
        "initializing",
        "loading_resume",
        "bc_warmstart",
        "final_eval",
    }:
        return "running"
    active_job = manifest.get("active_job")
    if isinstance(active_job, dict) and active_job.get("status") == "failed":
        return "failed"
    if manifest or run_dir.exists():
        return "partial"
    return "pending"


def backlog_statuses(runs_root: Path) -> dict[str, str]:
    direct = {
        experiment.id: direct_status(experiment, runs_root)
        for experiment in EXPERIMENTS
    }
    statuses = dict(direct)
    for experiment in EXPERIMENTS:
        if direct[experiment.id] != "pending":
            continue
        if any(
            direct.get(dependency) != "done" for dependency in experiment.depends_on
        ):
            statuses[experiment.id] = "blocked"
        else:
            statuses[experiment.id] = "ready"
    return statuses


def launch_environment(
    experiment: Experiment,
    *,
    supplied: Optional[Mapping[str, str]] = None,
    inherited: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    inherited = inherited or os.environ
    supplied = supplied or {}
    result = experiment.launch_env()
    result.update({key: value for key, value in supplied.items() if value})
    missing = [
        key
        for key in experiment.required_inputs
        if not result.get(key) and not inherited.get(key)
    ]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{experiment.id} requires: {names}")
    for key in experiment.required_inputs:
        if not result.get(key) and inherited.get(key):
            result[key] = str(inherited[key])
    result["EXPERIMENT_ID"] = experiment.id
    result["RUN_NAME"] = experiment.id
    return result


def launch_command(
    experiment: Experiment, supplied: Optional[Mapping[str, str]] = None
) -> str:
    env = launch_environment(experiment, supplied=supplied)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return f"{assignments} bash scripts/ucl_cs/start_run.sh"


def load_final_metrics(run_dir: Path) -> Optional[dict]:
    path = run_dir / "final_benchmark.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    rates = {
        row.get("opponent"): row.get("win_rate")
        for row in report.get("matchups", [])
        if row.get("opponent")
    }
    seat_gaps = [
        abs(row["win_rate_seat0"] - row["win_rate_seat1"])
        for row in report.get("matchups", [])
        if isinstance(row.get("win_rate_seat0"), (int, float))
        and isinstance(row.get("win_rate_seat1"), (int, float))
    ]
    return {
        "weighted_score": report.get("summary", {}).get("weighted_score"),
        "rates": rates,
        "max_seat_gap": max(seat_gaps) if seat_gaps else None,
        "all_gates_passed": report.get("all_gates_passed"),
    }


validate_backlog()

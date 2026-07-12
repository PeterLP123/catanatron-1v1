"""Executable definitions and evidence gates for the Colonist 1v1 backlog."""

from __future__ import annotations

import json
import math
import os
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


WEAK_OPPONENT_GATES: dict[str, float] = {"R": 0.90, "W": 0.70, "VP": 0.60}
REWARD_DELTA_GATE = 0.03
F_PROMOTION_GATE = 0.10
REWARD_PAIR_WEIGHTS: dict[str, float] = {
    "R": 0.08,
    "W": 0.12,
    "VP": 0.15,
    "F": 0.35,
    "G:25": 0.15,
    "M:200": 0.10,
    "AB:2": 0.05,
}
SEARCH_SWEEP_BUDGETS_MS: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)
SEARCH_SWEEP_OPPONENTS: tuple[str, ...] = ("F", "AB:2")
SEARCH_SWEEP_SEEDS: tuple[int, ...] = (20_260_711, 20_260_712, 20_260_713)


class ExperimentOutcome(str, Enum):
    """Scientific decision produced by completed experiment evidence."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class GateDecision:
    outcome: ExperimentOutcome
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


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
    gate: str = "report_complete"
    unlock: str = "all_accepted"
    command: tuple[str, ...] = ()

    def launch_env(self) -> dict[str, str]:
        return dict(self.env)


# Keep matched comparisons adjacent. Expensive scaling remains evidence-gated.
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
        gate="smoke_artifacts",
    ),
    Experiment(
        id="05-mcts-strength-sweep",
        stage="Validate",
        title="MCTS latency/strength sweep",
        hypothesis="Repaired MCTS has a measurable latency budget where it challenges F or AB:2.",
        gpu_hours=(0.0, 0.0),
        storage_gib=0.1,
        env=(),
        success_rule="Complete 10/25/50/100 ms, F/AB:2, two-seat held-out-seed evidence with p95 latency.",
        gate="search_sweep",
        command=(
            "python",
            "examples/colonist_1v1_search_benchmark.py",
            "--budgets",
            "10,25,50,100",
            "--opponents",
            "F,AB:2",
            "--seeds",
            "20260711,20260712,20260713",
            "--num-games",
            "20",
            "--report",
            "runs/05-mcts-strength-sweep/mcts_strength_sweep.json",
        ),
    ),
    Experiment(
        id="10-balanced-actual-s101",
        stage="Reward A/B",
        title="Balanced PPO · actual VP reward · seed 101",
        hypothesis="The current actual-VP shaping is the stronger reward baseline.",
        gpu_hours=(0.3, 1.0),
        storage_gib=0.4,
        env=(("TRAIN_PRESET", "standard"), ("SEED", "101")),
        success_rule="Produce a complete, comparable two-seat scorecard for experiment 11.",
        depends_on=("00-gpu-smoke", "05-mcts-strength-sweep"),
    ),
    Experiment(
        id="11-balanced-visible-s101",
        stage="Reward A/B",
        title="Balanced PPO · visible VP reward · seed 101",
        hypothesis="Public-score-only shaping improves robustness despite a weaker learning signal.",
        gpu_hours=(0.3, 1.0),
        storage_gib=0.4,
        env=(
            ("TRAIN_PRESET", "standard"),
            ("SEED", "101"),
            ("VISIBLE_VP_REWARD", "1"),
        ),
        success_rule="Paired outcome gain ≥0.03 over experiment 10 without regressing R/W/VP gates.",
        depends_on=("00-gpu-smoke", "05-mcts-strength-sweep"),
        gate="visible_reward",
    ),
    Experiment(
        id="12-balanced-actual-s202",
        stage="Replication",
        title="Balanced PPO · actual VP reward · seed 202",
        hypothesis="The seed-101 actual-VP winner remains credible under seed 202.",
        gpu_hours=(0.3, 1.0),
        storage_gib=0.4,
        env=(("TRAIN_PRESET", "standard"), ("SEED", "202")),
        success_rule="Run only when actual VP wins seed 101; retain all weak-opponent gates.",
        depends_on=("10-balanced-actual-s101", "11-balanced-visible-s101"),
        gate="weak_report",
        unlock="reward_actual_winner",
    ),
    Experiment(
        id="13-balanced-visible-s202",
        stage="Replication",
        title="Balanced PPO · visible VP reward · seed 202",
        hypothesis="The seed-101 visible-VP winner remains credible under seed 202.",
        gpu_hours=(0.3, 1.0),
        storage_gib=0.4,
        env=(
            ("TRAIN_PRESET", "standard"),
            ("SEED", "202"),
            ("VISIBLE_VP_REWARD", "1"),
        ),
        success_rule="Run only when visible VP wins seed 101; retain all weak-opponent gates.",
        depends_on=("10-balanced-actual-s101", "11-balanced-visible-s101"),
        gate="weak_report",
        unlock="reward_visible_winner",
    ),
    Experiment(
        id="20-hard-bc-actual-s101",
        stage="Hard states",
        title="Hard-state BC warm-start · actual VP",
        hypothesis="Choice-focused BC gives PPO a better decision-margin initialization.",
        gpu_hours=(0.4, 1.2),
        storage_gib=0.4,
        env=(("TRAIN_PRESET", "standard"), ("SEED", "101")),
        required_inputs=("BC_CHECKPOINT", "BC_BASELINE_CHECKPOINT"),
        success_rule="Lower held-out BC regret, then improve F rate or VP margin over experiment 10.",
        depends_on=("10-balanced-actual-s101",),
        gate="bc_actual",
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
        required_inputs=("BC_CHECKPOINT", "BC_BASELINE_CHECKPOINT"),
        success_rule="Lower held-out BC regret and improve the accepted visible-reward baseline.",
        depends_on=("10-balanced-actual-s101", "11-balanced-visible-s101"),
        gate="bc_visible",
        unlock="reward_visible_winner",
    ),
    Experiment(
        id="30-strong-promoted",
        stage="Scale",
        title="Promote the best checkpoint to strong curriculum",
        hypothesis="A model with an early F signal benefits from 5M strong-curriculum steps.",
        gpu_hours=(3.0, 10.0),
        storage_gib=3.5,
        env=(("TRAIN_PRESET", "strong"), ("SEED", "303")),
        required_inputs=("RESUME_CHECKPOINT",),
        success_rule="Start only after F ≥10% and all weak gates; retain both after scaling.",
        gate="promotion_strength",
        unlock="promotion_candidate",
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
        success_rule="F/search rises and each R/W/VP result stays within 2 points of the parent.",
        depends_on=("30-strong-promoted",),
        gate="selfplay_polish",
    ),
)


def experiments_by_id() -> dict[str, Experiment]:
    return {experiment.id: experiment for experiment in EXPERIMENTS}


def validate_backlog(experiments: Sequence[Experiment] = EXPERIMENTS) -> None:
    ids = [experiment.id for experiment in experiments]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique")
    known = set(ids)
    known_gates = {
        "smoke_artifacts",
        "search_sweep",
        "report_complete",
        "visible_reward",
        "weak_report",
        "bc_actual",
        "bc_visible",
        "promotion_strength",
        "selfplay_polish",
    }
    known_unlocks = {
        "all_accepted",
        "reward_actual_winner",
        "reward_visible_winner",
        "promotion_candidate",
    }
    for experiment in experiments:
        missing = set(experiment.depends_on) - known
        if missing:
            raise ValueError(
                f"{experiment.id} has unknown dependencies: {sorted(missing)}"
            )
        if experiment.gate not in known_gates:
            raise ValueError(f"{experiment.id} has unknown gate: {experiment.gate}")
        if experiment.unlock not in known_unlocks:
            raise ValueError(f"{experiment.id} has unknown unlock: {experiment.unlock}")

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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def read_manifest(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "run_manifest.json")


def _final_report(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "final_benchmark.json")


def _search_report(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "mcts_strength_sweep.json")


def _report_rates(report: Mapping[str, Any]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for row in report.get("matchups", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("opponent"), str):
            continue
        value = row.get("win_rate")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            rates[str(row["opponent"])] = float(value)
    return rates


def _report_vp_diffs(report: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in report.get("matchups", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("opponent"), str):
            continue
        value = row.get("avg_vp_diff")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[str(row["opponent"])] = float(value)
    return values


def _report_signature(report: Mapping[str, Any]) -> Optional[tuple[Any, ...]]:
    if report.get("schema_version") != "1.1" or report.get("colonist_1v1") is not True:
        return None
    meta = report.get("meta")
    if not isinstance(meta, Mapping):
        return None
    protocol = meta.get("protocol")
    if not isinstance(protocol, Mapping):
        return None
    opponents = protocol.get("opponents")
    if not isinstance(opponents, list) or not opponents:
        return None
    required = ("name", "num_games_per_matchup", "seed", "seed_suite", "gate_mode")
    if any(protocol.get(key) is None for key in required):
        return None
    if meta.get("both_seats") is not True:
        return None
    git_commit = meta.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit:
        return None
    return (
        report.get("schema_version"),
        report.get("colonist_1v1"),
        protocol.get("name"),
        tuple(opponents),
        protocol.get("num_games_per_matchup"),
        protocol.get("seed"),
        protocol.get("seed_suite"),
        protocol.get("gate_mode"),
        meta.get("both_seats"),
        git_commit,
    )


def reports_comparable(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """True only for reports produced by the same complete evaluation schedule."""
    left_signature = _report_signature(left)
    return left_signature is not None and left_signature == _report_signature(right)


def _report_complete(report: Mapping[str, Any]) -> tuple[bool, str]:
    signature = _report_signature(report)
    if signature is None:
        return False, "report lacks complete protocol, seat, seed, or commit metadata"
    matchups = report.get("matchups")
    if not isinstance(matchups, list) or not matchups:
        return False, "report has no matchups"
    requested = report.get("meta", {}).get("protocol", {}).get("num_games_per_matchup")
    expected_opponents = set(
        report.get("meta", {}).get("protocol", {}).get("opponents", [])
    )
    reported_opponents = {
        row.get("opponent") for row in matchups if isinstance(row, Mapping)
    }
    if reported_opponents != expected_opponents:
        return False, "report matchups do not match the declared opponent protocol"
    for row in matchups:
        if not isinstance(row, Mapping):
            return False, "report contains a malformed matchup"
        observed = row.get("requested_games", row.get("games"))
        if observed != requested:
            return (
                False,
                f"{row.get('opponent', '?')} accounts for {observed}/{requested} games",
            )
        game_rows = _game_rows(row)
        if len(game_rows) != requested:
            return False, f"{row.get('opponent', '?')} lacks per-game evidence"
        if any(str(game.get("outcome", "")).lower() == "error" for game in game_rows):
            return False, f"{row.get('opponent', '?')} contains errored games"
    return True, "complete comparable report"


def _weak_gate_result(
    report: Mapping[str, Any],
) -> tuple[Optional[bool], dict[str, float]]:
    rates = _report_rates(report)
    missing = [opponent for opponent in WEAK_OPPONENT_GATES if opponent not in rates]
    if missing:
        return None, {
            opponent: rates.get(opponent, float("nan")) for opponent in missing
        }
    failures = {
        opponent: rates[opponent]
        for opponent, threshold in WEAK_OPPONENT_GATES.items()
        if rates[opponent] < threshold
    }
    return not failures, failures


def _game_rows(matchup: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = matchup.get("game_results", matchup.get("per_game_results", []))
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _game_score(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("score")
    if isinstance(value, (int, float)):
        return float(value)
    outcome = str(row.get("outcome", row.get("result", ""))).lower()
    if outcome == "win":
        return 1.0
    if outcome in {"draw", "truncated"}:
        return 0.5
    if outcome == "loss":
        return 0.0
    return None


def _game_key(row: Mapping[str, Any], fallback_index: int) -> tuple[Any, ...]:
    schedule_id = row.get("schedule_id")
    if schedule_id is not None:
        return (schedule_id,)
    return (
        row.get("agent_seat", row.get("seat")),
        row.get("seed"),
        row.get("game_index", fallback_index),
    )


def paired_outcome_delta(
    treatment: Mapping[str, Any], control: Mapping[str, Any]
) -> Optional[float]:
    """Mean treatment-control game score on identical schedules/opponents."""
    treatment_rows = {
        row.get("opponent"): row
        for row in treatment.get("matchups", [])
        if isinstance(row, Mapping)
    }
    control_rows = {
        row.get("opponent"): row
        for row in control.get("matchups", [])
        if isinstance(row, Mapping)
    }
    if not treatment_rows or treatment_rows.keys() != control_rows.keys():
        return None
    opponent_deltas: dict[str, float] = {}
    for opponent in sorted(treatment_rows):
        treatment_games = _game_rows(treatment_rows[opponent])
        control_games = _game_rows(control_rows[opponent])
        treatment_by_key = {
            _game_key(row, index): _game_score(row)
            for index, row in enumerate(treatment_games)
        }
        control_by_key = {
            _game_key(row, index): _game_score(row)
            for index, row in enumerate(control_games)
        }
        if not treatment_by_key or treatment_by_key.keys() != control_by_key.keys():
            return None
        deltas: list[float] = []
        for key, treatment_score in treatment_by_key.items():
            control_score = control_by_key[key]
            if treatment_score is None or control_score is None:
                return None
            deltas.append(treatment_score - control_score)
        opponent_deltas[str(opponent)] = sum(deltas) / len(deltas)
    total_weight = sum(
        REWARD_PAIR_WEIGHTS.get(opponent, 0.0) for opponent in opponent_deltas
    )
    if total_weight <= 0:
        return sum(opponent_deltas.values()) / len(opponent_deltas)
    return (
        sum(
            REWARD_PAIR_WEIGHTS.get(opponent, 0.0) * delta
            for opponent, delta in opponent_deltas.items()
        )
        / total_weight
    )


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _heldout_regret(metadata: Mapping[str, Any]) -> Optional[float]:
    direct = _finite_number(metadata.get("mean_regret"))
    if direct is not None:
        return direct
    # Validation is the promotion split. Test metrics remain a fallback for old
    # sidecars that did not persist validation details.
    for section_name in ("val_metrics", "validation_metrics", "test_metrics"):
        section = metadata.get(section_name)
        if isinstance(section, Mapping):
            value = _finite_number(section.get("mean_regret"))
            if value is not None:
                return value
    return None


def bc_regret_pair(
    metadata: Mapping[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    candidate_meta = metadata.get("candidate")
    candidate = _heldout_regret(
        candidate_meta if isinstance(candidate_meta, Mapping) else metadata
    )
    baseline = None
    for key in (
        "baseline_mean_regret",
        "ce_baseline_mean_regret",
        "reference_mean_regret",
    ):
        baseline = _finite_number(metadata.get(key))
        if baseline is not None:
            break
    baseline_meta = metadata.get("baseline")
    if baseline is None and isinstance(baseline_meta, Mapping):
        baseline = _heldout_regret(baseline_meta)
    return candidate, baseline


def _accepted(reason: str, **evidence: Any) -> GateDecision:
    return GateDecision(ExperimentOutcome.ACCEPTED, reason, evidence)


def _rejected(reason: str, **evidence: Any) -> GateDecision:
    return GateDecision(ExperimentOutcome.REJECTED, reason, evidence)


def _inconclusive(reason: str, **evidence: Any) -> GateDecision:
    return GateDecision(ExperimentOutcome.INCONCLUSIVE, reason, evidence)


def _smoke_gate(run_dir: Path) -> GateDecision:
    manifest = read_manifest(run_dir)
    checkpoints = [
        path
        for path in (
            list(run_dir.glob("*.zip")) + list((run_dir / "checkpoints").glob("*.zip"))
        )
        if path.is_file() and path.stat().st_size > 0
    ]
    eval_reports = [
        path
        for path in (run_dir / "eval_reports").glob("*.json")
        if _read_json(path).get("meta", {}).get("eval_kind") == "dev"
    ]
    if manifest.get("phase") != "done":
        return _inconclusive("training manifest has not reached phase=done")
    training = manifest.get("training", {})
    if not isinstance(training, Mapping) or training.get("timesteps", 0) < 20_000:
        return _rejected("smoke run did not complete the declared 20k-step profile")
    provenance = manifest.get("provenance", {})
    hardware = provenance.get("hardware", {}) if isinstance(provenance, Mapping) else {}
    if not isinstance(hardware, Mapping) or hardware.get("cuda_available") is not True:
        return _rejected("smoke run did not record an available CUDA device")
    if not checkpoints or not eval_reports:
        return _rejected(
            "smoke run finished without required checkpoint/evaluation artifacts",
            checkpoints=len(checkpoints),
            evaluations=len(eval_reports),
        )
    return _accepted(
        "training completed with checkpoint and evaluation artifacts",
        checkpoints=len(checkpoints),
        evaluations=len(eval_reports),
    )


def _search_sweep_gate(run_dir: Path) -> GateDecision:
    report = _search_report(run_dir)
    meta = report.get("meta", {})
    if not report:
        return _inconclusive("MCTS strength sweep report is missing")
    if (
        report.get("schema_version") != "1.0"
        or report.get("kind") != "mcts_strength_sweep"
    ):
        return _rejected("search sweep has an unknown report schema or kind")
    if not isinstance(meta.get("git_commit"), str) or not meta.get("git_commit"):
        return _inconclusive("search sweep did not record its git commit")
    if meta.get("profile_only") is not False:
        return _rejected("profile-only output is not strength evidence")
    if meta.get("both_seats") is not True:
        return _rejected("search sweep did not use both seats")
    if tuple(float(v) for v in meta.get("budgets_ms", ())) != SEARCH_SWEEP_BUDGETS_MS:
        return _rejected("search sweep budgets differ from the backlog protocol")
    if tuple(meta.get("opponents", ())) != SEARCH_SWEEP_OPPONENTS:
        return _rejected("search sweep opponents differ from the backlog protocol")
    if tuple(meta.get("seeds", ())) != SEARCH_SWEEP_SEEDS:
        return _rejected("search sweep did not use the held-out seed suite")
    expected = (
        len(SEARCH_SWEEP_BUDGETS_MS)
        * len(SEARCH_SWEEP_OPPONENTS)
        * len(SEARCH_SWEEP_SEEDS)
    )
    rows = report.get("results", [])
    if not isinstance(rows, list) or len(rows) != expected:
        return _inconclusive(
            f"search sweep has {len(rows) if isinstance(rows, list) else 0}/{expected} cells"
        )
    expected_cells = {
        (budget, opponent, seed)
        for budget in SEARCH_SWEEP_BUDGETS_MS
        for opponent in SEARCH_SWEEP_OPPONENTS
        for seed in SEARCH_SWEEP_SEEDS
    }
    observed_cells = {
        (row.get("budget_ms"), row.get("opponent"), row.get("seed"))
        for row in rows
        if isinstance(row, Mapping)
    }
    if observed_cells != expected_cells:
        return _rejected("search sweep cells do not match the declared matrix")
    games_per_cell = meta.get("num_games_per_cell")
    if not isinstance(games_per_cell, int) or games_per_cell <= 0:
        return _inconclusive("search sweep did not record games per cell")
    for row in rows:
        if not isinstance(row, Mapping):
            return _rejected("search sweep contains a malformed result")
        profile = row.get("profile", {})
        matchup = row.get("matchup", {})
        p95 = profile.get("p95_latency_ms")
        if (
            not isinstance(p95, (int, float))
            or not math.isfinite(float(p95))
            or p95 <= 0
        ):
            return _inconclusive("search sweep is missing p95 latency")
        games = _game_rows(matchup)
        requested = matchup.get("requested_games", matchup.get("games"))
        if requested != games_per_cell or len(games) != requested:
            return _inconclusive("search sweep lacks complete per-game evidence")
        if matchup.get("opponent") != row.get("opponent"):
            return _rejected("search sweep matchup metadata disagrees with its cell")
        if {game.get("agent_seat") for game in games} != {0, 1}:
            return _rejected("search sweep cell does not cover both agent seats")
        if any(
            not game.get("schedule_id")
            or str(game.get("outcome", "")).lower() == "error"
            for game in games
        ):
            return _inconclusive("search sweep has missing schedules or errored games")
    return _accepted(
        "all latency/opponent/seed cells have complete two-seat evidence",
        cells=expected,
    )


def _complete_report_gate(run_dir: Path) -> GateDecision:
    report = _final_report(run_dir)
    complete, reason = _report_complete(report)
    return _accepted(reason) if complete else _inconclusive(reason)


def _weak_report_gate(run_dir: Path) -> GateDecision:
    report = _final_report(run_dir)
    complete, reason = _report_complete(report)
    if not complete:
        return _inconclusive(reason)
    weak_passed, details = _weak_gate_result(report)
    if weak_passed is None:
        return _inconclusive("weak-opponent matchups are missing", missing=details)
    if not weak_passed:
        return _rejected("one or more weak-opponent gates failed", failures=details)
    return _accepted("complete report retains all weak-opponent gates")


def _visible_reward_gate(runs_root: Path, run_dir: Path) -> GateDecision:
    treatment = _final_report(run_dir)
    control = _final_report(runs_root / "10-balanced-actual-s101")
    for report in (control, treatment):
        complete, reason = _report_complete(report)
        if not complete:
            return _inconclusive(reason)
    if not reports_comparable(treatment, control):
        return _inconclusive(
            "reward reports were not produced by the same protocol/commit"
        )
    delta = paired_outcome_delta(treatment, control)
    if delta is None:
        return _inconclusive("paired per-game outcomes are missing or schedules differ")
    weak_passed, failures = _weak_gate_result(treatment)
    if weak_passed is not True:
        return _rejected(
            "visible reward regressed a weak-opponent gate",
            delta=delta,
            failures=failures,
        )
    if delta >= REWARD_DELTA_GATE:
        return _accepted(
            "visible reward clears the paired +0.03 gate", paired_delta=delta
        )
    if delta <= -REWARD_DELTA_GATE:
        return _rejected(
            "actual reward wins by at least 0.03 on paired outcomes", paired_delta=delta
        )
    return _inconclusive(
        "paired reward difference is smaller than 0.03", paired_delta=delta
    )


def _bc_gate(runs_root: Path, run_dir: Path, baseline_id: str) -> GateDecision:
    manifest = read_manifest(run_dir)
    launch_evidence = _read_json(run_dir / "experiment_evidence.json")
    candidate_regret, baseline_regret = bc_regret_pair(launch_evidence.get("bc", {}))
    if candidate_regret is None or baseline_regret is None:
        candidate_regret, baseline_regret = bc_regret_pair(manifest.get("bc_meta", {}))
    if candidate_regret is None or baseline_regret is None:
        return _inconclusive("BC metadata lacks candidate and baseline held-out regret")
    if candidate_regret >= baseline_regret:
        return _rejected(
            "BC held-out decision regret did not improve",
            candidate_regret=candidate_regret,
            baseline_regret=baseline_regret,
        )
    candidate = _final_report(run_dir)
    baseline = _final_report(runs_root / baseline_id)
    for report in (baseline, candidate):
        complete, reason = _report_complete(report)
        if not complete:
            return _inconclusive(reason)
    if not reports_comparable(candidate, baseline):
        return _inconclusive("BC and baseline reports are not comparable")
    weak_passed, failures = _weak_gate_result(candidate)
    if weak_passed is not True:
        return _rejected(
            "BC candidate regressed a weak-opponent gate", failures=failures
        )
    rates = _report_rates(candidate)
    baseline_rates = _report_rates(baseline)
    vp_diffs = _report_vp_diffs(candidate)
    baseline_vp_diffs = _report_vp_diffs(baseline)
    rate_gain = rates.get("F", float("-inf")) - baseline_rates.get("F", float("-inf"))
    vp_gain = vp_diffs.get("F", float("-inf")) - baseline_vp_diffs.get(
        "F", float("-inf")
    )
    if rate_gain > 0 or vp_gain > 0:
        return _accepted(
            "lower-regret BC improves F win rate or VP margin",
            regret_delta=candidate_regret - baseline_regret,
            f_rate_gain=rate_gain,
            f_vp_gain=vp_gain,
        )
    return _rejected(
        "lower-regret BC did not improve F win rate or VP margin",
        regret_delta=candidate_regret - baseline_regret,
        f_rate_gain=rate_gain,
        f_vp_gain=vp_gain,
    )


def _promotion_strength_gate(run_dir: Path) -> GateDecision:
    report = _final_report(run_dir)
    complete, reason = _report_complete(report)
    if not complete:
        return _inconclusive(reason)
    weak_passed, failures = _weak_gate_result(report)
    f_rate = _report_rates(report).get("F")
    if weak_passed is not True:
        return _rejected("promoted run failed weak-opponent gates", failures=failures)
    if f_rate is None:
        return _inconclusive("promoted report has no F matchup")
    if f_rate < F_PROMOTION_GATE:
        return _rejected("promoted run fell below the 10% F signal gate", f_rate=f_rate)
    return _accepted("promoted run retains F ≥10% and all weak gates", f_rate=f_rate)


def _selfplay_gate(runs_root: Path, run_dir: Path) -> GateDecision:
    candidate = _final_report(run_dir)
    parent = _final_report(runs_root / "30-strong-promoted")
    for report in (parent, candidate):
        complete, reason = _report_complete(report)
        if not complete:
            return _inconclusive(reason)
    if not reports_comparable(candidate, parent):
        return _inconclusive("self-play and parent reports are not comparable")
    candidate_rates = _report_rates(candidate)
    parent_rates = _report_rates(parent)
    weak_regressions = {
        opponent: parent_rates[opponent] - candidate_rates.get(opponent, float("-inf"))
        for opponent in WEAK_OPPONENT_GATES
        if parent_rates.get(opponent, 0.0)
        - candidate_rates.get(opponent, float("-inf"))
        > 0.02
    }
    if weak_regressions:
        return _rejected(
            "self-play regressed a weak opponent by more than 2 points",
            regressions=weak_regressions,
        )
    strength_opponents = ("F", "G:25", "M:200", "AB:2")
    gains = {
        opponent: candidate_rates[opponent] - parent_rates[opponent]
        for opponent in strength_opponents
        if opponent in candidate_rates and opponent in parent_rates
    }
    if not gains:
        return _inconclusive("self-play comparison lacks F/search matchups")
    if max(gains.values()) <= 0:
        return _rejected("self-play did not improve F or search strength", gains=gains)
    return _accepted(
        "self-play improves F/search without weak-tier collapse", gains=gains
    )


def evaluate_experiment(experiment: Experiment, runs_root: Path) -> GateDecision:
    run_dir = runs_root / experiment.id
    if experiment.gate == "smoke_artifacts":
        return _smoke_gate(run_dir)
    if experiment.gate == "search_sweep":
        return _search_sweep_gate(run_dir)
    if experiment.gate == "report_complete":
        return _complete_report_gate(run_dir)
    if experiment.gate == "visible_reward":
        return _visible_reward_gate(runs_root, run_dir)
    if experiment.gate == "weak_report":
        return _weak_report_gate(run_dir)
    if experiment.gate == "bc_actual":
        return _bc_gate(runs_root, run_dir, "10-balanced-actual-s101")
    if experiment.gate == "bc_visible":
        return _bc_gate(runs_root, run_dir, "11-balanced-visible-s101")
    if experiment.gate == "promotion_strength":
        return _promotion_strength_gate(run_dir)
    if experiment.gate == "selfplay_polish":
        return _selfplay_gate(runs_root, run_dir)
    raise ValueError(f"Unknown gate {experiment.gate!r}")


def _execution_status(experiment: Experiment, runs_root: Path) -> str:
    run_dir = runs_root / experiment.id
    manifest = read_manifest(run_dir)
    if experiment.gate == "search_sweep" and _search_report(run_dir):
        return "complete"
    if manifest.get("phase") == "done":
        return "complete"
    if manifest.get("phase") in {
        "ppo_training",
        "initializing",
        "loading_resume",
        "bc_warmstart",
        "final_eval",
    }:
        return "running"
    active_job = manifest.get("active_job")
    if isinstance(active_job, Mapping) and active_job.get("status") == "failed":
        return "failed"
    if manifest or run_dir.exists():
        return "partial"
    return "pending"


def direct_status(experiment: Experiment, runs_root: Path) -> str:
    lifecycle = _execution_status(experiment, runs_root)
    if lifecycle != "complete":
        return lifecycle
    return evaluate_experiment(experiment, runs_root).outcome.value


def experiment_decisions(runs_root: Path) -> dict[str, Optional[GateDecision]]:
    decisions: dict[str, Optional[GateDecision]] = {}
    for experiment in EXPERIMENTS:
        if _execution_status(experiment, runs_root) == "complete":
            decisions[experiment.id] = evaluate_experiment(experiment, runs_root)
        else:
            decisions[experiment.id] = None
    return decisions


def _candidate_meets_promotion_gate(runs_root: Path) -> bool:
    candidate_ids = (
        "10-balanced-actual-s101",
        "11-balanced-visible-s101",
        "12-balanced-actual-s202",
        "13-balanced-visible-s202",
        "20-hard-bc-actual-s101",
        "21-hard-bc-visible-s101",
    )
    for candidate_id in candidate_ids:
        report = _final_report(runs_root / candidate_id)
        complete, _ = _report_complete(report)
        weak_passed, _ = _weak_gate_result(report)
        if (
            complete
            and weak_passed is True
            and _report_rates(report).get("F", 0.0) >= F_PROMOTION_GATE
        ):
            return True
    return False


def _pending_readiness(
    experiment: Experiment,
    decisions: Mapping[str, Optional[GateDecision]],
    runs_root: Path,
) -> str:
    if experiment.unlock == "promotion_candidate":
        return "ready" if _candidate_meets_promotion_gate(runs_root) else "blocked"
    if experiment.unlock in {"reward_actual_winner", "reward_visible_winner"}:
        control = decisions.get("10-balanced-actual-s101")
        treatment = decisions.get("11-balanced-visible-s101")
        if (
            control is None
            or control.outcome is not ExperimentOutcome.ACCEPTED
            or treatment is None
        ):
            return "blocked"
        if treatment.outcome is ExperimentOutcome.INCONCLUSIVE:
            return "blocked"
        visible_won = treatment.outcome is ExperimentOutcome.ACCEPTED
        selected = (
            visible_won
            if experiment.unlock == "reward_visible_winner"
            else not visible_won
        )
        return "ready" if selected else "skipped"
    if all(
        decisions.get(dependency) is not None
        and decisions[dependency].outcome is ExperimentOutcome.ACCEPTED
        for dependency in experiment.depends_on
    ):
        return "ready"
    return "blocked"


def backlog_statuses(runs_root: Path) -> dict[str, str]:
    decisions = experiment_decisions(runs_root)
    statuses: dict[str, str] = {}
    for experiment in EXPERIMENTS:
        direct = direct_status(experiment, runs_root)
        if direct == "pending":
            statuses[experiment.id] = _pending_readiness(
                experiment, decisions, runs_root
            )
        else:
            statuses[experiment.id] = direct
    return statuses


def _load_bc_launch_metadata(checkpoint: Path) -> dict[str, Any]:
    return _read_json(checkpoint.with_suffix(".meta.json"))


def launch_environment(
    experiment: Experiment,
    *,
    supplied: Optional[Mapping[str, str]] = None,
    inherited: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    inherited = os.environ if inherited is None else inherited
    supplied = {} if supplied is None else supplied
    result = experiment.launch_env()
    result.update({key: value for key, value in supplied.items() if value})
    missing = [
        key
        for key in experiment.required_inputs
        if not result.get(key) and not inherited.get(key)
    ]
    if missing:
        raise ValueError(f"{experiment.id} requires: {', '.join(missing)}")
    for key in experiment.required_inputs:
        if not result.get(key) and inherited.get(key):
            result[key] = str(inherited[key])
    if "BC_CHECKPOINT" in experiment.required_inputs:
        candidate_checkpoint = Path(result["BC_CHECKPOINT"])
        baseline_checkpoint = Path(result["BC_BASELINE_CHECKPOINT"])
        candidate = _heldout_regret(_load_bc_launch_metadata(candidate_checkpoint))
        baseline = _heldout_regret(_load_bc_launch_metadata(baseline_checkpoint))
        if candidate is None or baseline is None:
            raise ValueError(
                f"{experiment.id} requires candidate and baseline BC metadata with held-out mean_regret"
            )
        if candidate >= baseline:
            raise ValueError(
                f"{experiment.id} BC regret gate failed: {candidate:g} >= {baseline:g}"
            )
    result["EXPERIMENT_ID"] = experiment.id
    result["RUN_NAME"] = experiment.id
    return result


def write_launch_evidence(
    experiment: Experiment,
    runs_root: Path,
    launch_env: Mapping[str, str],
) -> Optional[Path]:
    """Persist preflight evidence needed to audit a completed experiment gate."""
    if "BC_CHECKPOINT" not in experiment.required_inputs:
        return None
    candidate_path = Path(launch_env["BC_CHECKPOINT"])
    baseline_path = Path(launch_env["BC_BASELINE_CHECKPOINT"])
    payload = {
        "schema_version": "1.0",
        "experiment_id": experiment.id,
        "bc": {
            "candidate": {
                "checkpoint": os.fspath(candidate_path),
                "mean_regret": _heldout_regret(
                    _load_bc_launch_metadata(candidate_path)
                ),
            },
            "baseline": {
                "checkpoint": os.fspath(baseline_path),
                "mean_regret": _heldout_regret(_load_bc_launch_metadata(baseline_path)),
            },
        },
    }
    path = runs_root / experiment.id / "experiment_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def launch_argv(experiment: Experiment) -> tuple[str, ...]:
    if experiment.command:
        return experiment.command
    return ("bash", "scripts/ucl_cs/start_run.sh")


def launch_command(
    experiment: Experiment, supplied: Optional[Mapping[str, str]] = None
) -> str:
    env = launch_environment(experiment, supplied=supplied)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    command = " ".join(shlex.quote(value) for value in launch_argv(experiment))
    return f"{assignments} {command}"


def load_final_metrics(run_dir: Path) -> Optional[dict[str, Any]]:
    report = _final_report(run_dir)
    if not report:
        return None
    rates = _report_rates(report)
    seat_gaps = [
        abs(float(row["win_rate_seat0"]) - float(row["win_rate_seat1"]))
        for row in report.get("matchups", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("win_rate_seat0"), (int, float))
        and isinstance(row.get("win_rate_seat1"), (int, float))
    ]
    return {
        "weighted_score": report.get("summary", {}).get("weighted_score"),
        "rates": rates,
        "max_seat_gap": max(seat_gaps) if seat_gaps else None,
        "all_gates_passed": report.get("all_gates_passed"),
        "protocol_signature": _report_signature(report),
    }


def render_backlog_markdown(
    experiments: Sequence[Experiment] = EXPERIMENTS,
) -> str:
    """Render the authoritative queue table deterministically from definitions."""

    lines = [
        "<!-- generated by: python examples/colonist_1v1_backlog.py render -->",
        "| ID | Stage | Question | Expected GPU time | Run storage | Promotion rule |",
        "|---|---|---|---:|---:|---|",
    ]
    for experiment in experiments:
        lo, hi = experiment.gpu_hours
        title = experiment.title.replace("|", "\\|")
        success = experiment.success_rule.replace("|", "\\|")
        lines.append(
            f"| `{experiment.id}` | {experiment.stage} | {title} | "
            f"{lo:g}–{hi:g} h | {experiment.storage_gib:g} GiB | {success} |"
        )
    return "\n".join(lines) + "\n"


def check_backlog_markdown(path: Path) -> bool:
    try:
        document = path.read_text(encoding="utf-8")
    except OSError:
        return False
    rendered = render_backlog_markdown()
    return document == rendered or rendered.rstrip() in document


validate_backlog()

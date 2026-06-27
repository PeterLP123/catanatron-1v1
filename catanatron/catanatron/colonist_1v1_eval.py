"""Colonist 1v1 evaluation: protocols, win rates, confidence intervals, and registry rows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from catanatron import Color
from catanatron.cli.cli_players import parse_cli_string
from catanatron.cli.play import GameConfigOptions, OutputOptions, play_batch

# Minimum win-rate gates for the learned agent (player index 0 / first CLI color).
DEFAULT_BENCHMARK_GATES: dict[str, float] = {
    "R": 0.90,
    "W": 0.70,
    "VP": 0.60,
    "F": 0.52,
    "G:25": 0.52,
    "M:200": 0.52,
    "AB:2": 0.52,
}

# Standard opponent battery for strength reports.
DEFAULT_BENCHMARK_OPPONENTS: tuple[str, ...] = (
    "R",
    "W",
    "VP",
    "F",
    "G:25",
    "M:200",
    "AB:2",
)


@dataclass(frozen=True)
class EvalProtocol:
    """Comparable evaluation protocol for checkpoints."""

    name: str
    opponents: tuple[str, ...]
    num_games: int
    description: str = ""


EVAL_PROTOCOLS: dict[str, EvalProtocol] = {
    "fast": EvalProtocol(
        name="fast",
        opponents=("R", "W", "VP", "F"),
        num_games=50,
        description="Frequent progress check against fast baselines plus ValueFunction.",
    ),
    "milestone": EvalProtocol(
        name="milestone",
        opponents=("R", "W", "VP", "F", "G:25"),
        num_games=100,
        description="Promotion-grade eval; includes shallow Greedy playouts.",
    ),
    "full": EvalProtocol(
        name="full",
        opponents=DEFAULT_BENCHMARK_OPPONENTS,
        num_games=200,
        description="Full strength report; can be expensive on search opponents.",
    ),
}

DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "R": 0.08,
    "W": 0.12,
    "VP": 0.15,
    "F": 0.35,
    "G:25": 0.15,
    "M:200": 0.10,
    "AB:2": 0.05,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_eval_protocol(name: str, *, num_games: Optional[int] = None) -> EvalProtocol:
    if name not in EVAL_PROTOCOLS:
        valid = ", ".join(sorted(EVAL_PROTOCOLS))
        raise ValueError(f"Unknown eval protocol {name!r}; expected one of: {valid}")
    proto = EVAL_PROTOCOLS[name]
    if num_games is None:
        return proto
    return EvalProtocol(
        name=proto.name,
        opponents=proto.opponents,
        num_games=num_games,
        description=proto.description,
    )


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.fspath(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_path_from_agent(agent_spec: str) -> Optional[Path]:
    if ":" not in agent_spec:
        return None
    code, path = agent_spec.split(":", 1)
    if code not in {"L", "T"} or not path:
        return None
    return Path(path)


def wilson_score_interval(
    wins: int,
    n: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (win rate)."""
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class MatchupResult:
    opponent: str
    agent_code: str
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    wilson_low: float
    wilson_high: float
    avg_agent_vp: float
    avg_opponent_vp: float
    avg_vp_diff: float
    avg_turns: float
    gate: Optional[float] = None
    passed_gate: Optional[bool] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    agent: str
    colonist_1v1: bool = True
    matchups: list[MatchupResult] = field(default_factory=list)
    all_gates_passed: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "agent": self.agent,
            "colonist_1v1": self.colonist_1v1,
            "all_gates_passed": self.all_gates_passed,
            "meta": self.meta,
            "summary": self.summary,
            "matchups": [m.to_dict() for m in self.matchups],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _agent_color_from_players(players) -> Color:
    return players[0].color


def evaluate_matchup(
    agent_spec: str,
    opponent_spec: str,
    *,
    num_games: int = 200,
    colonist_1v1: bool = True,
    gate: Optional[float] = None,
    quiet: bool = True,
) -> MatchupResult:
    """
    Play ``num_games`` Colonist 1v1 games with the agent as the first CLI seat.

    ``agent_spec`` examples: ``L:runs/ppo.zip``, ``F`` (for baseline bots).
    """
    started = time.monotonic()
    players = parse_cli_string(f"{agent_spec},{opponent_spec}")
    game_config = GameConfigOptions.from_cli(
        discard_limit=7,
        vps_to_win=10,
        map_type="BASE",
        number_placement="official_spiral",
        friendly_robber=False,
        colonist_1v1=colonist_1v1,
    )
    wins, vps_by_color, games = play_batch(
        num_games,
        players,
        OutputOptions(),
        game_config,
        quiet=quiet,
    )

    agent_color = _agent_color_from_players(players)
    opponent_color = players[1].color

    completed = len(games)
    agent_wins = wins.get(agent_color, 0)
    opponent_wins = wins.get(opponent_color, 0)
    draws = max(0, completed - agent_wins - opponent_wins)

    agent_vps = vps_by_color.get(agent_color, [])
    opp_vps = vps_by_color.get(opponent_color, [])
    avg_agent_vp = sum(agent_vps) / len(agent_vps) if agent_vps else 0.0
    avg_opp_vp = sum(opp_vps) / len(opp_vps) if opp_vps else 0.0
    avg_turns = sum(g.state.num_turns for g in games) / completed if completed else 0.0

    win_rate = agent_wins / completed if completed else 0.0
    lo, hi = wilson_score_interval(agent_wins, completed)

    passed = None
    if gate is not None:
        passed = win_rate >= gate

    return MatchupResult(
        opponent=opponent_spec,
        agent_code=agent_spec,
        games=completed,
        wins=agent_wins,
        losses=opponent_wins,
        draws=draws,
        win_rate=win_rate,
        wilson_low=lo,
        wilson_high=hi,
        avg_agent_vp=avg_agent_vp,
        avg_opponent_vp=avg_opp_vp,
        avg_vp_diff=avg_agent_vp - avg_opp_vp,
        avg_turns=avg_turns,
        gate=gate,
        passed_gate=passed,
        duration_seconds=time.monotonic() - started,
    )


def build_eval_meta(
    *,
    agent_spec: str,
    protocol: EvalProtocol,
    eval_kind: str = "manual",
    run_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_label: Optional[str] = None,
    training_timesteps: Optional[int] = None,
    command: Optional[Sequence[str]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path or checkpoint_path_from_agent(agent_spec)
    meta: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "eval_kind": eval_kind,
        "run_dir": os.fspath(run_dir) if run_dir else None,
        "git_commit": current_git_commit(),
        "command": list(command) if command is not None else sys.argv,
        "protocol": {
            "name": protocol.name,
            "description": protocol.description,
            "opponents": list(protocol.opponents),
            "num_games_per_matchup": protocol.num_games,
            "gates": DEFAULT_BENCHMARK_GATES,
        },
        "model": {
            "agent_spec": agent_spec,
            "checkpoint_path": os.fspath(checkpoint_path) if checkpoint_path else None,
            "checkpoint_label": checkpoint_label,
            "training_timesteps": training_timesteps,
            "file_sha256": sha256_file(checkpoint_path),
        },
    }
    if extra:
        meta.update(extra)
    return meta


def summarize_report(
    matchups: Sequence[MatchupResult],
    *,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    weights = weights or DEFAULT_SCORE_WEIGHTS
    gates_total = sum(1 for m in matchups if m.gate is not None)
    gates_passed = sum(1 for m in matchups if m.passed_gate is True)
    raw_weight = sum(weights.get(m.opponent, 0.0) for m in matchups)
    if raw_weight <= 0:
        weighted_score = (
            sum(m.win_rate for m in matchups) / len(matchups) if matchups else 0.0
        )
    else:
        # Penalize uncertainty by scoring the midpoint between observed win rate and CI lower bound.
        weighted_score = (
            sum(
                weights.get(m.opponent, 0.0) * ((m.win_rate + m.wilson_low) / 2.0)
                for m in matchups
            )
            / raw_weight
        )
    return {
        "gates_passed_count": gates_passed,
        "gates_total": gates_total,
        "mean_win_rate": (
            sum(m.win_rate for m in matchups) / len(matchups) if matchups else 0.0
        ),
        "weighted_score": weighted_score,
        "best_win_rate": max((m.win_rate for m in matchups), default=0.0),
        "worst_win_rate": min((m.win_rate for m in matchups), default=0.0),
    }


def run_benchmark(
    agent_spec: str,
    *,
    opponents: Optional[Sequence[str]] = None,
    gates: Optional[dict[str, float]] = None,
    num_games: Optional[int] = None,
    protocol: str | EvalProtocol = "full",
    colonist_1v1: bool = True,
    quiet: bool = True,
    eval_kind: str = "manual",
    run_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_label: Optional[str] = None,
    training_timesteps: Optional[int] = None,
    command: Optional[Sequence[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> EvaluationReport:
    """Run the full opponent battery and apply optional win-rate gates."""
    proto = (
        protocol if isinstance(protocol, EvalProtocol) else get_eval_protocol(protocol)
    )
    if opponents is None:
        opponents = proto.opponents
    if num_games is None:
        num_games = proto.num_games
    gates = gates or DEFAULT_BENCHMARK_GATES
    report = EvaluationReport(
        agent=agent_spec,
        colonist_1v1=colonist_1v1,
        meta=build_eval_meta(
            agent_spec=agent_spec,
            protocol=EvalProtocol(
                name=proto.name,
                opponents=tuple(opponents),
                num_games=num_games,
                description=proto.description,
            ),
            eval_kind=eval_kind,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_label=checkpoint_label,
            training_timesteps=training_timesteps,
            command=command,
            extra=metadata,
        ),
    )
    all_passed = True

    for opp in opponents:
        gate = gates.get(opp)
        result = evaluate_matchup(
            agent_spec,
            opp,
            num_games=num_games,
            colonist_1v1=colonist_1v1,
            gate=gate,
            quiet=quiet,
        )
        report.matchups.append(result)
        if gate is not None and result.passed_gate is False:
            all_passed = False

    report.all_gates_passed = all_passed
    report.summary = summarize_report(report.matchups)
    return report


def report_registry_row(
    report: EvaluationReport,
    *,
    report_path: Optional[Path] = None,
) -> dict[str, Any]:
    model = report.meta.get("model", {})
    row = {
        "created_at": report.meta.get("created_at", utc_now_iso()),
        "agent": report.agent,
        "checkpoint_path": model.get("checkpoint_path"),
        "checkpoint_label": model.get("checkpoint_label"),
        "training_timesteps": model.get("training_timesteps"),
        "file_sha256": model.get("file_sha256"),
        "protocol": report.meta.get("protocol", {}).get("name"),
        "report_path": os.fspath(report_path) if report_path else None,
        "all_gates_passed": report.all_gates_passed,
        "summary": report.summary,
        "win_rates": {m.opponent: m.win_rate for m in report.matchups},
        "wilson_low": {m.opponent: m.wilson_low for m in report.matchups},
        "gates": {
            m.opponent: m.passed_gate for m in report.matchups if m.gate is not None
        },
    }
    return row


def append_model_registry(
    registry_path: Path,
    report: EvaluationReport,
    *,
    report_path: Optional[Path] = None,
) -> dict[str, Any]:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    row = report_registry_row(report, report_path=report_path)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def format_matchup_line(result: MatchupResult) -> str:
    gate_str = ""
    if result.gate is not None:
        status = "PASS" if result.passed_gate else "FAIL"
        gate_str = f"  gate={result.gate:.0%} [{status}]"
    return (
        f"{result.opponent:8s}  "
        f"{result.wins:4d}/{result.games:<4d}  "
        f"win={result.win_rate:6.1%}  "
        f"CI=[{result.wilson_low:.1%}, {result.wilson_high:.1%}]  "
        f"vp_diff={result.avg_vp_diff:+.2f}  "
        f"turns={result.avg_turns:.1f}"
        f"{gate_str}"
    )


def print_report(report: EvaluationReport) -> None:
    print(f"Agent: {report.agent}")
    print(f"Colonist 1v1: {report.colonist_1v1}")
    if report.meta.get("protocol"):
        proto = report.meta["protocol"]
        print(
            f"Protocol: {proto.get('name')} "
            f"({proto.get('num_games_per_matchup')} games/opponent)"
        )
    for m in report.matchups:
        print(format_matchup_line(m))
    print(f"All gates passed: {report.all_gates_passed}")
    if report.summary:
        print(
            "Score: "
            f"{report.summary.get('weighted_score', 0.0):.3f} weighted, "
            f"{report.summary.get('gates_passed_count', 0)}/"
            f"{report.summary.get('gates_total', 0)} gates"
        )

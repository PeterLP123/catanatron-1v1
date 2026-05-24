"""
Colonist 1v1 evaluation: win rates, Wilson confidence intervals, and benchmark gates.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
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
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
        / denom
    )
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    agent: str
    colonist_1v1: bool = True
    matchups: list[MatchupResult] = field(default_factory=list)
    all_gates_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "colonist_1v1": self.colonist_1v1,
            "all_gates_passed": self.all_gates_passed,
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
    )


def run_benchmark(
    agent_spec: str,
    *,
    opponents: Sequence[str] = DEFAULT_BENCHMARK_OPPONENTS,
    gates: Optional[dict[str, float]] = None,
    num_games: int = 200,
    colonist_1v1: bool = True,
    quiet: bool = True,
) -> EvaluationReport:
    """Run the full opponent battery and apply optional win-rate gates."""
    gates = gates or DEFAULT_BENCHMARK_GATES
    report = EvaluationReport(agent=agent_spec, colonist_1v1=colonist_1v1)
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
    return report


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
    for m in report.matchups:
        print(format_matchup_line(m))
    print(f"All gates passed: {report.all_gates_passed}")

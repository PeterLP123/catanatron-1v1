"""Colonist 1v1 evaluation: protocols, win rates, confidence intervals, and registry rows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

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
DEFAULT_EVAL_SEED = 20_260_701

# Evaluation stages use disjoint deterministic seed namespaces. ``manual`` is
# deliberately offset-free so existing CLI invocations retain their historical
# seed. Training callers already label reports with ``eval_kind``, allowing dev
# and final evidence to separate automatically without a CLI compatibility break.
EVAL_SEED_SUITE_OFFSETS: dict[str, int] = {
    "manual": 0,
    "dev": 1_000_003,
    "promotion": 2_000_006,
    "final": 3_000_009,
}
EVAL_KIND_SEED_SUITES: dict[str, str] = {
    "manual": "manual",
    "mid_training": "dev",
    "dev": "dev",
    "promotion": "promotion",
    "final": "final",
    "final_benchmark": "final",
}


@dataclass(frozen=True)
class EvalProtocol:
    """Comparable evaluation protocol for checkpoints."""

    name: str
    opponents: tuple[str, ...]
    num_games: int
    description: str = ""
    seed: int = DEFAULT_EVAL_SEED


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
        seed=proto.seed,
    )


def seed_suite_for_eval_kind(eval_kind: str) -> str:
    """Return the deterministic seed namespace for an evaluation purpose."""
    return EVAL_KIND_SEED_SUITES.get(eval_kind, "manual")


def resolve_eval_seed(
    base_seed: int = DEFAULT_EVAL_SEED,
    *,
    suite: str = "manual",
) -> int:
    """Resolve a stable, disjoint base seed for dev/promotion/final evidence."""
    if suite not in EVAL_SEED_SUITE_OFFSETS:
        valid = ", ".join(sorted(EVAL_SEED_SUITE_OFFSETS))
        raise ValueError(f"Unknown evaluation seed suite {suite!r}; expected: {valid}")
    return base_seed + EVAL_SEED_SUITE_OFFSETS[suite]


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


def _quantile(values: Sequence[float], probability: float) -> float:
    """Small dependency-free linear quantile helper for bootstrap intervals."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, probability))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_interval(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 5_000,
    seed: int = DEFAULT_EVAL_SEED,
) -> tuple[float, float, float]:
    """Bootstrap a paired mean delta, preserving matched game/seed structure."""
    if len(candidate) != len(baseline):
        raise ValueError("Paired samples must have the same length")
    if not candidate:
        return (0.0, 0.0, 0.0)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    deltas = [float(a) - float(b) for a, b in zip(candidate, baseline)]
    estimate = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(resamples):
        bootstrapped.append(
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    tail = (1.0 - confidence) / 2.0
    return (
        estimate,
        _quantile(bootstrapped, tail),
        _quantile(bootstrapped, 1.0 - tail),
    )


def confidence_gate_passed(
    *,
    estimate: float,
    threshold: float,
    confidence_low: Optional[float] = None,
    mode: Literal["point", "lower_bound"] = "point",
) -> bool:
    """Apply either a legacy point gate or a confidence-lower-bound gate."""
    if mode == "point":
        return estimate >= threshold
    if mode == "lower_bound":
        if confidence_low is None:
            raise ValueError("confidence_low is required for a lower_bound gate")
        return confidence_low >= threshold
    raise ValueError(f"Unknown confidence gate mode: {mode!r}")


@dataclass(frozen=True)
class GameOutcome:
    """Auditable evidence for one requested game in a matchup."""

    game_index: int
    seat: int
    agent_first: bool
    status: Literal["completed", "truncated", "error"]
    result: Literal["win", "loss", "draw", "error"]
    game_id: Optional[str] = None
    seed: Optional[int] = None
    agent_color: Optional[str] = None
    opponent_color: Optional[str] = None
    winner_color: Optional[str] = None
    agent_vp: Optional[float] = None
    opponent_vp: Optional[float] = None
    vp_diff: Optional[float] = None
    turns: Optional[int] = None
    ticks: Optional[int] = None
    error: Optional[str] = None
    # Stable consumer-facing aliases used by backlog/promotion gates.
    schedule_id: Optional[str] = None
    agent_seat: Optional[int] = None
    outcome: Optional[Literal["win", "loss", "draw", "truncated", "error"]] = None
    truncated: Optional[bool] = None
    errored: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.schedule_id is None:
            identity = self.seed if self.seed is not None else self.game_index
            object.__setattr__(self, "schedule_id", f"seat-{self.seat}:game-{identity}")
        if self.agent_seat is None:
            object.__setattr__(self, "agent_seat", self.seat)
        if self.outcome is None:
            outcome = "truncated" if self.status == "truncated" else self.result
            object.__setattr__(self, "outcome", outcome)
        if self.truncated is None:
            object.__setattr__(self, "truncated", self.status == "truncated")
        if self.errored is None:
            object.__setattr__(self, "errored", self.status == "error")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GameOutcome":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})


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
    # Per-seat breakdown (None when the matchup was first-seat-only).
    win_rate_seat0: Optional[float] = None
    win_rate_seat1: Optional[float] = None
    vp_diff_seat0: Optional[float] = None
    vp_diff_seat1: Optional[float] = None
    games_seat0: Optional[int] = None
    games_seat1: Optional[int] = None
    # Additive integrity fields. ``games`` remains the legacy total-accounted
    # denominator; these fields make terminal, truncated, and failed requests
    # distinguishable without breaking old report readers or constructors.
    requested_games: Optional[int] = None
    observed_games: Optional[int] = None
    completed_games: Optional[int] = None
    truncated_games: Optional[int] = None
    error_games: int = 0
    game_results: list[GameOutcome] = field(default_factory=list)
    gate_mode: Literal["point", "lower_bound"] = "point"
    gate_value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.requested_games is None:
            self.requested_games = self.games
        if self.completed_games is None:
            self.completed_games = self.wins + self.losses
        if self.truncated_games is None:
            self.truncated_games = self.draws
        if self.observed_games is None:
            self.observed_games = self.completed_games + self.truncated_games
        self.game_results = [
            GameOutcome.from_dict(outcome) if isinstance(outcome, Mapping) else outcome
            for outcome in self.game_results
        ]
        if self.gate_value is None and self.gate is not None:
            self.gate_value = (
                self.wilson_low if self.gate_mode == "lower_bound" else self.win_rate
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MatchupResult":
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in allowed}
        raw_game_results = data.get("game_results", data.get("game_outcomes", []))
        values["game_results"] = [
            GameOutcome.from_dict(outcome) for outcome in raw_game_results
        ]
        return cls(**values)


@dataclass(frozen=True)
class PairedComparison:
    """Paired candidate-minus-baseline game score evidence."""

    matched_games: int
    mean_delta: float
    confidence_low: float
    confidence_high: float
    confidence: float
    resamples: int
    threshold: float
    passed_gate: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _outcome_score(outcome: GameOutcome) -> Optional[float]:
    if outcome.result == "win":
        return 1.0
    if outcome.result == "draw":
        return 0.5
    if outcome.result == "loss":
        return 0.0
    return None


def compare_paired_matchups(
    candidate: MatchupResult,
    baseline: MatchupResult,
    *,
    confidence: float = 0.95,
    resamples: int = 5_000,
    seed: int = DEFAULT_EVAL_SEED,
    threshold: float = 0.0,
) -> PairedComparison:
    """Compare reports on their shared ``(seat, seed)`` games via bootstrap."""
    if candidate.opponent != baseline.opponent:
        raise ValueError("Paired matchup reports must use the same opponent")

    def keyed_scores(matchup: MatchupResult) -> dict[str, float]:
        requested = int(matchup.requested_games or matchup.games)
        if requested <= 0 or len(matchup.game_results) != requested:
            raise ValueError(
                f"Paired {matchup.opponent} report lacks full requested-game evidence"
            )
        scores: dict[str, float] = {}
        for outcome in matchup.game_results:
            score = _outcome_score(outcome)
            if score is None:
                raise ValueError("Paired reports cannot contain errored games")
            key = outcome.schedule_id
            if key is None:
                raise ValueError("Paired game evidence is missing schedule_id")
            if key in scores:
                raise ValueError(f"Duplicate paired schedule_id: {key}")
            scores[key] = score
        return scores

    candidate_scores = keyed_scores(candidate)
    baseline_scores = keyed_scores(baseline)
    if candidate_scores.keys() != baseline_scores.keys():
        missing_candidate = sorted(baseline_scores.keys() - candidate_scores.keys())
        missing_baseline = sorted(candidate_scores.keys() - baseline_scores.keys())
        raise ValueError(
            "Paired reports must cover the exact same schedule; "
            f"missing_candidate={missing_candidate[:3]} "
            f"missing_baseline={missing_baseline[:3]}"
        )
    shared = sorted(candidate_scores, key=str)
    candidate_values = [candidate_scores[key] for key in shared]
    baseline_values = [baseline_scores[key] for key in shared]
    estimate, low, high = paired_bootstrap_interval(
        candidate_values,
        baseline_values,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return PairedComparison(
        matched_games=len(shared),
        mean_delta=estimate,
        confidence_low=low,
        confidence_high=high,
        confidence=confidence,
        resamples=resamples,
        threshold=threshold,
        passed_gate=bool(shared) and low >= threshold,
    )


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
            "schema_version": "1.1",
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationReport":
        """Read both legacy 1.0 aggregate reports and additive 1.1 reports."""
        return cls(
            agent=str(data["agent"]),
            colonist_1v1=bool(data.get("colonist_1v1", True)),
            matchups=[
                MatchupResult.from_dict(matchup) for matchup in data.get("matchups", [])
            ],
            all_gates_passed=bool(data.get("all_gates_passed", False)),
            meta=dict(data.get("meta", {})),
            summary=dict(data.get("summary", {})),
        )

    @classmethod
    def read_json(cls, path: Path) -> "EvaluationReport":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class _SeatStats:
    """Per-seat aggregates from one batch played with a fixed seat ordering."""

    requested_games: int
    observed_games: int
    completed_games: int
    truncated_games: int
    error_games: int
    games: int
    agent_wins: int
    opponent_wins: int
    draws: int
    agent_vp_sum: float
    opponent_vp_sum: float
    turns_sum: float
    game_results: list[GameOutcome] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.agent_wins / self.games if self.games else 0.0

    @property
    def vp_diff(self) -> float:
        if not self.observed_games:
            return 0.0
        return (self.agent_vp_sum - self.opponent_vp_sum) / self.observed_games


def _color_name(color: Any) -> Optional[str]:
    if color is None:
        return None
    return str(getattr(color, "value", color))


def _game_winner(game: Any, valid_colors: set[Any]) -> tuple[Any, bool]:
    """Return winner and whether the game exposed an authoritative value."""
    winner_fn = getattr(game, "winning_color", None)
    if not callable(winner_fn):
        return None, False
    winner = winner_fn()
    if winner is None or winner in valid_colors:
        return winner, True
    # Mock/legacy result objects sometimes expose an unconfigured MagicMock.
    # In that case reconcile from the aggregate win dictionary below.
    return None, False


def _indexed_value(values: Sequence[Any], index: int) -> Optional[float]:
    if index >= len(values):
        return None
    try:
        return float(values[index])
    except (TypeError, ValueError):
        return None


@contextmanager
def preserve_evaluation_rng_state():
    """Prevent model loading and matches from changing training RNG streams."""

    python_state = random.getstate()
    numpy_module = None
    numpy_state = None
    torch_module = None
    torch_state = None
    cuda_states = None
    try:
        try:
            import numpy as np

            numpy_module = np
            numpy_state = np.random.get_state()
        except ImportError:
            pass
        try:
            import torch

            torch_module = torch
            torch_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_states = torch.cuda.get_rng_state_all()
        except ImportError:
            pass
        yield
    finally:
        random.setstate(python_state)
        if numpy_module is not None and numpy_state is not None:
            numpy_module.random.set_state(numpy_state)
        if torch_module is not None and torch_state is not None:
            torch_module.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch_module.cuda.set_rng_state_all(cuda_states)


def _preserve_evaluation_rng(function):
    def guarded(*args, **kwargs):
        with preserve_evaluation_rng_state():
            return function(*args, **kwargs)

    return guarded


@_preserve_evaluation_rng
def _play_seat(
    agent_spec: str,
    opponent_spec: str,
    *,
    num_games: int,
    colonist_1v1: bool,
    quiet: bool,
    agent_first: bool,
    seed: int,
) -> _SeatStats:
    """Play ``num_games`` with the agent in a fixed seat and aggregate its results.

    The first CLI seat moves first, so ``agent_first`` controls whether the agent
    gets the first-player advantage. The agent's color is read off the parsed
    players rather than assumed, since swapping the seat swaps the color.
    """
    cli = (
        f"{agent_spec},{opponent_spec}"
        if agent_first
        else f"{opponent_spec},{agent_spec}"
    )
    players = parse_cli_string(cli)
    game_config = GameConfigOptions.from_cli(
        discard_limit=7,
        vps_to_win=10,
        map_type="BASE",
        number_placement="official_spiral",
        friendly_robber=False,
        colonist_1v1=colonist_1v1,
        seed=seed,
        shuffle_players=False,
    )
    random_state = random.getstate()
    try:
        wins, vps_by_color, games = play_batch(
            num_games,
            players,
            OutputOptions(),
            game_config,
            quiet=quiet,
        )
    finally:
        random.setstate(random_state)

    agent_player = players[0] if agent_first else players[1]
    opponent_player = players[1] if agent_first else players[0]
    agent_color = agent_player.color
    opponent_color = opponent_player.color

    agent_vps = vps_by_color.get(agent_color, [])
    opp_vps = vps_by_color.get(opponent_color, [])
    remaining_agent_wins = int(wins.get(agent_color, 0))
    remaining_opponent_wins = int(wins.get(opponent_color, 0))
    outcomes: list[GameOutcome] = []

    for game_index, game in enumerate(games[:num_games]):
        winner, authoritative = _game_winner(game, {agent_color, opponent_color})
        if authoritative and winner == agent_color:
            remaining_agent_wins = max(0, remaining_agent_wins - 1)
        elif authoritative and winner == opponent_color:
            remaining_opponent_wins = max(0, remaining_opponent_wins - 1)
        elif not authoritative:
            # Preserve compatibility with mocked/legacy batch results that only
            # expose aggregate winner counts, while real ``Game`` objects use
            # their authoritative ``winning_color`` (including explicit None).
            if remaining_agent_wins:
                winner = agent_color
                remaining_agent_wins -= 1
            elif remaining_opponent_wins:
                winner = opponent_color
                remaining_opponent_wins -= 1

        agent_vp = _indexed_value(agent_vps, game_index)
        opponent_vp = _indexed_value(opp_vps, game_index)
        vp_diff = (
            agent_vp - opponent_vp
            if agent_vp is not None and opponent_vp is not None
            else None
        )
        state = getattr(game, "state", None)
        turns_value = getattr(state, "num_turns", None)
        turns = turns_value if isinstance(turns_value, int) else None
        records = getattr(state, "action_records", None)
        ticks = len(records) if isinstance(records, (list, tuple)) else None
        game_id_value = getattr(game, "id", None)
        game_id = str(game_id_value) if isinstance(game_id_value, (str, int)) else None
        game_seed_value = getattr(game, "seed", None)
        game_seed = (
            game_seed_value if isinstance(game_seed_value, int) else seed + game_index
        )
        if winner == agent_color:
            result: Literal["win", "loss", "draw", "error"] = "win"
            status: Literal["completed", "truncated", "error"] = "completed"
        elif winner == opponent_color:
            result = "loss"
            status = "completed"
        else:
            result = "draw"
            status = "truncated"
        outcomes.append(
            GameOutcome(
                game_index=game_index,
                seat=0 if agent_first else 1,
                agent_first=agent_first,
                status=status,
                result=result,
                game_id=game_id,
                seed=game_seed,
                agent_color=_color_name(agent_color),
                opponent_color=_color_name(opponent_color),
                winner_color=_color_name(winner),
                agent_vp=agent_vp,
                opponent_vp=opponent_vp,
                vp_diff=vp_diff,
                turns=turns,
                ticks=ticks,
            )
        )

    observed_games = len(outcomes)
    for game_index in range(observed_games, num_games):
        outcomes.append(
            GameOutcome(
                game_index=game_index,
                seat=0 if agent_first else 1,
                agent_first=agent_first,
                status="error",
                result="error",
                seed=seed + game_index,
                agent_color=_color_name(agent_color),
                opponent_color=_color_name(opponent_color),
                error=(
                    f"play_batch returned {observed_games} of {num_games} "
                    "requested games"
                ),
            )
        )

    agent_wins = sum(outcome.result == "win" for outcome in outcomes)
    opponent_wins = sum(outcome.result == "loss" for outcome in outcomes)
    draws = sum(outcome.result == "draw" for outcome in outcomes)
    truncated_games = sum(outcome.status == "truncated" for outcome in outcomes)
    error_games = sum(outcome.status == "error" for outcome in outcomes)
    return _SeatStats(
        requested_games=num_games,
        observed_games=observed_games,
        completed_games=agent_wins + opponent_wins,
        truncated_games=truncated_games,
        error_games=error_games,
        games=num_games,
        agent_wins=agent_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        agent_vp_sum=sum(
            outcome.agent_vp or 0.0 for outcome in outcomes if outcome.status != "error"
        ),
        opponent_vp_sum=sum(
            outcome.opponent_vp or 0.0
            for outcome in outcomes
            if outcome.status != "error"
        ),
        turns_sum=float(
            sum(outcome.turns or 0 for outcome in outcomes if outcome.status != "error")
        ),
        game_results=outcomes,
    )


def evaluate_matchup(
    agent_spec: str,
    opponent_spec: str,
    *,
    num_games: int = 200,
    colonist_1v1: bool = True,
    gate: Optional[float] = None,
    quiet: bool = True,
    both_seats: bool = True,
    seed: int = DEFAULT_EVAL_SEED,
    gate_mode: Literal["point", "lower_bound"] = "point",
) -> MatchupResult:
    """
    Play ``num_games`` Colonist 1v1 games and report the agent's win rate.

    When ``both_seats`` (the default), ``num_games`` is split between the agent
    moving first and the agent moving second, so the result is not inflated by
    first-player advantage; per-seat win rate and VP margin are also recorded.
    With ``both_seats=False`` the agent only plays the first seat (legacy behavior).

    ``agent_spec`` examples: ``L:runs/ppo.zip``, ``F`` (for baseline bots).
    """
    started = time.monotonic()
    if both_seats:
        seat0_games = num_games // 2
        seat1_games = num_games - seat0_games
    else:
        seat0_games = num_games
        seat1_games = 0

    seat0 = _play_seat(
        agent_spec,
        opponent_spec,
        num_games=seat0_games,
        colonist_1v1=colonist_1v1,
        quiet=quiet,
        agent_first=True,
        seed=seed,
    )
    seat1 = (
        _play_seat(
            agent_spec,
            opponent_spec,
            num_games=seat1_games,
            colonist_1v1=colonist_1v1,
            quiet=quiet,
            agent_first=False,
            seed=seed,
        )
        if seat1_games > 0
        else None
    )

    seats = [s for s in (seat0, seat1) if s is not None]
    requested = sum(s.requested_games for s in seats)
    observed = sum(s.observed_games for s in seats)
    completed = sum(s.completed_games for s in seats)
    truncated = sum(s.truncated_games for s in seats)
    errors = sum(s.error_games for s in seats)
    agent_wins = sum(s.agent_wins for s in seats)
    opponent_wins = sum(s.opponent_wins for s in seats)
    draws = sum(s.draws for s in seats)
    accounted = agent_wins + opponent_wins + draws + errors
    if accounted != requested:
        raise RuntimeError(
            f"Evaluation accounting mismatch: requested={requested}, "
            f"accounted={accounted}"
        )

    agent_vp_sum = sum(s.agent_vp_sum for s in seats)
    opp_vp_sum = sum(s.opponent_vp_sum for s in seats)
    turns_sum = sum(s.turns_sum for s in seats)
    avg_agent_vp = agent_vp_sum / observed if observed else 0.0
    avg_opp_vp = opp_vp_sum / observed if observed else 0.0
    avg_turns = turns_sum / observed if observed else 0.0

    # Errors remain in the requested-game denominator so a broken evaluator
    # cannot improve a model's reported rate by censoring difficult games.
    win_rate = agent_wins / requested if requested else 0.0
    lo, hi = wilson_score_interval(agent_wins, requested)

    passed = None
    gate_value = None
    if gate is not None:
        gate_value = lo if gate_mode == "lower_bound" else win_rate
        passed = confidence_gate_passed(
            estimate=win_rate,
            threshold=gate,
            confidence_low=lo,
            mode=gate_mode,
        )

    return MatchupResult(
        opponent=opponent_spec,
        agent_code=agent_spec,
        games=requested,
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
        win_rate_seat0=seat0.win_rate,
        win_rate_seat1=seat1.win_rate if seat1 is not None else None,
        vp_diff_seat0=seat0.vp_diff,
        vp_diff_seat1=seat1.vp_diff if seat1 is not None else None,
        games_seat0=seat0.games,
        games_seat1=seat1.games if seat1 is not None else None,
        requested_games=requested,
        observed_games=observed,
        completed_games=completed,
        truncated_games=truncated,
        error_games=errors,
        game_results=[outcome for seat in seats for outcome in seat.game_results],
        gate_mode=gate_mode,
        gate_value=gate_value,
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
    both_seats: bool = True,
    seed_suite: str = "manual",
    base_seed: Optional[int] = None,
    gate_mode: Literal["point", "lower_bound"] = "point",
    gates: Optional[Mapping[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path or checkpoint_path_from_agent(agent_spec)
    meta: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "eval_kind": eval_kind,
        "run_dir": os.fspath(run_dir) if run_dir else None,
        "git_commit": current_git_commit(),
        "command": list(command) if command is not None else sys.argv,
        "both_seats": both_seats,
        "protocol": {
            "name": protocol.name,
            "description": protocol.description,
            "opponents": list(protocol.opponents),
            "num_games_per_matchup": protocol.num_games,
            "gates": dict(DEFAULT_BENCHMARK_GATES if gates is None else gates),
            "seed": protocol.seed,
            "base_seed": protocol.seed if base_seed is None else base_seed,
            "seed_suite": seed_suite,
            "gate_mode": gate_mode,
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
    requested_games = sum(m.requested_games or m.games for m in matchups)
    accounted_games = sum(m.wins + m.losses + m.draws + m.error_games for m in matchups)
    return {
        "gates_passed_count": gates_passed,
        "gates_total": gates_total,
        "mean_win_rate": (
            sum(m.win_rate for m in matchups) / len(matchups) if matchups else 0.0
        ),
        "weighted_score": weighted_score,
        "best_win_rate": max((m.win_rate for m in matchups), default=0.0),
        "worst_win_rate": min((m.win_rate for m in matchups), default=0.0),
        "requested_games": requested_games,
        "accounted_games": accounted_games,
        "all_games_accounted": requested_games == accounted_games,
        "observed_games": sum(m.observed_games or 0 for m in matchups),
        "completed_games": sum(m.completed_games or 0 for m in matchups),
        "truncated_games": sum(m.truncated_games or 0 for m in matchups),
        "error_games": sum(m.error_games for m in matchups),
    }


def run_benchmark(
    agent_spec: str,
    *,
    opponents: Optional[Sequence[str]] = None,
    gates: Optional[dict[str, float]] = None,
    num_games: Optional[int] = None,
    protocol: str | EvalProtocol = "full",
    colonist_1v1: bool = True,
    both_seats: bool = True,
    quiet: bool = True,
    eval_kind: str = "manual",
    run_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_label: Optional[str] = None,
    training_timesteps: Optional[int] = None,
    command: Optional[Sequence[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
    seed_suite: Optional[str] = None,
    gate_mode: Literal["point", "lower_bound"] = "point",
) -> EvaluationReport:
    """Run the full opponent battery and apply optional win-rate gates."""
    proto = (
        protocol if isinstance(protocol, EvalProtocol) else get_eval_protocol(protocol)
    )
    if opponents is None:
        opponents = proto.opponents
    if num_games is None:
        num_games = proto.num_games
    base_seed = proto.seed
    if seed is None:
        seed_suite = seed_suite or seed_suite_for_eval_kind(eval_kind)
        seed = resolve_eval_seed(base_seed, suite=seed_suite)
    else:
        seed_suite = seed_suite or "explicit"
    gates = gates or DEFAULT_BENCHMARK_GATES
    declared_gates = {
        opponent: gates[opponent] for opponent in opponents if opponent in gates
    }
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
                seed=seed,
            ),
            eval_kind=eval_kind,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_label=checkpoint_label,
            training_timesteps=training_timesteps,
            command=command,
            both_seats=both_seats,
            seed_suite=seed_suite,
            base_seed=base_seed,
            gate_mode=gate_mode,
            gates=declared_gates,
            extra=metadata,
        ),
    )
    all_passed = True

    for opp in opponents:
        gate = declared_gates.get(opp)
        result = evaluate_matchup(
            agent_spec,
            opp,
            num_games=num_games,
            colonist_1v1=colonist_1v1,
            gate=gate,
            quiet=quiet,
            both_seats=both_seats,
            seed=seed,
            gate_mode=gate_mode,
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
        "win_rates_seat0": {m.opponent: m.win_rate_seat0 for m in report.matchups},
        "win_rates_seat1": {m.opponent: m.win_rate_seat1 for m in report.matchups},
        "wilson_low": {m.opponent: m.wilson_low for m in report.matchups},
        "gate_modes": {m.opponent: m.gate_mode for m in report.matchups},
        "requested_games": {m.opponent: m.requested_games for m in report.matchups},
        "completed_games": {m.opponent: m.completed_games for m in report.matchups},
        "truncated_games": {m.opponent: m.truncated_games for m in report.matchups},
        "error_games": {m.opponent: m.error_games for m in report.matchups},
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
    seat_str = ""
    if result.win_rate_seat1 is not None and result.win_rate_seat0 is not None:
        seat_str = f"  seat[{result.win_rate_seat0:.0%}/{result.win_rate_seat1:.0%}]"
    integrity_str = ""
    if result.truncated_games or result.error_games:
        integrity_str = (
            f"  trunc={result.truncated_games or 0} err={result.error_games}"
        )
    return (
        f"{result.opponent:8s}  "
        f"{result.wins:4d}/{result.games:<4d}  "
        f"win={result.win_rate:6.1%}  "
        f"CI=[{result.wilson_low:.1%}, {result.wilson_high:.1%}]  "
        f"vp_diff={result.avg_vp_diff:+.2f}  "
        f"turns={result.avg_turns:.1f}"
        f"{seat_str}"
        f"{integrity_str}"
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

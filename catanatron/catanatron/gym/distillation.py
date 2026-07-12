"""Deterministic DAgger-style data collection for Colonist 1v1.

The behavior policy controls the game.  At every state it visits, an isolated
teacher policy labels the same legal-action set without perturbing the game's
random stream.  Iterations are written as immutable Parquet shards and a small
aggregate manifest makes the growing replay corpus auditable.

This module deliberately stops at data collection.  It gives BC or policy
training a trustworthy replay surface without pretending that a large expert
iteration run has already happened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from catanatron.features import create_sample
from catanatron.gym.envs.action_space import to_action_space
from catanatron.gym.model_schema import canonical_hash
from catanatron.gym.provenance import sha256_file
from catanatron.models.player import Color, Player
from catanatron.players.value import ValueFunctionPlayer, get_value_fn

DISTILLATION_FORMAT_VERSION = 1
PLAYER_COLORS = (Color.BLUE, Color.RED)
MAP_TYPE = "BASE"


@dataclass(frozen=True)
class AgentIdentity:
    """Stable identity for one participant in a distillation iteration."""

    spec: str
    agent_hash: str
    checkpoint_path: str | None = None
    checkpoint_hash: str | None = None

    @classmethod
    def from_spec(cls, spec: str) -> "AgentIdentity":
        checkpoint_path = _checkpoint_path_from_spec(spec)
        checkpoint_hash = None
        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
            checkpoint_path = os.fspath(checkpoint)
            checkpoint_hash = sha256_file(checkpoint)
        return cls(
            spec=spec,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=checkpoint_hash,
            agent_hash=canonical_hash(
                {"spec": spec, "checkpoint_hash": checkpoint_hash}
            ),
        )


@dataclass(frozen=True)
class DistillationConfig:
    """Inputs that determine one immutable data-collection iteration."""

    iteration: int
    games: int
    base_seed: int
    student_spec: str
    teacher_spec: str
    opponent_spec: str
    feature_profile: str = "raw"
    human_visible_obs: bool = False
    alternate_seats: bool = True
    include_forced: bool = False
    score_f_candidates: bool = True
    shard_games: int = 10

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.games < 1:
            raise ValueError("games must be at least 1")
        if self.shard_games < 1:
            raise ValueError("shard_games must be at least 1")
        validate_teacher_spec(self.teacher_spec)


def _checkpoint_path_from_spec(spec: str) -> str | None:
    code, separator, value = spec.partition(":")
    if code not in {"L", "T"}:
        return None
    if not separator or not value:
        raise ValueError(f"{code} player spec requires a checkpoint path")
    return value


def validate_teacher_spec(spec: str) -> None:
    """Accept F or fixed-simulation MCTS teachers only.

    Wall-clock MCTS budgets are intentionally rejected: machine load changes
    how many simulations fit in a time budget, so the resulting labels are not
    deterministic even when game and decision seeds are fixed.
    """

    parts = spec.split(":")
    code = parts[0]
    if code not in {"F", "M"}:
        raise ValueError("teacher must be F or fixed-simulation MCTS (M:...)")
    if code == "M":
        if len(parts) > 5:
            raise ValueError(f"Invalid MCTS teacher spec: {spec!r}")
        if len(parts) >= 5 and parts[4] not in {"", "None"}:
            raise ValueError(
                "MCTS distillation teachers must use a fixed simulation count, "
                "not max_time_ms"
            )
        if len(parts) >= 2:
            try:
                simulations = int(parts[1])
            except ValueError as exc:
                raise ValueError("MCTS simulation count must be an integer") from exc
            if simulations < 1:
                raise ValueError("MCTS simulation count must be at least 1")


def build_player(spec: str, color: Color) -> Player:
    """Build one player from the existing CLI player vocabulary."""

    from catanatron.cli.cli_players import CLI_PLAYERS

    parts = spec.split(":")
    code = parts[0]
    registration = next((row for row in CLI_PLAYERS if row.code == code), None)
    if registration is None:
        choices = ", ".join(row.code for row in CLI_PLAYERS)
        raise ValueError(f"Unknown player code {code!r}; choose one of: {choices}")
    try:
        return registration.import_fn(color, *parts[1:])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid player spec {spec!r}: {exc}") from exc


def derive_seed(
    namespace: str,
    *,
    base_seed: int,
    iteration: int,
    game_index: int,
    decision_index: int = 0,
) -> int:
    """Derive a stable uint32 seed without overlapping iteration namespaces."""

    payload = (
        f"{namespace}:{base_seed}:{iteration}:{game_index}:{decision_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@contextmanager
def isolated_random_seed(seed: int) -> Iterator[None]:
    """Run observational teacher work without changing behavior-policy RNG."""

    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _teacher_candidate_scores(
    teacher: Player,
    game,
    playable_actions: Sequence,
) -> list[float] | None:
    if not isinstance(teacher, ValueFunctionPlayer):
        return None

    from catanatron.players.leaf_evaluation import action_value

    value_fn = get_value_fn(teacher.value_fn_builder_name, teacher.params)
    return [
        float(action_value(game, action, teacher.color, value_fn))
        for action in playable_actions
    ]


def _action_index(action, *, context: str) -> int:
    try:
        return int(to_action_space(action, PLAYER_COLORS, MAP_TYPE))
    except KeyError as exc:
        raise ValueError(
            f"{context} action is outside the versioned Colonist action codec: {action}"
        ) from exc


def _human_visible_sample(sample: dict[str, Any], game, color: Color) -> None:
    if "P0_ACTUAL_VPS" not in sample:
        return
    from catanatron.state_functions import get_visible_victory_points

    sample["P0_ACTUAL_VPS"] = get_visible_victory_points(game.state, color)


class DistillationDecisionRecorder:
    """Ask the student for behavior and independently label its visited state."""

    def __init__(
        self,
        *,
        config: DistillationConfig,
        model_schema: Mapping[str, Any],
        student: AgentIdentity,
        teacher: AgentIdentity,
    ) -> None:
        self.config = config
        self.model_schema = dict(model_schema)
        self.student = student
        self.teacher = teacher

    def decide_and_label(
        self,
        *,
        behavior: Player,
        teacher: Player,
        game,
        playable_actions: Sequence,
        game_index: int,
        game_seed: int,
        decision_index: int,
        seat: int,
    ) -> tuple[Any, dict[str, Any] | None]:
        actions = list(playable_actions)
        if not actions:
            raise ValueError("Cannot label a state with no legal actions")
        if behavior.color != teacher.color:
            raise ValueError("behavior and teacher must label the same color")

        sample = create_sample(
            game,
            behavior.color,
            feature_profile=self.config.feature_profile,
        )
        if self.config.human_visible_obs:
            _human_visible_sample(sample, game, behavior.color)

        decision_seed = derive_seed(
            "teacher-decision",
            base_seed=self.config.base_seed,
            iteration=self.config.iteration,
            game_index=game_index,
            decision_index=decision_index,
        )
        with isolated_random_seed(decision_seed):
            teacher_action = teacher.decide(game, actions)
            candidate_scores = (
                _teacher_candidate_scores(teacher, game, actions)
                if self.config.score_f_candidates
                else None
            )
        if teacher_action not in actions:
            raise ValueError(f"Teacher returned an illegal action: {teacher_action}")

        # Teacher work restored the RNG stream, so stochastic student behavior
        # is exactly what it would have been without data collection.
        behavior_action = behavior.decide(game, actions)
        if behavior_action not in actions:
            raise ValueError(
                f"Behavior policy returned an illegal action: {behavior_action}"
            )

        if len(actions) == 1 and not self.config.include_forced:
            return behavior_action, None

        legal_indices = [_action_index(action, context="legal") for action in actions]
        teacher_index = _action_index(teacher_action, context="teacher")
        behavior_index = _action_index(behavior_action, context="behavior")
        teacher_distribution = [
            1.0 if action == teacher_action else 0.0 for action in actions
        ]
        scores_available = candidate_scores is not None
        if candidate_scores is None:
            # Keep a stable list<float> Parquet type for MCTS shards while the
            # availability flag distinguishes these placeholders from F scores.
            candidate_scores = [math.nan] * len(actions)

        prompt = getattr(game.state, "current_prompt", None)
        phase = getattr(prompt, "name", str(prompt))
        state_hash = canonical_hash(
            {
                "features": sample,
                "legal_actions": legal_indices,
                "color": behavior.color.name,
            }
        )
        row: dict[str, Any] = {
            "DISTILLATION_VERSION": DISTILLATION_FORMAT_VERSION,
            "ITERATION": self.config.iteration,
            "GAME_INDEX": game_index,
            "GAME_ID": str(game.id),
            "GAME_SEED": game_seed,
            "DECISION_INDEX": decision_index,
            "DECISION_SEED": decision_seed,
            "STATE_HASH": state_hash,
            "TURN": int(game.state.num_turns),
            "SEAT": seat,
            "COLOR": behavior.color.name,
            "PHASE": phase,
            "NUM_LEGAL": len(actions),
            "LEGAL_ACTIONS": legal_indices,
            "BEHAVIOR_ACTION": behavior_index,
            "TEACHER_ACTION": teacher_index,
            "TEACHER_DISTRIBUTION": teacher_distribution,
            "CANDIDATE_SCORES": [float(value) for value in candidate_scores],
            "CANDIDATE_SCORES_AVAILABLE": scores_available,
            "BEHAVIOR_MATCHES_TEACHER": behavior_index == teacher_index,
            "STUDENT_HASH": self.student.agent_hash,
            "TEACHER_HASH": self.teacher.agent_hash,
            "CHECKPOINT_HASH": self.student.checkpoint_hash,
            "SCHEMA_HASH": self.model_schema["schema_hash"],
            "FEATURE_HASH": self.model_schema["feature_hash"],
            "ACTION_HASH": self.model_schema["action_hash"],
            "RULES_HASH": self.model_schema["rules_hash"],
        }
        row.update({f"F_{key}": value for key, value in sample.items()})
        return behavior_action, row


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class DistillationDatasetWriter:
    """Stream games into immutable iteration shards and rebuild the replay index."""

    def __init__(
        self,
        root: str | Path,
        *,
        iteration: int,
        shard_games: int,
        metadata: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self.iteration = int(iteration)
        self.shard_games = int(shard_games)
        self.metadata = dict(metadata)
        self.iteration_dir = self.root / f"iteration-{self.iteration:04d}"
        if self.iteration_dir.exists() and any(self.iteration_dir.iterdir()):
            raise FileExistsError(
                f"Distillation iteration is immutable and already exists: "
                f"{self.iteration_dir}"
            )
        self._validate_existing_schema()
        self.iteration_dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []
        self._games_buffered = 0
        self._shard_index = 0
        self._shards: list[dict[str, Any]] = []
        self._games: list[dict[str, Any]] = []

    def _validate_existing_schema(self) -> None:
        aggregate_path = self.root / "manifest.json"
        if not aggregate_path.is_file():
            return
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        expected = aggregate.get("schema_hash")
        actual = self.metadata.get("schema", {}).get("schema_hash")
        if expected is not None and actual != expected:
            raise ValueError(
                "Cannot mix distillation iterations with different model schemas: "
                f"expected {expected}, got {actual}"
            )

    def add_game(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        game_index: int,
        game_seed: int,
        student_color: Color,
        winner: Color | None,
        truncated: bool,
    ) -> None:
        self._rows.extend(dict(row) for row in rows)
        self._games_buffered += 1
        self._games.append(
            {
                "game_index": int(game_index),
                "game_seed": int(game_seed),
                "student_color": student_color.name,
                "winner": winner.name if winner is not None else None,
                "truncated": bool(truncated),
                "rows": len(rows),
            }
        )
        if self._games_buffered >= self.shard_games:
            self._flush()

    def _flush(self) -> None:
        games = self._games_buffered
        if games == 0:
            return
        if not self._rows:
            self._games_buffered = 0
            return

        import pandas as pd

        filename = f"shard-{self._shard_index:05d}.parquet"
        path = self.iteration_dir / filename
        if path.exists():
            raise FileExistsError(f"Refusing to replace immutable shard: {path}")
        temporary = self.iteration_dir / f".{filename}.tmp.parquet"
        frame = pd.DataFrame.from_records(self._rows)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        self._shards.append(
            {
                "path": os.fspath(path.relative_to(self.root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(frame),
                "games": games,
            }
        )
        self._shard_index += 1
        self._rows = []
        self._games_buffered = 0

    def finalize(self) -> Path:
        self._flush()
        manifest_path = self.iteration_dir / "manifest.json"
        manifest = {
            "format_version": DISTILLATION_FORMAT_VERSION,
            "iteration": self.iteration,
            "rows": sum(int(shard["rows"]) for shard in self._shards),
            "games": self._games,
            "shards": self._shards,
            "metadata": self.metadata,
        }
        _atomic_write_json(manifest_path, manifest)
        rebuild_aggregate_manifest(self.root)
        return manifest_path


def rebuild_aggregate_manifest(root: str | Path) -> Path:
    """Index all immutable iterations, rejecting schema drift."""

    root_path = Path(root)
    iteration_rows = []
    schema_hash: str | None = None
    total_rows = 0
    total_games = 0
    for path in sorted(root_path.glob("iteration-*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        current_schema = (
            manifest.get("metadata", {}).get("schema", {}).get("schema_hash")
        )
        if schema_hash is None:
            schema_hash = current_schema
        elif current_schema != schema_hash:
            raise ValueError(
                f"Distillation schema drift in {path}: "
                f"expected {schema_hash}, got {current_schema}"
            )
        rows = int(manifest.get("rows", 0))
        games = len(manifest.get("games", []))
        total_rows += rows
        total_games += games
        iteration_rows.append(
            {
                "iteration": int(manifest["iteration"]),
                "manifest": os.fspath(path.relative_to(root_path)),
                "manifest_sha256": sha256_file(path),
                "rows": rows,
                "games": games,
                "shards": manifest.get("shards", []),
            }
        )
    aggregate = {
        "format_version": DISTILLATION_FORMAT_VERSION,
        "schema_hash": schema_hash,
        "iterations": iteration_rows,
        "rows": total_rows,
        "games": total_games,
    }
    output = root_path / "manifest.json"
    _atomic_write_json(output, aggregate)
    return output


def verify_distillation_dataset(root: str | Path) -> list[str]:
    """Return integrity problems for manifests and immutable Parquet shards."""

    root_path = Path(root)
    aggregate_path = root_path / "manifest.json"
    if not aggregate_path.is_file():
        return [f"missing aggregate manifest: {aggregate_path}"]
    problems: list[str] = []
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    for iteration in aggregate.get("iterations", []):
        manifest_path = root_path / iteration["manifest"]
        if not manifest_path.is_file():
            problems.append(f"missing iteration manifest: {manifest_path}")
            continue
        expected_manifest_hash = iteration.get("manifest_sha256")
        if expected_manifest_hash != sha256_file(manifest_path):
            problems.append(f"iteration manifest hash mismatch: {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for shard in manifest.get("shards", []):
            shard_path = root_path / shard["path"]
            if not shard_path.is_file():
                problems.append(f"missing shard: {shard_path}")
            elif shard.get("sha256") != sha256_file(shard_path):
                problems.append(f"shard hash mismatch: {shard_path}")
    return problems


def manifest_metadata(
    *,
    config: DistillationConfig,
    schema: Mapping[str, Any],
    student: AgentIdentity,
    teacher: AgentIdentity,
    opponent: AgentIdentity,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the reproducibility envelope shared by dry-run and execution."""

    return {
        "config": asdict(config),
        "schema": dict(schema),
        "student": asdict(student),
        "teacher": asdict(teacher),
        "opponent": asdict(opponent),
        "provenance": dict(provenance),
    }


def prepare_distillation_iteration(
    config: DistillationConfig,
    *,
    repo: str | Path = ".",
) -> tuple[dict[str, Any], AgentIdentity, AgentIdentity, AgentIdentity]:
    """Resolve hashes and schema before any game or output mutation starts."""

    from catanatron.gym.model_schema import build_model_schema
    from catanatron.gym.provenance import collect_run_provenance

    schema = build_model_schema(
        num_players=2,
        map_type=MAP_TYPE,
        player_colors=PLAYER_COLORS,
        feature_profile=config.feature_profile,
        human_visible_obs=config.human_visible_obs,
    )
    student = AgentIdentity.from_spec(config.student_spec)
    teacher = AgentIdentity.from_spec(config.teacher_spec)
    opponent = AgentIdentity.from_spec(config.opponent_spec)
    metadata = manifest_metadata(
        config=config,
        schema=schema,
        student=student,
        teacher=teacher,
        opponent=opponent,
        provenance=collect_run_provenance(repo),
    )
    return metadata, student, teacher, opponent


def distillation_plan(
    config: DistillationConfig,
    *,
    output: str | Path,
    repo: str | Path = ".",
) -> dict[str, Any]:
    """Return the fully resolved, non-mutating plan used by ``--dry-run``."""

    metadata, _, _, _ = prepare_distillation_iteration(config, repo=repo)
    return {
        "output": os.fspath(Path(output).resolve()),
        "iteration_dir": os.fspath(
            (Path(output) / f"iteration-{config.iteration:04d}").resolve()
        ),
        "game_seeds": [
            derive_seed(
                "game",
                base_seed=config.base_seed,
                iteration=config.iteration,
                game_index=game_index,
            )
            for game_index in range(config.games)
        ],
        "metadata": metadata,
    }


def run_distillation_iteration(
    config: DistillationConfig,
    *,
    output: str | Path,
    repo: str | Path = ".",
) -> Path:
    """Collect one student-visited iteration and return its manifest path."""

    from catanatron.colonist_1v1 import COLONIST_1V1_SETTINGS
    from catanatron.game import Game
    from catanatron.models.map import build_map

    metadata, student_identity, teacher_identity, _ = prepare_distillation_iteration(
        config, repo=repo
    )
    writer = DistillationDatasetWriter(
        output,
        iteration=config.iteration,
        shard_games=config.shard_games,
        metadata=metadata,
    )
    recorder = DistillationDecisionRecorder(
        config=config,
        model_schema=metadata["schema"],
        student=student_identity,
        teacher=teacher_identity,
    )

    for game_index in range(config.games):
        game_seed = derive_seed(
            "game",
            base_seed=config.base_seed,
            iteration=config.iteration,
            game_index=game_index,
        )
        # Resolve/import participant implementations before seeding the game.
        # First-use imports and checkpoint construction are not part of the
        # simulator RNG contract and must not change the first board only.
        student_seat = game_index % 2 if config.alternate_seats else 0
        student_color = PLAYER_COLORS[student_seat]
        blue_player = build_player(
            config.student_spec if student_seat == 0 else config.opponent_spec,
            Color.BLUE,
        )
        red_player = build_player(
            config.student_spec if student_seat == 1 else config.opponent_spec,
            Color.RED,
        )
        teacher = build_player(config.teacher_spec, student_color)
        # Map construction and the legacy engine both consume module RNGs.
        # Isolate the entire game so its derived seed controls board and play,
        # while nested teacher calls restore the exact behavior-policy stream.
        with isolated_random_seed(game_seed):
            catan_map = build_map(
                COLONIST_1V1_SETTINGS.map_type,
                COLONIST_1V1_SETTINGS.number_placement,
            )
            game = Game(
                [blue_player, red_player],
                seed=game_seed,
                catan_map=catan_map,
                colonist_1v1=True,
                shuffle_players=False,
            )

            rows: list[dict[str, Any]] = []
            decision_index = 0

            def decide(player, current_game, playable_actions):
                nonlocal decision_index
                if player.color != student_color:
                    return player.decide(current_game, playable_actions)
                action, row = recorder.decide_and_label(
                    behavior=player,
                    teacher=teacher,
                    game=current_game,
                    playable_actions=playable_actions,
                    game_index=game_index,
                    game_seed=game_seed,
                    decision_index=decision_index,
                    seat=student_seat,
                )
                decision_index += 1
                if row is not None:
                    rows.append(row)
                return action

            winner = game.play(decide_fn=decide)
            writer.add_game(
                rows,
                game_index=game_index,
                game_seed=game_seed,
                student_color=student_color,
                winner=winner,
                truncated=winner is None,
            )
    return writer.finalize()

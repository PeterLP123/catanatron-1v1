"""File-backed data helpers for the Colonist training TUI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from catanatron.gym.colonist_training import (
    EVENTS_NAME,
    MANIFEST_NAME,
    MODEL_REGISTRY_NAME,
)


OPPONENT_COLUMNS: tuple[str, ...] = ("R", "W", "VP", "F", "G:25", "M:200", "AB:2")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_safe(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def read_jsonl_safe(path: Path, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []
    if limit is not None:
        lines = lines[-limit:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Ignore a partially-written last line while training is appending.
            continue
    return rows


def append_event(run_dir: Path, event_type: str, **data: Any) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json_safe(run_dir / MANIFEST_NAME, {})
    row = {
        "time": utc_now_iso(),
        "run_id": manifest.get("run_id"),
        "type": event_type,
        **data,
    }
    with (run_dir / EVENTS_NAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def update_manifest(run_dir: Path, **updates: Any) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_NAME
    manifest = read_json_safe(path, {})
    manifest.update(updates)
    manifest["updated_at"] = utc_now_iso()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def score_value(row: dict[str, Any]) -> float:
    return float(row.get("summary", {}).get("weighted_score") or 0.0)


def load_registry(
    run_dir: Path, *, limit: Optional[int] = None
) -> list[dict[str, Any]]:
    rows = read_jsonl_safe(run_dir / MODEL_REGISTRY_NAME, limit=limit)
    return sorted(rows, key=score_value, reverse=True)


def best_registry_row(rows: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not rows:
        return None
    return max(rows, key=score_value)


def win_rate_style(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "dim"
    if value >= 0.75:
        return "success"
    if value >= 0.52:
        return "warning"
    return "danger"


def format_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{100 * value:.1f}%"


def format_score(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.3f}"


def sparkline(values: Sequence[float]) -> str:
    chars = "▁▂▃▄▅▆▇█"
    if not values:
        return "-"
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return chars[0] * len(values)
    out = []
    for value in values:
        idx = int((value - lo) / (hi - lo) * (len(chars) - 1))
        out.append(chars[max(0, min(idx, len(chars) - 1))])
    return "".join(out)


@dataclass
class RunSummary:
    run_dir: Path
    run_id: str = "-"
    phase: str = "unknown"
    preset: str = "-"
    updated_at: str = "-"
    final_model: Optional[str] = None
    timesteps: int = 0
    target_timesteps: Optional[int] = None
    latest_score: Optional[float] = None
    best_score: Optional[float] = None
    best_label: str = "-"
    latest_event: str = "-"
    warnings: list[str] = field(default_factory=list)
    active_job: Optional[dict[str, Any]] = None

    @property
    def progress_ratio(self) -> float:
        if not self.target_timesteps:
            return 0.0
        return min(1.0, max(0.0, self.timesteps / self.target_timesteps))


def summarize_run(run_dir: Path) -> RunSummary:
    manifest = read_json_safe(run_dir / MANIFEST_NAME, {})
    events = read_jsonl_safe(run_dir / EVENTS_NAME, limit=500)
    registry = load_registry(run_dir)
    training = manifest.get("training", {})
    latest_ppo = next(
        (e for e in reversed(events) if e.get("type") == "ppo_progress"), {}
    )
    latest_eval = next(
        (e for e in reversed(events) if e.get("type") == "evaluation"), {}
    )
    latest_event = events[-1] if events else {}
    best = best_registry_row(registry)
    warnings = detect_warnings(run_dir, manifest, events, registry)
    active_job = manifest.get("active_job")
    return RunSummary(
        run_dir=run_dir,
        run_id=str(manifest.get("run_id") or run_dir.name),
        phase=str(manifest.get("phase") or "unknown"),
        preset=str(manifest.get("preset") or "-"),
        updated_at=str(manifest.get("updated_at") or manifest.get("created_at") or "-"),
        final_model=manifest.get("final_model"),
        timesteps=int(latest_ppo.get("timesteps") or 0),
        target_timesteps=training.get("timesteps"),
        latest_score=latest_eval.get("weighted_score"),
        best_score=score_value(best) if best else None,
        best_label=str(
            (best or {}).get("checkpoint_label")
            or Path(str((best or {}).get("checkpoint_path", "-"))).stem
        ),
        latest_event=str(latest_event.get("type") or "-"),
        warnings=warnings,
        active_job=active_job if isinstance(active_job, dict) else None,
    )


def list_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and (
            (p / MANIFEST_NAME).exists()
            or (p / EVENTS_NAME).exists()
            or (p / MODEL_REGISTRY_NAME).exists()
            or (p / "colonist_maskable_ppo.zip").exists()
        )
    ]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def detect_warnings(
    run_dir: Path,
    manifest: dict[str, Any],
    events: Sequence[dict[str, Any]],
    registry: Sequence[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    phase = manifest.get("phase")
    if phase not in {"done", "training_complete"} and not events:
        warnings.append("No telemetry events yet")
    if not registry:
        warnings.append("No evaluated models yet")
    latest_eval = next(
        (e for e in reversed(events) if e.get("type") == "evaluation"), None
    )
    if latest_eval and latest_eval.get("all_gates_passed") is False:
        warnings.append("Latest eval failed one or more gates")
    active_job = manifest.get("active_job")
    if isinstance(active_job, dict) and active_job.get("status") == "failed":
        warnings.append(f"Last job failed: {active_job.get('name', 'job')}")
    if phase == "final_eval":
        warnings.append("Final eval may be expensive")
    return warnings


def build_data_commands(
    *,
    python: str,
    teacher_specs: Sequence[str],
    num_games: int,
    data_root: Path,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for spec in teacher_specs:
        safe = spec.replace(":", "_").replace(",", "_").replace(" ", "_")
        commands.append(
            [
                python,
                "examples/colonist_1v1_generate_data.py",
                "--num",
                str(num_games),
                "--teachers",
                spec,
                "--output",
                os.fspath(data_root / safe),
            ]
        )
    return commands


def build_bc_command(
    *,
    python: str,
    data_dirs: Sequence[Path],
    epochs: int,
    run_dir: Path,
) -> list[str]:
    return [
        python,
        "examples/colonist_1v1_bc.py",
        "--data-dir",
        *[os.fspath(p) for p in data_dirs],
        "--epochs",
        str(epochs),
        "--out",
        os.fspath(run_dir / "bc.pt"),
        "--run-dir",
        os.fspath(run_dir),
    ]


def build_train_command(
    *,
    python: str,
    run_dir: Path,
    preset: str,
    curriculum: str,
    n_envs: int,
    eval_protocol: str,
) -> list[str]:
    return [
        python,
        "examples/colonist_1v1_train.py",
        "--preset",
        preset,
        "--run-dir",
        os.fspath(run_dir),
        "--bc-checkpoint",
        os.fspath(run_dir / "bc.pt"),
        "--mixed-league",
        "--curriculum",
        curriculum,
        "--n-envs",
        str(n_envs),
        "--eval-protocol",
        eval_protocol,
        "--final-eval-protocol",
        eval_protocol,
        "--tensorboard",
    ]


def build_eval_command(
    *,
    python: str,
    run_dir: Path,
    agent: str,
    protocol: str,
    num_games: int,
    label: str,
) -> list[str]:
    return [
        python,
        "examples/colonist_1v1_evaluate.py",
        "--agent",
        agent,
        "--benchmark",
        "--protocol",
        protocol,
        "--num-games",
        str(num_games),
        "--gates",
        "--run-dir",
        os.fspath(run_dir),
        "--checkpoint-label",
        label,
        "--registry",
        os.fspath(run_dir / MODEL_REGISTRY_NAME),
        "--report",
        os.fspath(run_dir / f"{label}_{protocol}_benchmark.json"),
    ]


def data_dirs_for_specs(data_root: Path, teacher_specs: Iterable[str]) -> list[Path]:
    dirs = []
    for spec in teacher_specs:
        safe = spec.replace(":", "_").replace(",", "_").replace(" ", "_")
        dirs.append(data_root / safe)
    return dirs

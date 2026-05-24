#!/usr/bin/env python3
"""Rich terminal dashboard for Colonist 1v1 training runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from catanatron.gym.colonist_training import EVENTS_NAME, MANIFEST_NAME, MODEL_REGISTRY_NAME


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{100 * value:.1f}%"


def build_dashboard(run_dir: Path):
    from rich import box
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table

    manifest = _read_json(run_dir / MANIFEST_NAME, {})
    events = _read_jsonl(run_dir / EVENTS_NAME)
    registry = _read_jsonl(run_dir / MODEL_REGISTRY_NAME)
    latest_events = events[-12:]

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="body"),
        Layout(name="events", size=12),
    )
    layout["body"].split_row(Layout(name="progress"), Layout(name="models"))

    training = manifest.get("training", {})
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    header.add_row(
        f"[bold]Run[/bold] {manifest.get('run_id', '-')}",
        f"[bold]Phase[/bold] {manifest.get('phase', '-')}",
    )
    header.add_row(
        f"[bold]Preset[/bold] {manifest.get('preset', '-')}",
        f"[bold]Run dir[/bold] {manifest.get('run_dir', run_dir)}",
    )
    layout["header"].update(Panel(header, title="Colonist 1v1 Training"))

    progress = Table(title="Training Progress", box=box.SIMPLE_HEAVY)
    progress.add_column("Metric")
    progress.add_column("Value", justify="right")
    latest_ppo = next((e for e in reversed(events) if e.get("type") == "ppo_progress"), {})
    latest_bc = next((e for e in reversed(events) if e.get("type") == "bc_epoch"), {})
    latest_eval = next((e for e in reversed(events) if e.get("type") == "evaluation"), {})
    progress.add_row("Timesteps", f"{latest_ppo.get('timesteps', 0):,} / {training.get('timesteps', '-')}")
    progress.add_row("Envs", str(training.get("n_envs", "-")))
    progress.add_row("Curriculum", str(training.get("curriculum", "-")))
    progress.add_row("Eval protocol", str(training.get("eval_protocol", "-")))
    progress.add_row("BC val acc", _fmt_pct(latest_bc.get("val_accuracy")))
    progress.add_row("Latest score", f"{latest_eval.get('weighted_score', 0.0):.3f}" if latest_eval else "-")
    progress.add_row("Final model", str(manifest.get("final_model", "-")))
    layout["progress"].update(Panel(progress, title="Progress"))

    models = Table(title="Model Leaderboard", box=box.SIMPLE_HEAVY)
    models.add_column("Label")
    models.add_column("Step", justify="right")
    models.add_column("Score", justify="right")
    models.add_column("R", justify="right")
    models.add_column("W", justify="right")
    models.add_column("VP", justify="right")
    models.add_column("F", justify="right")
    for row in sorted(
        registry,
        key=lambda r: r.get("summary", {}).get("weighted_score", -1),
        reverse=True,
    )[:12]:
        wins = row.get("win_rates", {})
        models.add_row(
            str(row.get("checkpoint_label") or Path(str(row.get("checkpoint_path", "-"))).stem),
            f"{row.get('training_timesteps') or '-'}",
            f"{row.get('summary', {}).get('weighted_score', 0.0):.3f}",
            _fmt_pct(wins.get("R")),
            _fmt_pct(wins.get("W")),
            _fmt_pct(wins.get("VP")),
            _fmt_pct(wins.get("F")),
        )
    if not registry:
        models.add_row("-", "-", "-", "-", "-", "-", "-")
    layout["models"].update(Panel(models, title="Evaluated Models"))

    event_table = Table(title="Recent Events", box=box.SIMPLE)
    event_table.add_column("Time")
    event_table.add_column("Type")
    event_table.add_column("Details")
    for event in latest_events:
        details = {k: v for k, v in event.items() if k not in {"time", "type", "run_id"}}
        event_table.add_row(event.get("time", "-"), event.get("type", "-"), json.dumps(details)[:120])
    layout["events"].update(Panel(event_table, title="Event Log"))
    return layout


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("runs/colonist_1v1"))
    p.add_argument("--refresh", type=float, default=2.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args(argv)

    from rich.console import Console
    from rich.live import Live

    console = Console()
    if args.once:
        console.print(build_dashboard(args.run_dir))
        return 0

    with Live(build_dashboard(args.run_dir), console=console, screen=True, refresh_per_second=4) as live:
        while True:
            live.update(build_dashboard(args.run_dir))
            time.sleep(args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())

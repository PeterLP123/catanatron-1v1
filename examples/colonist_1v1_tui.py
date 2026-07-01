#!/usr/bin/env python3
"""Interactive terminal app for Colonist 1v1 training, evaluation, and ranking."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from catanatron.gym.colonist_training import EVENTS_NAME, MODEL_REGISTRY_NAME
from catanatron.gym.experiment_backlog import EXPERIMENTS, backlog_statuses
from catanatron.gym.tui_data import (
    OPPONENT_COLUMNS,
    build_bc_command,
    build_data_commands,
    build_eval_command,
    build_train_command,
    data_dirs_for_specs,
    format_pct,
    format_score,
    format_duration,
    list_run_dirs,
    load_registry,
    read_jsonl_safe,
    sparkline,
    summarize_run,
    win_rate_style,
)
from catanatron.gym.tui_jobs import JobRunner


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def build_dashboard(run_dir: Path):
    """Rich fallback/static renderable used by ``--once`` and tests."""
    from rich import box
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table

    summary = summarize_run(run_dir)
    events = read_jsonl_safe(run_dir / EVENTS_NAME, limit=12)
    registry = load_registry(run_dir, limit=200)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body"),
        Layout(name="events", size=8),
    )
    layout["body"].split_row(Layout(name="progress"), Layout(name="models"))

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    header.add_row(
        f"[bold cyan]Run[/] {summary.run_id}", f"[bold]Phase[/] {summary.phase}"
    )
    header.add_row(
        f"[bold]Preset[/] {summary.preset}",
        f"[bold]Best[/] {summary.best_label} {format_score(summary.best_score)}",
    )
    layout["header"].update(Panel(header, title="Colonist 1v1 Command Center"))

    progress = Table(title="Training Progress", box=box.SIMPLE_HEAVY)
    progress.add_column("Metric")
    progress.add_column("Value", justify="right")
    progress.add_row(
        "Timesteps", f"{summary.timesteps:,} / {_fmt(summary.target_timesteps)}"
    )
    progress.add_row("Progress", f"{summary.progress_ratio:.1%}")
    progress.add_row(
        "Rate",
        (
            "-"
            if summary.steps_per_second is None
            else f"{summary.steps_per_second:,.0f} steps/s"
        ),
    )
    progress.add_row("ETA", format_duration(summary.eta_seconds))
    progress.add_row("Elapsed", format_duration(summary.elapsed_seconds))
    progress.add_row(
        "Workers", f"{summary.vec_env} × {summary.n_envs or '-'} · seed {summary.seed}"
    )
    progress.add_row("Latest score", format_score(summary.latest_score))
    progress.add_row(
        "Warnings", "; ".join(summary.warnings) if summary.warnings else "none"
    )
    layout["progress"].update(Panel(progress, title="Progress"))

    models = Table(title="Model Leaderboard", box=box.SIMPLE_HEAVY)
    models.add_column("Label")
    models.add_column("Score", justify="right")
    for opp in ("R", "W", "VP", "F"):
        models.add_column(opp, justify="right")
    for row in registry[:12]:
        wins = row.get("win_rates", {})
        models.add_row(
            str(
                row.get("checkpoint_label")
                or Path(str(row.get("checkpoint_path", "-"))).stem
            ),
            format_score(row.get("summary", {}).get("weighted_score")),
            *[format_pct(wins.get(opp)) for opp in ("R", "W", "VP", "F")],
        )
    if not registry:
        models.add_row("-", "-", "-", "-", "-", "-")
    layout["models"].update(Panel(models, title="Ranking"))

    event_table = Table(title="Recent Events", box=box.SIMPLE)
    event_table.add_column("Time")
    event_table.add_column("Type")
    event_table.add_column("Details")
    for event in events:
        details = {
            k: v for k, v in event.items() if k not in {"time", "type", "run_id"}
        }
        event_table.add_row(
            event.get("time", "-"), event.get("type", "-"), str(details)[:120]
        )
    layout["events"].update(Panel(event_table, title="Event Log"))
    return layout


def make_textual_app(runs_root: Path, run_dir: Path, refresh: float):
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container, Horizontal, Vertical
        from textual.widgets import (
            Button,
            DataTable,
            Footer,
            Header,
            Input,
            Label,
            Log,
            ProgressBar,
            Select,
            Static,
            TabPane,
            TabbedContent,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Textual is not installed. Install it with: python3 -m pip install -e '.[tui]'"
        ) from exc

    class ColonistTrainingApp(App):
        CSS = """
        Screen {
            background: #07111f;
            color: #d9e6ff;
        }
        Header, Footer {
            background: #12335b;
            color: white;
        }
        TabbedContent {
            margin: 1;
        }
        .card {
            border: round #2f80ed;
            background: #0d1b2d;
            padding: 1 2;
            margin: 1;
            height: auto;
        }
        .hero {
            border: double #00d4ff;
            background: #081a2f;
            color: #e9fbff;
            padding: 1 2;
            margin: 1;
        }
        .danger {
            color: #ff6b6b;
        }
        .warning {
            color: #ffd166;
        }
        .success {
            color: #7bd88f;
        }
        .muted {
            color: #8aa1bd;
        }
        DataTable {
            margin: 1;
            height: 1fr;
        }
        Input, Select {
            margin: 0 1 1 1;
        }
        Button {
            margin: 0 1 1 1;
        }
        #log {
            height: 1fr;
            border: round #415a77;
            margin: 1;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "refresh", "Refresh"),
            ("1", "show_tab('runs')", "Runs"),
            ("2", "show_tab('launch')", "Launch"),
            ("3", "show_tab('monitor')", "Monitor"),
            ("4", "show_tab('ranking')", "Ranking"),
            ("5", "show_tab('evaluate')", "Evaluate"),
            ("6", "show_tab('logs')", "Logs"),
            ("7", "show_tab('backlog')", "Backlog"),
            ("c", "cancel_job", "Cancel Job"),
        ]

        def __init__(self, runs_root: Path, run_dir: Path, refresh: float):
            super().__init__()
            self.runs_root = runs_root
            self.run_dir = run_dir
            self.refresh = refresh
            self.repo_root = _repo_root()
            self.python = sys.executable
            self.runner = JobRunner(
                run_dir, cwd=self.repo_root, on_log=self._append_log
            )
            self._last_run_dirs: list[Path] = []

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with TabbedContent(initial="runs", id="tabs"):
                with TabPane("Runs", id="runs"):
                    yield Static(id="runs-hero", classes="hero")
                    yield DataTable(id="runs-table")
                with TabPane("Launch Training", id="launch"):
                    yield Static(
                        "Launch the full workflow: teacher data -> BC -> PPO. Defaults are safe smoke settings.",
                        classes="hero",
                    )
                    with Horizontal():
                        with Vertical(classes="card"):
                            yield Label("Run tag")
                            yield Input(
                                value=f"c1_tui_{int(time.time())}", id="launch-run-tag"
                            )
                            yield Label("Teacher specs")
                            yield Input(value="F,F VP,F", id="launch-teachers")
                            yield Label("Teacher games per spec")
                            yield Input(value="20", id="launch-games")
                            yield Label("BC epochs")
                            yield Input(value="2", id="launch-epochs")
                        with Vertical(classes="card"):
                            yield Label("Training preset")
                            yield Select(
                                [
                                    ("smoke", "smoke"),
                                    ("standard", "standard"),
                                    ("strong", "strong"),
                                    ("overnight", "overnight"),
                                ],
                                value="smoke",
                                id="launch-preset",
                            )
                            yield Label("Curriculum")
                            yield Select(
                                [
                                    ("balanced", "balanced"),
                                    ("strong", "strong"),
                                    ("self_play", "self_play"),
                                ],
                                value="balanced",
                                id="launch-curriculum",
                            )
                            yield Label("Eval protocol")
                            yield Select(
                                [
                                    ("fast", "fast"),
                                    ("milestone", "milestone"),
                                    ("full", "full"),
                                ],
                                value="fast",
                                id="launch-eval-protocol",
                            )
                            yield Label("Vector envs")
                            yield Input(value="1", id="launch-n-envs")
                    yield Static(id="launch-command", classes="card")
                    yield Button(
                        "Run Data -> BC -> PPO", id="launch-workflow", variant="success"
                    )
                with TabPane("Monitor", id="monitor"):
                    yield Static(id="monitor-hero", classes="hero")
                    yield ProgressBar(total=100, id="progress-bar")
                    with Horizontal():
                        yield Static(id="monitor-progress", classes="card")
                        yield Static(id="monitor-warnings", classes="card")
                    yield DataTable(id="eval-matrix")
                with TabPane("Ranking", id="ranking"):
                    yield Static(id="ranking-hero", classes="hero")
                    yield DataTable(id="ranking-table")
                with TabPane("Backlog", id="backlog"):
                    yield Static(
                        "Dependency-aware UCL GPU experiment queue.", classes="hero"
                    )
                    yield DataTable(id="backlog-table")
                with TabPane("Evaluate", id="evaluate"):
                    yield Static(
                        "Evaluate an existing checkpoint and append it to the registry.",
                        classes="hero",
                    )
                    yield Label("Agent spec")
                    yield Input(
                        value="",
                        placeholder="L:runs/.../colonist_maskable_ppo.zip",
                        id="eval-agent",
                    )
                    yield Label("Checkpoint label")
                    yield Input(value="manual", id="eval-label")
                    yield Label("Protocol")
                    yield Select(
                        [
                            ("fast", "fast"),
                            ("milestone", "milestone"),
                            ("full", "full"),
                        ],
                        value="fast",
                        id="eval-protocol",
                    )
                    yield Label("Games per opponent")
                    yield Input(value="20", id="eval-games")
                    yield Static(id="eval-command", classes="card")
                    yield Button("Run Evaluation", id="run-eval", variant="primary")
                with TabPane("Logs / Jobs", id="logs"):
                    yield Static(id="jobs-hero", classes="hero")
                    yield Log(id="log", highlight=True)
                    yield Button("Cancel Active Job", id="cancel-job", variant="error")
                with TabPane("Help", id="help"):
                    yield Static(
                        "Keyboard: 1 Runs, 2 Launch, 3 Monitor, 4 Ranking, 5 Evaluate, 6 Logs, 7 Backlog, r Refresh, c Cancel, q Quit\n\n"
                        "Protocols: fast = R/W/VP/F, milestone adds G:25, full adds expensive M/AB search.\n"
                        "Curricula move from teachers/baselines toward self-play and stronger opponents.\n"
                        "The TUI writes commands, job status, and metrics into run_manifest.json and training_events.jsonl.",
                        classes="card",
                    )
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(self.refresh, self.refresh_data)
            self.refresh_data()

        def action_show_tab(self, tab_id: str) -> None:
            self.query_one("#tabs", TabbedContent).active = tab_id

        def action_refresh(self) -> None:
            self.refresh_data()

        def action_cancel_job(self) -> None:
            self.runner.cancel()
            self._append_log("cancel requested")

        def _append_log(self, line: str) -> None:
            try:
                self.call_from_thread(self.query_one("#log", Log).write_line, line)
            except Exception:
                pass

        def _selected_run_dir(self) -> Path:
            return self.run_dir

        def _parse_int(self, selector: str, default: int) -> int:
            value = self.query_one(selector, Input).value.strip()
            try:
                return int(value)
            except ValueError:
                return default

        def _launch_values(self) -> dict[str, Any]:
            run_tag = (
                self.query_one("#launch-run-tag", Input).value.strip()
                or f"c1_tui_{int(time.time())}"
            )
            teacher_specs = self.query_one("#launch-teachers", Input).value.split()
            run_dir = self.runs_root / run_tag
            data_root = Path("data") / run_tag
            return {
                "run_dir": run_dir,
                "data_root": data_root,
                "teacher_specs": teacher_specs or ["F,F"],
                "num_games": self._parse_int("#launch-games", 20),
                "epochs": self._parse_int("#launch-epochs", 2),
                "preset": str(self.query_one("#launch-preset", Select).value),
                "curriculum": str(self.query_one("#launch-curriculum", Select).value),
                "eval_protocol": str(
                    self.query_one("#launch-eval-protocol", Select).value
                ),
                "n_envs": self._parse_int("#launch-n-envs", 1),
            }

        def _workflow_command(self) -> list[str]:
            values = self._launch_values()
            script = (
                "set -euo pipefail; "
                + " && ".join(
                    " ".join(map(_shell_quote, cmd))
                    for cmd in build_data_commands(
                        python=self.python,
                        teacher_specs=values["teacher_specs"],
                        num_games=values["num_games"],
                        data_root=values["data_root"],
                    )
                )
                + " && "
                + " ".join(
                    map(
                        _shell_quote,
                        build_bc_command(
                            python=self.python,
                            data_dirs=data_dirs_for_specs(
                                values["data_root"], values["teacher_specs"]
                            ),
                            epochs=values["epochs"],
                            run_dir=values["run_dir"],
                        ),
                    )
                )
                + " && "
                + " ".join(
                    map(
                        _shell_quote,
                        build_train_command(
                            python=self.python,
                            run_dir=values["run_dir"],
                            preset=values["preset"],
                            curriculum=values["curriculum"],
                            n_envs=values["n_envs"],
                            eval_protocol=values["eval_protocol"],
                        ),
                    )
                )
            )
            return ["bash", "-lc", script]

        def _eval_command(self) -> list[str]:
            protocol = str(self.query_one("#eval-protocol", Select).value)
            games = self._parse_int("#eval-games", 20)
            agent = self.query_one("#eval-agent", Input).value.strip()
            if not agent:
                agent = f"L:{self.run_dir / 'colonist_maskable_ppo.zip'}"
            label = self.query_one("#eval-label", Input).value.strip() or "manual"
            return build_eval_command(
                python=self.python,
                run_dir=self.run_dir,
                agent=agent,
                protocol=protocol,
                num_games=games,
                label=label,
            )

        def on_input_changed(self, event: Input.Changed) -> None:
            self.update_command_previews()

        def on_select_changed(self, event: Select.Changed) -> None:
            self.update_command_previews()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "launch-workflow":
                values = self._launch_values()
                self.run_dir = values["run_dir"]
                self.runner = JobRunner(
                    self.run_dir, cwd=self.repo_root, on_log=self._append_log
                )
                self.runner.start("training_workflow", self._workflow_command())
                self.query_one("#tabs", TabbedContent).active = "logs"
                self.refresh_data()
            elif event.button.id == "run-eval":
                self.runner = JobRunner(
                    self.run_dir, cwd=self.repo_root, on_log=self._append_log
                )
                self.runner.start("evaluation", self._eval_command())
                self.query_one("#tabs", TabbedContent).active = "logs"
            elif event.button.id == "cancel-job":
                self.action_cancel_job()

        def update_command_previews(self) -> None:
            try:
                launch_cmd = " ".join(map(_shell_quote, self._workflow_command()))
                try:
                    self.query_one("#launch-command", Static).update(
                        f"[bold]Command[/]\n{launch_cmd}"
                    )
                except Exception:
                    pass
                eval_cmd = " ".join(map(_shell_quote, self._eval_command()))
                try:
                    self.query_one("#eval-command", Static).update(
                        f"[bold]Command[/]\n{eval_cmd}"
                    )
                except Exception:
                    pass
            except Exception as exc:
                try:
                    self.query_one("#launch-command", Static).update(
                        f"Command preview unavailable: {exc}"
                    )
                except Exception:
                    pass

        def refresh_data(self) -> None:
            self.update_command_previews()
            self.refresh_runs()
            self.refresh_monitor()
            self.refresh_ranking()
            self.refresh_backlog()
            self.refresh_logs()

        def refresh_runs(self) -> None:
            runs = list_run_dirs(self.runs_root)
            if self.run_dir.exists() and self.run_dir not in runs:
                runs.insert(0, self.run_dir)
            self._last_run_dirs = runs
            try:
                hero = self.query_one("#runs-hero", Static)
                table = self.query_one("#runs-table", DataTable)
            except Exception:
                return
            hero.update(
                f"[bold cyan]Runs root[/] {self.runs_root}  [bold]Found[/] {len(runs)}"
            )
            table.clear(columns=True)
            for col in ("Run", "Phase", "Preset", "Progress", "Best", "Warnings"):
                table.add_column(col)
            for run in runs:
                s = summarize_run(run)
                table.add_row(
                    run.name,
                    s.phase,
                    s.preset,
                    f"{s.progress_ratio:.0%}",
                    f"{s.best_label} {format_score(s.best_score)}",
                    "; ".join(s.warnings) or "-",
                )

        def refresh_monitor(self) -> None:
            s = summarize_run(self.run_dir)
            try:
                self.query_one("#monitor-hero", Static).update(
                    f"[bold cyan]{s.run_id}[/]  phase=[bold]{s.phase}[/]  "
                    f"best=[bold green]{s.best_label} {format_score(s.best_score)}[/]"
                )
                self.query_one("#progress-bar", ProgressBar).progress = int(
                    s.progress_ratio * 100
                )
                self.query_one("#monitor-progress", Static).update(
                    f"[bold]Timesteps[/] {s.timesteps:,} / {_fmt(s.target_timesteps)}\n"
                    f"[bold]Latest score[/] {format_score(s.latest_score)}\n"
                    f"[bold]Rate[/] {_fmt(None if s.steps_per_second is None else f'{s.steps_per_second:,.0f} steps/s')}\n"
                    f"[bold]ETA[/] {format_duration(s.eta_seconds)}\n"
                    f"[bold]Elapsed[/] {format_duration(s.elapsed_seconds)}\n"
                    f"[bold]Workers[/] {s.vec_env} × {s.n_envs or '-'} · seed {s.seed}\n"
                    f"[bold]Latest event[/] {s.latest_event}\n"
                    f"[bold]Final model[/] {_fmt(s.final_model)}"
                )
                warnings = (
                    "\n".join(f"[red]WARN[/] {w}" for w in s.warnings)
                    or "[green]No warnings[/]"
                )
                self.query_one("#monitor-warnings", Static).update(warnings)
            except Exception:
                return
            self._fill_eval_matrix()

        def refresh_backlog(self) -> None:
            try:
                table = self.query_one("#backlog-table", DataTable)
            except Exception:
                return
            statuses = backlog_statuses(self.runs_root)
            table.clear(columns=True)
            for col in ("ID", "Status", "Stage", "GPU hours", "Disk"):
                table.add_column(col)
            for experiment in EXPERIMENTS:
                lo, hi = experiment.gpu_hours
                table.add_row(
                    experiment.id,
                    statuses[experiment.id],
                    experiment.stage,
                    f"{lo:g}–{hi:g}",
                    f"{experiment.storage_gib:g} GiB",
                )

        def _fill_eval_matrix(self) -> None:
            try:
                table = self.query_one("#eval-matrix", DataTable)
            except Exception:
                return
            table.clear(columns=True)
            for col in ("Model", "Score", *OPPONENT_COLUMNS, "Gates"):
                table.add_column(col)
            rows = load_registry(self.run_dir, limit=100)[:20]
            for row in rows:
                wins = row.get("win_rates", {})
                gates = row.get("gates", {})
                cells = []
                for opp in OPPONENT_COLUMNS:
                    value = wins.get(opp)
                    style = win_rate_style(value)
                    cells.append(f"[{style}]{format_pct(value)}[/]")
                passed = sum(1 for v in gates.values() if v is True)
                total = len(gates)
                table.add_row(
                    str(
                        row.get("checkpoint_label")
                        or Path(str(row.get("checkpoint_path", "-"))).stem
                    ),
                    format_score(row.get("summary", {}).get("weighted_score")),
                    *cells,
                    f"{passed}/{total}" if total else "-",
                )

        def refresh_ranking(self) -> None:
            rows = load_registry(self.run_dir, limit=500)
            scores = [
                float(r.get("summary", {}).get("weighted_score") or 0.0)
                for r in rows[:30]
            ]
            try:
                self.query_one("#ranking-hero", Static).update(
                    f"[bold]Score trend[/] {sparkline(list(reversed(scores[-20:])))}  "
                    f"[bold]Models[/] {len(rows)}"
                )
                table = self.query_one("#ranking-table", DataTable)
            except Exception:
                return
            table.clear(columns=True)
            for col in (
                "Rank",
                "Label",
                "Protocol",
                "Step",
                "Score",
                "R",
                "W",
                "VP",
                "F",
                "Report",
            ):
                table.add_column(col)
            for i, row in enumerate(rows[:50], start=1):
                wins = row.get("win_rates", {})
                table.add_row(
                    str(i),
                    str(
                        row.get("checkpoint_label")
                        or Path(str(row.get("checkpoint_path", "-"))).stem
                    ),
                    str(row.get("protocol") or "-"),
                    str(row.get("training_timesteps") or "-"),
                    format_score(row.get("summary", {}).get("weighted_score")),
                    *[
                        f"[{win_rate_style(wins.get(opp))}]{format_pct(wins.get(opp))}[/]"
                        for opp in ("R", "W", "VP", "F")
                    ],
                    str(row.get("report_path") or "-"),
                )

        def refresh_logs(self) -> None:
            s = summarize_run(self.run_dir)
            active = s.active_job or {}
            try:
                self.query_one("#jobs-hero", Static).update(
                    f"[bold]Active job[/] {active.get('name', '-')}  "
                    f"[bold]status[/] {active.get('status', '-')}  "
                    f"[bold]exit[/] {active.get('exit_code', '-')}"
                )
            except Exception:
                return

    def _shell_quote(value: Any) -> str:
        import shlex

        return shlex.quote(os.fspath(value))

    return ColonistTrainingApp(runs_root, run_dir, refresh)


def _run_textual_app(args: argparse.Namespace) -> int:
    make_textual_app(args.runs_root, args.run_dir, args.refresh).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("runs/colonist_1v1"))
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--refresh", type=float, default=2.0)
    p.add_argument(
        "--once", action="store_true", help="Render one Rich snapshot and exit."
    )
    args = p.parse_args(argv)

    if args.once:
        from rich.console import Console

        Console().print(build_dashboard(args.run_dir))
        return 0
    return _run_textual_app(args)


if __name__ == "__main__":
    raise SystemExit(main())

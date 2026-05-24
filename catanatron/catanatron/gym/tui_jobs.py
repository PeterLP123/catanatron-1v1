"""Background subprocess orchestration for the Colonist training TUI."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence
from uuid import uuid4

from catanatron.gym.tui_data import append_event, update_manifest, utc_now_iso


LogCallback = Callable[[str], None]


@dataclass
class BackgroundJob:
    """A subprocess launched from the TUI."""

    name: str
    command: list[str]
    run_dir: Path
    job_id: str = field(default_factory=lambda: uuid4().hex[:10])
    status: str = "pending"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    process: Optional[subprocess.Popen[str]] = field(default=None, repr=False)
    log_path: Optional[Path] = None

    def to_manifest(self) -> dict[str, object]:
        return {
            "id": self.job_id,
            "name": self.name,
            "command": self.command,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "log_path": os.fspath(self.log_path) if self.log_path else None,
        }


class JobRunner:
    """Run one long-lived subprocess at a time and write TUI telemetry."""

    def __init__(
        self,
        run_dir: Path,
        *,
        cwd: Optional[Path] = None,
        on_log: Optional[LogCallback] = None,
    ):
        self.run_dir = run_dir
        self.cwd = cwd or Path.cwd()
        self.on_log = on_log
        self.active: Optional[BackgroundJob] = None
        self._lock = threading.Lock()

    def start(self, name: str, command: Sequence[str]) -> BackgroundJob:
        with self._lock:
            if self.active is not None and self.active.status == "running":
                raise RuntimeError(f"Job already running: {self.active.name}")
            job = BackgroundJob(name=name, command=list(command), run_dir=self.run_dir)
            self.active = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def cancel(self) -> None:
        job = self.active
        if job is None or job.process is None or job.status != "running":
            return
        append_event(self.run_dir, "job_cancel_requested", job=job.to_manifest())
        try:
            job.process.send_signal(signal.SIGINT)
            time.sleep(1.0)
            if job.process.poll() is None:
                job.process.terminate()
        except OSError:
            pass

    def _emit_log(self, line: str) -> None:
        if self.on_log:
            self.on_log(line)

    def _run(self, job: BackgroundJob) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = self.run_dir / "tui_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        job.log_path = logs_dir / f"{job.job_id}_{job.name.replace(' ', '_')}.log"
        job.status = "running"
        job.started_at = utc_now_iso()
        update_manifest(self.run_dir, active_job=job.to_manifest())
        append_event(self.run_dir, "job_started", job=job.to_manifest())

        try:
            with job.log_path.open("a", encoding="utf-8") as log:
                log.write(f"$ {' '.join(job.command)}\n")
                job.process = subprocess.Popen(
                    job.command,
                    cwd=os.fspath(self.cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert job.process.stdout is not None
                for line in job.process.stdout:
                    log.write(line)
                    log.flush()
                    self._emit_log(line.rstrip())
                job.exit_code = job.process.wait()
            job.status = "succeeded" if job.exit_code == 0 else "failed"
        except Exception as exc:  # pragma: no cover - defensive guard for UI jobs
            job.status = "failed"
            job.exit_code = -1
            self._emit_log(f"job runner error: {exc}")
        finally:
            job.ended_at = utc_now_iso()
            update_manifest(self.run_dir, active_job=job.to_manifest())
            append_event(self.run_dir, "job_finished", job=job.to_manifest())

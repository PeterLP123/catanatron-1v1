"""Background subprocess orchestration for the Colonist training TUI."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence
from uuid import uuid4

from catanatron.file_utils import write_json_atomic
from catanatron.gym.tui_data import JOB_STATE_NAME, append_event, utc_now_iso


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
    error: Optional[str] = None
    finished: threading.Event = field(default_factory=threading.Event, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)
    _stop_requested: threading.Event = field(
        default_factory=threading.Event, repr=False
    )

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
            "error": self.error,
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
            if self.active is not None and not self.active.finished.is_set():
                raise RuntimeError(f"Job already running: {self.active.name}")
            job = BackgroundJob(name=name, command=list(command), run_dir=self.run_dir)
            self.active = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def cancel(self) -> None:
        """Request cancellation without doing I/O or waiting on the UI thread."""
        with self._lock:
            job = self.active
            if job is not None and job.status in {"pending", "running"}:
                job._cancel_requested = True
                job._stop_requested.set()

    def _run(self, job: BackgroundJob) -> None:
        stopper = None
        try:
            logs_dir = job.run_dir / "tui_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            job.log_path = logs_dir / f"{job.job_id}_{job.name.replace(' ', '_')}.log"
            if job._cancel_requested:
                return
            with job.log_path.open("a", encoding="utf-8") as log:
                log.write(f"$ {' '.join(job.command)}\n")
                job.process = subprocess.Popen(
                    job.command,
                    cwd=os.fspath(self.cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=os.name == "posix",
                )
                stopper = threading.Thread(
                    target=self._stop_process, args=(job,), daemon=True
                )
                stopper.start()
                job.status = "running"
                job.started_at = utc_now_iso()
                write_json_atomic(job.run_dir / JOB_STATE_NAME, job.to_manifest())
                append_event(job.run_dir, "job_started", job=job.to_manifest())
                assert job.process.stdout is not None
                for line in job.process.stdout:
                    log.write(line)
                    log.flush()
                    if self.on_log:
                        self.on_log(line.rstrip())
                job.exit_code = job.process.wait()
        except Exception as exc:
            job.error = str(exc)
            logging.getLogger(__name__).exception("Job %s failed", job.name)
        finally:
            # Also stop/reap the child on log, callback, and telemetry failures.
            # The stopper can interrupt a blocked stdout read without blocking the UI.
            job._stop_requested.set()
            if job.process is not None:
                if stopper is not None and stopper.ident is not None:
                    stopper.join()
                else:
                    self._stop_process(job)
                job.exit_code = job.process.wait()
                if job.process.stdout is not None:
                    job.process.stdout.close()
            try:
                with self._lock:
                    job.ended_at = utc_now_iso()
                    if job._cancel_requested:
                        job.status = "cancelled"
                    else:
                        job.status = (
                            "succeeded"
                            if job.exit_code == 0 and not job.error
                            else "failed"
                        )
                if job._cancel_requested:
                    append_event(
                        job.run_dir, "job_cancel_requested", job=job.to_manifest()
                    )
                append_event(job.run_dir, "job_finished", job=job.to_manifest())
            except Exception as exc:
                job.error = str(exc)
                job.status = "failed"
                logging.getLogger(__name__).exception(
                    "Could not record job %s", job.name
                )
            try:
                write_json_atomic(job.run_dir / JOB_STATE_NAME, job.to_manifest())
            except Exception as exc:
                job.error = str(exc)
                job.status = "failed"
                logging.getLogger(__name__).exception("Could not save job %s", job.name)
            finally:
                job.finished.set()

    def _stop_process(self, job: BackgroundJob) -> None:
        job._stop_requested.wait()
        process = job.process
        assert process is not None
        if os.name != "posix":
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            return

        # The session belongs to this job. Signal the whole group even if the
        # shell/parent already exited, since workers may still hold stdout open.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            if sig == signal.SIGKILL:
                return
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                process.poll()
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.05)

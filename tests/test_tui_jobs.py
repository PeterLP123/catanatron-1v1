import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from catanatron.gym.colonist_training import TrainingRunTracker
from catanatron.gym.experiment_backlog import direct_status, experiments_by_id
from catanatron.gym.tui_data import JOB_STATE_NAME, summarize_run
from catanatron.gym.tui_jobs import JobRunner


def test_pending_job_blocks_second_launch_and_can_be_cancelled(tmp_path, monkeypatch):
    queued = []
    monkeypatch.setattr(threading.Thread, "start", lambda thread: queued.append(thread))
    runner = JobRunner(tmp_path)
    job = runner.start("first", ["must-not-be-launched"])
    with pytest.raises(RuntimeError, match="Job already running"):
        runner.start("second", ["unused"])
    runner.cancel()
    runner.cancel()
    queued[0].run()
    assert job.finished.is_set()
    assert job.status == "cancelled"
    assert job.process is None
    assert summarize_run(tmp_path).active_job["status"] == "cancelled"


def test_job_status_survives_training_updates_and_reaches_backlog(tmp_path):
    run_dir = tmp_path / "00-gpu-smoke"
    tracker = TrainingRunTracker(run_dir, run_id="test-run")
    lines = []
    observed = []

    def on_log(line):
        lines.append(line)
        tracker.phase("ppo_training")
        observed.append(summarize_run(run_dir).active_job["status"])

    runner = JobRunner(run_dir, on_log=on_log)
    job = runner.start("short", [sys.executable, "-c", "print('hello from job')"])
    assert job.finished.wait(5)
    assert job.status == "succeeded"
    assert job.exit_code == 0
    assert lines == ["hello from job"]
    assert observed == ["running"]
    assert job.process.stdout.closed
    tracker.phase("training_complete")
    assert summarize_run(run_dir).active_job["status"] == "succeeded"
    assert "active_job" not in json.loads(tracker.manifest_path.read_text())

    failed = runner.start("bad command", [sys.executable, "-c", "raise SystemExit(3)"])
    assert failed.finished.wait(5)
    tracker.phase("ppo_training")
    summary = summarize_run(run_dir)
    assert summary.active_job["exit_code"] == 3
    assert "Last job failed: bad command" in summary.warnings
    assert direct_status(experiments_by_id()["00-gpu-smoke"], tmp_path) == "failed"
    events = [json.loads(line) for line in tracker.events_path.read_text().splitlines()]
    assert [e["job"]["status"] for e in events if e["type"] == "job_finished"] == [
        "succeeded",
        "failed",
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group shutdown")
@pytest.mark.parametrize("parent_exits", [False, True])
def test_cancel_is_nonblocking_and_kills_stubborn_workers(tmp_path, parent_exits):
    ready = threading.Event()
    child_pid = []

    def on_log(line):
        child_pid.append(int(line))
        ready.set()

    child_code = (
        "import os, signal, time; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(os.getpid(), flush=True); time.sleep(60)"
    )
    parent_code = (
        "import signal, subprocess, sys, time; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        + ("sys.exit(0)" if parent_exits else "time.sleep(60)")
    )
    runner = JobRunner(tmp_path, on_log=on_log)
    job = runner.start("workers", [sys.executable, "-c", parent_code])
    try:
        assert ready.wait(5)
        if parent_exits:
            assert job.process.wait(timeout=5) == 0
        started = time.monotonic()
        runner.cancel()
        runner.cancel()
        assert time.monotonic() - started < 0.2
        with pytest.raises(RuntimeError, match="Job already running"):
            runner.start("too soon", ["unused"])
        assert job.finished.wait(8)
        assert job.status == "cancelled"
        assert job.process.poll() is not None
        assert job.process.stdout.closed
        # An orphan can briefly remain a zombie awaiting the OS reaper.
        state = subprocess.run(
            ["ps", "-p", str(child_pid[0]), "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert not state or state.startswith("Z")
        saved = json.loads((tmp_path / JOB_STATE_NAME).read_text())
        assert saved["status"] == "cancelled"
    finally:
        runner.cancel()
        if job.process is not None:
            try:
                os.killpg(job.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job.process.wait(timeout=5)
        assert job.finished.wait(5)


@pytest.mark.parametrize("failure", ["callback", "telemetry"])
def test_runner_failure_reaps_process_and_releases_runner(
    tmp_path, monkeypatch, failure
):
    def broken_callback(line):
        raise RuntimeError("UI disappeared")

    runner = JobRunner(
        tmp_path, on_log=broken_callback if failure == "callback" else None
    )
    if failure == "telemetry":

        def broken_events(*args, **kwargs):
            raise OSError("events unavailable")

        monkeypatch.setattr("catanatron.gym.tui_jobs.append_event", broken_events)
    job = runner.start(
        "logging",
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(60)",
        ],
    )
    try:
        assert job.finished.wait(8)
        assert job.status == "failed"
        expected_error = (
            "UI disappeared" if failure == "callback" else "events unavailable"
        )
        assert job.error == expected_error
        assert job.process.poll() is not None
        assert job.process.stdout.closed
        assert summarize_run(tmp_path).active_job["error"] == expected_error
        monkeypatch.undo()
        next_job = runner.start("next", [sys.executable, "-c", "pass"])
        assert next_job.finished.wait(5)
        assert next_job.status == "succeeded"
    finally:
        runner.cancel()
        if not job.finished.is_set() and job.process is not None:
            job.process.kill()
            job.process.wait(timeout=5)


def test_setup_failure_is_recorded_as_finished(tmp_path):
    (tmp_path / "tui_logs").write_text("not a directory")
    job = JobRunner(tmp_path).start("cannot start", ["unused"])
    assert job.finished.wait(5)
    assert job.status == "failed"
    assert job.process is None
    assert summarize_run(tmp_path).active_job["error"]

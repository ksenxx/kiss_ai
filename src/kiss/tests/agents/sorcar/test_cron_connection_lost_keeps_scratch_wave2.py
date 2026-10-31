# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A prompt job whose daemon connection fails while its task runs keeps its scratch directory.

A prompt job's task runs on the daemon in the run's scratch directory.
When the connection to the daemon fails after the ``run`` was sent, the
task MAY still be running there, exactly as when the daemon never
confirms a stop.  ``_run_prompt_job`` raised the keep-the-directory
``TimeoutError`` only for the unconfirmed stop; a connection lost while
the call waited, or while cron stopped the timed-out job, came back as
an ordinary error and ``_execute_job`` deleted the directory under the
task.  Real dispatches against local-WSS daemon stand-ins; nothing is
mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs, tick
from kiss.tests.agents.sorcar.test_dispatch_stop_cascade import _RecordingDaemon


class _DropOnStopDaemon(_RecordingDaemon):
    """Runs the task silently; closes the connection when the client sends ``stop``."""

    def __init__(self) -> None:
        super().__init__("silent")

    async def _handle(self, ws: ServerConnection) -> None:
        """Serve one client: report the task running, never confirm a stop."""
        try:
            run_cmd = json.loads(await ws.recv())
            self.run_cmd = run_cmd
            await ws.send(json.dumps(
                {"type": "status", "running": True, "tabId": run_cmd.get("tabId", "")}
            ))
            async for message in ws:
                command = json.loads(message)
                self.commands.append(command)
                if command.get("type") == "stop":
                    return  # Returning from the handler closes the connection.
        except ConnectionClosed:
            return


def _run_job(
    daemon: _RecordingDaemon, monkeypatch: pytest.MonkeyPatch, name: str, prompt: str,
    timeout: str = "",
) -> dict:
    """Create the prompt job *name*, run it once against *daemon*; return its record."""
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    try:
        created = yaml.safe_load(cron_job(
            "create", name=name, prompt=prompt, schedule="every 1d", timeout=timeout,
        ))["created"]
        jobs = load_jobs()
        for job in jobs:
            if job["id"] == created["id"]:
                job["next_run_at"] = 1.0
        cron_agent.save_jobs(jobs)
        assert tick(2.0) == 1
    finally:
        daemon.close()
    assert daemon.run_cmd is not None, "the run never reached the daemon"
    (stored,) = load_jobs()
    return stored


def _kept_run_dirs(home: Path) -> list[Path]:
    """Return the run scratch directories left under ``$KISS_HOME/cron/runs``."""
    runs = home / "cron" / "runs"
    return [p for p in runs.iterdir() if p.is_dir()] if runs.is_dir() else []


def test_connection_lost_while_the_task_runs_keeps_the_scratch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    stored = _run_job(_RecordingDaemon(mode="drop"), monkeypatch, name="dropped", prompt="say hi")

    (kept,) = _kept_run_dirs(tmp_path)
    assert stored["last_status"] == "error"
    summary = stored["last_summary"]
    assert f"may still be running in {kept} (its scratch directory {kept} is kept)" in summary
    assert "the connection to the daemon failed while the" in summary
    assert "MAY STILL BE RUNNING" in summary


def test_connection_lost_while_cron_stops_the_timed_out_task_keeps_the_scratch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    daemon = _DropOnStopDaemon()
    stored = _run_job(daemon, monkeypatch, name="slow", prompt="think hard", timeout="0.5")

    assert [c for c in daemon.commands if c.get("type") == "stop"], "cron never stopped the task"
    (kept,) = _kept_run_dirs(tmp_path)
    assert stored["last_status"] == "error"
    summary = stored["last_summary"]
    assert "timed out after 0.5s" in summary
    assert f"scratch directory {kept} is kept" in summary
    assert "MAY STILL BE RUNNING" in summary

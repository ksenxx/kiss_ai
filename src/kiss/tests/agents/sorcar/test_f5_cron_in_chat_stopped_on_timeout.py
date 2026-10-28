# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cron (audit F5): an ``in_chat`` job whose timeout stop the daemon confirmed.

``_run_in_chat_job`` runs the prompt with ``stop_on_timeout=True``.  A
timed-out run the daemon confirms stopped raises
``daemon_client.StoppedOnTimeoutError`` — a ``TimeoutError`` subclass
that ``_run_in_chat_job`` did not handle, so it escaped to
``_execute_job``'s ``except TimeoutError``, the branch reserved for an
UNCONFIRMED stop: the run's scratch directory was kept (as if the task
might still be running) and the bare client message was recorded.
The confirmed stop is now the ordinary ``"error"`` outcome, as a
prompt job's confirmed stop is, and the scratch directory is removed.

Real job store under an isolated ``KISS_HOME``; the daemon is a local
WebSocket stand-in (:class:`_StopConfirmingDaemon`) that never finishes
the task and confirms the client's ``stop`` with the terminal status.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs
from kiss.tests.agents.sorcar.test_dispatch_timeout import _StopConfirmingDaemon


def test_in_chat_job_stopped_on_timeout_is_a_recorded_error_without_a_kept_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmed timeout stop records ``error`` and frees the run's scratch directory."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    daemon = _StopConfirmingDaemon()
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", str(daemon.endpoint_file))
    try:
        created = yaml.safe_load(cron_job(
            "create", name="chatty", prompt="never finishes", schedule="0 8 * * *",
            deliver="none", in_chat=True, timeout="0.5",
        ))["created"]
        reply = yaml.safe_load(cron_job("run_now", job_id=created["id"]))
        assert daemon.wait_for_command("stop"), daemon.commands
    finally:
        daemon.close()
    assert reply["ran"]["last_status"] == "error", reply
    stored = next(job for job in load_jobs() if job["id"] == created["id"])
    assert stored["last_status"] == "error"
    assert stored["last_summary"] == "in-chat job timed out after 0.5s and was stopped"
    runs_dir = tmp_path / "cron" / "runs"
    assert not runs_dir.exists() or list(runs_dir.iterdir()) == [], (
        "a confirmed stop must not keep the run's scratch directory"
    )

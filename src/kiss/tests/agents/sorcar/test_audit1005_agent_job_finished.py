# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-05 (scope D): a ``run_agent`` job is finished once its result is recorded.

``_finish_agent_job`` writes the job's ``result``, then sets its
``running`` event (its ``finally``), and only then does the thread
exit.  ``run_agent(wait="false")`` returns as soon as ``running`` is
set and ``agent_job`` used to decide "still running" by
``thread.is_alive()``, so a dispatch that failed before any tab existed
(no daemon reachable) was reported as "Started ... its tab is open",
and an immediate ``agent_job(id, "tail")`` could still say "is still
running" although the error was already on the job.  ``AgentJob.finished``
(the result is recorded) is now the one predicate.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    make_agent_job_tool,
    make_run_agent_tool,
)
from kiss.tests.agents.sorcar.test_dispatch_timeout import _detached_job, _StopConfirmingDaemon


@pytest.fixture(autouse=True)
def _no_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the dispatch at a missing endpoint file and drop leftover jobs."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    yield
    agent_dispatch.kill_jobs_of(None)


def test_wait_false_returns_the_error_when_the_dispatch_already_failed(tmp_path: Path) -> None:
    started = time.monotonic()
    out = make_run_agent_tool(str(tmp_path))("say hi", wait="false")
    assert time.monotonic() - started < 10
    assert out.startswith("Error: the sorcar agent task could not run: Cannot connect"), out
    assert "its tab is open" not in out
    # Nothing is left for ``agent_job``: the error was the answer.
    assert agent_dispatch.agent_jobs_of(None) == {}


def test_a_recorded_result_is_never_reported_as_still_running(tmp_path: Path) -> None:
    job = agent_dispatch.start_agent_job(
        "sorcar",
        {
            "name": "sorcar", "prompt": "say hi", "agent_path": DEFAULT_AGENT_PATH,
            "work_dir": str(tmp_path), "model_name": "", "budget": None, "timeout": 5.0,
            "alias": "",
        },
        None,
    )
    assert job.running.wait(10)
    # Woken by the event, a caller sees the result at once, whether or
    # not the job's thread has exited yet.
    assert job.finished
    assert job.result.startswith("Error: the sorcar agent task could not run: Cannot connect")
    tool = make_agent_job_tool()
    assert tool(job.job_id, "tail") == job.result
    assert tool(job.job_id, "kill") == job.result


def test_wait_false_keeps_its_notice_for_a_task_that_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    try:
        out = make_run_agent_tool(str(tmp_path))("never finishes", wait="false")
        assert out.startswith("Started the sorcar agent task as job agent-"), out
        assert "its tab is open" in out
        job = _detached_job(out)
        assert not job.finished
        tool = make_agent_job_tool()
        assert tool(job.job_id, "tail") == f"Job {job.job_id} (sorcar agent task) is still running."
        killed = tool(job.job_id, "kill")
        assert "was stopped before it finished" in killed, killed
        assert job.finished and tool(job.job_id, "tail") == killed
    finally:
        daemon.close()

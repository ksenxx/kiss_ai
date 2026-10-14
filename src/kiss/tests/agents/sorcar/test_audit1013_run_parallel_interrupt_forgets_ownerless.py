# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-13 (R4): an interrupted ``run_agents_parallel`` drops its owner-less jobs.

``_run_agent`` forgets a job without an owner when the call is
interrupted (no run end ever collects it); ``run_agents_parallel``
cancelled its jobs on an interrupt but left owner-less ones in the
daemon-wide registry for good.  Both now behave the same.

The interrupt is injected with a trace function the moment the call
reaches ``join_agent_job``, i.e. once every child is started and
registered; the daemon stand-in confirms the cancellation's stop.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.tests.agents.sorcar.test_dispatch_timeout import _StopConfirmingDaemon


def _stop_at_the_join(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace function: raise KeyboardInterrupt on entering ``join_agent_job``."""
    if frame.f_code is agent_dispatch.join_agent_job.__code__:
        sys.settrace(None)
        raise KeyboardInterrupt
    return _stop_at_the_join


def test_interrupted_parallel_call_forgets_its_ownerless_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    before = set(agent_dispatch.agent_jobs_of(None))
    outcome: list[str] = []

    def call() -> None:
        sys.settrace(_stop_at_the_join)
        try:
            agent_dispatch.run_agents_parallel(
                str(tmp_path), ["never finishes", "never finishes either"], timeout="30",
            )
            outcome.append("returned")
        except KeyboardInterrupt:
            outcome.append("interrupted")
        finally:
            sys.settrace(None)

    try:
        worker = threading.Thread(target=call, name="stopped-caller")
        worker.start()
        worker.join(30)
        assert not worker.is_alive()
        assert outcome == ["interrupted"], outcome
        leaked = set(agent_dispatch.agent_jobs_of(None)) - before
        assert leaked == set(), "owner-less jobs stayed registered after the interrupt"
        # The children were cancelled: their threads end once the daemon confirms the stop.
        for thread in threading.enumerate():
            if thread.name.startswith("agent-job-"):
                thread.join(30)
                assert not thread.is_alive()
    finally:
        for job_id, job in agent_dispatch.agent_jobs_of(None).items():
            if job_id not in before:
                job.cancel.set()
                agent_dispatch.forget_agent_job(job)
        daemon.close()

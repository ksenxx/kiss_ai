# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A settling ``run_agent`` job is counted once in a sibling's budget share.

Audit 2026-10-30 (S6), RACE: a job's spend reached its caller's ledger
(``_attribute_dispatch_usage``) before its result was recorded, and its
budget reservation lapsed only with the result.  A job of another call
resolving its inherited share in between saw the settling job both as
a reservation and as spend: with a $10 caller, a $5 job that spent
$4.90 left the next job $0.05 instead of half of the $5.10 left.

The test pauses the settling job's thread right after its dispatch
returned (at its call of ``format_dispatch_result``, the step that
precedes the result write) and dispatches the next job meanwhile.  Real
job threads, a real local-WSS daemon stand-in and a real caller ledger;
nothing is mocked.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    live_agent_jobs,
    make_run_agent_tool,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.third_party_agents.muse_test_utils import wait_until
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon


@pytest.fixture()
def daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[RecordingDaemon]:
    """A daemon stand-in whose every sub-task reports a $4.90 spend."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    stand_in = RecordingDaemon(cost=4.9)
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", str(stand_in.endpoint_file))
    try:
        yield stand_in
    finally:
        stand_in.close()


class _PauseAtFormat:
    """Thread trace hook: park the first job thread that formats its result.

    ``_finish_agent_job`` records a job's result right after
    ``format_dispatch_result`` returns, so a thread parked at that call
    has finished its dispatch (its spend is settled) but not yet
    recorded its result.
    """

    def __init__(self) -> None:
        self.parked = threading.Event()
        self.release = threading.Event()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if (
            event == "call"
            and frame.f_code is agent_dispatch.format_dispatch_result.__code__
            and not self.parked.is_set()
        ):
            self.parked.set()
            self.release.wait(30)


def test_a_settling_job_is_counted_once_in_the_next_share(
    tmp_path: Path, daemon: RecordingDaemon,
) -> None:
    parent = SorcarAgent("settle-parent")
    parent.max_budget = 10.0
    parent.budget_used = 0.0
    parent._current_executor = None
    run_agent = make_run_agent_tool(str(tmp_path), parent)
    pause = _PauseAtFormat()
    saved_trace = threading.gettrace()
    threading.settrace(pause)
    try:
        first = run_agent("first", DEFAULT_AGENT_PATH, max_budget="5", wait="false", timeout="60")
        assert pause.parked.wait(30), first
        (settling,) = live_agent_jobs(parent)
        assert not settling.finished
        assert parent._live_budget_used() == pytest.approx(4.9)
        run_agent("second", DEFAULT_AGENT_PATH, wait="false", timeout="60")
        assert wait_until(lambda: len(daemon.run_commands) == 2, 30.0), daemon.run_commands
        # The first job holds nothing any more: its $4.90 is on the ledger,
        # so the second gets half of the $5.10 left.
        assert float(daemon.run_commands[1]["maxBudget"]) == pytest.approx(2.55)
    finally:
        threading.settrace(saved_trace)  # type: ignore[arg-type]
        pause.release.set()
        for job in live_agent_jobs(parent):
            assert agent_dispatch.join_agent_job(job, 30.0), job.job_id
        agent_dispatch.kill_jobs_of(parent)

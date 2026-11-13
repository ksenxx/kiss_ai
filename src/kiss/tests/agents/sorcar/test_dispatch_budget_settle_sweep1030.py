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

2026-11-10: the spend landed and the reservation lapsed in one hold of
the job registry's lock, and landing the spend flushes the caller's
queued events.  With the event writer blocked, one settling job kept
the registry from every ``finish()``, ``agent_job`` call and job
completion of the daemon.  The spend now lands outside that lock,
between two short holds that mark the job as settling and lapse its
reservation; the last two tests hold a settling job's thread in between
and check that the registry answers and that a share resolved meanwhile
waits for the spend, so the job is still counted once.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent, persistence, sorcar_agent
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    live_agent_jobs,
    make_run_agent_tool,
)
from kiss.agents.sorcar.daemon_client import TaskResult
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.third_party_agents.muse_test_utils import wait_until
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon
from kiss.tests.persistence_helpers import close_db
from kiss.tests.server.pause import Pause


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


def _parent() -> SorcarAgent:
    """A caller with a $10 budget and nothing spent."""
    parent = SorcarAgent("settle-parent")
    parent.max_budget = 10.0
    parent.budget_used = 0.0
    parent._current_executor = None
    return parent


def test_a_settling_job_is_counted_once_in_the_next_share(
    tmp_path: Path, daemon: RecordingDaemon,
) -> None:
    parent = _parent()
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


def test_a_share_resolved_during_a_settle_waits_for_the_spend(
    tmp_path: Path, daemon: RecordingDaemon,
) -> None:
    """A job of another call is not given its share while the first job's spend is landing.

    The first job's thread is held at its ledger bank: the job is marked
    settling, its $5 is still reserved and its $4.90 is not on the ledger.
    A share resolved now would count the $5, and one resolved right after
    the bank both.  The second call's job waits instead, with the registry
    free, and gets half of the $5.10 left once the first has settled.
    """
    parent = _parent()
    run_agent = make_run_agent_tool(str(tmp_path), parent)
    second = threading.Thread(
        target=run_agent, args=("second", DEFAULT_AGENT_PATH),
        kwargs={"wait": "false", "timeout": "60"}, name="second-call", daemon=True,
    )
    try:
        with Pause(sorcar_agent._attribute_sub_usage.__code__, lambda _locals: True) as held:
            run_agent("first", DEFAULT_AGENT_PATH, max_budget="5", wait="false", timeout="60")
            assert held.paused.wait(30), "the first job never banked its spend"
            (settling,) = live_agent_jobs(parent)
            assert settling.settling and settling.budget == pytest.approx(5.0)
            second.start()
            assert wait_until(lambda: len(live_agent_jobs(parent)) == 2, 30.0)
            time.sleep(1.0)  # the second job's thread has asked for its share by now
            assert len(daemon.run_commands) == 1, daemon.run_commands
        assert wait_until(lambda: len(daemon.run_commands) == 2, 30.0), daemon.run_commands
        assert float(daemon.run_commands[1]["maxBudget"]) == pytest.approx(2.55)
        assert not settling.settling and settling.budget is None
    finally:
        second.join(60)
        for job in live_agent_jobs(parent):
            assert agent_dispatch.join_agent_job(job, 30.0), job.job_id
        agent_dispatch.kill_jobs_of(parent)


@pytest.fixture()
def history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point persistence at a fresh database under *tmp_path*."""
    monkeypatch.setattr(persistence, "_KISS_DIR", tmp_path)
    monkeypatch.setattr(persistence, "_DB_PATH", tmp_path / "history.db")
    close_db()
    yield
    close_db()


def test_a_settling_job_frees_the_registry_while_its_charge_flushes(history: None) -> None:
    """The registry answers while a finished sub-task's charge waits for the database.

    The caller has a persisted row, so the charge flushes its queued
    events first.  The settling thread is held at that flush: another
    thread still lists the caller's jobs, and the job's $5 stays reserved
    until its $4.90 is on the ledger.
    """
    parent = _parent()
    task_id, _chat = persistence._add_task("caller of the settling job")
    job = agent_dispatch.new_agent_job("sorcar", {"timeout": 60.0, "budget": 5.0}, parent)
    result = TaskResult(text="done", success=True, cost=4.9, tokens=7, steps=1)
    fold = threading.Thread(
        target=agent_dispatch._attribute_dispatch_usage,
        args=(parent, result, None, task_id, job), name="settling-fold", daemon=True,
    )
    listing = threading.Thread(
        target=agent_dispatch.agent_jobs_of, args=(parent,), name="registry-reader", daemon=True,
    )
    flushing = persistence._flush_chat_events.__code__
    with Pause(flushing, lambda _locals: threading.current_thread() is fold) as held:
        fold.start()
        assert held.paused.wait(30), "the charge never flushed the caller's events"
        listing.start()
        listing.join(5)
        assert not listing.is_alive(), "the job registry is locked while the charge flushes"
        assert job.settling and job.budget == pytest.approx(5.0)
    fold.join(30)
    assert not fold.is_alive(), "the fold did not finish"
    assert parent._live_budget_used() == pytest.approx(4.9)
    assert not job.settling and job.budget is None

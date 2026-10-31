# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Budgets handed to still-running sub-tasks are reserved against the caller.

Audit 2026-10-20, finding B4-F1: a ``run_agent(wait="false")`` child's
budget share was computed from the caller's remainder alone, so three
overlapping non-blocking calls on a $10 parent each got $5 ($15 against
a $10 remainder) and the parent could die of budget exhaustion caused
by budgets it handed out.  Now each dispatched job records the budget
it was given (:attr:`AgentJob.budget`) and a later call takes what
the caller's still-running jobs hold off the remainder before
splitting it.

The children run against a daemon stand-in that keeps every sub-task
running until the test releases it, so the reservations are live while
the next calls plan.  Real threads and a real local-WSS daemon
connection; nothing is mocked.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    live_agent_jobs,
    make_run_agent_tool,
    make_run_parallel_tool,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.third_party_agents.muse_test_utils import wait_until
from kiss.tests.agents.third_party_agents.recording_daemon import (
    RecordingDaemon,
    serve_recording_daemon,
)


class HoldingDaemon(RecordingDaemon):
    """A :class:`RecordingDaemon` whose sub-tasks stay running until :meth:`release`."""

    def __init__(self, **kwargs: object) -> None:
        self.gate = threading.Event()
        super().__init__(**kwargs)  # type: ignore[arg-type]

    async def _handle(self, ws: ServerConnection) -> None:
        try:
            command = json.loads(await ws.recv())
        except (ConnectionClosed, ValueError):
            return
        self.commands.append(command)
        tab_id = command.get("tabId", "")
        try:
            await ws.send(json.dumps({"type": "status", "running": True, "tabId": tab_id}))
            while not self.gate.is_set():
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({**self.result, "tabId": tab_id}))
            await ws.send(json.dumps({"type": "status", "running": False, "tabId": tab_id}))
        except ConnectionClosed:
            pass

    def release(self) -> None:
        """Let every held sub-task finish."""
        self.gate.set()

    def close(self) -> None:
        """Release the held sub-tasks, then stop serving.

        A handler parked on the gate keeps the server from shutting down,
        so without the release the base ``close`` waits out its 30 s join
        and leaves the loop thread running.
        """
        self.release()
        super().close()


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)


@pytest.fixture()
def daemon() -> Iterator[HoldingDaemon]:
    with serve_recording_daemon(cron_agent, "_daemon_endpoint_file", HoldingDaemon) as stand_in:
        assert isinstance(stand_in, HoldingDaemon)
        yield stand_in


def _parent(budget: float) -> SorcarAgent:
    """A parent with *budget* to spend and nothing spent yet."""
    parent = SorcarAgent("reserve-parent")
    parent.max_budget = budget
    parent.budget_used = 0.0
    parent._current_executor = None
    return parent


def _settle(parent: SorcarAgent, daemon: HoldingDaemon) -> None:
    """Release the held sub-tasks and wait for every job of *parent* to finish."""
    daemon.release()
    for job in live_agent_jobs(parent):
        assert agent_dispatch.join_agent_job(job, 30.0), job.job_id
    agent_dispatch.kill_jobs_of(parent)


def _budgets(daemon: HoldingDaemon) -> list[float]:
    return [float(command["maxBudget"]) for command in daemon.run_commands]


def test_overlapping_non_blocking_children_split_what_is_left(
    tmp_path: Path,
    daemon: HoldingDaemon,
) -> None:
    """Three live ``wait="false"`` children of a $10 parent get 5, 2.5 and 1.25, not 5, 5, 5."""
    parent = _parent(10.0)
    run_agent = make_run_agent_tool(str(tmp_path), parent)
    try:
        notices = [
            run_agent(task, DEFAULT_AGENT_PATH, wait="false", timeout="60")
            for task in ("first", "second", "third")
        ]
        assert all(notice.startswith("Started the ") for notice in notices), notices
        jobs = live_agent_jobs(parent)
        assert len(jobs) == 3
        assert _budgets(daemon) == pytest.approx([5.0, 2.5, 1.25])
        assert sorted(job.budget or 0.0 for job in jobs) == pytest.approx([1.25, 2.5, 5.0])
        assert sum(_budgets(daemon)) < parent.max_budget
    finally:
        _settle(parent, daemon)
    # The reservations are released with the jobs: a child dispatched
    # after they finished (their stand-in spend is $0) gets half of the
    # whole remainder again.
    assert live_agent_jobs(parent) == []
    daemon.gate.clear()
    try:
        run_agent("fourth", DEFAULT_AGENT_PATH, wait="false", timeout="60")
        assert _budgets(daemon)[-1] == pytest.approx(5.0)
    finally:
        _settle(parent, daemon)


def _plan(parent: SorcarAgent, work_dir: Path, task: str, max_budget: str = "") -> dict[str, Any]:
    """Plan a ``run_agent`` call of *parent* without starting it."""
    plan = agent_dispatch.plan_run_agent(
        str(work_dir), task, DEFAULT_AGENT_PATH, max_budget=max_budget, parent_agent=parent
    )
    assert isinstance(plan, dict), plan
    return plan


def test_calls_planned_concurrently_still_see_each_others_reservation(
    tmp_path: Path,
    daemon: HoldingDaemon,
) -> None:
    """Two ``run_agent`` calls of one owner that overlap do not both take half.

    Two tool calls executing at the same time (the Codex MCP bridge
    runs each on its own thread) both plan before either job exists,
    so a snapshot of the owner's live jobs taken at plan time is
    empty for both and each would get half of the $10 remainder.
    The reservation is read where the budget is resolved, under the
    job registry's lock, so whichever job resolves second sees the
    first one's $5 and gets half of the $5 left.
    """
    parent = _parent(10.0)
    plans = [_plan(parent, tmp_path, task) for task in ("first", "second")]
    try:
        jobs = [agent_dispatch.start_agent_job(plan["name"], plan, parent) for plan in plans]
        for job in jobs:
            agent_dispatch.wait_until_started(job)
            assert not job.finished, job.result
        assert sorted(_budgets(daemon)) == pytest.approx([2.5, 5.0])
    finally:
        _settle(parent, daemon)


def test_an_explicit_budget_is_reserved_from_registration(
    tmp_path: Path,
    daemon: HoldingDaemon,
) -> None:
    """A concurrent call with its own ``max_budget`` holds it before its job thread runs.

    The explicit $1 child of a $1 parent is registered, and an
    inherited sibling planned at the same time is started right after
    it: the sibling's resolution finds the $1 already held and gets
    the exhaustion error, not half of a remainder that is spoken for.
    """
    parent = _parent(1.0)
    greedy = _plan(parent, tmp_path, "greedy", max_budget="1")
    starved = _plan(parent, tmp_path, "starved")
    try:
        greedy_job = agent_dispatch.start_agent_job(greedy["name"], greedy, parent)
        assert greedy_job.budget == pytest.approx(1.0)
        starved_job = agent_dispatch.start_agent_job(starved["name"], starved, parent)
        assert agent_dispatch.join_agent_job(starved_job, 30.0)
        assert "reserve-parent has no remaining budget" in starved_job.result, starved_job.result
        assert "$1.0000 of its remainder is reserved" in starved_job.result, starved_job.result
        agent_dispatch.wait_until_started(greedy_job)
        assert _budgets(daemon) == pytest.approx([1.0])
    finally:
        _settle(parent, daemon)


def test_a_child_that_finished_between_planning_and_dispatch_is_not_charged_twice(
    tmp_path: Path,
    daemon: HoldingDaemon,
) -> None:
    """R4 review issue 1: a reservation lapses once the child's spend is on the ledger.

    A $10 parent starts a $5 child; a second call is planned while the
    first runs (so the plan holds its $5 reservation); the first then
    finishes spending its whole $5 before the second is dispatched.
    The ledger says $5 spent and nothing is held any more, so the
    second child gets half of the $5 left — not ``BudgetExceededError``
    from the stale reservation being taken off the remainder a second
    time.
    """
    parent = _parent(10.0)
    run_agent = make_run_agent_tool(str(tmp_path), parent)
    daemon.result["cost"] = "5.000000"
    try:
        run_agent("first", DEFAULT_AGENT_PATH, wait="false", timeout="60")
        first = live_agent_jobs(parent)
        assert [job.budget for job in first] == pytest.approx([5.0])
        plan = _plan(parent, tmp_path, "second")
    finally:
        _settle(parent, daemon)
    assert parent._live_budget_used() == pytest.approx(5.0)
    assert all(job.finished for job in first)
    daemon.gate.clear()
    job = agent_dispatch.start_agent_job(plan["name"], plan, parent)
    try:
        agent_dispatch.wait_until_started(job)
        assert not job.finished, job.result
        assert _budgets(daemon) == pytest.approx([5.0, 2.5])
    finally:
        _settle(parent, daemon)


class PausingParent(SorcarAgent):
    """A parent whose ledger read for a sub-task's share can be held open.

    The real ``SorcarAgent`` ledger and share arithmetic are used; the
    override only adds a synchronization point right after the ledger
    was read, so a test can let another child settle in that window.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pause = False
        self.ledger_read = threading.Event()
        self.resume = threading.Event()

    def _subagent_budget_share(self, num_tasks: int) -> float | None:
        share = super()._subagent_budget_share(num_tasks)
        if self.pause:
            self.ledger_read.set()
            assert self.resume.wait(30.0)
        return share


def test_a_child_settling_between_the_ledger_and_reservation_reads_is_counted(
    tmp_path: Path,
    daemon: HoldingDaemon,
) -> None:
    """Review round 1, item 2: a child that finishes mid-resolution is never dropped from both.

    A $10 parent runs a $5 child.  A second child resolves its share
    while the first still runs; the first then finishes (its $8 spend
    arrives) before the second child's resolution is done.  Reading the
    ledger first and the reservations second saw the old ledger ($0
    spent) and no reservation, and handed the second child $5 of the
    $2 actually left.  A child's spend now lands, and its reservation
    lapses, in the critical section the resolution holds (audit
    2026-10-30 S6), so the first child settles only after the second
    child's share was resolved against its live $5 reservation:
    (10 - 5) / 2 = $2.50.
    """
    parent = PausingParent("reserve-parent")
    parent.max_budget = 10.0
    parent.budget_used = 0.0
    parent._current_executor = None
    run_agent = make_run_agent_tool(str(tmp_path), parent)
    daemon.result["cost"] = "8.000000"
    try:
        run_agent("first", DEFAULT_AGENT_PATH, wait="false", timeout="60")
        first = live_agent_jobs(parent)
        assert [job.budget for job in first] == pytest.approx([5.0])
        plan = _plan(parent, tmp_path, "second")
        parent.pause = True
        second = agent_dispatch.start_agent_job(plan["name"], plan, parent)
        assert parent.ledger_read.wait(30.0)
        daemon.release()
        # The first child's spend waits for the resolution in progress.
        assert not agent_dispatch.join_agent_job(first[0], 0.5)
        assert parent._live_budget_used() == pytest.approx(0.0)
        parent.resume.set()
        assert agent_dispatch.join_agent_job(first[0], 30.0)
        assert agent_dispatch.join_agent_job(second, 30.0)
        assert _budgets(daemon) == pytest.approx([5.0, 2.5])
    finally:
        parent.resume.set()
        _settle(parent, daemon)


def test_fan_out_siblings_split_the_unreserved_remainder_equally(
    tmp_path: Path,
    daemon: HoldingDaemon,
) -> None:
    """A ``run_parallel`` after a live child splits (remainder - reserved) evenly, N+1 ways."""
    parent = _parent(10.0)
    run_agent = make_run_agent_tool(str(tmp_path), parent)
    run_parallel = make_run_parallel_tool(str(tmp_path), parent)
    try:
        run_agent("first", DEFAULT_AGENT_PATH, wait="false", timeout="60")
        # The call blocks for its timeout: short, so the two children are
        # still held when it returns their notices.
        out = run_parallel(json.dumps(["second", "third"]), DEFAULT_AGENT_PATH, timeout="0.5")
        entries = yaml.safe_load(out)
        assert len(entries) == 2 and all("still running" in e for e in entries), out
        assert wait_until(lambda: len(daemon.run_commands) == 3, 10.0), daemon.run_commands
        # (10 - 5) / (2 + 1): the siblings of one call never subtract
        # each other, whichever thread resolves its budget first.
        assert _budgets(daemon) == pytest.approx([5.0, 5.0 / 3, 5.0 / 3])
    finally:
        _settle(parent, daemon)


def test_child_whose_reservations_exhaust_the_caller_gets_the_error(
    tmp_path: Path,
    daemon: HoldingDaemon,
) -> None:
    """An explicit child budget as large as the remainder leaves the next child nothing."""
    parent = _parent(1.0)
    run_agent = make_run_agent_tool(str(tmp_path), parent)
    try:
        run_agent("greedy", DEFAULT_AGENT_PATH, max_budget="1", wait="false", timeout="60")
        out = run_agent("starved", DEFAULT_AGENT_PATH, wait="false", timeout="60")
        assert out.startswith("Error: the sorcar agent task could not run: "), out
        assert "reserve-parent has no remaining budget" in out, out
        assert "$1.0000 of its remainder is reserved" in out, out
        assert len(daemon.run_commands) == 1
        assert len(live_agent_jobs(parent)) == 1
    finally:
        _settle(parent, daemon)

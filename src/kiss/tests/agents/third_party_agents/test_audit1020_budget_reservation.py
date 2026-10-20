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

import pytest
import yaml
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    live_agent_jobs,
    make_run_agent_tool,
    make_run_parallel_tool,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon

DUMMY_SEA = str(
    Path(agent_dispatch.__file__).resolve().parents[1] / "seas" / "sorcar" / "sorcar_sea.py"
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


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)


@pytest.fixture()
def daemon() -> Iterator[HoldingDaemon]:
    stand_in = HoldingDaemon()
    saved = cron_agent._daemon_endpoint_file
    cron_agent._daemon_endpoint_file = str(stand_in.endpoint_file)
    try:
        yield stand_in
    finally:
        cron_agent._daemon_endpoint_file = saved
        stand_in.release()
        stand_in.close()


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
            run_agent(task, DUMMY_SEA, wait="false", timeout="60")
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
        run_agent("fourth", DUMMY_SEA, wait="false", timeout="60")
        assert _budgets(daemon)[-1] == pytest.approx(5.0)
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
        run_agent("first", DUMMY_SEA, wait="false", timeout="60")
        plan = agent_dispatch.plan_run_agent(
            str(tmp_path), "second", DUMMY_SEA, parent_agent=parent,
        )
        assert isinstance(plan, dict), plan
        assert [job.budget for job in plan["reserving"]] == pytest.approx([5.0])
    finally:
        _settle(parent, daemon)
    assert parent._live_budget_used() == pytest.approx(5.0)
    assert all(job.finished for job in plan["reserving"])
    daemon.gate.clear()
    job = agent_dispatch.start_agent_job(plan["name"], plan, parent)
    try:
        agent_dispatch.wait_until_started(job)
        assert not job.finished, job.result
        assert _budgets(daemon) == pytest.approx([5.0, 2.5])
    finally:
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
        run_agent("first", DUMMY_SEA, wait="false", timeout="60")
        # The call blocks for its timeout: short, so the two children are
        # still held when it returns their notices.
        out = run_parallel(json.dumps(["second", "third"]), DUMMY_SEA, timeout="2")
        entries = yaml.safe_load(out)
        assert len(entries) == 2 and all("still running" in e for e in entries), out
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
        run_agent("greedy", DUMMY_SEA, max_budget="1", wait="false", timeout="60")
        out = run_agent("starved", DUMMY_SEA, wait="false", timeout="60")
        assert out.startswith("Error: the sorcar agent task could not run: "), out
        assert "reserve-parent has no remaining budget" in out, out
        assert "$1.0000 of its remainder is reserved" in out, out
        assert len(daemon.run_commands) == 1
        assert len(live_agent_jobs(parent)) == 1
    finally:
        _settle(parent, daemon)

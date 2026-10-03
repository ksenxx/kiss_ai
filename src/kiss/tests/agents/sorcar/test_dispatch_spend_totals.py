# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A dispatched task's full spend must reach the task that dispatched it.

Found by the 2026-09-29 cost audit of ``~/.kiss/history.db``:

1. Post-result spend was dropped.  ``SorcarAgent.run`` folds the pre-run
   classifier's spend into the task's totals AFTER the agent emitted
   its ``result`` event and announces the new totals with a later
   ``usage_info`` (merge-agent and late side-channel spend arrive the
   same way).  ``daemon_client.run`` took the spend from the ``result``
   event alone, so every ``run_agent`` child folded about $0.02 less
   into its parent than its own row showed (task 96743718: three
   ``github_sea`` children, $0.058 missing).
2. A stopped caller dropped the child's whole spend.  When the calling
   task was stopped or interrupted by a server restart while blocked
   in a ``run_agent`` dispatch, the injected ``KeyboardInterrupt``
   skipped the fold entirely (task bc92cb57: a $56.47
   ``revise_and_review_paper`` child, parent row $0.52).

Two side-channel interleavings are covered as well: spend that a late
side channel banks directly on the waiting caller must not be folded a
second time from the child's totals, and spend banked on the child
after its last usage event must still be announced and folded.

The tests drive the real ``daemon_client.run`` and ``run_agent`` tool
against a local-WSS daemon stand-in that streams a scripted event
sequence, with a real ``SorcarAgent`` as the calling agent.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import cron_agent, daemon_client
from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.local_ws import fake_daemon


@pytest.fixture(autouse=True)
def _standalone_daemon_endpoint(monkeypatch: pytest.MonkeyPatch):
    """Keep a daemon endpoint recorded by another test from diverting dispatch."""
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    yield


def _usage(cost: str, tokens: int, steps: int) -> dict[str, Any]:
    """Return a ``usage_info`` event carrying the given task totals."""
    return {
        "type": "usage_info", "text": "", "cost": cost,
        "total_tokens": tokens, "total_steps": steps,
    }


def _result(cost: str, tokens: int, steps: int, success: bool) -> dict[str, Any]:
    """Return a terminal ``result`` event carrying the given task totals."""
    return {
        "type": "result", "taskId": "task-child-1", "success": success,
        "summary": "child done", "text": "child done", "cost": cost,
        "total_tokens": tokens, "step_count": steps,
    }


class _ScriptedDaemon:
    """A local-WSS daemon stand-in that streams *events* for the client's tab.

    Served by :func:`fake_daemon` on an event loop of its own thread, so
    the synchronous ``daemon_client.run`` under test can block the test
    thread.  After the ``run`` command it sends ``status running=true``,
    then every scripted event (stamped with the client's tab id), then
    ends as *end* says: ``"finish"`` sends ``status running=false``,
    ``"hold"`` keeps the connection open (recording every later client
    command), ``"drop"`` closes the connection.
    """

    def __init__(self, events: list[dict[str, Any]], end: str) -> None:
        """Start serving; returns once the endpoint file is written."""
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_spend_"))
        self.endpoint_file = self.tmp / "sorcar-local.json"
        self.events = events
        self.end = end
        self.commands: list[dict[str, Any]] = []
        self.sent_all = threading.Event()
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._closed = asyncio.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._serve(),), daemon=True,
        )
        self._thread.start()
        assert self._ready.wait(10), "the scripted daemon never came up"

    async def _serve(self) -> None:
        """Keep the fake daemon up until :meth:`close`."""
        async with fake_daemon(self.tmp, self._handle, endpoint_file=self.endpoint_file):
            self._ready.set()
            await self._closed.wait()

    async def _handle(self, ws: ServerConnection) -> None:
        """Serve one authenticated client connection."""
        tab_id = json.loads(await ws.recv())["tabId"]
        stream = [{"type": "status", "running": True}, *self.events]
        if self.end == "finish":
            stream.append({"type": "status", "running": False})
        for event in stream:
            await ws.send(json.dumps({**event, "tabId": tab_id}))
        self.sent_all.set()
        if self.end == "drop":
            return
        try:
            async for line in ws:
                self.commands.append(json.loads(line))
        except ConnectionClosed:
            pass

    def close(self) -> None:
        """Stop serving and close every connection."""
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(timeout=10)
        self._loop.close()


def test_post_result_usage_is_part_of_the_task_result() -> None:
    """Spend announced after the ``result`` counts; text/success come from the result."""
    daemon = _ScriptedDaemon([
        _usage("$0.2621", 52921, 3),
        _result("$0.2621", 52921, 3, success=True),
        _usage("$0.2827", 54512, 4),  # the classifier fold
    ], end="finish")
    try:
        result = daemon_client.run("child", endpoint_file=daemon.endpoint_file, timeout=30)
    finally:
        daemon.close()
    assert (result.cost, result.tokens, result.steps) == (0.2827, 54512, 4)
    assert (result.text, result.success) == ("child done", True)
    assert result.task_id == "task-child-1"


def test_run_agent_folds_post_result_spend_into_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calling agent is charged the child's final totals, not its result's."""
    daemon = _ScriptedDaemon([
        _result("$0.2872", 61859, 3, success=True),
        _usage("$0.3039", 63139, 4),
    ], end="finish")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    parent = SorcarAgent("spend-parent")
    try:
        out = make_run_agent_tool(str(tmp_path), parent_agent=parent)(
            "child", str(script), timeout="30",
        )
    finally:
        daemon.close()
    assert "child done" in out
    assert parent.budget_used == pytest.approx(0.3039)
    assert (parent.total_tokens_used, parent.total_steps) == (63139, 4)


def _interrupt_blocked_dispatch(daemon: _ScriptedDaemon, call: Any) -> BaseException:
    """Run *call* in a thread, interrupt it once the daemon sent everything.

    Returns:
        The exception *call* raised.
    """
    outcome: dict[str, BaseException] = {}

    def target() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 — captured for the asserts
            outcome["exc"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    assert daemon.sent_all.wait(5)
    time.sleep(0.2)  # let the client read the stream
    assert worker.ident is not None
    assert inject_keyboard_interrupt(worker.ident) == 1
    worker.join(timeout=15)
    assert not worker.is_alive(), "the dispatch wait never aborted"
    return outcome["exc"]


def test_interrupted_wait_reports_the_spend_seen_so_far() -> None:
    """An aborted wait's exception carries the child's latest totals."""
    daemon = _ScriptedDaemon([_usage("$1.2500", 9000, 6)], end="hold")
    try:
        exc = _interrupt_blocked_dispatch(
            daemon,
            lambda: daemon_client.run("child", endpoint_file=daemon.endpoint_file, timeout=60),
        )
    finally:
        daemon.close()
    assert isinstance(exc, KeyboardInterrupt)
    spent = exc.task_result  # type: ignore[attr-defined]
    assert (spent.cost, spent.tokens, spent.steps) == (1.25, 9000, 6)
    assert spent.success is False


def test_stopped_caller_is_still_charged_the_childs_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller stopped mid-dispatch keeps the stopped child's spend."""
    daemon = _ScriptedDaemon([_usage("$56.4700", 700000, 120)], end="hold")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    parent = SorcarAgent("stopped-parent")
    tool = make_run_agent_tool(str(tmp_path), parent_agent=parent)
    try:
        exc = _interrupt_blocked_dispatch(
            daemon, lambda: tool("child", str(script), timeout="60"),
        )
        assert isinstance(exc, KeyboardInterrupt)
        deadline = time.monotonic() + 5
        while not any(c.get("type") == "stop" for c in daemon.commands):
            assert time.monotonic() < deadline, "the child was never stopped"
            time.sleep(0.02)
    finally:
        daemon.close()
    assert parent.budget_used == pytest.approx(56.47)
    assert (parent.total_tokens_used, parent.total_steps) == (700000, 120)


def test_interrupt_before_any_spend_charges_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no spend reported yet, the stopped caller is charged nothing."""
    daemon = _ScriptedDaemon([], end="hold")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    parent = SorcarAgent("idle-parent")
    tool = make_run_agent_tool(str(tmp_path), parent_agent=parent)
    try:
        exc = _interrupt_blocked_dispatch(
            daemon, lambda: tool("child", str(script), timeout="60"),
        )
    finally:
        daemon.close()
    assert isinstance(exc, KeyboardInterrupt)
    assert exc.task_result.cost == 0.0  # type: ignore[attr-defined]
    assert (parent.budget_used, parent.total_tokens_used, parent.total_steps) == (0, 0, 0)


def test_dropped_connection_charges_the_spend_seen_so_far(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon that dies mid-dispatch still leaves the child's spend charged."""
    daemon = _ScriptedDaemon([_usage("$0.5000", 1000, 2)], end="drop")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    parent = SorcarAgent("dropped-parent")
    try:
        out = make_run_agent_tool(str(tmp_path), parent_agent=parent)(
            "child", str(script), timeout="60",
        )
    finally:
        daemon.close()
    assert "could not run" in out
    assert "closed the connection" in out
    assert parent.budget_used == pytest.approx(0.5)
    assert (parent.total_tokens_used, parent.total_steps) == (1000, 2)


def _usage_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the ``usage_info`` events among *events*, without ``tabId``."""
    return [
        {k: v for k, v in e.items() if k != "tabId"}
        for e in events if e.get("type") == "usage_info"
    ]


def _fold_into(
    parent: SorcarAgent,
    stream: list[dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch a child whose daemon streams *stream*; fold it into *parent*."""
    daemon = _ScriptedDaemon(stream, end="finish")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text("def model() -> str:\n    return 'm'\n")
    try:
        make_run_agent_tool(str(tmp_path), parent_agent=parent)(
            "child", str(script), timeout="30",
        )
    finally:
        daemon.close()


def test_late_side_channel_spend_banked_on_the_caller_is_not_folded_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spend banked on the waiting caller is subtracted, also from later totals.

    The child's row is saved ($1.00) and it is still tearing down when a
    side channel of it ends ($0.75): the spend is added to the child's
    row and banked on the running caller directly.  A merge agent then
    adds $0.50 to the child's row.  The caller must end at the child's
    row total, $2.25, not $3.00.
    """
    from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
    from kiss.server import agent_state
    from kiss.server.agent_state import AgentState
    from kiss.server.task_update import charge_side_channel_usage
    from kiss.tests.server.test_side_channel_spend import _row, _server, _task

    root = _task(finished=False, cost=0.0)
    child = _task(finished=True, parent=root, cost=1.0)
    server, events = _server()  # before registering: it resets the registry
    parent = WorktreeSorcarAgent("late-parent")
    state = AgentState(root, agent=parent, tab_id="tab-late-parent", server_owned=True)
    agent_state.register(state)
    try:
        charge_side_channel_usage(server.printer, SorcarAgent("c"), child, 0.75, 10, 1)
        assert parent.budget_used == pytest.approx(0.75)
        # First the caller's own new totals (for the caller's tabs), then
        # the child's row totals (for the child's tabs, which the
        # waiting dispatch reads).
        own, late = _usage_events(events)
        assert (own["cost"], "ancestor_charged" in own) == ("$0.7500", False)
        assert late["cost"] == "$1.7500"
        assert late["ancestor_charged"] == {"cost": 0.75, "tokens": 10, "steps": 1}
        _fold_into(parent, [
            _result("$1.0000", 100, 2, success=True),
            late,
            _usage("$2.2500", 160, 4),  # the merge agent's row total
        ], tmp_path, monkeypatch)
    finally:
        agent_state.unregister(root, state)
    assert _row(child) == (110, 1.75, 3)
    assert parent.budget_used == pytest.approx(2.25)
    assert (parent.total_tokens_used, parent.total_steps) == (160, 4)


def test_side_channel_spend_banked_on_a_finishing_child_reaches_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live bank on the child after its last event is announced and folded.

    The announced total covers the banked spend ($1.00), the in-flight
    session's ($10.00) and the side channel's ($0.75).
    """
    from kiss.core.kiss_agent import KISSAgent
    from kiss.server.task_update import charge_side_channel_usage
    from kiss.tests.server.test_side_channel_spend import _server, _task

    child = _task(finished=False, cost=0.0)
    child_agent = SorcarAgent("finishing-child")
    child_agent.budget_used, child_agent.total_tokens_used, child_agent.total_steps = 1.0, 100, 2
    executor = KISSAgent("finishing-child-session")
    executor.budget_used, executor.total_tokens_used, executor.step_count = 10.0, 1000, 5
    child_agent._current_executor = executor
    server, events = _server()
    charge_side_channel_usage(server.printer, child_agent, child, 0.75, 10, 1)
    live = _usage_events(events)
    assert live == [{
        "type": "usage_info", "text": "", "total_tokens": 1110,
        "cost": "$11.7500", "total_steps": 8,
    }]
    parent = SorcarAgent("caller")
    _fold_into(
        parent, [_result("$11.0000", 1100, 7, success=True), *live],
        tmp_path, monkeypatch,
    )
    assert parent.budget_used == pytest.approx(11.75)
    assert (parent.total_tokens_used, parent.total_steps) == (1110, 8)

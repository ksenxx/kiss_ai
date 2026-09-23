# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the task-info panel's task update
(:mod:`kiss.server.task_update` and the ``getTaskUpdate`` command).

* :class:`TaskUpdateRunner` is driven through ``poll`` with a scripted
  agent runner whose completion the test controls, so the lazy
  10-minute schedule, the refresh path, the single-run-in-flight rule,
  the failure path and pruning are all observed from the outside.
* :func:`run_task_update_sea` runs a real :class:`ChatSorcarAgent`
  child against the scripted local chat-completions server, proving
  the child lands in the parent's chat as its sub-agent and that its
  spend is charged to the parent.
* ``getTaskUpdate`` is exercised over a live WSS connection and over
  the UDS transport of a real :class:`RemoteAccessServer`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import ssl
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import pytest
from websockets.asyncio.client import connect

from kiss.agents.seas import task_update_sea
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _get_db,
    _rw_lock,
)
from kiss.agents.sorcar.sorcar_agent import _agent_usage
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import agent_state, task_update
from kiss.server.task_update import TaskUpdateRunner, run_task_update_sea
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.conftest import posix_only, requires_unix_sockets


class _ScriptedSea:
    """A stand-in for :func:`run_task_update_sea` the test releases by hand.

    Each call records ``(parent_agent, task_id)``, blocks until
    :meth:`release` (at most 10 s), then returns the queued outcome or
    raises it.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []
        self.outcomes: list[tuple[str, float] | Exception] = []
        self._gate = threading.Event()

    def __call__(self, parent_agent: Any, task_id: str) -> tuple[str, float]:
        self.calls.append((parent_agent, task_id))
        self._gate.wait(10)
        self._gate.clear()
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def release(self, outcome: tuple[str, float] | Exception) -> None:
        """Let the blocked call finish with *outcome*."""
        self.outcomes.append(outcome)
        self._gate.set()


def _wait_until(predicate: Any, timeout: float = 10.0) -> None:
    """Block until *predicate* is true (polling), or fail after *timeout*."""
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


class TestTaskUpdateRunner(unittest.TestCase):
    """The lazy schedule behind the panel's poll."""

    def test_first_poll_runs_once_and_fresh_reports_wait(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()

        first = runner.poll("task-a", parent)
        self.assertTrue(first.running)
        self.assertEqual(first.text, "")
        self.assertEqual(first.finished_at, 0.0)
        self.assertEqual(first.sig, "0.000:1")
        self.assertEqual(
            first.payload(),
            {
                "exists": True, "content": "", "error": "", "running": True,
                "cost": 0.0, "updatedAt": 0, "sig": "0.000:1",
            },
        )
        _wait_until(lambda: len(sea.calls) == 1)
        self.assertEqual(sea.calls[0], (parent, "task-a"))

        # A poll while the run is in flight — even a forced one — never
        # doubles the run.
        again = runner.poll("task-a", parent, force=True)
        self.assertTrue(again.running)
        self.assertEqual(len(sea.calls), 1)

        sea.release(("<h4>Report</h4>", 0.25))
        _wait_until(lambda: not runner.poll("task-a", parent).running)
        done = runner.poll("task-a", parent)
        self.assertEqual(done.text, "<h4>Report</h4>")
        self.assertEqual(done.cost, 0.25)
        self.assertEqual(done.error, "")
        self.assertGreater(done.finished_at, 0.0)
        self.assertEqual(done.sig, f"{done.finished_at:.3f}:0")
        self.assertEqual(done.payload()["updatedAt"], int(done.finished_at * 1000))
        # A fresh report is served from the cache: no new run.
        self.assertEqual(len(sea.calls), 1)

    def test_refresh_reruns_and_a_failure_keeps_the_previous_report(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        runner.poll("task-b", parent)
        sea.release(("first report", 0.1))
        _wait_until(lambda: not runner.poll("task-b", parent).running)

        forced = runner.poll("task-b", parent, force=True)
        self.assertTrue(forced.running)
        self.assertEqual(forced.text, "first report")
        _wait_until(lambda: len(sea.calls) == 2)
        sea.release(RuntimeError("model down"))
        _wait_until(lambda: not runner.poll("task-b", parent).running)
        failed = runner.poll("task-b", parent)
        self.assertEqual(failed.text, "first report")
        self.assertEqual(failed.error, "RuntimeError: model down")
        self.assertEqual(failed.cost, 0.0)
        self.assertFalse(failed.payload()["running"])
        self.assertTrue(failed.payload()["exists"])

    def test_interval_elapsed_reruns_and_idle_reports_are_pruned(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        saved = task_update.UPDATE_INTERVAL_S
        task_update.UPDATE_INTERVAL_S = 0.05
        self.addCleanup(setattr, task_update, "UPDATE_INTERVAL_S", saved)
        try:
            runner.poll("task-c", parent)
            sea.release(("r1", 0.0))
            _wait_until(lambda: not runner.poll("task-c", parent).running)
            time.sleep(0.06)
            due = runner.poll("task-c", parent)
            self.assertTrue(due.running)
            self.assertEqual(len(sea.calls), 2)
            sea.release(("r2", 0.0))
            _wait_until(lambda: runner.poll("task-c", parent).text == "r2")
        finally:
            task_update.UPDATE_INTERVAL_S = saved

        # A report nobody polled for an hour is dropped on the next poll
        # of any task; a later poll of the pruned task starts afresh.
        with runner._lock:
            runner._updates["task-c"].polled_at -= task_update._PRUNE_AFTER_S
        runner.poll("task-other", parent)
        with runner._lock:
            self.assertNotIn("task-c", runner._updates)
            self.assertIn("task-other", runner._updates)
        fresh = runner.poll("task-c", parent)
        self.assertEqual(fresh.text, "")
        self.assertTrue(fresh.running)
        sea.release(("other", 0.0))
        sea.release(("c-again", 0.0))
        _wait_until(lambda: len(sea.calls) == 4 and not any(
            runner.poll(t, parent).running for t in ("task-other", "task-c")
        ))


def _chat_rows(chat_id: str) -> list[dict[str, Any]]:
    """Return ``(id, task, parent_task_id, cost)`` of every row in *chat_id*."""
    with _rw_lock.read_lock():
        rows = _get_db().execute(
            "SELECT id, task, parent_task_id, cost FROM task_history "
            "WHERE chat_id = ? ORDER BY timestamp ASC, rowid ASC",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def test_run_task_update_sea_runs_as_a_subagent_in_the_parents_chat(tmp_path: Path) -> None:
    """The child reports on the parent, nests under it in the same chat, and bills it.

    The parent is built the way the task runner leaves it mid-run: a
    persisted task row in a chat, exposed through ``last_task_id``, and
    configured for the scripted local model (which the child inherits).
    """
    parent = ChatSorcarAgent("task-update-parent")
    parent.work_dir = str(tmp_path)
    task_id, chat_id = _add_task(
        "Parent task prompt", chat_id="",
        extra={"model": MODEL, "work_dir": str(tmp_path)},
    )
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = task_id
    budget_before = _agent_usage(parent)[0]

    report = "<h4>Done so far</h4><ul><li>read parser.py</li></ul>"
    script = [
        tool_call_body("task_transcript", {"task_id": task_id}, prompt_tokens=500),
        finish_body(report, prompt_tokens=800),
    ]
    with serve(script) as (url, requests):
        parent.model_name = MODEL
        parent.model_config = {"base_url": url, "api_key": "local"}
        text, cost = run_task_update_sea(parent, task_id)

    assert text == report
    assert cost > 0.0
    assert _agent_usage(parent)[0] == pytest.approx(budget_before + cost)

    rows = _chat_rows(chat_id)
    assert [r["task"] for r in rows] == [
        "Parent task prompt", task_update_sea.build_prompt(task_id),
    ]
    assert rows[1]["parent_task_id"] == task_id
    assert rows[1]["cost"] == pytest.approx(cost)
    # The parent is still running: its row keeps the totals its own
    # final save will write (from the live counters charged above).
    assert rows[0]["cost"] == 0.0

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, [list(r) for r in requests]
    names = {t["function"]["name"] for t in agentic[0]["tools"]}
    assert names == {"Bash", "finish", "task_transcript"}
    system = next(m for m in agentic[0]["messages"] if m["role"] == "system")
    assert str(system["content"]).startswith(task_update_sea.SYSTEM_PROMPT)
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert task_update_sea.build_prompt(task_id) in str(user["content"])
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1
    digest = str(tool_results[0]["content"])
    assert f"Task id: {task_id}" in digest
    assert "Task prompt: Parent task prompt" in digest
    assert "(no transcript entries yet)" in digest


def test_run_task_update_sea_adds_its_spend_to_a_finished_parents_row(
    tmp_path: Path,
) -> None:
    """A parent that finished while the update ran gets the spend on its row.

    The parent's final save has already written its totals (``endTs``
    set), so banking on the live agent alone would lose the update's
    cost; it is added to the ``task_history`` row instead.
    """
    parent = ChatSorcarAgent("task-update-finished-parent")
    parent.work_dir = str(tmp_path)
    task_id, chat_id = _add_task(
        "Finished parent", chat_id="",
        extra={
            "model": MODEL, "work_dir": str(tmp_path), "cost": 0.5,
            "tokens": 1000, "steps": 3, "endTs": int(time.time() * 1000),
        },
    )
    parent.resume_chat_by_id(chat_id)
    with parent._task_id_lock:
        parent._last_task_id = task_id
    budget_before = _agent_usage(parent)[0]
    script = [
        tool_call_body("task_transcript", {"task_id": task_id}, prompt_tokens=500),
        finish_body("<p>finished</p>", prompt_tokens=800),
    ]
    with serve(script) as (url, _requests):
        parent.model_name = MODEL
        parent.model_config = {"base_url": url, "api_key": "local"}
        text, cost = run_task_update_sea(parent, task_id)
    assert text == "<p>finished</p>"
    assert cost > 0.0
    rows = _chat_rows(chat_id)
    assert rows[0]["id"] == task_id
    assert rows[0]["cost"] == pytest.approx(0.5 + cost)
    assert rows[1]["parent_task_id"] == task_id
    # Not ALSO banked on the live counters: the row is the single account.
    assert _agent_usage(parent)[0] == pytest.approx(budget_before)


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestGetTaskUpdateOverWss(IsolatedAsyncioTestCase):
    """``getTaskUpdate`` over a live WSS connection."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": ""})
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        self.sea = _ScriptedSea()
        self._agents: dict[str, Any] = {}
        self.server._task_updates = TaskUpdateRunner(run_sea=self.sea)
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    def _register_task(
        self, tab_id: str, *, active: bool = True, state_key: str = "",
    ) -> str:
        """Register a running agent state for *tab_id*; return its task id.

        *state_key* is the registry key (the persisted task id once the
        run allocated its row; ``""`` uses the task id).
        """
        agent = WorktreeSorcarAgent(f"task-update-wss {tab_id}")
        task_id, chat_id = _add_task(f"prompt for {tab_id}")
        agent.resume_chat_by_id(chat_id)
        with agent._task_id_lock:
            agent._last_task_id = task_id
        state = agent_state.AgentState(
            state_key or task_id, agent=agent, tab_id=tab_id, server_owned=True,
            is_task_active=active,
        )
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, state_key or task_id, state)
        self._agents[task_id] = agent
        return task_id

    async def _poll(self, ws: Any, fields: dict[str, Any]) -> dict[str, Any]:
        """Send one ``getTaskUpdate`` and return its ``taskUpdate`` reply."""
        await ws.send(json.dumps({"type": "getTaskUpdate", **fields}))
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            ev: dict[str, Any] = json.loads(raw)
            if ev.get("type") == "taskUpdate":
                return ev
        raise AssertionError("no taskUpdate reply received")

    async def _connect(self) -> Any:
        ws = await connect(f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl())
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        await asyncio.wait_for(ws.recv(), timeout=5)
        return ws

    async def test_tab_without_a_running_task_replies_nothing(self) -> None:
        idle_task = self._register_task("t-idle", active=False)
        ws = await self._connect()
        try:
            unknown = await self._poll(ws, {"tabId": "t-unknown", "token": "1"})
            idle = await self._poll(ws, {"tabId": "t-idle", "token": "2"})
            no_tab = await self._poll(ws, {"token": "3"})
        finally:
            await ws.close()
        for reply in (unknown, idle, no_tab):
            self.assertIs(reply["exists"], False)
            self.assertEqual(reply["content"], "")
            self.assertEqual(reply["sig"], "")
            self.assertEqual(reply["taskId"], "")
            self.assertNotIn("unchanged", reply)
        self.assertEqual(unknown["tabId"], "t-unknown")
        self.assertEqual(unknown["token"], "1")
        self.assertEqual(idle["token"], "2")
        self.assertEqual(no_tab["tabId"], "")
        self.assertEqual(self.sea.calls, [])
        self.assertIsNotNone(idle_task)

    async def test_previous_task_id_before_row_allocation_replies_nothing(self) -> None:
        """A reused tab's agent still names the PREVIOUS task before its row exists.

        The task runner marks the state active before the run allocates
        its history row (which re-keys the state to the new id), so a
        state keyed otherwise than the agent's ``last_task_id`` is a run
        in that window: nothing is shown and no agent is started.
        """
        self._register_task("t-reused", state_key="pre-allocation-key")
        ws = await self._connect()
        try:
            reply = await self._poll(ws, {"tabId": "t-reused", "token": "4"})
        finally:
            await ws.close()
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["taskId"], "")
        self.assertEqual(reply["token"], "4")
        self.assertEqual(self.sea.calls, [])

    async def test_running_task_is_reported_polled_and_refreshed(self) -> None:
        task_id = self._register_task("t-run")
        ws = await self._connect()
        try:
            first = await self._poll(ws, {"tabId": "t-run", "token": "7"})
            self.assertIs(first["exists"], True)
            self.assertIs(first["running"], True)
            self.assertEqual(first["taskId"], task_id)
            self.assertEqual(first["content"], "")
            self.assertEqual(first["token"], "7")
            self.assertEqual(first["tabId"], "t-run")
            self.assertEqual(first["sig"], "0.000:1")
            await asyncio.to_thread(_wait_until, lambda: len(self.sea.calls) == 1)
            self.assertEqual(self.sea.calls[0][1], task_id)
            self.assertIs(self.sea.calls[0][0], self._agents[task_id])

            same = await self._poll(
                ws, {"tabId": "t-run", "knownSig": first["sig"], "token": "7"},
            )
            self.assertIs(same["unchanged"], True)
            self.assertNotIn("content", same)
            self.assertEqual(same["sig"], first["sig"])

            self.sea.release(("<h4>Progress</h4><p>read 3 files</p>", 0.07))
            await asyncio.to_thread(
                _wait_until,
                lambda: not self.server._task_updates.poll(
                    task_id, self._agents[task_id],
                ).running,
            )
            done = await self._poll(
                ws, {"tabId": "t-run", "knownSig": first["sig"], "token": "7"},
            )
            self.assertNotIn("unchanged", done)
            self.assertIs(done["running"], False)
            self.assertEqual(done["content"], "<h4>Progress</h4><p>read 3 files</p>")
            self.assertEqual(done["error"], "")
            self.assertEqual(done["cost"], 0.07)
            self.assertGreater(done["updatedAt"], 0)
            self.assertNotEqual(done["sig"], first["sig"])
            # Still fresh: no new run.
            self.assertEqual(len(self.sea.calls), 1)

            refreshed = await self._poll(
                ws, {"tabId": "t-run", "knownSig": done["sig"], "token": "7",
                     "refresh": True},
            )
            self.assertIs(refreshed["running"], True)
            self.assertEqual(refreshed["content"], done["content"])
            self.assertNotEqual(refreshed["sig"], done["sig"])
            await asyncio.to_thread(_wait_until, lambda: len(self.sea.calls) == 2)
            self.sea.release(("<h4>Progress</h4><p>read 5 files</p>", 0.02))
            await asyncio.to_thread(
                _wait_until,
                lambda: self.server._task_updates.poll(
                    task_id, self._agents[task_id],
                ).text.endswith("5 files</p>"),
            )
        finally:
            await ws.close()


@posix_only("UDS transport")
@requires_unix_sockets
class TestGetTaskUpdateOverUds(unittest.TestCase):
    """A UDS-delivered ``getTaskUpdate`` gets a direct ``taskUpdate`` reply.

    Editor-tab chat panels of the VS Code extension (UDS clients) poll
    ``getTaskUpdate`` to fill the secondary sidebar's Task Info view,
    so the command is served on both transports and the reply must
    come back on the requesting UDS connection.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.sock_path = os.path.join(self.tmp.name, "sorcar-test.sock")
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        self.server._printer._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(self.server._uds_handler, path=self.sock_path),
            self.loop,
        ).result(timeout=5)

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            self.uds_server.close()
            await self.uds_server.wait_closed()

        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)], timeout=5,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def test_uds_get_task_update_gets_direct_reply(self) -> None:
        """A chat editor panel's poll under an idle tab id is answered on its connection."""

        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(self.sock_path)
            try:
                writer.write(
                    json.dumps({
                        "type": "getTaskUpdate", "tabId": "editor-panel-tab",
                        "knownSig": "", "token": "9", "refresh": False,
                    }).encode() + b"\n"
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                event: dict[str, Any] = json.loads(line)
                return event
            finally:
                writer.close()
                await writer.wait_closed()

        event = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(timeout=15)
        self.assertEqual(event.get("type"), "taskUpdate")
        self.assertIs(event.get("exists"), False)
        self.assertEqual(event.get("content"), "")
        self.assertEqual(event.get("tabId"), "editor-panel-tab")
        self.assertEqual(event.get("token"), "9")


if __name__ == "__main__":
    unittest.main()

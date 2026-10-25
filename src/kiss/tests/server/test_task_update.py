# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the task-info panel's task update
(:mod:`kiss.server.task_update` and the ``getTaskUpdate`` command).

* :class:`TaskUpdateRunner` is driven through ``poll`` with a scripted
  agent runner whose completion the test controls, so the one-minute
  wait after the task's start, the lazy 10-minute schedule, the
  refresh path, the single-run-in-flight rule, the failure path and
  pruning are all observed from the outside.
* :func:`run_task_update_sea` runs a real :class:`ChatSorcarAgent`
  child against the scripted local chat-completions server, proving
  the child is the ``/ask`` agent (its system prompt, playbook and
  ``task_context`` tool), lands in the parent's chat as its sub-agent
  and has its spend charged to the parent.
* ``getTaskUpdate`` is exercised over a live WSS connection and over
  the local channel of a real :class:`RemoteAccessServer`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import pytest
from websockets.asyncio.client import connect

from kiss.agents.seas.ask import ask_sea
from kiss.agents.seas.task_update import task_update_sea
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _get_db,
    _rw_lock,
)
from kiss.agents.sorcar.sorcar_agent import _agent_usage
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server import agent_state, task_update
from kiss.server.task_update import (
    TaskUpdateRunner,
    mark_legacy_updates_as_side_channels,
    run_task_update_sea,
)
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.conftest import find_free_port, no_verify_ssl
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    StandInModelServer,
    finish_response,
    tool_call_response,
)


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


def _started_task(age_s: float, **extra: Any) -> str:
    """Persist a task row that started *age_s* seconds ago; return its id.

    The task runner allocates every run's row with its own ``startTs``;
    the runner's first-update delay counts from it.
    """
    task_id, _chat_id = _add_task(
        f"task started {age_s}s ago", chat_id="",
        extra={"startTs": int((time.time() - age_s) * 1000), **extra},
    )
    return task_id


def _wait_until(predicate: Any, timeout: float = 10.0) -> None:
    """Block until *predicate* is true (polling), or fail after *timeout*."""
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def _add_events(task_id: str, events: list[dict[str, Any]]) -> None:
    """Persist *events* on the task's row in sequence order."""
    from kiss.agents.sorcar import persistence

    for ev in events:
        persistence._append_chat_event(ev, task_id=task_id)


class TestTaskUpdateRunner(unittest.TestCase):
    """The digest-on-poll, agent-on-refresh behaviour behind the panel's poll.

    Not covered: ``poll`` marks an update ``running`` only after its
    worker thread has started, so a ``Thread.start()`` that raises
    (``RuntimeError: can't start new thread`` under thread exhaustion)
    cannot pin the update on "running" for good.  That branch needs the
    interpreter to fail creating a thread, which no end-to-end test can
    provoke without a test double.
    """

    def test_poll_renders_the_digest_and_never_runs_the_agent_by_itself(self) -> None:
        """An unforced poll shows the task's latest summary and the calls since; no run starts."""
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_id = _started_task(120.0)
        _add_events(task_id, [
            {"type": "tool_call", "name": "Read", "file_path": "/repo/a.py"},
            {"type": "tool_result", "tool_name": "Read", "content": "..."},
            {"type": "tool_call", "name": "summary",
             "description": "- **Read** `a.py`\n- planned the fix"},
            {"type": "tool_result", "tool_name": "summary", "content": "ok"},
            {"type": "tool_call", "name": "Bash", "command": "uv run pytest -q",
             "extras": {"description": "run the <tests>"}},
            {"type": "tool_result", "tool_name": "Bash", "content": "1 passed"},
        ])
        upd = runner.poll(task_id, parent)
        self.assertFalse(upd.running)
        self.assertEqual(upd.text, "")
        self.assertEqual(upd.finished_at, 0.0)
        digest = upd.digest
        self.assertIn("<h4>Latest summary (after 1 tool calls)</h4>", digest)
        self.assertIn("<ul><li><strong>Read</strong> <code>a.py</code></li>"
                      "<li>planned the fix</li></ul>", digest)
        self.assertIn("<h4>Since then (1 tool calls)</h4>", digest)
        self.assertIn("<li><code>Bash</code> run the &lt;tests&gt;</li>", digest)
        payload = upd.payload()
        self.assertIs(payload["exists"], True)
        self.assertEqual(payload["content"], digest)
        self.assertEqual(payload["updatedAt"], 0)
        self.assertNotIn("dueAt", payload)
        self.assertTrue(payload["sig"].startswith("0.000:0:"))
        time.sleep(0.05)
        self.assertFalse(runner.poll(task_id, parent).running)
        self.assertEqual(sea.calls, [])

        # A task without a row or without events renders an empty digest
        # and "no summary yet" respectively.
        self.assertEqual(runner.poll("task-without-a-row", parent).digest, "")
        bare = runner.poll(_started_task(1.0), parent).digest
        self.assertIn("<h4>No summary yet</h4>", bare)
        self.assertIn("<p>No tool calls since.</p>", bare)

    def test_digest_is_reread_after_the_refresh_interval(self) -> None:
        """New events reach the panel once DIGEST_REFRESH_S passed; the sig changes with them."""
        runner = TaskUpdateRunner(run_sea=_ScriptedSea())
        parent = object()
        task_id = _started_task(120.0)
        first = runner.poll(task_id, parent)
        _add_events(task_id, [{"type": "tool_call", "name": "Bash", "command": "ls"}])
        self.assertEqual(runner.poll(task_id, parent).sig, first.sig)
        with runner._lock:
            runner._updates[task_id].digest_at -= task_update.DIGEST_REFRESH_S
        later = runner.poll(task_id, parent)
        self.assertNotEqual(later.sig, first.sig)
        self.assertIn("<li><code>Bash</code> ls</li>", later.digest)

    def test_refresh_runs_once_and_the_answer_sits_above_the_digest(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_a = _started_task(120.0)

        first = runner.poll(task_a, parent, force=True)
        self.assertTrue(first.running)
        self.assertEqual(first.text, "")
        self.assertEqual(first.finished_at, 0.0)
        self.assertTrue(first.sig.startswith("0.000:1:"))
        self.assertEqual(first.payload()["content"], first.digest)
        _wait_until(lambda: len(sea.calls) == 1)
        self.assertEqual(sea.calls[0], (parent, task_a))

        # A poll while the run is in flight — even a forced one — never
        # doubles the run.
        again = runner.poll(task_a, parent, force=True)
        self.assertTrue(again.running)
        self.assertEqual(len(sea.calls), 1)

        sea.release(("<h4>Report</h4>", 0.25))
        _wait_until(lambda: not runner.poll(task_a, parent).running)
        done = runner.poll(task_a, parent)
        self.assertEqual(done.text, "<h4>Report</h4>")
        self.assertEqual(done.cost, 0.25)
        self.assertEqual(done.error, "")
        self.assertGreater(done.finished_at, 0.0)
        self.assertTrue(done.sig.startswith(f"{done.finished_at:.3f}:0:"))
        self.assertEqual(done.payload()["updatedAt"], int(done.finished_at * 1000))
        self.assertEqual(
            done.payload()["content"],
            '<div class="task-update-answer"><h4>Report</h4></div>\n' + done.digest,
        )
        # An unforced poll serves the cached answer: no new run.
        time.sleep(0.02)
        runner.poll(task_a, parent)
        self.assertEqual(len(sea.calls), 1)

    def test_refresh_reruns_and_a_failure_keeps_the_previous_report(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_b = _started_task(120.0)
        runner.poll(task_b, parent, force=True)
        sea.release(("first report", 0.1))
        _wait_until(lambda: not runner.poll(task_b, parent).running)

        forced = runner.poll(task_b, parent, force=True)
        self.assertTrue(forced.running)
        self.assertEqual(forced.text, "first report")
        _wait_until(lambda: len(sea.calls) == 2)
        sea.release(RuntimeError("model down"))
        _wait_until(lambda: not runner.poll(task_b, parent).running)
        failed = runner.poll(task_b, parent)
        self.assertEqual(failed.text, "first report")
        self.assertEqual(failed.error, "RuntimeError: model down")
        self.assertEqual(failed.cost, 0.0)
        self.assertFalse(failed.payload()["running"])
        self.assertTrue(failed.payload()["exists"])

    def test_idle_reports_are_pruned(self) -> None:
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_c = _started_task(120.0)
        task_other = _started_task(120.0)
        runner.poll(task_c, parent, force=True)
        sea.release(("r1", 0.0))
        _wait_until(lambda: not runner.poll(task_c, parent).running)

        # A report nobody polled for an hour is dropped on the next poll
        # of any task; a later poll of the pruned task starts afresh
        # (digest only, no run).
        with runner._lock:
            runner._updates[task_c].polled_at -= task_update._PRUNE_AFTER_S
        runner.poll(task_other, parent)
        with runner._lock:
            self.assertNotIn(task_c, runner._updates)
            self.assertIn(task_other, runner._updates)
        fresh = runner.poll(task_c, parent)
        self.assertEqual(fresh.text, "")
        self.assertFalse(fresh.running)
        self.assertEqual(len(sea.calls), 1)

    def test_concurrent_refreshes_create_one_update_and_one_run(self) -> None:
        """Simultaneous refreshes (several surfaces showing one tab) share one update."""
        sea = _ScriptedSea()
        runner = TaskUpdateRunner(run_sea=sea)
        parent = object()
        task_id = _started_task(120.0)
        results: list[task_update.TaskUpdate] = []
        go = threading.Barrier(8)

        def poll_once() -> None:
            go.wait(5)
            results.append(runner.poll(task_id, parent, force=True))

        threads = [threading.Thread(target=poll_once) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r.running for r in results))
        with runner._lock:
            self.assertEqual(list(runner._updates), [task_id])
        _wait_until(lambda: len(sea.calls) == 1)
        sea.release(("one", 0.0))
        _wait_until(lambda: runner.poll(task_id, parent).text == "one")
        self.assertEqual(len(sea.calls), 1)


def _chat_rows(chat_id: str) -> list[dict[str, Any]]:
    """Return ``(id, task, parent_task_id, cost, is_side_channel)`` of every row in *chat_id*."""
    with _rw_lock.read_lock():
        rows = _get_db().execute(
            "SELECT id, task, parent_task_id, cost, is_side_channel "
            "FROM task_history "
            "WHERE chat_id = ? ORDER BY timestamp ASC, rowid ASC",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def test_run_task_update_sea_runs_as_a_subagent_in_the_parents_chat(tmp_path: Path) -> None:
    """The child is the /ask agent asked about the parent; it nests under it and bills it.

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

    report = "<p>Read parser.py; no results yet.</p>"
    script = [
        tool_call_body("task_context", {"task_id": task_id}, prompt_tokens=500),
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
    assert len(rows) == 2 and rows[0]["task"] == "Parent task prompt"
    # The child's row holds the prompt the update run actually used: the
    # ``/ask`` question with the ask SEA's ``prompt(task)`` suffix and
    # ``{task_id}`` filled in.
    prompt = str(rows[1]["task"])
    assert prompt.startswith(task_update.UPDATE_QUESTION)
    assert prompt.endswith(ask_sea.ADD_TO_PROMPT.replace("{task_id}", task_id))
    assert "{task_id}" not in prompt
    assert rows[1]["parent_task_id"] == task_id
    assert rows[1]["cost"] == pytest.approx(cost)
    # A side channel: the daemon replays the finished child as
    # ``subagentDone`` rather than re-opening its tab on every reload.
    assert rows[1]["is_side_channel"] == 1
    assert rows[0]["is_side_channel"] == 0
    # The parent is still running: its row keeps the totals its own
    # final save will write (from the live counters charged above).
    assert rows[0]["cost"] == 0.0

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, [list(r) for r in requests]
    names = {t["function"]["name"] for t in agentic[0]["tools"]}
    assert names == {"finish", "task_context"}
    system = next(m for m in agentic[0]["messages"] if m["role"] == "system")
    # The ask SEA's ``system_prompt`` replaces the assembled prompt: the
    # default base prompt the run would otherwise start from is gone.
    assert str(system["content"]).startswith(ask_sea.AskSea().system_prompt("ASSEMBLED"))
    assert SYSTEM_PROMPT not in str(system["content"])
    assert SYSTEM_PROMPT_LITE not in str(system["content"])
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert prompt in str(user["content"])
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1
    digest = str(tool_results[0]["content"])
    assert f"Task id: {task_id}" in digest
    assert "Task prompt: Parent task prompt" in digest


def test_update_of_a_subagent_hangs_off_the_subagents_client_tab(
    tmp_path: Path,
) -> None:
    """A task update requested on a running SUB-agent announces itself
    under the tab every client shows that sub-agent in.

    The sub-agent's own ``_tab_id`` is the fan-out's synthetic
    ``task-…__sub_N`` id, which no client has: a ``new_tab`` carrying
    it is dropped by every client (``main.js`` ``case 'new_tab'``) and
    the update's tab never opens anywhere.  The clients show the
    sub-agent as ``{parent_tab_id}__sub_{task_id}``, and that is what
    the update's ``new_tab`` must carry — before any client has
    subscribed to the sub-agent's stream.
    """
    printer = CapturePrinter()
    grandparent_id, chat_id = _add_task(
        "Grandparent prompt", chat_id="",
        extra={"model": MODEL, "work_dir": str(tmp_path)},
    )
    subagent = ChatSorcarAgent("Parallel-child")
    subagent.work_dir = str(tmp_path)
    subagent.printer = printer
    subagent._tab_id = f"task-{grandparent_id}__sub_0"
    subagent._subagent_info = {
        "parent_task_id": grandparent_id,
        "parent_tab_id": "root-tab",
        "reviewer": False,
    }
    sub_task_id, _ = _add_task(
        "Sub-agent prompt", chat_id=chat_id,
        extra={
            "model": MODEL, "work_dir": str(tmp_path),
            "subagent": subagent._subagent_info,
        },
    )
    subagent.resume_chat_by_id(chat_id)
    with subagent._task_id_lock:
        subagent._last_task_id = sub_task_id

    # The child streams (its parent has a printer), so it needs the
    # streaming-capable stand-in endpoint rather than the scripted one.
    def respond(request: dict[str, Any]) -> dict[str, Any]:
        if any(m.get("role") == "tool" for m in request.get("messages", [])):
            return finish_response("<h4>Working</h4>")
        return tool_call_response("task_context", {"task_id": sub_task_id})

    standin = StandInModelServer(respond)
    try:
        subagent.model_name = STANDIN_MODEL
        subagent.model_config = standin.model_config
        text, _cost = run_task_update_sea(subagent, sub_task_id)
    finally:
        standin.stop()

    assert text == "<h4>Working</h4>"
    (new_tab,) = printer.events_of_type("new_tab")
    assert new_tab["parent_tab_id"] == f"root-tab__sub_{sub_task_id}"
    update_row = next(
        r for r in _chat_rows(chat_id) if r["parent_task_id"] == sub_task_id
    )
    assert update_row["is_side_channel"] == 1
    # Completion names the tab every surface opened on the ``new_tab``
    # (``{parent_tab_id}__sub_{task_id}``) and the update's synthetic tab.
    derived_tab, synthetic_tab = (
        e["tab_id"] for e in printer.events_of_type("subagentDone")
    )
    assert derived_tab == f"{new_tab['parent_tab_id']}__sub_{new_tab['task_id']}"
    assert synthetic_tab.startswith(f"task-{sub_task_id}__update-")


def test_mark_legacy_updates_as_side_channels_stamps_only_update_children() -> None:
    """Pre-flag task-update rows get stamped; other children and stamped rows are untouched."""
    parent_id, chat_id = _add_task("Parent task prompt", chat_id="")
    child: dict[str, Any] = {"subagent": {"parent_task_id": parent_id}}
    legacy_update, _ = _add_task(
        task_update_sea.build_prompt(parent_id), chat_id=chat_id, extra=child,
    )
    other_child, _ = _add_task("Review the diff", chat_id=chat_id, extra=child)
    already_stamped, _ = _add_task(
        task_update_sea.build_prompt(parent_id), chat_id=chat_id,
        extra={"subagent": {"parent_task_id": parent_id, "side_channel": True}},
    )
    # An update prompt naming ANOTHER task is not this parent's update.
    foreign_update, _ = _add_task(
        task_update_sea.build_prompt("0" * 32), chat_id=chat_id, extra=child,
    )

    assert mark_legacy_updates_as_side_channels() == 1
    flags = {r["id"]: r["is_side_channel"] for r in _chat_rows(chat_id)}
    assert flags == {
        parent_id: 0,
        legacy_update: 1,
        other_child: 0,
        already_stamped: 1,
        foreign_update: 0,
    }
    # Idempotent: a second daemon start stamps nothing.
    assert mark_legacy_updates_as_side_channels() == 0


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
        tool_call_body("task_context", {"task_id": task_id}, prompt_tokens=500),
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


class TestGetTaskUpdateOverWss(IsolatedAsyncioTestCase):
    """``getTaskUpdate`` over a live WSS connection."""

    async def asyncSetUp(self) -> None:
        self.port = find_free_port()
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
        age_s: float = 120.0,
    ) -> str:
        """Register a running agent state for *tab_id*; return its task id.

        *state_key* is the registry key (the persisted task id once the
        run allocated its row; ``""`` uses the task id).  The task's row
        started *age_s* seconds ago; two minutes by default, past the
        runner's first-update delay.
        """
        agent = ChatSorcarAgent(f"task-update-wss {tab_id}")
        task_id, chat_id = _add_task(
            f"prompt for {tab_id}",
            extra={"startTs": int((time.time() - age_s) * 1000)},
        )
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
        ws = await connect(f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl())
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

    async def test_young_task_replies_with_its_digest_and_no_run(self) -> None:
        """A task seconds old gets its digest at once and no agent run."""
        task_id = self._register_task("t-young", age_s=5.0)
        ws = await self._connect()
        try:
            reply = await self._poll(ws, {"tabId": "t-young", "token": "5"})
        finally:
            await ws.close()
        self.assertIs(reply["exists"], True)
        self.assertIs(reply["running"], False)
        self.assertIn("<h4>No summary yet</h4>", reply["content"])
        self.assertEqual(reply["taskId"], task_id)
        self.assertEqual(reply["updatedAt"], 0)
        self.assertNotIn("dueAt", reply)
        self.assertTrue(reply["sig"].startswith("0.000:0:"))
        self.assertEqual(self.sea.calls, [])

    async def test_running_task_is_reported_polled_and_refreshed(self) -> None:
        task_id = self._register_task("t-run")
        ws = await self._connect()
        try:
            first = await self._poll(ws, {"tabId": "t-run", "token": "7", "refresh": True})
            self.assertIs(first["exists"], True)
            self.assertIs(first["running"], True)
            self.assertEqual(first["taskId"], task_id)
            self.assertIn("<h4>No summary yet</h4>", first["content"])
            self.assertEqual(first["token"], "7")
            self.assertEqual(first["tabId"], "t-run")
            self.assertTrue(first["sig"].startswith("0.000:1:"))
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
            self.assertTrue(done["content"].startswith(
                '<div class="task-update-answer"><h4>Progress</h4><p>read 3 files</p></div>'
            ))
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


class TestGetTaskUpdateOverLocalChannel(unittest.TestCase):
    """A local-channel ``getTaskUpdate`` gets a direct ``taskUpdate`` reply.

    Chat pages on the local WSS connection poll ``getTaskUpdate`` to
    fill the Task Info view, so the command is served to local and remote clients alike and
    the reply must come back on the requesting connection.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=self.tmp.name,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
            local_endpoint_file=os.path.join(self.tmp.name, "sorcar-local.json"),
        )
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=15)

    def tearDown(self) -> None:
        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop)],
            timeout=10,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def test_local_get_task_update_gets_direct_reply(self) -> None:
        """A chat editor panel's poll under an idle tab id is answered on its connection."""

        async def _talk() -> dict[str, Any]:
            reader, writer = await open_local_connection(self.server)
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

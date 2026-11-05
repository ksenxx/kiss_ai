# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing BETWEEN subtasks is acknowledged and labelled stopped.

Concurrency audit 2026 (C1): ``_run_task_inner``'s outer
``except BaseException`` gated its cancellation handling on
``result_summary == "Agent Failed Abruptly"``.  When a stop-injected
``KeyboardInterrupt`` landed AFTER a subtask's ``agent.run`` returned
(in the between-subtask bookkeeping — draining follow-ups, persisting
the intermediate row, where SQLite's busy timeout alone allows a 30 s
wait) the summary already carried the finished subtask's SUCCESS text,
so the handler took the else branch: ``_cancel_outcome`` was never
called, therefore

* ``stop_acknowledged`` stayed False and the Stop watchdog re-injected
  a second ``KeyboardInterrupt`` into the end-of-run cleanup
  ("Cleanup interrupted") — the exact failure mode the flag exists to
  prevent;
* the stopped run was persisted with the success summary and a
  ``task_done`` end event, so ``task_failed`` stayed False and
  ``effective_auto_commit`` stayed True: a run the user stopped was
  auto-merged as a completed success.

The test drives the REAL ``_run_task`` worker, the REAL stop watchdog
and the REAL SQLite persistence.  A two-``<task>`` prompt puts the run
into the between-subtask window; a second SQLite connection holds the
write lock (``BEGIN IMMEDIATE``) so the intermediate
``_persist_subtask_row`` genuinely blocks there while the stop is
requested and the watchdog's first interrupt is injected (it fires the
moment the blocked write returns).  The only substitution is the
agent's LLM loop (``agent.run``), which is external to the code under
test (the technique of
``test_audit0902_fix_server_stop_single_injection.py``).
"""

from __future__ import annotations

import os
import queue
import sqlite3
import tempfile
import threading
import time
from typing import Any, cast
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState

_LOCK_RELEASED_AT = 2.5  # after the watchdog's 1 s injection, before its 5 s retry


def _make_remote_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.web_server import RemoteAccessServer

    tmp = tempfile.mkdtemp(prefix="kiss-conc2026-between-")
    return RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        local_endpoint_file=os.path.join(tmp, "sorcar-local.json"),
    )


def _persisted(task_id: str) -> tuple[dict[str, Any], list[str]]:
    _persistence._flush_chat_events()
    db = _persistence._get_db()
    row = db.execute(
        "SELECT result, tokens, cost, steps FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row is not None
    loaded = _persistence._load_chat_events_by_task_id(task_id)
    events = cast(list[dict[str, Any]], loaded["events"]) if loaded else []
    return dict(row), [str(e.get("type")) for e in events]


class TestStopBetweenSubtasks(TestCase):
    """The between-subtask window acknowledges the stop like any other."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.work_dir = tempfile.mkdtemp(prefix="kiss-conc2026-between-wd-")

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_stop_in_between_subtask_bookkeeping_is_acknowledged(self) -> None:
        remote = _make_remote_server()
        daemon = remote._agent_server
        tab_id = "between-subtasks-1"
        agent = ChatSorcarAgent("Sorcar")
        state = AgentState(
            f"pre-{tab_id}",
            agent=agent,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
        )
        state.user_answer_queue = queue.Queue()
        agent_state.register(state)
        captured: dict[str, str] = {}
        added = threading.Event()
        proceed = threading.Event()
        run_calls: list[str] = []

        def fake_run(**kwargs: Any) -> str:
            run_calls.append(kwargs.get("prompt_template", ""))
            if len(run_calls) > 1:
                # The second subtask must never start on a stopped run.
                return "summary: subtask two done\nsuccess: true"
            agent.total_tokens_used = 7
            agent.budget_used = 0.001
            agent.step_count = 1
            agent._chat_id = agent._chat_id or f"between-chat-{tab_id}"
            task_id, _ = _persistence._add_task(
                kwargs.get("prompt_template", ""),
                chat_id=agent._chat_id,
                extra={
                    "model": kwargs.get("model_name", ""),
                    "work_dir": kwargs.get("work_dir", ""),
                    "version": "test",
                    "is_parallel": False,
                    "is_worktree": False,
                },
            )
            agent._last_task_id = task_id
            captured["id"] = task_id
            added.set()
            # Return only once the test holds the SQLite write lock, so
            # the between-subtask ``_persist_subtask_row`` blocks on it.
            assert proceed.wait(timeout=30), "test never released subtask one"
            return "summary: subtask one done\nsuccess: true"

        agent.run = fake_run  # type: ignore[method-assign, assignment]
        worker = threading.Thread(
            target=daemon._run_task,
            args=(
                {
                    "type": "run",
                    "prompt": "<task>first part</task><task>second part</task>",
                    "tabId": tab_id,
                    "workDir": self.work_dir,
                    "isParallel": False,
                    "useWorktree": False,
                    "autoCommit": False,
                    "_state_key": state.task_id,
                },
            ),
            daemon=True,
        )
        state.task_thread = worker
        worker.start()
        self.assertTrue(added.wait(timeout=15), "subtask one never started")
        # Make sure the runner's own connection exists before the lock
        # is taken, so its writes wait on the busy handler instead of
        # failing to open the database.
        _persistence._get_db()
        locker = sqlite3.connect(str(_persistence._DB_PATH), timeout=30)
        locker.execute("BEGIN IMMEDIATE")
        try:
            with self.assertLogs("kiss.server.task_runner", level="DEBUG") as logs:
                proceed.set()
                # Give the worker time to finish subtask one and block
                # inside the intermediate persist on the held lock.
                time.sleep(0.4)
                t0 = time.monotonic()
                daemon._stop_task(tab_id)
                # Watchdog injects the first KeyboardInterrupt at
                # t0 + 1 s; it pends inside the blocked C-level SQLite
                # wait and fires between the subtasks once the lock is
                # released below.
                time.sleep(max(0.0, _LOCK_RELEASED_AT - (time.monotonic() - t0)))
                locker.execute("ROLLBACK")
                worker.join(timeout=30)
                self.assertFalse(worker.is_alive(), "worker never finished")
        finally:
            locker.close()

        messages = "\n".join(logs.output)
        self.assertNotIn(
            "Cleanup interrupted",
            messages,
            "BUG: the unacknowledged stop let the watchdog re-inject into the end-of-run cleanup",
        )
        self.assertIn("Task lifecycle complete", messages)
        self.assertEqual(
            len(run_calls),
            1,
            "the second subtask of a stopped run must never start",
        )
        row, types = _persisted(captured["id"])
        self.assertEqual(
            row["result"],
            "Task stopped by user",
            "BUG: a run stopped between subtasks was persisted with the "
            "finished subtask's success summary",
        )
        self.assertIn(
            "task_stopped",
            types,
            "BUG: the stopped run kept its task_done end event, so "
            "task_failed/auto-commit treated it as a success",
        )
        self.assertFalse(state.stop_acknowledged, "flag is per run and reset")
        self.assertIsNone(state.task_thread)

    def test_stop_after_intermediate_row_persisted_leaves_it_complete(self) -> None:
        """A Stop before the next subtask allocates its row keeps the finished one.

        Audit V6 (F1): after the intermediate ``_persist_subtask_row``
        the loop kept ``task_history_id`` on the finished row, so a
        stop landing in the next subtask's pre-``agent.run`` bookkeeping
        (its ``_zero_usage_counters``, before the per-subtask ``try``
        whose ``finally`` would have re-resolved the id) made the outer
        handler re-persist that completed row as ``Task stopped by
        user`` with the zeroed tokens, cost and steps.  The worker is
        parked in the agent's second ``reset_usage`` (polling, so the
        watchdog's asynchronous interrupt fires right there) while the
        real Stop is requested.
        """
        remote = _make_remote_server()
        daemon = remote._agent_server
        tab_id = "between-subtasks-2"
        agent = ChatSorcarAgent("Sorcar")
        state = AgentState(
            f"pre-{tab_id}",
            agent=agent,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
        )
        state.user_answer_queue = queue.Queue()
        agent_state.register(state)
        events: list[dict[str, Any]] = []
        events_lock = threading.Lock()
        orig_broadcast = daemon.printer.broadcast

        def capture(event: dict[str, Any]) -> None:
            with events_lock:
                events.append(dict(event))
            orig_broadcast(event)

        daemon.printer.broadcast = capture  # type: ignore[method-assign]
        captured: dict[str, str] = {}
        parked = threading.Event()
        release = threading.Event()
        run_calls: list[str] = []
        real_reset_usage = agent.reset_usage

        def fake_run(**kwargs: Any) -> str:
            run_calls.append(kwargs.get("prompt_template", ""))
            if len(run_calls) > 1:
                return "summary: subtask two done\nsuccess: true"
            agent.total_tokens_used = 7
            agent.budget_used = 0.001
            agent.step_count = 1
            agent._chat_id = agent._chat_id or f"between-chat-{tab_id}"
            task_id, _ = _persistence._add_task(
                kwargs.get("prompt_template", ""),
                chat_id=agent._chat_id,
                extra={
                    "model": kwargs.get("model_name", ""),
                    "work_dir": kwargs.get("work_dir", ""),
                    "version": "test",
                    "is_parallel": False,
                    "is_worktree": False,
                },
            )
            agent._last_task_id = task_id
            captured["id"] = task_id
            return "summary: subtask one done\nsuccess: true"

        def parking_reset_usage() -> None:
            real_reset_usage()
            if captured and not parked.is_set():
                # The next subtask's reset, after subtask one's row was
                # persisted: hold the worker here until the stop lands.
                parked.set()
                deadline = time.monotonic() + 30
                while not release.is_set() and time.monotonic() < deadline:
                    release.wait(0.05)

        agent.run = fake_run  # type: ignore[method-assign, assignment]
        agent.reset_usage = parking_reset_usage  # type: ignore[method-assign]
        worker = threading.Thread(
            target=daemon._run_task,
            args=(
                {
                    "type": "run",
                    "prompt": "<task>first part</task><task>second part</task>",
                    "tabId": tab_id,
                    "workDir": self.work_dir,
                    "isParallel": False,
                    "useWorktree": False,
                    "autoCommit": False,
                    "_state_key": state.task_id,
                },
            ),
            daemon=True,
        )
        state.task_thread = worker
        with self.assertLogs("kiss.server.task_runner", level="DEBUG") as logs:
            worker.start()
            self.assertTrue(parked.wait(timeout=15), "second subtask's reset never ran")
            before = _persisted(captured["id"])
            try:
                daemon._stop_task(tab_id)
                worker.join(timeout=30)
            finally:
                release.set()
            self.assertFalse(worker.is_alive(), "worker never finished")

        messages = "\n".join(logs.output)
        self.assertNotIn("Cleanup interrupted", messages)
        self.assertIn("Task lifecycle complete", messages)
        self.assertEqual(len(run_calls), 1, "the second subtask of a stopped run never starts")
        self.assertEqual(
            before,
            (
                {"result": "subtask one done", "tokens": 7, "cost": 0.001, "steps": 1},
                [
                    "task_done",
                ],
            ),
        )
        self.assertEqual(
            _persisted(captured["id"]),
            before,
            "BUG: the stop rewrote the completed, already persisted row",
        )
        with events_lock:
            terminal = [
                (e["type"], e.get("tabId"), e.get("taskId"), e.get("text"))
                for e in events
                if e["type"] in ("result", "task_stopped", "task_done")
            ]
        self.assertEqual(
            terminal,
            [
                ("result", tab_id, None, "Task stopped by user"),
                ("task_stopped", tab_id, None, None),
            ],
            "the launcher tab gets the stopped result and end event, unaddressed by row",
        )
        self.assertFalse(state.stop_acknowledged, "flag is per run and reset")
        self.assertIsNone(state.task_thread)


if __name__ == "__main__":
    import unittest

    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the Stop watchdog never re-interrupts an already-acknowledged stop.

``_force_stop_thread`` injects ``KeyboardInterrupt`` into the worker
one second after Stop and, when the thread is still alive five
seconds later, injects AGAIN.  "Still alive" does not mean "the first
interrupt was swallowed": the worker may be executing the legitimate
cancellation cleanup of the first one — persisting the stopped row
(SQLite ``busy_timeout`` allows a 30 s wait for a write lock),
presenting the worktree, broadcasting the end event.  The second
interrupt then lands inside that cleanup, the runner logs "Cleanup
interrupted" and the row's metrics, the end event and the
post-task refresh are skipped.

The fix records the acknowledgement on the :class:`AgentState`
(``stop_acknowledged``) at the moment the first interrupt is turned
into the stopped result; the watchdog's ownership predicate treats an
acknowledged stop as "nothing left to interrupt".

The test drives the REAL ``_run_task`` worker, the REAL stop watchdog,
the REAL SQLite persistence and a REAL second SQLite connection that
holds the write lock (``BEGIN IMMEDIATE``) from before the first
injection until well after the point of the buggy second one.  The
only substitution is the agent's LLM loop (``agent.run``), which is
external to the code under test and must block like an agent wedged in
a model call until interrupted.
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
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.task_runner import _state_owns_thread

# Watchdog schedule: first injection after 1 s, retry after 5 more.
_SECOND_INJECTION_AT = 6.0
_LOCK_HELD_FOR = 8.5


def _make_remote_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.web_server import RemoteAccessServer

    tmp = tempfile.mkdtemp(prefix="kiss-stop-once-")
    return RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        uds_path=os.path.join(tmp, "sorcar.sock"),
    )


def _start_blocked_worker(
    vscode: Any, tab_id: str, work_dir: str,
) -> tuple[threading.Thread, AgentState, str]:
    """Start a real ``_run_task`` worker blocked inside ``agent.run``."""
    agent = WorktreeSorcarAgent("Sorcar VS Code")
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
    entered = threading.Event()

    def blocked_run(**kwargs: Any) -> None:
        agent.total_tokens_used = 123
        agent.budget_used = 0.01
        agent.step_count = 3
        agent._chat_id = agent._chat_id or f"stop-once-chat-{tab_id}"
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
        entered.set()
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:  # non-cooperative: ignores stop_event
            time.sleep(0.05)

    agent.run = blocked_run  # type: ignore[method-assign, assignment]
    worker = threading.Thread(
        target=vscode._run_task,
        args=({
            "type": "run",
            "prompt": f"stop-once-{tab_id}",
            "tabId": tab_id,
            "workDir": work_dir,
            "useParallel": False,
            "useWorktree": False,
            "autoCommit": False,
            "_state_key": state.task_id,
        },),
        daemon=True,
    )
    state.task_thread = worker
    worker.start()
    assert entered.wait(timeout=10), "worker never entered agent.run"
    for _ in range(200):
        if state.is_task_active:
            break
        time.sleep(0.02)
    assert state.is_task_active
    return worker, state, captured["id"]


def _persisted(task_id: str) -> tuple[dict[str, Any], list[str]]:
    _persistence._flush_chat_events()
    db = _persistence._get_db()
    row = db.execute(
        "SELECT result, tokens, steps FROM task_history WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row is not None
    loaded = _persistence._load_chat_events_by_task_id(task_id)
    events = cast(list[dict[str, Any]], loaded["events"]) if loaded else []
    return dict(row), [str(e.get("type")) for e in events]


class TestStopInjectsOnce(TestCase):
    """One acknowledged interrupt; cleanup blocked past 5 s completes intact."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.work_dir = tempfile.mkdtemp(prefix="kiss-stop-once-wd-")

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_slow_cleanup_is_not_reinterrupted(self) -> None:
        remote = _make_remote_server()
        vscode = remote._vscode_server
        worker, state, task_id = _start_blocked_worker(
            vscode, "stop-once-1", self.work_dir,
        )
        # Make sure the runner's connection exists before the lock is
        # taken, so the row update below waits on the busy handler
        # instead of failing to open the database.
        _persistence._get_db()

        locker = sqlite3.connect(str(_persistence._DB_PATH), timeout=30)
        locker.execute("BEGIN IMMEDIATE")
        try:
            with self.assertLogs("kiss.server.task_runner", level="DEBUG") as logs:
                t0 = time.monotonic()
                vscode._stop_task("stop-once-1")
                # Just before the watchdog's retry moment the worker must
                # be in its (lock-blocked) cleanup: the first interrupt
                # was handled and the stop acknowledged.
                time.sleep(_SECOND_INJECTION_AT - 0.5 - (time.monotonic() - t0))
                self.assertTrue(worker.is_alive(), "cleanup finished too early")
                self.assertTrue(
                    state.stop_acknowledged,
                    "the runner must record that the stop was acknowledged",
                )
                with agent_state.STATE_LOCK:
                    watchdog_would_inject = _state_owns_thread(state, worker)
                time.sleep(max(0.0, _LOCK_HELD_FOR - (time.monotonic() - t0)))
                locker.execute("ROLLBACK")
                worker.join(timeout=30)
                self.assertFalse(worker.is_alive(), "worker never finished")
        finally:
            locker.close()

        messages = "\n".join(logs.output)
        self.assertNotIn(
            "Cleanup interrupted", messages,
            "BUG: the watchdog re-injected KeyboardInterrupt into the "
            "cleanup of an already-acknowledged stop",
        )
        self.assertIn("Task lifecycle complete", messages)
        row, types = _persisted(task_id)
        self.assertEqual(row["result"], "Task stopped by user")
        self.assertEqual((row["tokens"], row["steps"]), (123, 3))
        self.assertIn("task_stopped", types)
        self.assertFalse(
            watchdog_would_inject,
            "an acknowledged stop leaves nothing to interrupt",
        )
        self.assertFalse(state.stop_acknowledged, "flag is per run and reset")

    def test_unacknowledged_stop_is_retried(self) -> None:
        """An interrupt that never reaches the runner IS retried.

        A worker that swallows the first ``KeyboardInterrupt`` (here:
        a bare ``except BaseException`` inside the model loop) has
        not acknowledged anything, so the watchdog's second injection
        must still happen and the run must end as stopped.
        """
        remote = _make_remote_server()
        vscode = remote._vscode_server
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        state = AgentState(
            "pre-stop-swallow", agent=agent, tab_id="stop-swallow",
            server_owned=True, stop_event=threading.Event(),
        )
        state.user_answer_queue = queue.Queue()
        agent_state.register(state)
        entered = threading.Event()
        swallowed = threading.Event()

        def swallowing_run(**kwargs: Any) -> None:
            agent._chat_id = agent._chat_id or "stop-swallow-chat"
            task_id, _ = _persistence._add_task(
                kwargs.get("prompt_template", ""), chat_id=agent._chat_id,
                extra={"model": "", "work_dir": self.work_dir, "version": "test",
                       "is_parallel": False, "is_worktree": False},
            )
            agent._last_task_id = task_id
            entered.set()
            try:
                while True:
                    time.sleep(0.05)
            except BaseException:
                swallowed.set()
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                time.sleep(0.05)

        agent.run = swallowing_run  # type: ignore[method-assign, assignment]
        worker = threading.Thread(
            target=vscode._run_task,
            args=({
                "type": "run", "prompt": "stop-swallow", "tabId": "stop-swallow",
                "workDir": self.work_dir, "useParallel": False,
                "useWorktree": False, "autoCommit": False,
                "_state_key": state.task_id,
            },),
            daemon=True,
        )
        state.task_thread = worker
        worker.start()
        self.assertTrue(entered.wait(timeout=10))
        vscode._stop_task("stop-swallow")
        self.assertTrue(swallowed.wait(timeout=10), "first injection not delivered")
        self.assertFalse(state.stop_acknowledged)
        with agent_state.STATE_LOCK:
            self.assertTrue(_state_owns_thread(state, worker))
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "second injection never came")
        self.assertIsNotNone(agent._last_task_id)
        row, types = _persisted(str(agent._last_task_id))
        self.assertEqual(row["result"], "Task stopped by user")
        self.assertIn("task_stopped", types)

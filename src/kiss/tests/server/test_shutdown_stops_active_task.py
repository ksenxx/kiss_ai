# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reproduction + fix: shutdown must not silently kill an in-flight task.

Production failure (task 2968 in the bundled ``sorcar.db``): a task was
running in its worker thread (step 9 of 100) when ``kiss-web`` received
a routine restart ``SIGTERM``.  The signal handler raised
``KeyboardInterrupt`` in the *main* thread, ``asyncio.run`` unwound, and
:meth:`RemoteAccessServer.start` returned — terminating the process.

The agent ran in a **daemon worker thread**, which the interpreter kills
abruptly on process exit.  That thread therefore never ran the cleanup
``finally`` in :meth:`VSCodeServer._run_task` that persists a real
``task_history.result`` and broadcasts the outcome.  The row was left at
the ``"Agent Failed Abruptly"`` sentinel and the next startup's orphan
sweep rewrote it to ``"Task terminated unexpectedly (process killed)"`` —
a *silent* failure: the user saw the task simply stop with no error.

The log proof::

    Signal SIGTERM received: pid=35487 active_tasks=[df7d8ef8...(task=None)]
    Server shutting down: pid=35487 (KeyboardInterrupt)
    Server stopped: pid=35487
    Orphaned task recovered: id=2968 ... last_events=[seq=44 ...]

The fix adds :meth:`RemoteAccessServer._stop_active_agent_tasks`, invoked
from ``start()``'s shutdown ``finally``.  It mirrors the user-facing
"stop" button — set the cooperative stop event, then inject a
``KeyboardInterrupt`` into the worker thread — but **joins** each worker
so its cleanup ``finally`` runs to completion before the process exits.

This test stands up a real worker thread running :meth:`_run_task` with a
stub agent that blocks in an uninterruptible-style ``sleep`` loop (it does
NOT poll the stop event, mimicking an agent wedged in an LLM API call),
then verifies that ``_stop_active_agent_tasks`` (a) joins the worker and
(b) leaves a meaningful result — never the sentinel.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState


def _make_remote_server() -> Any:
    """Build a :class:`RemoteAccessServer` with no tunnel / sockets bound.

    Constructing the remote server also constructs its owned
    :class:`VSCodeServer`, whose ``__init__`` *clears* the process-global
    ``kiss.server.agent_state`` registry.  Tests must therefore build
    the remote server FIRST and then run their task through
    ``remote._vscode_server`` so the worker registers into the same
    registry the shutdown helper later scans — exactly as in production
    where a single server owns both.
    """
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.web_server import RemoteAccessServer

    tmp = tempfile.mkdtemp(prefix="kiss-shutdown-test-")
    return RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        uds_path=os.path.join(tmp, "sorcar.sock"),
    )


class TestShutdownStopsActiveTask(TestCase):
    """The shutdown path must stop in-flight workers, not abandon them."""

    def test_shutdown_stops_blocked_worker_without_sentinel(self) -> None:
        """An agent blocked in a non-cooperative ``sleep`` (standing in for
        a wedged LLM call) must be stopped and joined by
        ``_stop_active_agent_tasks``, leaving a meaningful result rather
        than the ``"Agent Failed Abruptly"`` sentinel.
        """
        tab_id = "shutdown-active-1"
        remote = _make_remote_server()
        vscode = remote._vscode_server
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
        self.addCleanup(agent_state.agent_states.clear)

        captured: dict[str, str] = {}
        entered = threading.Event()

        def fake_run(**kwargs: Any) -> None:
            agent.total_tokens_used = 123
            agent.budget_used = 0.01
            agent.step_count = 9
            agent._chat_id = agent._chat_id or "shutdown-chat-id"
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
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                time.sleep(0.05)

        agent.run = fake_run  # type: ignore[assignment]

        worker = threading.Thread(
            target=vscode._run_task,
            args=({
                "type": "run",
                "prompt": "shutdown-while-running",
                "tabId": tab_id,
                "workDir": "/tmp",
                "useParallel": False,
                "useWorktree": False,
                "autoCommit": False,
            },),
            daemon=True,
        )
        state.task_thread = worker
        worker.start()

        assert entered.wait(timeout=10), "worker never entered agent.run"
        for _ in range(100):
            if state.is_task_active:
                break
            time.sleep(0.02)
        assert state.is_task_active, (
            "is_task_active was never set on the state"
        )

        remote._stop_active_agent_tasks(timeout=12.0)

        worker.join(timeout=5)
        assert not worker.is_alive(), (
            "regression: shutdown left the agent worker thread running — "
            "the in-flight task would be killed abruptly on process exit"
        )

        task_id = captured["id"]
        _persistence._flush_chat_events()
        db = _persistence._get_db()
        row = db.execute(
            "SELECT result FROM task_history WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row is not None
        assert row["result"] != "Agent Failed Abruptly", (
            "regression: shutdown left the task at the sentinel result; "
            f"got {row['result']!r}"
        )
        assert row["result"] == "Task interrupted by server restart/shutdown", (
            "shutdown should persist the restart/shutdown-specific result, "
            "not the user-stop label; "
            f"got {row['result']!r}"
        )

    def test_stop_active_agent_tasks_is_noop_without_active_tasks(self) -> None:
        """With no active worker the shutdown helper must be a safe no-op."""
        with agent_state.STATE_LOCK:
            for st in agent_state.agent_states.values():
                thread = st.task_thread
                if thread is not None and thread.is_alive():
                    self.skipTest("another test left an active worker running")
        remote = _make_remote_server()
        remote._stop_active_agent_tasks(timeout=1.0)

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reproduction + fix: shutdown must not silently kill an in-flight task.

Production failure (task 2968 in the bundled ``history.db``): a task was
running in its worker thread (step 9 of 100) when ``kiss-web`` received
a routine restart ``SIGTERM``.  The signal handler raised
``KeyboardInterrupt`` in the *main* thread, ``asyncio.run`` unwound, and
:meth:`RemoteAccessServer.start` returned — terminating the process.

The agent ran in a **daemon worker thread**, which the interpreter kills
abruptly on process exit.  That thread therefore never ran the cleanup
``finally`` in :meth:`AgentServer._run_task` that persists a real
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
import signal
import tempfile
import threading
import time
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState


def _make_remote_server() -> Any:
    """Build a :class:`RemoteAccessServer` with no tunnel / sockets bound.

    Constructing the remote server also constructs its owned
    :class:`AgentServer`, whose ``__init__`` *clears* the process-global
    ``kiss.server.agent_state`` registry.  Tests must therefore build
    the remote server FIRST and then run their task through
    ``remote._agent_server`` so the worker registers into the same
    registry the shutdown helper later scans — exactly as in production
    where a single server owns both.
    """
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.web_server import RemoteAccessServer

    tmp = tempfile.mkdtemp(prefix="kiss-shutdown-test-")
    return RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        local_endpoint_file=os.path.join(tmp, "sorcar-local.json"),
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
        daemon = remote._agent_server
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
            target=daemon._run_task,
            args=({
                "type": "run",
                "prompt": "shutdown-while-running",
                "tabId": tab_id,
                "workDir": "/tmp",
                "isParallel": False,
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

    def test_signal_handler_does_not_block_on_the_registry_lock(self) -> None:
        """The SIGTERM handler must return while a worker holds ``STATE_LOCK``.

        Lock inversion: the handler runs on the event-loop thread, which
        can be inside a ``WebPrinter._ws_lock`` section when the signal
        lands; a task worker that holds ``agent_state.STATE_LOCK`` while
        broadcasting a status (``AgentServer._broadcast_replay_status``)
        blocks on that same ``_ws_lock``.  A handler that then takes
        ``STATE_LOCK`` (the registry snapshot) waits for the worker, the
        worker waits for the handler's thread, and the daemon never
        shuts down.  The handler must not block on the registry lock.
        """
        remote = _make_remote_server()
        printer = remote._printer
        worker_holds_registry = threading.Event()
        release_worker = threading.Event()
        handler_returned = threading.Event()

        def worker() -> None:
            with agent_state.STATE_LOCK:
                worker_holds_registry.set()
                release_worker.wait(timeout=30)
                printer.broadcast({"type": "status", "running": False, "tabId": "t1"})

        def interrupted_loop_thread() -> None:
            with printer._ws_lock:
                assert worker_holds_registry.wait(timeout=10)
                try:
                    remote._handle_shutdown_signal(int(signal.SIGTERM))
                except KeyboardInterrupt:
                    pass  # No loop is running: the documented fallback.
                handler_returned.set()

        worker_thread = threading.Thread(target=worker, daemon=True)
        loop_thread = threading.Thread(target=interrupted_loop_thread, daemon=True)
        self.addCleanup(worker_thread.join, 5)
        self.addCleanup(loop_thread.join, 5)
        worker_thread.start()
        loop_thread.start()
        assert worker_holds_registry.wait(timeout=10)
        release_worker.set()  # The worker now blocks on ``_ws_lock``.
        assert handler_returned.wait(timeout=5), (
            "regression: the signal handler blocked on STATE_LOCK while the "
            "worker holding it waited for the handler thread's _ws_lock"
        )
        assert remote._shutdown_initiated

    def test_workers_are_signalled_before_the_shutdown_persistence(self) -> None:
        """The sweep sets the stop events before its pre-emptive persistence.

        That persistence takes the history write lock and a ``BEGIN
        IMMEDIATE`` transaction, so behind another writer it waits out
        the in-process lock and SQLite's 30 s busy timeout; a cooperative
        worker must already be stopping during that wait, not running on
        unsignalled until the write goes through.
        """
        remote = _make_remote_server()
        stop_event = threading.Event()
        state = AgentState(
            "signalled-before-persist",
            agent=ChatSorcarAgent("Sorcar"),
            tab_id="tab-signalled-before-persist",
            server_owned=True,
            stop_event=stop_event,
        )
        worker = threading.Thread(target=stop_event.wait, args=(30,), daemon=True)
        state.task_thread = worker
        agent_state.register(state)
        self.addCleanup(agent_state.agent_states.clear)
        worker.start()
        sweep = threading.Thread(
            target=remote._stop_active_agent_tasks, kwargs={"timeout": 2.0}, daemon=True,
        )
        with _persistence._rw_lock.write_lock():  # another history-DB writer
            sweep.start()
            assert stop_event.wait(timeout=5), (
                "regression: the shutdown sweep waited for the history DB "
                "before signalling the in-flight worker to stop"
            )
        sweep.join(timeout=30)
        worker.join(timeout=5)
        assert not sweep.is_alive() and not worker.is_alive()

    def test_stop_active_agent_tasks_is_noop_without_active_tasks(self) -> None:
        """With no active worker the shutdown helper must be a safe no-op."""
        with agent_state.STATE_LOCK:
            for st in agent_state.agent_states.values():
                thread = st.task_thread
                if thread is not None and thread.is_alive():
                    self.skipTest("another test left an active worker running")
        remote = _make_remote_server()
        remote._stop_active_agent_tasks(timeout=1.0)

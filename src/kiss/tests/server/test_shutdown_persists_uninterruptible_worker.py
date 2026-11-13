# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Intermittent "agent was killed" — root cause + safety-net fix.

User-visible symptom (intermittent, hard to reproduce in dev):

    Running an agent on a task occasionally ends with the task row
    re-written to ``"Task terminated unexpectedly (process killed)"``
    on the next ``kiss-web`` startup, even though no SIGKILL / OOM
    actually fired.  The user perceives the agent as "killed" mid-run.

Root cause traced through the codebase:

1. :meth:`ChatSorcarAgent.run` inserts the ``task_history`` row via
   :func:`_persistence._add_task` with the sentinel
   ``result = "Agent Failed Abruptly"`` BEFORE the agent's main work
   loop starts.
2. :meth:`_TaskRunnerMixin._run_task_inner`'s cleanup ``finally``
   normally overwrites that sentinel with a meaningful result via
   :func:`_save_task_result`.
3. When the daemon is shut down (SIGTERM / install restart / IP
   change in :meth:`RemoteAccessServer._watchdog`),
   :meth:`RemoteAccessServer._stop_active_agent_tasks` signals each
   worker thread to stop and joins it with a bounded *timeout*.
4. If the worker is wedged inside an LLM API call (a blocking C-level
   socket read that does not honour ``KeyboardInterrupt``), the join
   times out, the daemon worker is killed at process exit, and the
   cleanup ``finally`` never runs.  The sentinel survives in the DB.
5. On the next process startup,
   :func:`_persistence._recover_orphaned_tasks` (invoked from
   :meth:`AgentServer.__init__`) rewrites every surviving sentinel
   row to ``"Task terminated unexpectedly (process killed)"`` —
   silently, several seconds after the fact, after the user has
   already reopened the browser.

The fix is a pre-emptive persistence safety net:
:func:`_persistence._shutdown_persist_in_flight_results`, invoked from
:meth:`RemoteAccessServer._stop_active_agent_tasks` beside the joins
(on a thread of its own, so a stuck history writer cannot hold the
shutdown).  It rewrites the sentinel row to
``"Task interrupted by server restart/shutdown"`` so that even if the
worker is wedged and the daemon thread is killed at process exit, the
DB already carries the truthful, non-alarming result.

This module spins up a real worker thread that *ignores*
``KeyboardInterrupt`` and drives the shutdown through a real
:class:`RemoteAccessServer`.  The helper's scoping rules (only the
supplied ids, never a completed row, empty set is a no-op), which
depend only on ``kiss.core`` and ``kiss.agents.sorcar``, live in
``kiss.tests.agents.sorcar.test_shutdown_persists_uninterruptible_worker``
together with the row helpers imported below.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence
from kiss.server import agent_state
from kiss.tests.agents.sorcar.test_shutdown_persists_uninterruptible_worker import (
    _insert_sentinel_row,
    _row_result,
)


def _make_remote_server() -> Any:
    """Build a :class:`RemoteAccessServer` with no tunnel / sockets bound.

    Constructing the remote server also constructs its owned
    :class:`AgentServer`, whose ``__init__`` sweeps the process-global
    ``agent_state.agent_states`` registry.  Tests must therefore build
    the remote server FIRST and then mutate the registry — so the
    shutdown helper scans the same registry the test populated.
    """
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.web_server import RemoteAccessServer

    tmp = tempfile.mkdtemp(prefix="kiss-shutdown-uninterruptible-")
    return RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        local_endpoint_file=os.path.join(tmp, "sorcar-local.json"),
    )


def _swallow_interrupts_until(cleanup: threading.Event) -> None:
    """Worker body: ignore ``KeyboardInterrupt`` until *cleanup*, like an
    LLM call wedged in a C-level socket read."""
    while not cleanup.is_set():
        try:
            time.sleep(0.02)
        except KeyboardInterrupt:
            pass


@contextmanager
def _wedged_task() -> Iterator[tuple[Any, str]]:
    """A real server and one registered task whose worker cannot be interrupted.

    Yields the server and the id of the task's history row, still at
    the creation sentinel; checks on exit that the sweep flagged the task.
    """
    remote = _make_remote_server()
    chat_id = "shutdown-uninterruptible-chat-1"
    task_id = _insert_sentinel_row(chat_id)
    assert _row_result(task_id) == "Agent Failed Abruptly", (
        "precondition: row must start with the sentinel"
    )
    cleanup = threading.Event()
    state = agent_state.AgentState(
        str(task_id),
        chat_id=chat_id,
        tab_id="shutdown-uninterruptible-1",
        server_owned=True,
        stop_event=threading.Event(),
        is_task_active=True,
    )
    state.user_answer_queue = queue.Queue()
    worker = threading.Thread(target=_swallow_interrupts_until, args=(cleanup,), daemon=True)
    state.task_thread = worker
    agent_state.register(state)
    worker.start()
    try:
        yield remote, task_id
        assert state.interrupted_by_shutdown is True, (
            "shutdown helper must set interrupted_by_shutdown on in-flight states"
        )
    finally:
        cleanup.set()
        worker.join(timeout=2)
        agent_state.unregister(str(task_id), state)


def _hold_history_write_lock(held: threading.Event, release: threading.Event) -> None:
    """Writer body: keep the history write lock until *release* (5 s at most)."""
    with persistence._rw_lock.write_lock():
        held.set()
        release.wait(5)


class TestShutdownPersistsUninterruptibleWorker(TestCase):
    """Pre-emptive persistence must guard against truly-wedged workers."""

    def test_uninterruptible_worker_row_is_rewritten_before_timeout(
        self,
    ) -> None:
        """Without the pre-emptive save, ``_stop_active_agent_tasks``
        would return after its join timeout with the row still at the
        sentinel — so the next startup's orphan sweep rewrites it to
        "process killed" and the user perceives the agent as killed.
        With the safety net, the row carries the truthful "Task
        interrupted by server restart/shutdown" by the time the helper
        returns, regardless of whether the worker ever unwinds.
        """
        with _wedged_task() as (remote, task_id):
            start = time.monotonic()
            remote._stop_active_agent_tasks(timeout=0.5)
            elapsed = time.monotonic() - start

            assert elapsed < 3.0, (
                f"helper hung past its timeout: {elapsed:.2f}s"
            )

            result = _row_result(task_id)
            assert result == "Task interrupted by server restart/shutdown", (
                "regression: pre-emptive shutdown persistence did not "
                "rewrite the sentinel; the next startup's orphan sweep "
                "would rewrite it to 'process killed' and the user "
                f"would perceive the agent as killed; got {result!r}"
            )

    def test_stuck_history_writer_does_not_hold_the_shutdown(self) -> None:
        """Another history writer keeps the write lock: the sweep still
        returns within its timeout, and the rewrite, which goes on
        waiting for the lock, lands once that writer is done."""
        held, release = threading.Event(), threading.Event()
        writer = threading.Thread(
            target=_hold_history_write_lock, args=(held, release), daemon=True,
        )
        with _wedged_task() as (remote, task_id):
            writer.start()
            try:
                assert held.wait(5)
                start = time.monotonic()
                remote._stop_active_agent_tasks(timeout=0.5)
                elapsed = time.monotonic() - start
            finally:
                release.set()
                writer.join(10)
            assert elapsed < 3.0, f"sweep waited for the history writer: {elapsed:.2f}s"
            deadline = time.monotonic() + 10
            while (
                _row_result(task_id) == "Agent Failed Abruptly"
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert _row_result(task_id) == "Task interrupted by server restart/shutdown"

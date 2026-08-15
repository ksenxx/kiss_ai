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
3. When the host process is shut down (SIGTERM / VS Code reload / IP
   change in :meth:`RemoteAccessServer._watchdog`),
   :meth:`RemoteAccessServer._stop_active_agent_tasks` signals each
   worker thread to stop and joins it with a bounded *timeout*.
4. If the worker is wedged inside an LLM API call (a blocking C-level
   socket read that does not honour ``KeyboardInterrupt``), the join
   times out, the daemon worker is killed at process exit, and the
   cleanup ``finally`` never runs.  The sentinel survives in the DB.
5. On the next process startup,
   :func:`_persistence._recover_orphaned_tasks` (invoked from
   :meth:`VSCodeServer.__init__`) rewrites every surviving sentinel
   row to ``"Task terminated unexpectedly (process killed)"`` —
   silently, several seconds after the fact, after the user has
   already reopened the browser.

The fix is a pre-emptive persistence safety net:
:func:`_persistence._shutdown_persist_in_flight_results`, invoked from
:meth:`RemoteAccessServer._stop_active_agent_tasks` BEFORE the
cooperative stop is signalled and BEFORE the join timeout has any
chance to expire.  It rewrites the sentinel row to
``"Task interrupted by server restart/shutdown"`` so that even if the
worker is wedged and the daemon thread is killed at process exit, the
DB already carries the truthful, non-alarming result.

This module spins up real worker threads — one that *ignores*
``KeyboardInterrupt`` and one that responds to it — and verifies that
both cases land at the same, truthful row-result.
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
from kiss.server import agent_state


def _make_remote_server() -> Any:
    """Build a :class:`RemoteAccessServer` with no tunnel / sockets bound.

    Constructing the remote server also constructs its owned
    :class:`VSCodeServer`, whose ``__init__`` sweeps the process-global
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
        uds_path=os.path.join(tmp, "sorcar.sock"),
    )


def _insert_sentinel_row(chat_id: str) -> str:
    """Insert a fresh ``task_history`` row carrying the sentinel.

    Mirrors what :meth:`ChatSorcarAgent.run` does at the start of every
    task — the same shape :func:`_recover_orphaned_tasks` looks at when
    rewriting orphans.
    """
    task_id, _ = _persistence._add_task(
        "shutdown-while-wedged",
        chat_id=chat_id,
        extra={
            "model": "test/model",
            "work_dir": "/tmp",
            "version": "test",
            "is_parallel": False,
            "is_worktree": False,
        },
    )
    return task_id


def _row_result(task_id: str) -> str:
    db = _persistence._get_db()
    row = db.execute(
        "SELECT result FROM task_history WHERE id = ?", (task_id,),
    ).fetchone()
    assert row is not None, f"row {task_id} disappeared"
    return str(row["result"])


class TestShutdownPersistsUninterruptibleWorker(TestCase):
    """Pre-emptive persistence must guard against truly-wedged workers."""

    def test_uninterruptible_worker_row_is_rewritten_before_timeout(
        self,
    ) -> None:
        """A worker that *swallows* ``KeyboardInterrupt`` mimics an LLM
        call wedged in a C-level socket read.  Without the pre-emptive
        save, ``_stop_active_agent_tasks`` would return after its join
        timeout with the row still at the sentinel — so the next
        startup's orphan sweep rewrites it to "process killed" and the
        user perceives the agent as killed.  With the safety net, the
        row carries the truthful "Task interrupted by server
        restart/shutdown" by the time the helper returns, regardless
        of whether the worker ever unwinds.
        """
        remote = _make_remote_server()

        tab_id = "shutdown-uninterruptible-1"
        chat_id = "shutdown-uninterruptible-chat-1"
        task_id = _insert_sentinel_row(chat_id)
        assert _row_result(task_id) == "Agent Failed Abruptly", (
            "precondition: row must start with the sentinel"
        )

        cleanup = threading.Event()

        def _uninterruptible_worker() -> None:
            while not cleanup.is_set():
                try:
                    time.sleep(0.02)
                except KeyboardInterrupt:
                    pass

        state = agent_state.AgentState(
            str(task_id),
            chat_id=chat_id,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
            is_task_active=True,
        )
        state.user_answer_queue = queue.Queue()
        worker = threading.Thread(target=_uninterruptible_worker, daemon=True)
        state.task_thread = worker
        agent_state.register(state)
        worker.start()
        try:
            for _ in range(100):
                if worker.is_alive():
                    break
                time.sleep(0.01)
            assert worker.is_alive(), "worker never started"

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

            assert state.interrupted_by_shutdown is True, (
                "shutdown helper must set interrupted_by_shutdown on "
                "in-flight states"
            )
        finally:
            cleanup.set()
            worker.join(timeout=2)
            agent_state.unregister(str(task_id), state)

    def test_pre_emptive_rewrite_only_touches_active_ids(self) -> None:
        """The safety net must be tightly scoped: an unrelated row
        that also still carries the sentinel (e.g. a true orphan from
        a previous crash that has not yet been swept) must NOT be
        clobbered to "restart/shutdown" — that is the
        ``_recover_orphaned_tasks`` sweep's job, with its own,
        different, message.
        """
        active_id = _insert_sentinel_row("shutdown-uninterruptible-chat-2a")
        bystander_id = _insert_sentinel_row("shutdown-uninterruptible-chat-2b")
        assert _row_result(active_id) == "Agent Failed Abruptly"
        assert _row_result(bystander_id) == "Agent Failed Abruptly"

        affected = _persistence._shutdown_persist_in_flight_results(
            {active_id},
        )
        assert affected == 1, f"expected exactly 1 rewrite, got {affected}"
        assert _row_result(active_id) == (
            "Task interrupted by server restart/shutdown"
        )
        assert _row_result(bystander_id) == "Agent Failed Abruptly", (
            "bystander row was clobbered; helper must be scoped to "
            "the supplied id set"
        )

    def test_pre_emptive_rewrite_does_not_clobber_completed_row(self) -> None:
        """A row that the worker *did* manage to overwrite with a
        real result before the shutdown call must NEVER be
        downgraded to "restart/shutdown" — the helper conditions on
        ``result = 'Agent Failed Abruptly'`` exactly to avoid this.
        """
        completed_id = _insert_sentinel_row("shutdown-uninterruptible-chat-3")
        _persistence._save_task_result(
            "Task completed successfully", task_id=completed_id,
        )
        assert _row_result(completed_id) == "Task completed successfully"

        affected = _persistence._shutdown_persist_in_flight_results(
            {completed_id},
        )
        assert affected == 0, (
            "helper must not rewrite rows that already have a real "
            f"result; got affected={affected}"
        )
        assert _row_result(completed_id) == "Task completed successfully"

    def test_empty_id_set_is_noop(self) -> None:
        """Calling the helper with no ids must be a safe no-op."""
        affected = _persistence._shutdown_persist_in_flight_results(set())
        assert affected == 0

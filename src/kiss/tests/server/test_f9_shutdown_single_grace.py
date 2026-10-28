# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""F9: a SIGTERM shutdown waits one grace period per worker and merge, not two.

``_shutdown_on_sigterm`` runs :meth:`RemoteAccessServer._stop_active_agent_tasks`
(12 s) and :meth:`RemoteAccessServer._await_active_merges` (30 s), and
``start()``'s ``finally`` runs both again.  A worker wedged in
uninterruptible code, or a merge stuck in git, used to cost both
timeouts (up to 84 s) before the process exited.  These tests drive
the two helpers twice over a real registry entry and a real thread
and check that the second call returns at once.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from typing import Any
from unittest import TestCase

from kiss.server import agent_state


def _make_remote_server() -> Any:
    """Build a :class:`RemoteAccessServer` with no tunnel or sockets bound."""
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.web_server import RemoteAccessServer

    tmp = tempfile.mkdtemp(prefix="kiss-f9-single-grace-")
    return RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        local_endpoint_file=os.path.join(tmp, "sorcar-local.json"),
    )


def _wait_alive(thread: threading.Thread) -> None:
    for _ in range(100):
        if thread.is_alive():
            return
        time.sleep(0.01)
    raise AssertionError("thread never started")


class TestShutdownSingleGrace(TestCase):
    """The second sweep of one shutdown does not wait again."""

    def test_stop_active_agent_tasks_waits_once_per_worker(self) -> None:
        """A worker that survived its grace period is skipped by the next sweep."""
        remote = _make_remote_server()
        cleanup = threading.Event()

        def _uninterruptible_worker() -> None:
            while not cleanup.is_set():
                try:
                    time.sleep(0.02)
                except KeyboardInterrupt:
                    pass

        state = agent_state.AgentState(
            "f9-single-grace-task",
            chat_id="f9-single-grace-chat",
            tab_id="f9-single-grace-tab",
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
            _wait_alive(worker)
            start = time.monotonic()
            remote._stop_active_agent_tasks(timeout=0.5)
            first = time.monotonic() - start
            self.assertGreaterEqual(first, 0.4, "first sweep must wait its grace period")
            self.assertTrue(worker.is_alive())
            self.assertTrue(state.interrupted_by_shutdown)

            start = time.monotonic()
            remote._stop_active_agent_tasks(timeout=0.5)
            second = time.monotonic() - start
            self.assertLess(
                second, 0.2,
                f"second sweep waited again ({second:.2f}s) for a worker "
                "the first sweep already gave up on",
            )
        finally:
            cleanup.set()
            worker.join(timeout=2)
            agent_state.unregister(state.task_id, state)

    def test_await_active_merges_waits_once_per_merge(self) -> None:
        """A merge claim that outlived its grace period is not waited for again."""
        remote = _make_remote_server()
        release = threading.Event()

        def _stuck_merge() -> None:
            release.wait(5)

        state = agent_state.AgentState(
            "f9-single-grace-merge-task",
            chat_id="f9-single-grace-merge-chat",
            tab_id="f9-single-grace-merge-tab",
            server_owned=True,
        )
        merger = threading.Thread(target=_stuck_merge, daemon=True)
        state.is_merging = True
        state.merge_thread = merger
        agent_state.register(state)
        merger.start()
        try:
            _wait_alive(merger)
            start = time.monotonic()
            remote._await_active_merges(timeout=0.5)
            first = time.monotonic() - start
            self.assertGreaterEqual(first, 0.4, "first wait must last its grace period")
            self.assertTrue(state.merge_in_progress())

            start = time.monotonic()
            remote._await_active_merges(timeout=0.5)
            second = time.monotonic() - start
            self.assertLess(
                second, 0.2,
                f"second wait lasted {second:.2f}s for a merge the first "
                "wait already gave up on",
            )
        finally:
            release.set()
            merger.join(timeout=2)
            state.is_merging = False
            state.merge_thread = None
            agent_state.unregister(state.task_id, state)

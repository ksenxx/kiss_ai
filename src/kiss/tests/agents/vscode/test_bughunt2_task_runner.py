# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: ``task_runner.py`` viewer-stop and answer-queue routing bugs.

BUG-TR2-1 — ``_find_source_tab_for_viewer`` stops at the FIRST task in
``printer._subscribers`` whose viewer set contains the viewer tab, then
only inspects that one task's peers.  ``JsonPrinter.cleanup_task``
*intentionally preserves* subscriber sets when a task ends (so post-task
broadcasts still fan out), which means a viewer tab subscribed earlier
to a now-FINISHED task keeps that stale subscription.  When the same
viewer later subscribes to a RUNNING task launched from a *different*
tab, the scan resolves ``task_key`` to the finished task (dict insertion
order), finds no peer with a live ``stop_event`` there, and returns
``None`` — so ``_stop_task`` from the viewer tab is a silent no-op and
the running task cannot be stopped from that viewer.

BUG-TR2-2 — ``_resolve_task_answer_queue`` returns the first subscribed
tab that has a live ``user_answer_queue``, on the (documented but false)
assumption that "the queue always lives on the task-owner tab".  A
viewer tab subscribed to task X that is itself running its OWN task Y
also carries a live ``user_answer_queue`` — owned by task Y.  When the
owner tab's queue is gone (tab closed → queue set to ``None``), task X's
``ask_user_question`` resolves to task Y's queue and steals the answer
the user submits for task Y's question (and task Y's agent never
receives it).  The correct behaviour is to resolve ``None`` so
``_await_user_response`` raises ``KeyboardInterrupt`` via its M4 guard
instead of hijacking another task's queue.

Both tests use the real ``VSCodeServer``, real ``JsonPrinter``
subscription state, and real ``_RunningAgentState`` registry entries —
no mocks or patched methods.
"""

from __future__ import annotations

import queue
import threading
import unittest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer


def _register_state(tab_id: str) -> _RunningAgentState:
    """Create and register a bare per-tab state (no agent, idle)."""
    state = _RunningAgentState(tab_id, "")
    _RunningAgentState.register(tab_id, state)
    return state


class TestViewerStopWithStaleSubscription(unittest.TestCase):
    """BUG-TR2-1: stop from a viewer holding a stale finished-task sub."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_stop_resolves_running_task_despite_stale_subscription(self) -> None:
        """A viewer subscribed to an old finished task AND a new running
        task must still be able to stop the running task."""
        viewer = "bh2-viewer"
        old_launcher = "bh2-launcher-old"
        new_launcher = "bh2-launcher-new"

        old_state = _register_state(old_launcher)
        new_state = _register_state(new_launcher)
        _register_state(viewer)

        # Task 8101 ran earlier from old_launcher and FINISHED: its
        # stop_event was cleared by _run_task's finally, but the
        # subscriber set survives (cleanup_task preserves it).
        self.server.printer.subscribe_tab("8101", old_launcher)
        self.server.printer.subscribe_tab("8101", viewer)
        old_state.stop_event = None
        old_state.task_thread = None

        # Task 8202 is RUNNING from new_launcher; the viewer is
        # subscribed to it too (e.g. via _reattach_running_chat).
        self.server.printer.subscribe_tab("8202", new_launcher)
        self.server.printer.subscribe_tab("8202", viewer)
        stop_event = threading.Event()
        new_state.stop_event = stop_event
        worker = threading.Thread(
            target=stop_event.wait, args=(15,), daemon=True,
        )
        worker.start()
        new_state.task_thread = worker

        try:
            self.server._stop_task(viewer)
            self.assertTrue(
                stop_event.wait(2.0),
                "Stop from the viewer tab was silently dropped: "
                "_find_source_tab_for_viewer resolved the stale finished "
                "task's subscriber set and never reached the running task",
            )
        finally:
            stop_event.set()
            worker.join(timeout=5)


class TestAnswerQueueCrossTaskHijack(unittest.TestCase):
    """BUG-TR2-2: askUser must not steal another task's answer queue."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        self.server.printer._thread_local.task_id = None
        _RunningAgentState.running_agent_states.clear()

    def test_resolution_skips_viewer_running_its_own_task(self) -> None:
        """When the owner tab's queue is gone, resolution must NOT fall
        back to a co-subscriber that is running a different task."""
        owner = "bh2-owner-x"
        viewer = "bh2-viewer-b"

        owner_state = _register_state(owner)
        # Real per-tab state + real WorktreeSorcarAgent via _get_tab.
        viewer_state = self.server._get_tab(viewer)

        # Both tabs are subscribed to task 8301 (owner launched it,
        # viewer watches it).  The owner tab has since been closed:
        # its user_answer_queue was reset to None.
        self.server.printer.subscribe_tab("8301", owner)
        self.server.printer.subscribe_tab("8301", viewer)
        owner_state.user_answer_queue = None

        # The viewer tab is itself running its OWN task 8999 and owns
        # a live answer queue for it.
        assert viewer_state.agent is not None
        viewer_state.agent._last_task_id = "8999"
        viewer_state.is_task_active = True
        viewer_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        viewer_state.user_answer_queue = viewer_queue
        self.server.printer.subscribe_tab("8999", viewer)

        # Task 8301's agent thread asks a question.
        self.server.printer._thread_local.task_id = "8301"
        resolved = self.server._resolve_task_answer_queue()

        self.assertIsNot(
            resolved,
            viewer_queue,
            "Task 8301 hijacked the answer queue owned by the viewer "
            "tab's own task 8999 — answers meant for 8999 would be "
            "stolen by 8301",
        )
        self.assertIsNone(resolved)

    def test_owner_queue_still_resolves(self) -> None:
        """Regression guard: the launcher tab's own live queue (its
        agent owns the asking task) must still resolve."""
        owner = "bh2-owner-y"
        owner_state = self.server._get_tab(owner)
        assert owner_state.agent is not None
        owner_state.agent._last_task_id = "8302"
        owner_state.is_task_active = True
        owner_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        owner_state.user_answer_queue = owner_queue
        self.server.printer.subscribe_tab("8302", owner)

        self.server.printer._thread_local.task_id = "8302"
        self.assertIs(self.server._resolve_task_answer_queue(), owner_queue)


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: ``task_runner.py`` viewer-stop and answer-queue routing bugs.

BUG-TR2-1 — a viewer tab subscribed earlier to a now-FINISHED task
keeps that stale subscription (``JsonPrinter.cleanup_task``
intentionally preserves subscriber sets so post-task broadcasts still
fan out).  When the same viewer later subscribes to a RUNNING task
launched from a *different* tab, ``_stop_task`` from the viewer tab
must skip the stale finished task's state and stop the running one —
not silently no-op.

BUG-TR2-2 — the user-answer queue lives on the asking TASK's own
registered agent state, resolved by the calling thread's task id.  A
viewer tab subscribed to task X that is itself running its OWN task Y
carries a live ``user_answer_queue`` owned by task Y; when task X's
queue is gone (tab closed → queue set to ``None``), task X's
``ask_user_question`` must resolve ``None`` — never task Y's queue —
so ``_await_user_response`` raises ``KeyboardInterrupt`` via its M4
guard instead of hijacking another task's answer.

Both tests use the real ``VSCodeServer``, real ``JsonPrinter``
subscription state, and real ``agent_state`` registry entries —
no mocks or patched methods.
"""

from __future__ import annotations

import queue
import threading
import unittest

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer


def _register_state(task_id: str, tab_id: str) -> AgentState:
    """Create and register a bare server-owned state (no agent, idle)."""
    state = AgentState(task_id, tab_id=tab_id, server_owned=True)
    agent_state.register(state)
    return state


class TestViewerStopWithStaleSubscription(unittest.TestCase):
    """BUG-TR2-1: stop from a viewer holding a stale finished-task sub."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_stop_resolves_running_task_despite_stale_subscription(self) -> None:
        """A viewer subscribed to an old finished task AND a new running
        task must still be able to stop the running task."""
        viewer = "bh2-viewer"
        old_launcher = "bh2-launcher-old"
        new_launcher = "bh2-launcher-new"

        old_state = _register_state("8101", old_launcher)
        new_state = _register_state("8202", new_launcher)

        self.server.printer.subscribe_tab("8101", old_launcher)
        self.server.printer.subscribe_tab("8101", viewer)
        old_state.stop_event = None
        old_state.task_thread = None

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
                "Stop from the viewer tab was silently dropped: the "
                "viewer-task resolution picked the stale finished "
                "task's state and never reached the running task",
            )
        finally:
            stop_event.set()
            worker.join(timeout=5)


class TestAnswerQueueCrossTaskHijack(unittest.TestCase):
    """BUG-TR2-2: askUser must not steal another task's answer queue."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        self.server.printer._thread_local.task_id = None
        agent_state.agent_states.clear()

    def test_resolution_skips_viewer_running_its_own_task(self) -> None:
        """When the asking task's queue is gone, resolution must NOT fall
        back to a co-subscriber that is running a different task."""
        owner = "bh2-owner-x"
        viewer = "bh2-viewer-b"

        owner_state = _register_state("8301", owner)
        viewer_state = _register_state("8999", viewer)

        self.server.printer.subscribe_tab("8301", owner)
        self.server.printer.subscribe_tab("8301", viewer)
        owner_state.user_answer_queue = None

        viewer_state.is_task_active = True
        viewer_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        viewer_state.user_answer_queue = viewer_queue
        self.server.printer.subscribe_tab("8999", viewer)

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
        """Regression guard: the asking task's own live queue must
        still resolve."""
        owner = "bh2-owner-y"
        owner_state = _register_state("8302", owner)
        owner_state.is_task_active = True
        owner_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        owner_state.user_answer_queue = owner_queue
        self.server.printer.subscribe_tab("8302", owner)

        self.server.printer._thread_local.task_id = "8302"
        self.assertIs(self.server._resolve_task_answer_queue(), owner_queue)


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group C): viewer Stop must not kill an unrelated task.

BUG-TR8-1 — the old per-tab viewer resolution returned the FIRST peer
tab (in subscriber-set scan order) that carried a live ``stop_event``,
without checking that the peer's running task was actually one of the
tasks the viewer is subscribed to.  ``JsonPrinter.cleanup_task``
intentionally preserves subscriber sets after a task finishes, so a
viewer typically holds a stale subscription to a FINISHED task X whose
launcher tab has since started a brand-new, unrelated task Y (which the
viewer is NOT subscribed to).  When the viewer is also watching a
RUNNING task Z launched from a third tab and clicks Stop, the stale
task-X subscription must not route the stop to unrelated task Y.

In the task-keyed registry this is structurally prevented:
``_find_viewer_task_states`` resolves each subscribed task id directly
in ``kiss.server.agent_state``, so a stale subscription to a finished
task resolves to nothing and the stop lands on the task the viewer is
actually watching.

The test uses the real ``VSCodeServer``, real ``JsonPrinter``
subscription state, real ``agent_state`` registry entries, real
threads and real ``threading.Event`` objects — no mocks or patches.
"""

from __future__ import annotations

import threading
import unittest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class TestViewerStopCrossTaskHijack(unittest.TestCase):
    """BUG-TR8-1: Stop from a viewer must target the viewer's task."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_stop_skips_peer_running_unrelated_task(self) -> None:
        """A stale finished-task co-subscriber that is now running its
        OWN unrelated task must not be stopped by the viewer's Stop."""
        viewer = "bh8c-viewer"
        old_launcher = "bh8c-launcher-old"
        z_launcher = "bh8c-launcher-z"

        # The viewer tab has an idle state of its own (no stop_event).
        viewer_state = agent_state.AgentState(
            "bh8c-viewer-key", tab_id=viewer, server_owned=True,
        )
        agent_state.register(viewer_state)

        # Task 9101 finished long ago: its subscriber set lingers in the
        # printer, but no registry entry exists for it any more.
        self.server.printer.subscribe_tab("9101", old_launcher)
        self.server.printer.subscribe_tab("9101", viewer)

        # The old launcher is now running a brand-new, unrelated task
        # 9555 that the viewer is NOT subscribed to.
        unrelated_stop = threading.Event()
        unrelated_worker = threading.Thread(
            target=unrelated_stop.wait, args=(15,), daemon=True,
        )
        unrelated_worker.start()
        old_state = agent_state.AgentState(
            "9555",
            tab_id=old_launcher,
            server_owned=True,
            is_task_active=True,
            stop_event=unrelated_stop,
        )
        old_state.task_thread = unrelated_worker
        agent_state.register(old_state)
        self.server.printer.subscribe_tab("9555", old_launcher)

        # A third tab launched task 9202 (task Z) which the viewer IS
        # subscribed to (history-click multi-view).
        target_stop = threading.Event()
        target_worker = threading.Thread(
            target=target_stop.wait, args=(15,), daemon=True,
        )
        target_worker.start()
        z_state = agent_state.AgentState(
            "9202",
            tab_id=z_launcher,
            server_owned=True,
            is_task_active=True,
            stop_event=target_stop,
        )
        z_state.task_thread = target_worker
        agent_state.register(z_state)
        self.server.printer.subscribe_tab("9202", z_launcher)
        self.server.printer.subscribe_tab("9202", viewer)

        try:
            self.server._stop_task(viewer)
            self.assertTrue(
                target_stop.wait(2.0),
                "Stop from the viewer tab never reached the task the "
                "viewer is actually watching (task 9202)",
            )
            self.assertFalse(
                unrelated_stop.is_set(),
                "Stop from the viewer tab hijacked the old launcher's "
                "unrelated task 9555: viewer resolution matched the "
                "stale finished-task subscriber set and stopped a "
                "task the viewer is not subscribed to",
            )
        finally:
            unrelated_stop.set()
            target_stop.set()
            unrelated_worker.join(timeout=5)
            target_worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

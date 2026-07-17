# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group C): viewer Stop must not kill an unrelated task.

BUG-TR8-1 — ``_find_source_tab_for_viewer`` returns the FIRST peer tab
(in subscriber-set scan order) that carries a live ``stop_event``,
without checking that the peer's running task is actually one of the
tasks the viewer is subscribed to.  ``JsonPrinter.cleanup_task``
intentionally preserves subscriber sets after a task finishes, so a
viewer typically holds a stale subscription to a FINISHED task X whose
launcher tab has since started a brand-new, unrelated task Y (which the
viewer is NOT subscribed to).  When the viewer is also watching a
RUNNING task Z launched from a third tab and clicks Stop, the scan hits
the stale task-X subscriber set first, sees the old launcher's live
``stop_event`` (owned by unrelated task Y), and stops task Y instead of
task Z — a cross-task stop hijack symmetric to the answer-queue hijack
fixed in bug-hunt 2 (BUG-TR2-2).

The test uses the real ``VSCodeServer``, real ``JsonPrinter``
subscription state, real ``_RunningAgentState`` registry entries, real
threads and real ``threading.Event`` objects — no mocks or patches.
"""

from __future__ import annotations

import threading
import unittest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer


def _register_state(tab_id: str) -> _RunningAgentState:
    """Create and register a bare per-tab state (no agent, idle)."""
    state = _RunningAgentState(tab_id, "")
    _RunningAgentState.register(tab_id, state)
    return state


class TestViewerStopCrossTaskHijack(unittest.TestCase):
    """BUG-TR8-1: Stop from a viewer must target the viewer's task."""

    def setUp(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_stop_skips_peer_running_unrelated_task(self) -> None:
        """A stale finished-task co-subscriber that is now running its
        OWN unrelated task must not be stopped by the viewer's Stop."""
        viewer = "bh8c-viewer"
        old_launcher = "bh8c-launcher-old"
        z_launcher = "bh8c-launcher-z"

        old_state = self.server._get_tab(old_launcher)
        z_state = self.server._get_tab(z_launcher)
        _register_state(viewer)

        # Task 9101 ran earlier from old_launcher and FINISHED; the
        # viewer watched it.  cleanup_task preserves the subscriber
        # set, so the stale subscription survives.
        self.server.printer.subscribe_tab("9101", old_launcher)
        self.server.printer.subscribe_tab("9101", viewer)

        # old_launcher has since started a NEW unrelated task 9555
        # (viewer NOT subscribed): it owns a live stop_event again.
        assert old_state.agent is not None
        old_state.agent._last_task_id = "9555"
        old_state.is_task_active = True
        unrelated_stop = threading.Event()
        old_state.stop_event = unrelated_stop
        unrelated_worker = threading.Thread(
            target=unrelated_stop.wait, args=(15,), daemon=True,
        )
        unrelated_worker.start()
        old_state.task_thread = unrelated_worker
        self.server.printer.subscribe_tab("9555", old_launcher)

        # Task 9202 is RUNNING from z_launcher and the viewer is
        # subscribed to it (e.g. via _reattach_running_chat).
        assert z_state.agent is not None
        z_state.agent._last_task_id = "9202"
        z_state.is_task_active = True
        target_stop = threading.Event()
        z_state.stop_event = target_stop
        target_worker = threading.Thread(
            target=target_stop.wait, args=(15,), daemon=True,
        )
        target_worker.start()
        z_state.task_thread = target_worker
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
                "unrelated task 9555: _find_source_tab_for_viewer "
                "matched the stale finished-task subscriber set and "
                "returned a peer whose live stop_event belongs to a "
                "task the viewer is not subscribed to",
            )
        finally:
            unrelated_stop.set()
            target_stop.set()
            unrelated_worker.join(timeout=5)
            target_worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

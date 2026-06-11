# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt: stale task-stream subscriptions when a tab navigates away.

``JsonPrinter._subscribers`` (task_id -> {tab_id}) drives the per-tab
fan-out of live task events.  A tab subscribes when it opens a chat
backed by a running task (``_reattach_running_chat``), but the ONLY
unsubscribe path is tab CLOSE (``_teardown_tab_resources`` →
``cleanup_tab``).  Two navigation paths leave the subscription behind:

* ``_new_chat`` — the tab now shows the welcome screen, yet the old
  task's live events keep streaming into it (rendered on top of the
  welcome view).
* ``_replay_session`` of a DIFFERENT chat — the tab now displays chat
  B, yet chat A's running task keeps streaming its events into the
  tab, mixing two unrelated task streams in one webview.
"""

from __future__ import annotations

import time
import unittest
from typing import Any

from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


class _StaleSubscriptionBase(unittest.TestCase):
    """Scaffolding: one live (simulated) task on chat A, owned by src-tab."""

    def setUp(self) -> None:
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(self.printer)
        # Persist a task row for chat A and register a live-looking
        # owner state for it (mirrors the state an in-flight
        # ``_run_task_inner`` maintains: ``is_task_active=True`` and
        # ``task_history_id`` set, exactly what
        # ``_reattach_running_chat`` matches on).
        self.task_a, self.chat_a = _add_task("live task on chat A")
        self.src_tab = "srv2-src-tab"
        src = self.server._get_tab(self.src_tab)
        src.chat_id = self.chat_a
        src.task_history_id = self.task_a
        src.is_task_active = True
        # The launcher tab is subscribed to its own task's stream
        # (done by ChatSorcarAgent.run via ``_subscribe_tab_id`` in
        # production).
        self.printer.subscribe_tab(self.task_a, self.src_tab)

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def _emit_live_event_for_task_a(self) -> None:
        """Broadcast a live agent event attributed to task A."""
        self.printer._thread_local.task_id = str(self.task_a)
        try:
            self.printer.broadcast({"type": "text_delta", "text": "live"})
        finally:
            self.printer._thread_local.task_id = None

    def _copies_for(self, tab_id: str) -> list[dict[str, Any]]:
        return [
            e for e in self.printer.emitted
            if e.get("tabId") == tab_id and e.get("type") == "text_delta"
        ]

    def _resume(self, chat_id: str, task_id: int, tab_id: str) -> None:
        self.server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "taskId": task_id,
            "tabId": tab_id,
        })


class TestNewChatDropsOldStream(_StaleSubscriptionBase):
    """New Chat must stop the previously viewed task's live stream."""

    def test_new_chat_unsubscribes_tab_from_old_task(self) -> None:
        viewer = "srv2-viewer-newchat"
        self._resume(self.chat_a, self.task_a, viewer)
        # Sanity: the viewer is now receiving task A's live events.
        self.printer.emitted.clear()
        self._emit_live_event_for_task_a()
        assert self._copies_for(viewer), "viewer never subscribed to task A"

        # The user clicks "New Chat" in that tab: the webview shows the
        # welcome screen, so task A's events must no longer reach it.
        self.server._handle_command({"type": "newChat", "tabId": viewer})
        self.printer.emitted.clear()
        self._emit_live_event_for_task_a()
        leaked = self._copies_for(viewer)
        assert not leaked, (
            "task A's live events still stream into a tab that was "
            f"reset to a new chat: {leaked!r}"
        )


class TestReplayOtherChatDropsOldStream(_StaleSubscriptionBase):
    """Loading a different chat must stop the old task's live stream."""

    def test_replay_of_other_chat_unsubscribes_old_task(self) -> None:
        viewer = "srv2-viewer-switch"
        self._resume(self.chat_a, self.task_a, viewer)
        self.printer.emitted.clear()
        self._emit_live_event_for_task_a()
        assert self._copies_for(viewer), "viewer never subscribed to task A"

        # The user loads a COMPLETED task of an unrelated chat B into
        # the same tab.
        time.sleep(0.02)
        task_b, chat_b = _add_task("completed task on chat B")
        self._resume(chat_b, task_b, viewer)

        self.printer.emitted.clear()
        self._emit_live_event_for_task_a()
        leaked = self._copies_for(viewer)
        assert not leaked, (
            "chat A's live task events still stream into a tab that now "
            f"displays chat B: {leaked!r}"
        )

    def test_replay_back_to_running_chat_restores_stream(self) -> None:
        """Regression guard: switching back to chat A re-subscribes."""
        viewer = "srv2-viewer-back"
        self._resume(self.chat_a, self.task_a, viewer)
        time.sleep(0.02)
        task_b, chat_b = _add_task("completed task on chat B")
        self._resume(chat_b, task_b, viewer)
        self._resume(self.chat_a, self.task_a, viewer)
        self.printer.emitted.clear()
        self._emit_live_event_for_task_a()
        assert self._copies_for(viewer), (
            "viewer no longer receives task A's events after navigating "
            "back to chat A"
        )

    def test_owner_tab_self_replay_keeps_own_stream(self) -> None:
        """Regression guard: the launcher tab replaying its OWN running
        chat (e.g. webview restore) must keep receiving the stream."""
        self._resume(self.chat_a, self.task_a, self.src_tab)
        self.printer.emitted.clear()
        self._emit_live_event_for_task_a()
        assert self._copies_for(self.src_tab), (
            "owner tab lost its own task stream after replaying its "
            "own running chat"
        )


if __name__ == "__main__":
    unittest.main()

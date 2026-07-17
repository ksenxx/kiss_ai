# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt: adjacent-task navigation from a pure-viewer tab.

``_replay_session`` (server.py) deliberately does NOT create a
``_RunningAgentState`` registry entry for a pure-viewer tab (the
C2/C3 fix) — it only records the tab in ``_tab_chat_views``.  But
``_cmd_get_adjacent_task`` (commands.py) resolves the chat id via
``self._get_tab(tab_id)``, which CREATES a fresh registry entry whose
``chat_id`` is ``""``.  ``_get_adjacent_task_by_chat_id`` returns
``None`` for an empty chat id, so arrow-key navigation in any tab
opened from the history sidebar (after a daemon restart, or any tab
that never ran a task itself) always broadcasts an EMPTY
``adjacent_task_events`` payload even though the chat has adjacent
tasks.
"""

from __future__ import annotations

import time
import unittest

from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


class TestAdjacentTaskFromViewerTab(unittest.TestCase):
    """getAdjacentTask must work in a tab that only VIEWS a chat."""

    def setUp(self) -> None:
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(self.printer)

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_adjacent_prev_resolves_chat_of_viewer_tab(self) -> None:
        """A history-opened viewer tab can navigate to the previous task."""
        t1, chat_id = _add_task("first task in chat")
        # Distinct timestamps: the adjacent lookup compares strictly.
        time.sleep(0.02)
        t2, _ = _add_task("second task in chat", chat_id)

        viewer_tab = "srv2-viewer-tab"
        # Simulate the user clicking task t2 in the history sidebar:
        # the frontend allocates a fresh tab and round-trips a
        # resumeSession command.  No task ever ran in this tab, so no
        # registry entry exists — only the chat-viewer association.
        self.server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "taskId": t2,
            "tabId": viewer_tab,
        })
        self.printer.emitted.clear()

        # The user presses the "previous task" arrow in that tab.
        self.server._handle_command({
            "type": "getAdjacentTask",
            "tabId": viewer_tab,
            "taskId": t2,
            "direction": "prev",
        })

        adj = [
            e for e in self.printer.emitted
            if e.get("type") == "adjacent_task_events"
            and e.get("tabId") == viewer_tab
        ]
        assert adj, "no adjacent_task_events broadcast for the viewer tab"
        assert adj[0].get("task_id") == t1, (
            "adjacent-task navigation returned an empty payload for a "
            f"pure-viewer tab: expected task_id={t1}, got "
            f"task_id={adj[0].get('task_id')!r} task={adj[0].get('task')!r}"
        )
        assert adj[0].get("task") == "first task in chat"


if __name__ == "__main__":
    unittest.main()

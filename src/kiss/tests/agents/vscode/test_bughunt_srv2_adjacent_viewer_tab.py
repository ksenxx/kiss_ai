# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt: adjacent-task navigation from a pure-viewer tab.

``_replay_session`` (server.py) deliberately does NOT create an
agent-state registry entry for a pure-viewer tab (the C2/C3 fix) —
it only records the tab in ``_tab_chat_views``.
``_cmd_get_adjacent_task`` (commands.py) must therefore resolve the
chat id through ``_tab_chat_views`` rather than the registry (a
viewer tab has no registered state, hence no ``chat_id``).
``_get_adjacent_task_by_chat_id`` returns ``None`` for an empty chat
id, so getting this wrong makes arrow-key navigation in any tab
opened from the history sidebar (after a daemon restart, or any tab
that never ran a task itself) broadcast an EMPTY
``adjacent_task_events`` payload even though the chat has adjacent
tasks.
"""

from __future__ import annotations

import time
import unittest

from kiss.agents.sorcar.persistence import _add_task
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


class TestAdjacentTaskFromViewerTab(unittest.TestCase):
    """getAdjacentTask must work in a tab that only VIEWS a chat."""

    def setUp(self) -> None:
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(self.printer)

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_adjacent_prev_resolves_chat_of_viewer_tab(self) -> None:
        """A history-opened viewer tab can navigate to the previous task."""
        t1, chat_id = _add_task("first task in chat")
        time.sleep(0.02)
        t2, _ = _add_task("second task in chat", chat_id)

        viewer_tab = "srv2-viewer-tab"
        self.server._handle_command({
            "type": "resumeSession",
            "chatId": chat_id,
            "taskId": t2,
            "tabId": viewer_tab,
        })
        self.printer.emitted.clear()

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

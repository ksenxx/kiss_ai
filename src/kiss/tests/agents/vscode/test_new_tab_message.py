# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``VSCodeServer.broadcast_new_tab`` emits the expected payload.

The Python server can ask the frontend to open a fresh chat tab and
resume an existing task into it by broadcasting::

    {"type": "new_tab", "task_id": <int>}

This module pins the Python-side helper ``VSCodeServer.broadcast_new_tab``
emits the exact dict shape the frontend handler expects.
"""

from __future__ import annotations

import unittest
from typing import Any

from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer


class _CapturingPrinter(JsonPrinter):
    """Records every ``broadcast`` call verbatim."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class TestBroadcastNewTabHelper(unittest.TestCase):
    """``VSCodeServer.broadcast_new_tab`` emits the expected payload."""

    def test_helper_broadcasts_expected_shape(self) -> None:
        printer = _CapturingPrinter()
        server = VSCodeServer(printer=printer)
        server.broadcast_new_tab(7)
        # Explicit empty ``taskId`` keeps this a global system event
        # so the broadcast reaches every connected client (the
        # frontend needs the event to allocate the new tab; only
        # after allocation does it subscribe to the task's stream).
        self.assertEqual(
            printer.events,
            [{"type": "new_tab", "task_id": 7, "taskId": ""}],
        )

    def test_helper_coerces_task_id_to_int(self) -> None:
        printer = _CapturingPrinter()
        server = VSCodeServer(printer=printer)
        # Numeric-string task ids (e.g. read from JSON) should be
        # normalised to a plain int so the frontend doesn't have to
        # branch on the field's runtime type.
        server.broadcast_new_tab("11")  # type: ignore[arg-type]
        self.assertEqual(
            printer.events,
            [{"type": "new_tab", "task_id": 11, "taskId": ""}],
        )


if __name__ == "__main__":
    unittest.main()

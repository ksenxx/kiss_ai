# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: dispatcher / numeric-field robustness (BUG-A, BUG-B).

BUG-A: ``VSCodeServer._handle_command`` used the raw ``type`` field as
a dict key, so an unhashable value (``{"type": []}``) raised TypeError.
The websocket/UDS handlers wrap the whole receive loop in one try, so
the exception killed the ENTIRE client connection (whole VS Code
window).

BUG-B: four handlers called ``int()`` on frontend-supplied fields
without a guard — ``deleteTask``/``setFavorite``/``resumeSession``
(``taskId``) and ``getFrequentTasks`` (``limit``) — so garbage like
``"abc"`` raised ValueError and killed the connection.  Inconsistent
with ``_cmd_get_adjacent_task``, which guards its parse.
"""

from __future__ import annotations

import unittest
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer


class TestDispatchMalformed(unittest.TestCase):
    """Malformed payloads must never raise out of ``_handle_command``."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_unhashable_type_field_does_not_raise(self) -> None:
        self.server._handle_command({"type": []})
        self.server._handle_command({"type": {}})
        errors = [e for e in self.events if e.get("type") == "error"]
        assert len(errors) == 2, (
            f"expected 2 unknown-command error events, got {self.events!r}"
        )

    def test_delete_task_garbage_task_id_does_not_raise(self) -> None:
        self.server._handle_command({"type": "deleteTask", "taskId": "abc"})

    def test_set_favorite_garbage_task_id_does_not_raise(self) -> None:
        self.server._handle_command(
            {"type": "setFavorite", "taskId": "abc", "isFavorite": True},
        )

    def test_resume_session_garbage_task_id_does_not_raise(self) -> None:
        self.server._handle_command(
            {"type": "resumeSession", "taskId": "abc", "tabId": "t1"},
        )

    def test_get_frequent_tasks_garbage_limit_does_not_raise(self) -> None:
        self.server._handle_command(
            {"type": "getFrequentTasks", "limit": "abc"},
        )
        freq = [e for e in self.events if e.get("type") == "frequentTasks"]
        assert freq, "getFrequentTasks must still reply with default limit"


if __name__ == "__main__":
    unittest.main()

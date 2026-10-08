# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: every Stop click is acknowledged.

Post-mortem ``reports/stop_button_delay_2026-08-05.html`` (task
``709ebce3``): a stop the daemon could not route used to disappear
behind a disabled ``logger.debug``, so the UI could not tell a pending
stop from a discarded one.  ``_stop_task`` now answers every click
with a ``stop_ack`` event, accepted or not.

Real threads, the real agent-state registry and the production
``VSCodeServer._stop_task``.
"""

from __future__ import annotations

import threading
from typing import Any

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class TestStopIsAlwaysAcknowledged:
    """No Stop click may vanish without a word."""

    def _server_with_capture(self) -> tuple[VSCodeServer, list[dict[str, Any]]]:
        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        return server, events

    def test_stop_on_a_running_tab_is_acknowledged(self) -> None:
        """The click is acknowledged before the task even reacts."""
        server, events = self._server_with_capture()
        tab_id = "stop-ack-running"
        stop_event = threading.Event()
        state = agent_state.AgentState(
            "task-stop-ack",
            tab_id=tab_id,
            server_owned=True,
            stop_event=stop_event,
            is_task_active=True,
        )
        agent_state.register(state)
        try:
            server._stop_task(tab_id)
        finally:
            agent_state.unregister(state.task_id, state)
        acks = [e for e in events if e.get("type") == "stop_ack"]
        assert acks == [
            {"type": "stop_ack", "accepted": True, "tabId": tab_id},
        ]
        assert stop_event.is_set()

    def test_stop_with_nothing_to_stop_says_so(self) -> None:
        """A click on a tab whose task already ended is reported back.

        It used to be discarded in silence — indistinguishable from a
        stop the agent had simply not reached yet, which is what makes
        people click again.
        """
        server, events = self._server_with_capture()
        server._stop_task("stop-ack-nothing-running")
        acks = [e for e in events if e.get("type") == "stop_ack"]
        assert acks == [
            {
                "type": "stop_ack",
                "accepted": False,
                "tabId": "stop-ack-nothing-running",
            },
        ]

    def test_stop_without_a_tab_id_is_still_ignored(self) -> None:
        """A missing tabId is a frontend bug, not a stop-everything."""
        server, events = self._server_with_capture()
        server._stop_task("")
        assert [e for e in events if e.get("type") == "stop_ack"] == []

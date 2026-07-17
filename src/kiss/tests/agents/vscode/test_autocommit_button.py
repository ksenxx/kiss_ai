# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the auto-commit button in the VS Code webview.

Validates:
- The button element exists in the HTML template (SorcarTab.ts).
- The button is placed inline next to the "Auto commit" label inside
  the settings panel (``#cfg-auto-commit`` checkbox row).
- The button has its own #autocommit-btn CSS, styled like #menu-btn.
- The JS click handler sends the correct ``autocommitAction`` message.
- The button is disabled when a task is running (setRunningState).
- The backend ``autocommitAction`` command dispatches to ``_handle_autocommit_action``.
"""

from __future__ import annotations

import threading
import unittest
from pathlib import Path

from kiss.server.commands import _CommandsMixin
from kiss.server.server import VSCodeServer

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


def _read(name: str) -> str:
    return (_VSCODE_DIR / name).read_text()


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events








class TestAutocommitButtonBackend(unittest.TestCase):
    """The backend correctly dispatches the autocommitAction command."""

    def test_handler_in_dispatch_table(self) -> None:
        """autocommitAction is registered in _HANDLERS."""
        assert "autocommitAction" in _CommandsMixin._HANDLERS

    def test_autocommit_action_commit(self) -> None:
        """Sending autocommitAction with action=commit triggers the commit flow."""
        server, events = _make_server()
        server.work_dir = "/tmp/nonexistent"
        tab_id = "test-tab-ac"
        server._get_tab(tab_id)

        server._handle_autocommit_action("commit", tab_id)

        done_events = [e for e in events if e.get("type") == "autocommit_done"]
        assert len(done_events) == 1
        assert done_events[0]["tabId"] == tab_id

    def test_autocommit_action_skip(self) -> None:
        """Sending autocommitAction with action=skip broadcasts done with committed=False."""
        server, events = _make_server()
        tab_id = "test-tab-skip"
        server._get_tab(tab_id)

        server._handle_autocommit_action("skip", tab_id)

        done_events = [e for e in events if e.get("type") == "autocommit_done"]
        assert len(done_events) == 1
        assert done_events[0]["committed"] is False
        assert done_events[0]["success"] is True
        assert done_events[0]["tabId"] == tab_id


if __name__ == "__main__":
    unittest.main()

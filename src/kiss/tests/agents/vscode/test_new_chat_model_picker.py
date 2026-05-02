"""Integration test: new chat model picker shows last user-picked model.

Bug: When a new chat is opened, the model picker shows the model from the
currently active tab instead of the last model explicitly picked by the user
(saved in the database). This happens because the frontend copies the JS
global ``selectedModel`` (which tracks the current tab) into the new tab,
and the backend's ``showWelcome`` event did not include the correct model.

Fix: ``_new_chat`` re-reads the last-picked model from the database and
includes it in the ``showWelcome`` event. The frontend ``showWelcome``
handler updates the model picker accordingly.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from kiss.agents.sorcar.persistence import (
    _close_db,
    _load_last_model,
    _record_model_usage,
)
from kiss.agents.vscode.server import VSCodeServer

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Point persistence at a temp dir so tests don't touch real data."""
    import kiss.agents.sorcar.persistence as pm

    _close_db()
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setattr(pm, "_KISS_DIR", type(pm._KISS_DIR)(tmpdir))
    monkeypatch.setattr(pm, "_DB_PATH", type(pm._DB_PATH)(os.path.join(tmpdir, "sorcar.db")))
    yield
    _close_db()


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
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


class TestNewChatModelPicker:
    """New chat must show the last user-picked model from DB, not the current tab's model."""

    def test_show_welcome_includes_last_picked_model(self) -> None:
        """When selectModel saves model-B to DB and a new chat is opened,
        the showWelcome event must contain model="model-B"."""
        server, events = _make_server()

        # Simulate user picking "model-A" then "model-B" via the picker
        server._handle_command({
            "type": "selectModel",
            "model": "model-A",
            "tabId": "tab-1",
        })
        server._handle_command({
            "type": "selectModel",
            "model": "model-B",
            "tabId": "tab-1",
        })

        # Verify DB has "model-B" as last picked
        assert _load_last_model() == "model-B"

        # Clear captured events
        events.clear()

        # Open new chat on a fresh tab
        server._handle_command({
            "type": "newChat",
            "tabId": "tab-2",
        })

        # Find the showWelcome event
        welcome_events = [e for e in events if e.get("type") == "showWelcome"]
        assert len(welcome_events) == 1
        welcome = welcome_events[0]

        # The showWelcome event must include the last-picked model
        assert welcome.get("model") == "model-B", (
            f"showWelcome should include model='model-B' from DB, "
            f"got model={welcome.get('model')!r}"
        )

    def test_new_tab_state_uses_db_model_not_stale_default(self) -> None:
        """The backend tab state for the new chat must use the DB model,
        not a stale _default_model from server init."""
        server, _events = _make_server()

        # Server init _default_model might be empty or from an old DB.
        # User picks "fresh-model" via the picker.
        server._handle_command({
            "type": "selectModel",
            "model": "fresh-model",
            "tabId": "tab-1",
        })

        # Simulate the _default_model being stale by overwriting it
        # (as if a different code path changed it)
        server._default_model = "stale-model"

        # Write "fresh-model" to DB to simulate that the user's pick was
        # persisted but the in-memory default diverged
        _record_model_usage("fresh-model")

        # Open new chat — should read from DB, not from stale _default_model
        server._handle_command({
            "type": "newChat",
            "tabId": "tab-new",
        })

        tab = server._tab_states.get("tab-new")
        assert tab is not None
        assert tab.selected_model == "fresh-model", (
            f"New tab should use DB model 'fresh-model', "
            f"got '{tab.selected_model}'"
        )

    def test_new_chat_model_differs_from_current_tab(self) -> None:
        """When the current tab has model-A but DB says model-B,
        the new chat must use model-B."""
        server, events = _make_server()

        # Tab-1 runs with model-A
        tab1 = server._get_tab("tab-1")
        tab1.selected_model = "model-A"

        # User picks model-B via picker (saved to DB and _default_model)
        server._handle_command({
            "type": "selectModel",
            "model": "model-B",
            "tabId": "tab-1",
        })

        # Now change tab-1 back to model-A in memory only (simulates
        # switching to a tab that was using model-A)
        tab1.selected_model = "model-A"
        # Also set _default_model to model-A (simulating the bug where
        # _default_model tracks the current tab)
        server._default_model = "model-A"

        events.clear()

        # Open new chat — should read from DB where model-B is saved
        server._handle_command({
            "type": "newChat",
            "tabId": "tab-new",
        })

        welcome_events = [e for e in events if e.get("type") == "showWelcome"]
        assert len(welcome_events) == 1
        assert welcome_events[0].get("model") == "model-B"

        tab = server._tab_states.get("tab-new")
        assert tab is not None
        assert tab.selected_model == "model-B"


class TestShowWelcomeHandlerUpdatesModel:
    """Frontend showWelcome handler must update model picker from ev.model."""

    def test_show_welcome_handler_uses_ev_model(self) -> None:
        """The showWelcome case in main.js must set selectedModel from ev.model."""
        source = MAIN_JS.read_text()

        # Find the showWelcome case block
        m = re.search(r"case\s+'showWelcome'\s*:", source)
        assert m is not None, "showWelcome case not found in main.js"

        # Extract the case block (until next case or closing brace)
        start = m.start()
        # Find the block - look for next case or closing brace at same level
        block_end = source.find("case '", start + 20)
        if block_end == -1:
            block_end = len(source)
        block = source[start:block_end]

        assert "ev.model" in block, (
            "showWelcome handler must read ev.model to update model picker"
        )
        assert "selectedModel" in block, (
            "showWelcome handler must update selectedModel from ev.model"
        )

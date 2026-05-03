"""Integration test: new chat model picker shows last user-picked model.

Bug: When a new chat tab is opened, the model picker shows the model from the
currently active tab instead of the last model explicitly picked by the user
(saved in the database).

Root cause: ``createNewTab()`` in main.js never sent a ``newChat`` command to
the backend, so the backend's ``_new_chat()`` (which reads the DB model and
sends it back in a ``showWelcome`` event) was never invoked.

Fix: ``createNewTab()`` now posts ``{type: 'newChat', tabId: tab.id}`` so the
backend reads the last-picked model from the database and includes it in the
``showWelcome`` event. The frontend ``showWelcome`` handler updates the model
picker accordingly.
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

        assert _load_last_model() == "model-B"

        events.clear()

        server._handle_command({
            "type": "newChat",
            "tabId": "tab-2",
        })

        welcome_events = [e for e in events if e.get("type") == "showWelcome"]
        assert len(welcome_events) == 1
        welcome = welcome_events[0]

        assert welcome.get("model") == "model-B", (
            f"showWelcome should include model='model-B' from DB, "
            f"got model={welcome.get('model')!r}"
        )

    def test_new_tab_state_uses_db_model_not_stale_default(self) -> None:
        """The backend tab state for the new chat must use the DB model,
        not a stale _default_model from server init."""
        server, _events = _make_server()

        server._handle_command({
            "type": "selectModel",
            "model": "fresh-model",
            "tabId": "tab-1",
        })

        server._default_model = "stale-model"

        _record_model_usage("fresh-model")

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

        tab1 = server._get_tab("tab-1")
        tab1.selected_model = "model-A"

        server._handle_command({
            "type": "selectModel",
            "model": "model-B",
            "tabId": "tab-1",
        })

        tab1.selected_model = "model-A"
        server._default_model = "model-A"

        events.clear()

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


class TestFrontendSendsNewChatCommand:
    """Frontend createNewTab() must post a newChat command to the backend."""

    def test_create_new_tab_posts_new_chat(self) -> None:
        """createNewTab() in main.js must include vscode.postMessage({type: 'newChat', ...})."""
        source = MAIN_JS.read_text()

        m = re.search(r"function\s+createNewTab\s*\(\s*\)", source)
        assert m is not None, "createNewTab function not found in main.js"

        start = source.index("{", m.start())
        depth = 0
        end = start
        for i in range(start, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        body = source[start:end]

        assert "newChat" in body, (
            "createNewTab() must post a 'newChat' command to trigger "
            "backend model lookup from DB"
        )

    def test_show_welcome_handler_uses_ev_model(self) -> None:
        """The showWelcome case in main.js must set selectedModel from ev.model."""
        source = MAIN_JS.read_text()

        m = re.search(r"case\s+'showWelcome'\s*:", source)
        assert m is not None, "showWelcome case not found in main.js"

        start = m.start()
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

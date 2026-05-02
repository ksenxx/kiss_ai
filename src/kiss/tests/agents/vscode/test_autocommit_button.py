"""Integration tests for the auto-commit button in the VS Code webview.

Validates:
- The button element exists in the HTML template (SorcarTab.ts).
- The button is placed between parallel-toggle-btn and demo-toggle-btn in #model-picker.
- The button uses the shared dropdown menu-item CSS.
- The JS click handler sends the correct ``autocommitAction`` message.
- The button is disabled when a task is running (setRunningState).
- The backend ``autocommitAction`` command dispatches to ``_handle_autocommit_action``.
"""

from __future__ import annotations

import threading
import unittest
from pathlib import Path

from kiss.agents.vscode.commands import _CommandsMixin
from kiss.agents.vscode.server import VSCodeServer

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


# ===================================================================
# HTML template tests
# ===================================================================


class TestAutocommitButtonInTemplate(unittest.TestCase):
    """The autocommit button exists in the SorcarTab HTML template."""

    def test_button_element_exists(self) -> None:
        html = _read("src/SorcarTab.ts")
        assert 'id="autocommit-btn"' in html, (
            "autocommit-btn button not found in SorcarTab.ts"
        )

    def test_button_has_menu_label(self) -> None:
        html = _read("src/SorcarTab.ts")
        btn_start = html.index('id="autocommit-btn"')
        btn_end = html.index("</button>", btn_start)
        btn_html = html[btn_start:btn_end]
        assert "Auto commit" in btn_html

    def test_button_between_parallel_and_demo(self) -> None:
        """The autocommit button is between parallel-toggle-btn and demo-toggle-btn."""
        html = _read("src/SorcarTab.ts")
        parallel_pos = html.index('id="parallel-toggle-btn"')
        btn_pos = html.index('id="autocommit-btn"')
        demo_pos = html.index('id="demo-toggle-btn"')
        assert parallel_pos < btn_pos < demo_pos, (
            "autocommit-btn should be between parallel-toggle-btn and demo-toggle-btn"
        )

    def test_button_inside_model_picker(self) -> None:
        """The button is a child of the #model-picker div."""
        html = _read("src/SorcarTab.ts")
        picker_start = html.index('id="model-picker"')
        btn_pos = html.index('id="autocommit-btn"')
        assert btn_pos > picker_start, (
            "autocommit-btn should be inside #model-picker"
        )

    def test_button_has_svg_icon(self) -> None:
        """The button contains an SVG icon."""
        html = _read("src/SorcarTab.ts")
        btn_start = html.index('id="autocommit-btn"')
        # Find the closing </button> after the btn
        btn_end = html.index("</button>", btn_start)
        btn_html = html[btn_start:btn_end]
        assert "<svg" in btn_html, "autocommit-btn should contain an SVG icon"


# ===================================================================
# CSS tests
# ===================================================================


class TestAutocommitButtonCSS(unittest.TestCase):
    """The autocommit button uses shared menu-item CSS."""

    def test_base_styles_exist(self) -> None:
        css = _read("media/main.css")
        assert ".menu-item" in css
        assert "#autocommit-btn" not in css

    def test_hover_style_exists(self) -> None:
        css = _read("media/main.css")
        assert ".menu-item:hover:not(:disabled)" in css

    def test_disabled_style_exists(self) -> None:
        css = _read("media/main.css")
        assert ".menu-item:disabled" in css


# ===================================================================
# JavaScript tests
# ===================================================================


class TestAutocommitButtonJS(unittest.TestCase):
    """The JS code references the autocommit button and wires it correctly."""

    def test_element_reference(self) -> None:
        js = _read("media/main.js")
        assert "getElementById('autocommit-btn')" in js

    def test_click_sends_autocommit_action(self) -> None:
        """The click handler posts an autocommitAction message with action 'commit'."""
        js = _read("media/main.js")
        # Find the autocommitBtn click listener
        assert "autocommitBtn.addEventListener('click'" in js or \
               "autocommitBtn.addEventListener(\"click\"" in js
        # Verify it sends the right message type
        click_idx = js.index("autocommitBtn.addEventListener")
        # Find the postMessage call within the next ~300 chars
        snippet = js[click_idx:click_idx + 500]
        assert "type: 'autocommitAction'" in snippet or \
               'type: "autocommitAction"' in snippet
        assert "action: 'commit'" in snippet or \
               'action: "commit"' in snippet

    def test_disabled_when_running(self) -> None:
        """The button is disabled in setRunningState when running."""
        js = _read("media/main.js")
        # Find the setRunningState function
        assert "autocommitBtn" in js
        idx = js.index("function setRunningState")
        # Get the function body (up to next top-level function)
        end = js.index("\n  function ", idx + 1)
        fn_body = js[idx:end]
        assert "autocommitBtn" in fn_body, (
            "setRunningState should reference autocommitBtn"
        )
        assert "autocommitBtn.disabled" in fn_body or \
               "autocommitBtn) autocommitBtn.disabled" in fn_body


# ===================================================================
# Backend dispatch test
# ===================================================================


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

        # Should get autocommit_done (likely "Not a git repository" since
        # /tmp/nonexistent isn't a git repo)
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

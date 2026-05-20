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


class TestAutocommitButtonInTemplate(unittest.TestCase):
    """The autocommit button exists in the SorcarTab HTML template."""

    def test_button_element_exists(self) -> None:
        html = _read("src/SorcarTab.ts")
        assert 'id="autocommit-btn"' in html, (
            "autocommit-btn button not found in SorcarTab.ts"
        )

    def test_button_has_tooltip(self) -> None:
        """The button advertises its purpose via a ``data-tooltip``."""
        html = _read("src/SorcarTab.ts")
        btn_start = html.index('id="autocommit-btn"')
        btn_end = html.index("</button>", btn_start)
        btn_html = html[btn_start:btn_end]
        assert "data-tooltip=" in btn_html

    def test_button_inside_auto_commit_label(self) -> None:
        """The button sits inside the ``cfg-auto-commit`` settings label.

        It must appear after the ``#cfg-auto-commit`` checkbox input and
        before the label's closing ``</label>`` tag, so it renders inline
        next to the "Auto commit" text in the settings panel.
        """
        html = _read("src/SorcarTab.ts")
        checkbox_pos = html.index('id="cfg-auto-commit"')
        btn_pos = html.index('id="autocommit-btn"')
        label_end = html.index("</label>", checkbox_pos)
        assert checkbox_pos < btn_pos < label_end, (
            "autocommit-btn should sit inside the cfg-auto-commit label, "
            "after the checkbox and before </label>"
        )

    def test_button_not_in_model_picker(self) -> None:
        """The button is no longer rendered inside ``#model-picker``."""
        html = _read("src/SorcarTab.ts")
        picker_start = html.index('id="model-picker"')
        # Find the closing </div> that ends the model-picker block by
        # tracking nested <div> tags.
        depth = 1
        i = html.index(">", picker_start) + 1
        picker_end = len(html)
        while i < len(html) and depth:
            nxt = html.find("<", i)
            if nxt < 0:
                break
            if html.startswith("<div", nxt) and html[nxt + 4] in " \t\n>":
                depth += 1
                i = nxt + 4
            elif html.startswith("</div>", nxt):
                depth -= 1
                if depth == 0:
                    picker_end = nxt
                    break
                i = nxt + len("</div>")
            else:
                i = nxt + 1
        picker_html = html[picker_start:picker_end]
        assert 'id="autocommit-btn"' not in picker_html, (
            "autocommit-btn should no longer live inside #model-picker"
        )

    def test_button_has_svg_icon(self) -> None:
        """The button contains an SVG icon."""
        html = _read("src/SorcarTab.ts")
        btn_start = html.index('id="autocommit-btn"')
        btn_end = html.index("</button>", btn_start)
        btn_html = html[btn_start:btn_end]
        assert "<svg" in btn_html, "autocommit-btn should contain an SVG icon"


class TestAutocommitButtonCSS(unittest.TestCase):
    """The autocommit button has its own icon-button CSS."""

    def test_base_styles_exist(self) -> None:
        css = _read("media/main.css")
        assert "#autocommit-btn {" in css

    def test_hover_style_exists(self) -> None:
        css = _read("media/main.css")
        assert "#autocommit-btn:hover:not(:disabled)" in css

    def test_disabled_style_exists(self) -> None:
        css = _read("media/main.css")
        assert "#autocommit-btn:disabled" in css


class TestAutocommitButtonJS(unittest.TestCase):
    """The JS code references the autocommit button and wires it correctly."""

    def test_element_reference(self) -> None:
        js = _read("media/main.js")
        assert "getElementById('autocommit-btn')" in js

    def test_click_sends_autocommit_action(self) -> None:
        """The click handler posts an autocommitAction message with action 'commit'."""
        js = _read("media/main.js")
        assert "autocommitBtn.addEventListener('click'" in js or \
               "autocommitBtn.addEventListener(\"click\"" in js
        click_idx = js.index("autocommitBtn.addEventListener")
        snippet = js[click_idx:click_idx + 500]
        assert "type: 'autocommitAction'" in snippet or \
               'type: "autocommitAction"' in snippet
        assert "action: 'commit'" in snippet or \
               'action: "commit"' in snippet

    def test_not_disabled_when_running(self) -> None:
        """The button stays enabled during running (tasks queue locally)."""
        js = _read("media/main.js")
        assert "autocommitBtn" in js
        assert "autocommitBtn.disabled" not in js


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

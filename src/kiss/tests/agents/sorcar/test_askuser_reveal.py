"""Test that the sidebar is revealed when an askUser event arrives.

The bug: when the user switched to another sidebar panel (e.g., File Explorer)
while a task was running, the askUser modal was rendered in the hidden webview
but the user never saw it. The fix: call this._view.show(true) before
forwarding askUser events to the webview.
"""

from pathlib import Path

import pytest

_TS_PATH = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "src"
    / "SorcarSidebarView.ts"
)

_JS_PATH = (
    Path.home()
    / ".vscode"
    / "extensions"
    / "ksenxx.kiss-sorcar-2026.4.4"
    / "out"
    / "SorcarSidebarView.js"
)


class TestAskUserSidebarReveal:
    """Structural test: _setupProcessListeners reveals the sidebar on askUser."""

    def test_setup_process_listeners_reveals_sidebar_on_askuser(self) -> None:
        """The TS source must call view.show before sendToWebview for askUser."""
        if not _TS_PATH.exists():
            pytest.skip("TS source not available")
        source = _TS_PATH.read_text()

        start = source.index("_setupProcessListeners(proc:")
        # Find the end of the method (next `private ` at indent level 2)
        end = source.find("\n  private ", start + 1)
        if end == -1:
            end = start + 10000
        body = source[start:end]

        assert "msg.type === 'askUser'" in body, (
            "_setupProcessListeners must check for askUser message type"
        )
        assert ".show(true)" in body, (
            "_setupProcessListeners must call view.show(true) for askUser"
        )

        askuser_pos = body.index("msg.type === 'askUser'")
        send_pos = body.index("this._sendToWebview(msg)")
        assert askuser_pos < send_pos, (
            "view.show(true) for askUser must happen before _sendToWebview"
        )

    def test_compiled_js_has_askuser_reveal(self) -> None:
        """The compiled JS must also have the askUser reveal logic."""
        if not _JS_PATH.exists():
            pytest.skip("Compiled extension not available")
        source = _JS_PATH.read_text()
        assert "askUser" in source and ".show(true)" in source, (
            "Compiled JS must contain askUser reveal logic"
        )

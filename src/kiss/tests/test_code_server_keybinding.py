"""Tests for code-server extension Cmd+K toggle focus keybinding."""

from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS
from kiss.agents.sorcar.code_server import _CS_EXTENSION_JS


def test_extension_js_has_toggle_focus_command():
    """Test that the VS Code extension registers kiss.toggleFocus command."""
    assert "kiss.toggleFocus" in _CS_EXTENSION_JS
    assert "registerCommand('kiss.toggleFocus'" in _CS_EXTENSION_JS


def test_extension_js_toggle_focus_calls_focus_chatbox():
    """Test that toggleFocus command calls the /focus-chatbox endpoint."""
    assert "/focus-chatbox" in _CS_EXTENSION_JS


def test_extension_js_polls_for_focus_editor_file():
    """Test that extension polls for pending-focus-editor.json and focuses editor."""
    assert "pending-focus-editor.json" in _CS_EXTENSION_JS
    assert "focusActiveEditorGroup" in _CS_EXTENSION_JS


def test_chatbot_js_posts_to_focus_editor():
    """Test that Cmd+K in chatbox POSTs to /focus-editor instead of direct iframe focus."""
    assert "/focus-editor" in CHATBOT_JS
    assert "frame.contentWindow.focus" not in CHATBOT_JS


def test_chatbot_js_focus_chatbox_calls_window_focus():
    """Test that focus_chatbox SSE handler calls window.focus() before inp.focus()."""
    assert "case'focus_chatbox':window.focus();inp.focus();break;" in CHATBOT_JS


def test_chatbot_js_cmdk_handler_exists():
    """Test that the Cmd+K / Ctrl+K keydown handler exists."""
    assert "e.key==='k'" in CHATBOT_JS
    assert "e.metaKey" in CHATBOT_JS
    assert "e.ctrlKey" in CHATBOT_JS

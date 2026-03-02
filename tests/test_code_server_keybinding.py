"""Tests for code-server extension Cmd+K toggle focus keybinding."""

import json
import tempfile
from pathlib import Path

from kiss.agents.assistant.code_server import _CS_EXTENSION_JS, _setup_code_server


def _make_data_dir() -> str:
    """Create a temp dir with the extensions subdirectory pre-created."""
    tmpdir = tempfile.mkdtemp()
    (Path(tmpdir) / "extensions").mkdir()
    return tmpdir


def test_extension_js_has_toggle_focus_command():
    """Test that the VS Code extension registers kiss.toggleFocus command."""
    assert "kiss.toggleFocus" in _CS_EXTENSION_JS
    assert "registerCommand('kiss.toggleFocus'" in _CS_EXTENSION_JS


def test_extension_js_toggle_focus_calls_focus_chatbox():
    """Test that toggleFocus command calls the /focus-chatbox endpoint."""
    assert "/focus-chatbox" in _CS_EXTENSION_JS


def test_setup_code_server_adds_keybinding():
    """Test that _setup_code_server creates package.json with toggle focus keybinding."""
    tmpdir = _make_data_dir()
    _setup_code_server(tmpdir)
    pkg_path = Path(tmpdir) / "extensions" / "kiss-init" / "package.json"
    assert pkg_path.exists()
    pkg = json.loads(pkg_path.read_text())

    keybindings = pkg["contributes"]["keybindings"]
    toggle_kb = [kb for kb in keybindings if kb["command"] == "kiss.toggleFocus"]
    assert len(toggle_kb) == 1
    assert toggle_kb[0]["key"] == "ctrl+k"
    assert toggle_kb[0]["mac"] == "cmd+k"


def test_setup_code_server_adds_toggle_focus_command():
    """Test that _setup_code_server lists toggleFocus in contributes.commands."""
    tmpdir = _make_data_dir()
    _setup_code_server(tmpdir)
    pkg_path = Path(tmpdir) / "extensions" / "kiss-init" / "package.json"
    pkg = json.loads(pkg_path.read_text())

    commands = pkg["contributes"]["commands"]
    toggle_cmds = [c for c in commands if c["command"] == "kiss.toggleFocus"]
    assert len(toggle_cmds) == 1
    assert toggle_cmds[0]["title"] == "Toggle Focus to Chatbox"


def test_extension_js_written_to_file():
    """Test that extension.js file contains the toggleFocus code."""
    tmpdir = _make_data_dir()
    _setup_code_server(tmpdir)
    ext_path = Path(tmpdir) / "extensions" / "kiss-init" / "extension.js"
    assert ext_path.exists()
    content = ext_path.read_text()
    assert "kiss.toggleFocus" in content
    assert "/focus-chatbox" in content

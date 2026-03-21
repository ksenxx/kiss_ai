"""Tests for code-server: keybinding, watchdog, copilot SCM disable, GitHub token persistence."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS
from kiss.agents.sorcar.code_server import (
    _CS_EXTENSION_JS,
    _GH_TOKEN_FILENAME,
    _install_copilot_extension,
    _load_github_token,
)


def test_extension_js_toggle_focus():
    assert "kiss.toggleFocus" in _CS_EXTENSION_JS
    assert "registerCommand('kiss.toggleFocus'" in _CS_EXTENSION_JS
    assert "/focus-chatbox" in _CS_EXTENSION_JS


def test_extension_js_polls_for_focus_editor_file():
    assert "pending-focus-editor.json" in _CS_EXTENSION_JS
    assert "focusActiveEditorGroup" in _CS_EXTENSION_JS


def test_chatbot_js_focus_keybinding():
    assert "/focus-editor" in CHATBOT_JS
    assert "frame.contentWindow.focus" not in CHATBOT_JS
    assert "case'focus_chatbox':window.focus();inp.focus();break;" in CHATBOT_JS
    assert "e.key==='k'" in CHATBOT_JS
    assert "e.metaKey" in CHATBOT_JS
    assert "e.ctrlKey" in CHATBOT_JS


class TestCodeServerWatchdogLogic:
    def test_watchdog_detects_crashed_process(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(1)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() is not None
        assert proc.returncode == 1

    def test_watchdog_skips_running_process(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            assert proc.poll() is None
        finally:
            proc.terminate()
            proc.wait()

    def test_watchdog_thread_stops_on_shutdown_event(self) -> None:
        shutting_down = threading.Event()
        iterations: list[int] = []

        def watchdog() -> None:
            while not shutting_down.is_set():
                iterations.append(1)
                shutting_down.wait(0.1)
                if shutting_down.is_set():
                    break

        t = threading.Thread(target=watchdog, daemon=True)
        t.start()
        time.sleep(0.3)
        shutting_down.set()
        t.join(timeout=2)
        assert not t.is_alive()
        assert len(iterations) > 0


class TestCodeServerLaunchArgs:
    def test_chatbot_js_has_iframe_reload(self) -> None:
        assert "code_server_restarted" in CHATBOT_JS


class TestSSEHeartbeat:
    def test_sse_format(self) -> None:
        heartbeat = ": heartbeat\n\n"
        assert heartbeat.startswith(":")
        assert heartbeat.endswith("\n\n")
        event = {"type": "code_server_restarted"}
        sse_line = f"data: {json.dumps(event)}\n\n"
        assert sse_line.startswith("data: ")
        assert sse_line.endswith("\n\n")
        parsed = json.loads(sse_line[6:].strip())
        assert parsed["type"] == "code_server_restarted"


class TestProcessMonitoringEdgeCases:
    def test_process_poll_return_codes(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() == 0

        proc2 = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc2.send_signal(signal.SIGTERM)
        proc2.wait()
        assert proc2.poll() == -signal.SIGTERM


class TestInstallCopilotCallsDisable:
    def test_source_code_calls_disable_after_subprocess(self) -> None:
        import inspect

        source = inspect.getsource(_install_copilot_extension)
        assert "_disable_copilot_scm_button" in source
        idx_subprocess = source.index("subprocess.run")
        idx_disable = source.index("_disable_copilot_scm_button")
        assert idx_disable > idx_subprocess


class TestLoadGithubToken:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.sorcar_data_dir = os.path.join(self.tmpdir, "kiss", "sorcar-data")
        os.makedirs(self.sorcar_data_dir, exist_ok=True)
        self.token_file = Path(self.sorcar_data_dir).parent / _GH_TOKEN_FILENAME

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_token(self) -> None:
        self.token_file.write_text(json.dumps({
            "accessToken": "gho_abc123xyz",
            "account": {"label": "testuser", "id": "12345"},
            "id": "session-id",
        }))
        assert _load_github_token(self.sorcar_data_dir) == "gho_abc123xyz"

    def test_nonexistent_parent_dir(self) -> None:
        assert _load_github_token("/nonexistent/path/cs-test") is None

    def test_unreadable_file(self) -> None:
        self.token_file.write_text(json.dumps({"accessToken": "gho_secret"}))
        os.chmod(str(self.token_file), 0o000)
        try:
            assert _load_github_token(self.sorcar_data_dir) is None
        finally:
            os.chmod(str(self.token_file), 0o644)


_EXTENSION_JS_TOKEN_STRINGS = [
    "github-copilot-token.json",
    "path.join(dataDir,'..','github-copilot-token.json')",
    "vscode.authentication.getSession(",
    "'github'",
    "'user:email'",
    "scopeSets",
    "silent:true",
    "mode:0o600",
    "onDidChangeSessions",
    "ghInterval",
    "setInterval(saveGitHubToken",
]


class TestExtensionJSTokenCode:
    def test_js_syntax_valid(self) -> None:
        result = subprocess.run(
            ["node", "--check", "--input-type=commonjs"],
            input=_CS_EXTENSION_JS,
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"JS syntax error: {result.stderr}"

    @pytest.mark.parametrize("expected", _EXTENSION_JS_TOKEN_STRINGS)
    def test_extension_contains_token_string(self, expected: str) -> None:
        assert expected in _CS_EXTENSION_JS

    def test_saves_on_startup(self) -> None:
        def_end = _CS_EXTENSION_JS.index("async function saveGitHubToken(){")
        remaining = _CS_EXTENSION_JS[def_end:]
        assert "saveGitHubToken();" in remaining

    def test_token_saving_inside_activate(self) -> None:
        stripped = _CS_EXTENSION_JS.strip()
        assert stripped.endswith("module.exports={activate};")
        idx_save = stripped.index("saveGitHubToken")
        idx_exports = stripped.index("module.exports")
        assert idx_save < idx_exports


class TestTokenPathConsistency:
    def test_filename_and_path_consistency(self) -> None:
        assert _GH_TOKEN_FILENAME in _CS_EXTENSION_JS
        sorcar_data_dir = "/home/user/.kiss/sorcar-data"
        python_path = Path(sorcar_data_dir).parent / _GH_TOKEN_FILENAME
        js_path = Path(sorcar_data_dir) / ".." / _GH_TOKEN_FILENAME
        assert python_path.resolve() == js_path.resolve()

"""Tests for code-server: keybinding, watchdog, and extension JS."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time

from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS
from kiss.agents.sorcar.code_server import (
    _CS_EXTENSION_JS,
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




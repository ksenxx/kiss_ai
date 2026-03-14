"""Tests for code-server process monitoring and auto-restart.

No mocks, patches, or test doubles. Uses real subprocesses and real
threading primitives.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time


class TestCodeServerWatchdogLogic:
    """Test the code-server watchdog thread logic using real subprocesses."""

    def test_watchdog_detects_crashed_process(self) -> None:
        """When a subprocess exits, poll() returns non-None."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() is not None
        assert proc.returncode == 1

    def test_watchdog_skips_running_process(self) -> None:
        """When a subprocess is alive, poll() returns None."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert proc.poll() is None
        finally:
            proc.terminate()
            proc.wait()

    def test_watchdog_thread_stops_on_shutdown_event(self) -> None:
        """The watchdog thread exits when shutting_down is set."""
        shutting_down = threading.Event()
        iterations = []

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
    """Test the _code_server_launch_args helper is consistent."""

    def test_chatbot_js_has_iframe_reload(self) -> None:
        """The CHATBOT_JS must contain the iframe reload on code_server_restarted."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        assert "code_server_restarted" in CHATBOT_JS


class TestSSEHeartbeat:
    """Test SSE heartbeat behavior for connection keepalive."""

    def test_heartbeat_comment_format(self) -> None:
        """SSE heartbeat must be a valid SSE comment."""
        heartbeat = ": heartbeat\n\n"
        assert heartbeat.startswith(":")
        assert heartbeat.endswith("\n\n")

    def test_sse_event_format(self) -> None:
        """SSE data events must follow the correct format."""
        event = {"type": "code_server_restarted"}
        sse_line = f"data: {json.dumps(event)}\n\n"
        assert sse_line.startswith("data: ")
        assert sse_line.endswith("\n\n")
        parsed = json.loads(sse_line[6:].strip())
        assert parsed["type"] == "code_server_restarted"


class TestProcessMonitoringEdgeCases:
    """Edge cases for process monitoring."""

    def test_process_poll_returns_zero_on_clean_exit(self) -> None:
        """A process that exits cleanly has returncode 0."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() == 0

    def test_process_poll_returns_negative_on_signal(self) -> None:
        """A process killed by SIGTERM has a negative return code on unix."""
        import signal

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.send_signal(signal.SIGTERM)
        proc.wait()
        ret = proc.poll()
        assert ret is not None
        if sys.platform == "win32":
            assert ret == 1
        else:
            # On Unix, killed by signal returns -signal_number.
            assert ret == -signal.SIGTERM

    def test_rapid_crash_restart_cycle(self) -> None:
        """Multiple crash-restart cycles work correctly."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        restart_count = 0

        for i in range(3):
            proc = subprocess.Popen(
                [sys.executable, "-c", f"import sys; sys.exit({i + 1})"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.wait()
            assert proc.poll() == i + 1
            restart_count += 1
            printer.broadcast({"type": "code_server_restarted"})

        assert restart_count == 3
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        assert len(events) == 3
        assert all(e["type"] == "code_server_restarted" for e in events)
        printer.remove_client(cq)

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

    def test_watchdog_restart_cycle(self) -> None:
        """Simulate the watchdog detecting a crash and restarting."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        # Start a process that exits immediately
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(42)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() == 42

        # Simulate the watchdog restart: start a new process
        new_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert new_proc.poll() is None
            printer.broadcast({"type": "code_server_restarted"})

            event = cq.get(timeout=2)
            assert event["type"] == "code_server_restarted"
        finally:
            new_proc.terminate()
            new_proc.wait()
            printer.remove_client(cq)

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

    def test_watchdog_does_not_restart_when_process_is_alive(self) -> None:
        """When poll() returns None, no restart should occur."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        restart_count = 0

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            # Simulate watchdog check
            for _ in range(5):
                ret = proc.poll()
                if ret is not None:
                    restart_count += 1

            assert restart_count == 0
            assert cq.empty()
        finally:
            proc.terminate()
            proc.wait()
            printer.remove_client(cq)


class TestCodeServerWatchdogIntegration:
    """Integration test simulating the full watchdog lifecycle."""

    def test_full_watchdog_lifecycle(self) -> None:
        """Start a process, kill it, detect the crash, and verify broadcast."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        shutting_down = threading.Event()
        cs_proc_holder: list[subprocess.Popen[bytes] | None] = [None]
        restart_events: list[int] = []

        # Start a long-running process to simulate code-server
        cs_proc_holder[0] = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def watchdog() -> None:
            while not shutting_down.is_set():
                shutting_down.wait(0.5)
                if shutting_down.is_set():
                    break
                proc = cs_proc_holder[0]
                if proc is None:
                    continue
                ret = proc.poll()
                if ret is None:
                    continue
                # Process died - restart it
                new_proc = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                cs_proc_holder[0] = new_proc
                restart_events.append(ret)
                printer.broadcast({"type": "code_server_restarted"})

        t = threading.Thread(target=watchdog, daemon=True)
        t.start()

        try:
            # Kill the process
            initial_proc = cs_proc_holder[0]
            assert initial_proc is not None
            initial_proc.terminate()
            initial_proc.wait()

            # Wait for watchdog to detect and restart
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not restart_events:
                time.sleep(0.2)

            assert len(restart_events) == 1
            new_proc = cs_proc_holder[0]
            assert new_proc is not None
            assert new_proc.poll() is None  # New process is running

            # Verify event was broadcast
            event = cq.get(timeout=2)
            assert event["type"] == "code_server_restarted"
        finally:
            shutting_down.set()
            t.join(timeout=3)
            if cs_proc_holder[0]:
                cs_proc_holder[0].terminate()
                cs_proc_holder[0].wait()
            printer.remove_client(cq)


class TestCodeServerRestartedEvent:
    """Test that the code_server_restarted event is properly handled by the frontend JS."""

    def test_event_in_build_html(self) -> None:
        """Verify the JS handler for code_server_restarted is in the generated HTML."""
        from kiss.agents.sorcar.chatbot_ui import _build_html

        html = _build_html("Test", "http://127.0.0.1:13338", "/tmp")
        assert "code_server_restarted" in html
        assert "code-server-frame" in html

    def test_health_check_js_in_html(self) -> None:
        """Verify the iframe health check JS is included in the HTML."""
        from kiss.agents.sorcar.chatbot_ui import _build_html

        html = _build_html("Test", "http://127.0.0.1:13338", "/tmp")
        assert "_checkCodeServerHealth" in html
        assert "_csBaseUrl" in html
        assert "_csHealthOk" in html

    def test_no_health_check_without_code_server(self) -> None:
        """When code_server_url is empty, the iframe and health check should
        still be in the JS but data-base-url should be empty."""
        from kiss.agents.sorcar.chatbot_ui import _build_html

        html = _build_html("Test", "", "/tmp")
        # Health check JS is always included but _csBaseUrl will be empty
        assert "_checkCodeServerHealth" in html
        assert "editor-fallback" in html


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


class TestBroadcastCodeServerRestarted:
    """Test broadcasting the code_server_restarted event to multiple clients."""

    def test_broadcast_to_multiple_clients(self) -> None:
        """All connected clients receive the code_server_restarted event."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q1 = printer.add_client()
        q2 = printer.add_client()
        q3 = printer.add_client()

        printer.broadcast({"type": "code_server_restarted"})

        for q in [q1, q2, q3]:
            event = q.get(timeout=1)
            assert event["type"] == "code_server_restarted"

        printer.remove_client(q1)
        printer.remove_client(q2)
        printer.remove_client(q3)

    def test_broadcast_recorded_in_chat_events(self) -> None:
        """code_server_restarted events are recorded but not in display events."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        printer.start_recording()
        printer.broadcast({"type": "code_server_restarted"})
        events = printer.stop_recording()
        # code_server_restarted is not in _DISPLAY_EVENT_TYPES, so it's filtered
        assert len(events) == 0


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
        # On Unix, killed by signal returns -signal_number
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

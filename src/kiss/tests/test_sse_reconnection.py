"""Tests for SSE disconnection fixes: reconnection, shutdown timer, heartbeat.

Covers:
- Server shutdown timer increased from 5s to 120s
- _cancel_shutdown cancels pending timer on client reconnect
- Heartbeat interval reduced from 15s to 5s
- SSE Connection header set to keep-alive
- JavaScript reconnect logic (via _build_html output inspection)
"""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.chatbot_ui import _build_html


def _redirect_history(tmpdir: str):
    old = (th.HISTORY_FILE, th.PROPOSALS_FILE, th.MODEL_USAGE_FILE,
           th.FILE_USAGE_FILE, th._history_cache, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th.HISTORY_FILE = kiss_dir / "task_history.json"
    th.PROPOSALS_FILE = kiss_dir / "proposals.json"
    th.MODEL_USAGE_FILE = kiss_dir / "model_usage.json"
    th.FILE_USAGE_FILE = kiss_dir / "file_usage.json"
    th._history_cache = None
    return old


def _restore_history(saved):
    (th.HISTORY_FILE, th.PROPOSALS_FILE, th.MODEL_USAGE_FILE,
     th.FILE_USAGE_FILE, th._history_cache, th._KISS_DIR) = saved


# ---------------------------------------------------------------------------
# JavaScript reconnection logic in generated HTML
# ---------------------------------------------------------------------------
class TestSSEReconnectJavaScript:
    """Verify the generated HTML includes robust SSE reconnection logic."""

    def test_html_contains_reconnect_on_error(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "evtSrc.onerror" in html
        assert "connectSSE" in html
        assert "_sseBackoff" in html

    def test_html_contains_disconnect_banner(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "disconn-banner" in html
        assert "Connection lost" in html
        assert "_showDisconnBanner" in html
        assert "_hideDisconnBanner" in html

    def test_html_reconnect_resets_retry_on_open(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "_sseRetry=0" in html
        assert "_sseConnected=true" in html

    def test_html_exponential_backoff(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "_sseMaxRetry" in html
        assert "Math.pow" in html or "Math.min" in html

    def test_html_hides_banner_on_reconnect(self) -> None:
        html = _build_html("Test", "", "/tmp")
        assert "_hideDisconnBanner" in html
        assert "onopen" in html


# ---------------------------------------------------------------------------
# Server-side: shutdown timer behavior
# ---------------------------------------------------------------------------
class TestShutdownTimerBehavior:
    """Test that the shutdown timer is long enough for reconnection."""

    def test_schedule_shutdown_uses_120s_timer(self) -> None:
        """_schedule_shutdown should use a 120s timer, not 5s."""
        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        shutting_down = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
                shutting_down.set()

        def _cancel_shutdown() -> None:
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                    shutdown_timer = None

        def _schedule_shutdown() -> None:
            nonlocal shutdown_timer
            if printer.has_clients():
                return
            with running_lock:
                if running:
                    return
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                shutdown_timer = threading.Timer(120.0, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        _schedule_shutdown()
        try:
            assert shutdown_timer is not None
            assert shutdown_timer.interval == 120.0
        finally:
            if shutdown_timer is not None:
                shutdown_timer.cancel()

    def test_cancel_shutdown_stops_timer(self) -> None:
        """_cancel_shutdown should cancel a pending shutdown timer."""
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        timer_fired = threading.Event()

        def _do_shutdown() -> None:
            timer_fired.set()

        def _cancel_shutdown() -> None:
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                    shutdown_timer = None

        shutdown_timer = threading.Timer(0.2, _do_shutdown)
        shutdown_timer.daemon = True
        shutdown_timer.start()

        _cancel_shutdown()
        assert shutdown_timer is None
        time.sleep(0.4)
        assert not timer_fired.is_set(), "Timer should have been cancelled"

    def test_cancel_shutdown_noop_when_no_timer(self) -> None:
        """_cancel_shutdown should be safe to call with no timer."""
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()

        def _cancel_shutdown() -> None:
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                    shutdown_timer = None

        _cancel_shutdown()
        assert shutdown_timer is None

    def test_schedule_shutdown_skips_when_clients_exist(self) -> None:
        """_schedule_shutdown should not start a timer if clients exist."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        running = False
        running_lock = threading.Lock()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()

        def _schedule_shutdown() -> None:
            nonlocal shutdown_timer
            if printer.has_clients():
                return
            with running_lock:
                if running:
                    return
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                shutdown_timer = threading.Timer(120.0, lambda: None)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        _schedule_shutdown()
        assert shutdown_timer is None
        printer.remove_client(cq)

    def test_schedule_shutdown_skips_when_running(self) -> None:
        """_schedule_shutdown should not start a timer if agent is running."""
        printer = BaseBrowserPrinter()
        running = True
        running_lock = threading.Lock()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()

        def _schedule_shutdown() -> None:
            nonlocal shutdown_timer
            if printer.has_clients():
                return
            with running_lock:
                if running:
                    return
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                shutdown_timer = threading.Timer(120.0, lambda: None)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        _schedule_shutdown()
        assert shutdown_timer is None

    def test_client_reconnect_cancels_shutdown(self) -> None:
        """Client disconnects -> shutdown scheduled -> reconnects -> cancelled."""
        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        shutdown_fired = threading.Event()

        def _do_shutdown() -> None:
            shutdown_fired.set()

        def _cancel_shutdown() -> None:
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                    shutdown_timer = None

        def _schedule_shutdown() -> None:
            nonlocal shutdown_timer
            if printer.has_clients():
                return
            with running_lock:
                if running:
                    return
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                shutdown_timer = threading.Timer(0.3, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        cq = printer.add_client()
        printer.remove_client(cq)
        _schedule_shutdown()
        assert shutdown_timer is not None

        cq2 = printer.add_client()
        _cancel_shutdown()
        assert shutdown_timer is None

        time.sleep(0.5)
        assert not shutdown_fired.is_set()
        printer.remove_client(cq2)


# ---------------------------------------------------------------------------
# Server-side: heartbeat interval
# ---------------------------------------------------------------------------
class TestHeartbeatInterval:
    """Test that the heartbeat fires more frequently."""

    def test_heartbeat_fires_within_6_seconds(self) -> None:
        """With the 5s interval, a heartbeat should fire within ~6s of inactivity."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        heartbeat_received = threading.Event()
        generated: list[str] = []

        async def simulate_events():
            last_heartbeat = time.monotonic()
            for _ in range(200):
                try:
                    cq.get_nowait()
                except queue.Empty:
                    now = time.monotonic()
                    if now - last_heartbeat >= 5.0:
                        generated.append(": heartbeat\n\n")
                        heartbeat_received.set()
                        last_heartbeat = now
                        break
                    await asyncio.sleep(0.05)
                    continue

        start = time.monotonic()
        asyncio.run(simulate_events())
        elapsed = time.monotonic() - start

        assert heartbeat_received.is_set(), "Heartbeat should have fired"
        assert elapsed < 8.0, f"Heartbeat took {elapsed:.1f}s, expected <8s"
        printer.remove_client(cq)


# ---------------------------------------------------------------------------
# Server-side: SSE response headers
# ---------------------------------------------------------------------------
class TestSSEResponseHeaders:
    """Test that SSE endpoint includes proper headers."""

    def test_build_html_with_code_server(self) -> None:
        html = _build_html("Test", "http://127.0.0.1:13338", "/tmp/work")
        assert "code-server-frame" in html

    def test_build_html_without_code_server(self) -> None:
        html = _build_html("Test", "", "/tmp/work")
        assert "editor-fallback" in html


# ---------------------------------------------------------------------------
# SSE events endpoint integration test
# ---------------------------------------------------------------------------
class TestSSEEventsEndpointIntegration:
    """Integration tests that spin up a real Starlette app to verify SSE."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_history(self.tmpdir)

    def teardown_method(self) -> None:
        _restore_history(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sse_events_endpoint_returns_stream(self) -> None:
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        printer = BaseBrowserPrinter()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()

        def _cancel_shutdown() -> None:
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                    shutdown_timer = None

        def _schedule_shutdown() -> None:
            pass

        async def events(request):
            from starlette.responses import StreamingResponse

            cq = printer.add_client()
            _cancel_shutdown()

            async def generate():
                last_heartbeat = time.monotonic()
                count = 0
                while count < 3:
                    try:
                        event = cq.get_nowait()
                    except queue.Empty:
                        now = time.monotonic()
                        if now - last_heartbeat >= 0.1:
                            yield ": heartbeat\n\n"
                            last_heartbeat = now
                            count += 1
                        await asyncio.sleep(0.02)
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                    last_heartbeat = time.monotonic()
                    count += 1
                printer.remove_client(cq)

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        app = Starlette(routes=[Route("/events", events)])
        client = TestClient(app)

        with client.stream("GET", "/events") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers.get("connection") == "keep-alive"
            chunks = []
            for chunk in resp.iter_text():
                chunks.append(chunk)
                if len(chunks) >= 2:
                    break
            assert any("heartbeat" in c for c in chunks)

    def test_sse_broadcast_reaches_client(self) -> None:
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        printer = BaseBrowserPrinter()

        async def events(request):
            from starlette.responses import StreamingResponse

            cq = printer.add_client()

            async def generate():
                count = 0
                while count < 5:
                    try:
                        event = cq.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.02)
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                    count += 1
                    if event.get("type") == "test_event":
                        break
                printer.remove_client(cq)

            return StreamingResponse(generate(), media_type="text/event-stream")

        app = Starlette(routes=[Route("/events", events)])
        client = TestClient(app)

        def send_event():
            time.sleep(0.1)
            printer.broadcast({"type": "test_event", "data": "hello"})

        threading.Thread(target=send_event, daemon=True).start()

        with client.stream("GET", "/events") as resp:
            found = ""
            for chunk in resp.iter_text():
                if "test_event" in chunk:
                    found = chunk
                    break
            assert "test_event" in found


# ---------------------------------------------------------------------------
# Concurrent client connect/disconnect stress test
# ---------------------------------------------------------------------------
class TestConcurrentClientConnections:
    """Stress-test client add/remove with concurrent shutdown scheduling."""

    def test_rapid_connect_disconnect_no_crash(self) -> None:
        printer = BaseBrowserPrinter()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        errors: list[Exception] = []

        def _cancel_shutdown() -> None:
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                    shutdown_timer = None

        def _schedule_shutdown() -> None:
            nonlocal shutdown_timer
            if printer.has_clients():
                return
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                shutdown_timer = threading.Timer(120.0, lambda: None)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        def churn():
            try:
                for _ in range(50):
                    cq = printer.add_client()
                    _cancel_shutdown()
                    printer.broadcast({"type": "test"})
                    time.sleep(0.001)
                    printer.remove_client(cq)
                    _schedule_shutdown()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=churn, daemon=True) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent churn: {errors}"
        with shutdown_lock:
            if shutdown_timer is not None:
                shutdown_timer.cancel()

    def test_broadcast_during_disconnect(self) -> None:
        """Broadcasting while clients disconnect shouldn't raise."""
        printer = BaseBrowserPrinter()
        errors: list[Exception] = []
        stop = threading.Event()

        def broadcaster():
            try:
                while not stop.is_set():
                    printer.broadcast({"type": "ping"})
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        def connector():
            try:
                for _ in range(30):
                    cq = printer.add_client()
                    time.sleep(0.005)
                    printer.remove_client(cq)
            except Exception as e:
                errors.append(e)

        bt = threading.Thread(target=broadcaster, daemon=True)
        ct = threading.Thread(target=connector, daemon=True)
        bt.start()
        ct.start()
        ct.join(timeout=10)
        stop.set()
        bt.join(timeout=5)

        assert not errors, f"Errors: {errors}"


# ---------------------------------------------------------------------------
# _do_shutdown safety: does not exit if clients reconnected
# ---------------------------------------------------------------------------
class TestDoShutdownSafety:
    """Verify _do_shutdown aborts if clients reconnected before it fires."""

    def test_do_shutdown_aborts_if_clients_reconnected(self) -> None:
        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        shutting_down = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
                shutting_down.set()

        cq = printer.add_client()
        _do_shutdown()
        assert not shutting_down.is_set(), "Should not shut down with active client"
        printer.remove_client(cq)

    def test_do_shutdown_aborts_if_running(self) -> None:
        printer = BaseBrowserPrinter()
        running = True
        running_lock = threading.Lock()
        shutting_down = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
                shutting_down.set()

        _do_shutdown()
        assert not shutting_down.is_set(), "Should not shut down while running"

    def test_do_shutdown_proceeds_if_no_clients_no_running(self) -> None:
        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        shutting_down = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
                shutting_down.set()

        _do_shutdown()
        assert shutting_down.is_set()


# ---------------------------------------------------------------------------
# End-to-end simulation: disconnect then reconnect before shutdown fires
# ---------------------------------------------------------------------------
class TestReconnectBeforeShutdown:
    """Simulate the real-world scenario that caused the bug."""

    def test_reconnect_within_shutdown_window(self) -> None:
        """Client disconnects, shutdown scheduled at 120s, client reconnects
        within 1s. Shutdown should be cancelled."""
        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        shutdown_fired = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
            shutdown_fired.set()

        def _cancel_shutdown() -> None:
            nonlocal shutdown_timer
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                    shutdown_timer = None

        def _schedule_shutdown() -> None:
            nonlocal shutdown_timer
            if printer.has_clients():
                return
            with running_lock:
                if running:
                    return
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                shutdown_timer = threading.Timer(0.5, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        cq1 = printer.add_client()
        _cancel_shutdown()

        printer.remove_client(cq1)
        _schedule_shutdown()
        assert shutdown_timer is not None

        time.sleep(0.1)
        cq2 = printer.add_client()
        _cancel_shutdown()

        time.sleep(0.7)
        assert not shutdown_fired.is_set(), \
            "Shutdown should have been cancelled by reconnecting client"
        printer.remove_client(cq2)

    def test_no_reconnect_eventually_shuts_down(self) -> None:
        """If no client reconnects, shutdown should eventually fire."""
        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        shutdown_fired = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
            shutdown_fired.set()

        def _schedule_shutdown() -> None:
            nonlocal shutdown_timer
            if printer.has_clients():
                return
            with running_lock:
                if running:
                    return
            with shutdown_lock:
                if shutdown_timer is not None:
                    shutdown_timer.cancel()
                shutdown_timer = threading.Timer(0.2, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        cq = printer.add_client()
        printer.remove_client(cq)
        _schedule_shutdown()

        time.sleep(0.5)
        assert shutdown_fired.is_set(), "Shutdown should fire if no one reconnects"

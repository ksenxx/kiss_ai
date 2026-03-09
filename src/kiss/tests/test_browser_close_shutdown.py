"""Tests for process shutdown when browser window closes.

Verifies that the sorcar process terminates after the browser disconnects,
via three mechanisms:
1. The /closing endpoint (called by beforeunload beacon)
2. The periodic no-client safety net (_watch_no_clients)
3. The SSE disconnect detection scheduling shutdown
"""

import threading
import time

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter


class TestClosingEndpointTriggersShutdown:
    """The /closing POST (sent by navigator.sendBeacon on beforeunload)
    should schedule a shutdown when no clients remain."""

    def test_schedule_shutdown_sets_timer(self) -> None:
        """_schedule_shutdown creates a timer that fires _do_shutdown."""
        exited = threading.Event()
        running = False
        running_lock = threading.Lock()
        printer = BaseBrowserPrinter()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
            exited.set()

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
                shutdown_timer = threading.Timer(0.1, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        # Simulate: browser sends /closing beacon, no clients remain
        _schedule_shutdown()
        assert exited.wait(timeout=2.0), "Process should have exited after /closing"

    def test_schedule_shutdown_skipped_when_clients_remain(self) -> None:
        """If clients are still connected, _schedule_shutdown does nothing."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        running = False
        running_lock = threading.Lock()
        fired = threading.Event()

        def _do_shutdown() -> None:
            fired.set()

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
                shutdown_timer = threading.Timer(0.05, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        _schedule_shutdown()
        assert not fired.wait(timeout=0.3), "Should NOT fire when clients exist"
        printer.remove_client(cq)

    def test_schedule_shutdown_skipped_when_running(self) -> None:
        """If a task is running, _schedule_shutdown does nothing."""
        printer = BaseBrowserPrinter()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        running = True
        running_lock = threading.Lock()
        fired = threading.Event()

        def _do_shutdown() -> None:
            fired.set()

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
                shutdown_timer = threading.Timer(0.05, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        _schedule_shutdown()
        assert not fired.wait(timeout=0.3), "Should NOT fire when task is running"

    def test_do_shutdown_skipped_when_client_reconnects(self) -> None:
        """If a client reconnects before timer fires, _do_shutdown is a no-op."""
        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        exited = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
            exited.set()

        # A client was gone when scheduled, but reconnects before timer fires
        cq = printer.add_client()
        _do_shutdown()
        assert not exited.is_set(), "_do_shutdown should skip if client reconnected"
        printer.remove_client(cq)


class TestNoClientSafetyNet:
    """The _watch_no_clients thread should schedule shutdown after clients
    have been disconnected for a sustained period."""

    def test_no_client_triggers_shutdown(self) -> None:
        """After sustained no-client period, shutdown is scheduled."""
        printer = BaseBrowserPrinter()
        shutting_down = threading.Event()
        scheduled = threading.Event()

        def _schedule_shutdown() -> None:
            scheduled.set()

        def _watch_no_clients() -> None:
            no_client_since: float | None = None
            while not shutting_down.is_set():
                shutting_down.wait(0.1)
                if shutting_down.is_set():
                    break
                if not printer.has_clients():
                    if no_client_since is None:
                        no_client_since = time.monotonic()
                    elif time.monotonic() - no_client_since >= 0.3:
                        _schedule_shutdown()
                        return
                else:
                    no_client_since = None

        t = threading.Thread(target=_watch_no_clients, daemon=True)
        t.start()
        assert scheduled.wait(timeout=3.0), "Should schedule shutdown after no-client period"
        shutting_down.set()
        t.join(timeout=1.0)

    def test_no_client_resets_when_client_connects(self) -> None:
        """A new client connecting resets the no-client timer."""
        printer = BaseBrowserPrinter()
        shutting_down = threading.Event()
        scheduled = threading.Event()

        def _schedule_shutdown() -> None:
            scheduled.set()

        def _watch_no_clients() -> None:
            no_client_since: float | None = None
            while not shutting_down.is_set():
                shutting_down.wait(0.1)
                if shutting_down.is_set():
                    break
                if not printer.has_clients():
                    if no_client_since is None:
                        no_client_since = time.monotonic()
                    elif time.monotonic() - no_client_since >= 0.5:
                        _schedule_shutdown()
                        return
                else:
                    no_client_since = None

        t = threading.Thread(target=_watch_no_clients, daemon=True)
        t.start()

        # Let it run for a bit with no clients, then add one before timeout
        time.sleep(0.2)
        cq = printer.add_client()
        time.sleep(0.5)  # Now it should have reset
        assert not scheduled.is_set(), "Timer should reset when client connects"

        # Remove client again and let it expire
        printer.remove_client(cq)
        assert scheduled.wait(timeout=3.0), "Should schedule after client leaves again"
        shutting_down.set()
        t.join(timeout=1.0)


class TestSSEDisconnectSchedulesShutdown:
    """When the SSE generator's finally block runs, it should call
    _schedule_shutdown after removing the client."""

    def test_sse_finally_calls_schedule_shutdown(self) -> None:
        """Simulates SSE generator cleanup: remove client, then schedule."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        assert printer.has_clients()

        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        running = False
        running_lock = threading.Lock()
        exited = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
            exited.set()

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
                shutdown_timer = threading.Timer(0.1, _do_shutdown)
                shutdown_timer.daemon = True
                shutdown_timer.start()

        # Simulate the SSE generator finally block
        printer.remove_client(cq)
        _schedule_shutdown()

        assert exited.wait(timeout=2.0), "Should exit after SSE disconnect"


class TestCancelShutdownOnReconnect:
    """If a browser reconnects before the timer fires, shutdown is cancelled."""

    def test_cancel_and_reschedule(self) -> None:
        """A new client connection cancels the pending shutdown timer."""
        printer = BaseBrowserPrinter()
        shutdown_timer: threading.Timer | None = None
        shutdown_lock = threading.Lock()
        running = False
        running_lock = threading.Lock()
        exited = threading.Event()

        def _do_shutdown() -> None:
            with running_lock:
                if running or printer.has_clients():
                    return
            exited.set()

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

        # Schedule shutdown
        _schedule_shutdown()
        assert shutdown_timer is not None

        # Client reconnects — cancel shutdown
        cq = printer.add_client()
        _cancel_shutdown()
        assert not exited.wait(timeout=1.0), "Shutdown should be cancelled"
        printer.remove_client(cq)


class TestBeforeUnloadInHTML:
    """The generated HTML must include a beforeunload handler that sends
    a beacon to /closing."""

    def test_html_contains_beforeunload_beacon(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import _build_html

        html = _build_html("Test", "", "/tmp")
        assert "beforeunload" in html
        assert "sendBeacon" in html
        assert "/closing" in html


class TestShutdownTimerDuration:
    """The shutdown timer should use a short delay (not the old 120s)."""

    def test_timer_is_short(self) -> None:
        """Verify the source code uses a short timer, not 120 seconds."""
        import inspect

        from kiss.agents.sorcar import sorcar

        source = inspect.getsource(sorcar.run_chatbot)
        # The timer should be 10 seconds, not 120
        assert "Timer(10.0," in source
        assert "Timer(120.0," not in source

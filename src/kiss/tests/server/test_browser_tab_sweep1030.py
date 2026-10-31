# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `daemon` / `page_server` fixtures are imported from
# the service test module and re-exported for pytest)
"""The daemon's Browser tab after the 2026-10-30 sweep.

1. ``open_for_agent`` / ``open_for_user`` run on an agent's task thread
   and waited with one ``future.result(120)``: a C-level wait the task
   runner's Stop (an injected ``KeyboardInterrupt``) cannot interrupt.
   Stop during a browser launch (or behind a slow close or teardown)
   waited out the whole open, and the interrupt then fired right after
   it, leaving the focused tab it had just opened on every surface with
   nobody to use it.  The wait is now sliced, and an abandoned open
   closes its tab (never focusing it) instead of handing it over.
2. Address-bar input that is an IP address (``192.168.1.1``,
   ``127.0.0.1:8000/app``, ``[::1]:3000``) went to a Google search: only
   names with a top-level domain and ``localhost`` counted as hosts.

The Stop is the production one (``inject_async_exception``, what the
task runner's Stop calls), aimed at a real thread blocked in the real
service.  The open is held back by keeping the service's lifecycle lock,
as a launch, close or teardown in progress does.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import http.server
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.core.tool_interrupt import inject_async_exception
from kiss.server.browser_tab import BrowserTabService, _result_in_slices, normalize_url
from kiss.tests.server.test_browser_tab_service import (
    _PLAYWRIGHT_CACHE,
    ConnPrinter,
    _events,
    _wait,
    daemon,  # noqa: F401 - pytest fixture
    page_server,  # noqa: F401 - pytest fixture
)

_HOLD = 3.0
"""Seconds the tests keep the service's lifecycle lock after the Stop."""

_SLICE_MARGIN = 1.5
"""How soon after the Stop the task thread must have left the service."""


class _Caller(threading.Thread):
    """A task thread making one blocking open call; records how it ended."""

    def __init__(self, call: Callable[[], object]) -> None:
        super().__init__(daemon=True)
        self.call = call
        self.error: BaseException | None = None
        self.ended_at = 0.0

    def run(self) -> None:
        try:
            self.call()
        except BaseException as exc:  # noqa: BLE001 - the injected Stop is the outcome
            self.error = exc
        self.ended_at = time.monotonic()


async def _hold_lifecycle_lock(service: BrowserTabService, seconds: float) -> None:
    """Keep the service's lifecycle lock, as a launch, close or teardown does."""
    async with service._launch_lock:
        await asyncio.sleep(seconds)


def _stop_while_queued(
    service: BrowserTabService, call: Callable[[], object],
) -> concurrent.futures.Future[None]:
    """Stop a task thread whose *call* waits behind the held lifecycle lock.

    Returns the future of the hold (done :data:`_HOLD` seconds after the
    Stop), once the thread has ended with the injected interrupt.
    """
    loop = service._ensure_loop()
    held = asyncio.run_coroutine_threadsafe(_hold_lifecycle_lock(service, _HOLD), loop)
    _wait(service._launch_lock.locked, "the lock held")
    caller = _Caller(call)
    caller.start()
    _wait(lambda: service._launch_lock._waiters, "the open queued behind the lock")
    stopped_at = time.monotonic()
    assert caller.ident is not None
    assert inject_async_exception(caller.ident, KeyboardInterrupt) == 1
    caller.join(_HOLD + 120)
    assert isinstance(caller.error, KeyboardInterrupt), caller.error
    assert caller.ended_at - stopped_at < _SLICE_MARGIN, "the Stop waited for the open"
    assert not held.done(), "the thread left before the open could go on"
    return held


def test_stop_reaches_an_agent_waiting_for_the_browser(tmp_path: Path) -> None:
    service = BrowserTabService(ConnPrinter(), tmp_path / "profile")
    try:
        _stop_while_queued(service, service.open_for_agent)
    finally:
        # Shut down while the open is still queued: it then finds the
        # service closed and launches nothing.
        service.shutdown()


def _assert_abandoned_tab_closed(service: BrowserTabService, printer: ConnPrinter) -> None:
    """The open went on after the Stop; its tab was closed again, never focused."""
    opened = _wait(lambda: _events(printer, "openBrowserTab"), "the tab announced", 90)
    tab_id = opened[0]["tab_id"]
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "the tab closed", 60)
    assert [e["focus"] for e in _events(printer, "openBrowserTab")] == [False]
    _wait(lambda: service._context is None, "the browser gone with its last tab", 60)
    assert service._pages == {}


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_a_stopped_agent_open_closes_the_tab_it_still_opens(daemon: Any) -> None:
    server, printer = daemon
    service = server.browser_tabs
    held = _stop_while_queued(service, service.open_for_agent)
    held.result(_HOLD + 10)
    _assert_abandoned_tab_closed(service, printer)


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_a_stopped_user_hand_off_closes_the_tab_it_still_opens(
    daemon: Any, page_server: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    held = _stop_while_queued(service, functools.partial(service.open_for_user, page_server))
    held.result(_HOLD + 10)
    _assert_abandoned_tab_closed(service, printer)
    # The service is unharmed: the next hand-off opens and keeps its tab.
    assert service.open_for_user(page_server) is True
    assert len(service._pages) == 1


class _SlowPage(http.server.BaseHTTPRequestHandler):
    """A sign-in page that answers only once :attr:`release` is set (or after a minute)."""

    release = threading.Event()

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.release.wait(60)
        body = b"<html><head><title>Slow</title></head></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_a_user_hand_off_stopped_while_its_page_loads_closes_the_tab(
    daemon: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    _SlowPage.release.clear()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowPage)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/sign-in"
        caller = _Caller(functools.partial(service.open_for_user, url))
        caller.start()
        # The tab is the user's to act on: focused everywhere, page loading.
        focused = _wait(lambda: _events(printer, "openBrowserTab", focus=True), "focus", 90)
        tab_id = focused[0]["tab_id"]
        stopped_at = time.monotonic()
        assert caller.ident is not None
        assert inject_async_exception(caller.ident, KeyboardInterrupt) == 1
        caller.join(120)
        assert isinstance(caller.error, KeyboardInterrupt), caller.error
        assert caller.ended_at - stopped_at < _SLICE_MARGIN, "the Stop waited for the page"
        _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "the tab closed", 60)
        assert tab_id not in service._pages
    finally:
        _SlowPage.release.set()
        httpd.shutdown()
        httpd.server_close()


def test_a_wait_that_runs_out_raises_timeout_error() -> None:
    # The open's own deadline (``_OPEN_TIMEOUT`` + ``_CALL_TIMEOUT``, two
    # minutes) is only reached with a wedged browser loop; the helper is
    # driven directly with a future nothing ever completes.
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _result_in_slices(concurrent.futures.Future(), 0.3)
    assert 0.3 <= time.monotonic() - started < 2


def test_ip_addresses_typed_in_the_address_bar_open_as_urls() -> None:
    assert normalize_url("192.168.1.10") == "http://192.168.1.10"
    assert normalize_url("127.0.0.1:8000/app?x=1") == "http://127.0.0.1:8000/app?x=1"
    assert normalize_url("[::1]:3000/x") == "http://[::1]:3000/x"
    assert normalize_url("[fe80::1]") == "http://[fe80::1]"
    # Not addresses: searched as before.
    for words in ("1.2", "3.14", "10.0.0.1:port", "999.1.1.1", "::1", "[::1", "1.2.3.4/a b"):
        assert normalize_url(words).startswith("https://www.google.com/search?q="), words


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_an_ip_address_in_the_address_bar_loads_the_page(daemon: Any, page_server: str) -> None:
    server, printer = daemon
    server._handle_command({"type": "browserOpen", "url": "about:blank", "connId": "c1"})
    tab_id = _wait(lambda: _events(printer, "openBrowserTab"), "tab opened", 60)[0]["tab_id"]
    typed = page_server.removeprefix("http://")  # what a user types: 127.0.0.1:PORT/
    server._handle_command(
        {"type": "browserNavigate", "tab_id": tab_id, "action": "go", "url": typed}
    )
    state = _wait(lambda: _events(printer, "browserState", tab_id=tab_id, title="Tall"), "page")
    assert state[-1]["url"] == page_server

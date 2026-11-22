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
import gc
import http.server
import socket
import sys
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from playwright._impl._transport import PipeTransport

from kiss.agents.sorcar.web_use_tool import _Watchdog
from kiss.core.tool_interrupt import inject_async_exception
from kiss.server.browser_tab import (
    _INTERRUPT_PROBE_TIMEOUT,
    BrowserTabService,
    _result_in_slices,
    _run_loop,
    normalize_url,
)
from kiss.tests.lock_injection import stop_as_it_takes_its_lock
from kiss.tests.server.conftest import ensure_loop
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.test_browser_tab_service import (
    _PLAYWRIGHT_CACHE,
    ConnPrinter,
    _events,
    _unclosed_event_loops,
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
    loop = ensure_loop(service)
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


def _loop_threads() -> set[threading.Thread]:
    """The running loop threads of every browser-tab service of this process."""
    return {thread for thread in threading.enumerate() if thread.name == "kiss-browser-tab"}


@pytest.mark.parametrize(
    ("held_in", "thread_held"),
    [
        (asyncio.new_event_loop, False),
        (socket.socketpair, False),
        (threading.Thread.__init__, False),
        (threading.Thread.start, False),
        (threading.Event.wait, True),
        (threading.Event.wait, False),
    ],
    ids=[
        "no_loop_yet",
        "loop_half_built",
        "no_thread_built",
        "no_thread_yet",
        "thread_waits",
        "thread_runs",
    ],
)
def test_a_stop_while_the_loop_thread_starts_leaves_no_thread_behind(
    tmp_path: Path, held_in: Callable[..., object], thread_held: bool
) -> None:
    """A Stop that lands while the first open starts the loop thread orphans nothing.

    The first open starts the service's loop thread and recorded it
    only once ``Thread.start()`` had returned.  A Stop raised inside
    that call skipped the record: the thread ran its loop for the life
    of the daemon, where no ``shutdown`` could find it, and the open's
    coroutine, never handed to a loop, was never closed.  The task
    thread is stopped before it has a loop, while the loop is being
    made (it makes its self-pipe last), where it builds the thread and
    where it starts it, no thread started yet (the loop is then
    nobody's to run, and must not wait for the garbage collector to be
    closed), and where it waits for the new thread to come up, the loop
    thread alive: once held back before it takes the loop, which the
    task thread then takes back and closes, once free to run it.
    """
    service = BrowserTabService(ConnPrinter(), tmp_path / "profile")
    loops_before = _unclosed_event_loops()
    threads_before = _loop_threads()
    caller = _Caller(service.open_for_agent)
    starting = Pause(held_in.__code__, lambda local: threading.current_thread() is caller)
    taking = Pause(_run_loop.__code__, lambda local: thread_held)
    try:
        with warnings.catch_warnings(record=True) as caught, Pauses(starting, taking):
            warnings.simplefilter("always")
            caller.start()
            assert starting.paused.wait(10), "the open never started its loop thread"
            assert (_loop_threads() <= threads_before) is (held_in is not threading.Event.wait)
            assert not thread_held or taking.paused.wait(10), "the loop thread never came up"
            assert caller.ident is not None
            assert inject_async_exception(caller.ident, KeyboardInterrupt) == 1
            starting.release()
            caller.join(10)
            assert isinstance(caller.error, KeyboardInterrupt), caller.error
            taking.release()
            _wait(lambda: _loop_threads() <= threads_before, "the unrecorded thread to end", 10)
            caller.error = None  # its traceback holds the loop
            gc.collect()
        assert [
            str(w.message) for w in caught
            if "never awaited" in str(w.message) or "unclosed event loop" in str(w.message)
        ] == []
        assert service._loop is None and service._thread is None
        assert [loop for loop in _unclosed_event_loops() if loop not in loops_before] == []
        # The service is unharmed: the next call starts a loop thread of its own.
        ran = functools.partial(asyncio.sleep, 0, "ran")
        assert service._run_threadsafe(ran).result(10) == "ran"
    finally:
        service.shutdown()
        for loop in _unclosed_event_loops():  # the orphan of a failed run
            if loop not in loops_before:
                loop.call_soon_threadsafe(loop.stop)


def test_a_stop_while_the_first_open_is_handed_to_the_loop_leaves_nothing_unawaited(
    tmp_path: Path,
) -> None:
    """A Stop that lands as the open is handed to the loop leaves no coroutine behind.

    The open's coroutine was made on the task thread, and counted as
    the loop's before it was queued there: a Stop raised in between
    left it, and the ``wait_for`` around it, never awaited.  The
    coroutine is now made on the loop's thread, so the task thread,
    stopped where it queues the request, holds none: the service has
    started, nothing was queued, and no browser is launched for an
    agent that is gone.
    """
    printer = ConnPrinter()
    service = BrowserTabService(printer, tmp_path / "profile")
    caller = _Caller(service.open_for_agent)
    queueing = Pause(
        asyncio.BaseEventLoop.call_soon_threadsafe.__code__,
        lambda local: threading.current_thread() is caller,
    )
    try:
        with warnings.catch_warnings(record=True) as caught, queueing:
            warnings.simplefilter("always")
            caller.start()
            assert queueing.paused.wait(10), "the open never reached the loop"
            assert caller.ident is not None
            assert inject_async_exception(caller.ident, KeyboardInterrupt) == 1
            queueing.release()
            caller.join(10)
            assert isinstance(caller.error, KeyboardInterrupt), caller.error
            caller.error = None  # its traceback holds what the open had made
            gc.collect()
        assert [str(w.message) for w in caught if "never awaited" in str(w.message)] == []
        # The loop the open started is the service's, and still serves it.
        ran = functools.partial(asyncio.sleep, 0, "ran")
        assert service._run_threadsafe(ran).result(10) == "ran"
        assert service._context is None and not _events(printer, "openBrowserTab")
    finally:
        service.shutdown()


def _profiled(hook: Callable[..., None], call: Callable[[], object]) -> None:
    """Make *call* with the profile *hook* set on the calling thread."""
    sys.setprofile(hook)
    try:
        call()
    finally:
        sys.setprofile(None)


def _stopped_as_it_takes_its_lock(call: Callable[[], object], taker: Any, *under: Any) -> None:
    """Make *call* on a task thread stopped where *taker*, running in *under*, took its lock."""
    caller = _Caller(functools.partial(_profiled, stop_as_it_takes_its_lock(taker, *under), call))
    caller.start()
    caller.join(20)
    assert isinstance(caller.error, KeyboardInterrupt), caller.error


def test_a_stop_as_the_start_of_the_loop_thread_takes_its_lock_strands_no_thread(
    tmp_path: Path,
) -> None:
    """A Stop landing where the start of the loop thread has taken its lock strands no thread.

    ``Thread.start`` ends in a wait for the thread to come up, which
    enters a ``threading.Condition``: a Stop raised as its ``__enter__``
    returns leaves the lock held (gh-74174), and the new thread, which
    sets the event that lock belongs to, blocked there for good.  The
    Stop is now raised once the start is over.
    """
    service = BrowserTabService(ConnPrinter(), tmp_path / "profile")
    threads_before = _loop_threads()
    try:
        _stopped_as_it_takes_its_lock(
            service.open_for_agent,
            threading.Event.wait, threading.Thread.start, BrowserTabService._ensure_loop_locked,
        )
        assert service._loop is None and service._context is None
    finally:
        service.shutdown()
    _wait(lambda: _loop_threads() <= threads_before, "the loop thread to end", 10)


@pytest.mark.parametrize(
    ("taker", "under"),
    [
        (threading.Event.wait, (concurrent.futures.wait, _result_in_slices)),
        (concurrent.futures.Future.done, (_result_in_slices,)),
        (concurrent.futures.Future.add_done_callback, (BrowserTabService._submit,)),
    ],
    ids=["slice", "done", "log"],
)
def test_a_stop_as_a_wait_for_the_loop_takes_its_lock_leaves_the_loop_free(
    tmp_path: Path, taker: Any, under: tuple[Any, ...]
) -> None:
    """A Stop landing where a task thread's wait for the loop has taken its lock blocks no loop.

    An open waits for its request in slices, and a request nobody waits
    for has its failure logged: both enter the condition of the
    request's future, or of the event the wait is told through.  A Stop
    raised as ``Condition.__enter__`` returns leaves that lock held
    (gh-74174), and the loop thread, which takes it to end the future,
    blocked for good: the daemon's browser answered nothing ever after.
    """
    service = BrowserTabService(ConnPrinter(), tmp_path / "profile")
    try:
        # The loop is busy: the request is pending when it is waited for.
        ensure_loop(service).call_soon_threadsafe(time.sleep, 0.2)
        if under[-1] is _result_in_slices:
            future = service._run_threadsafe(functools.partial(asyncio.sleep, 0))
            call: Callable[[], object] = functools.partial(_result_in_slices, future, 10)
        else:
            call = functools.partial(service.viewer_gone, "nobody")
        _stopped_as_it_takes_its_lock(call, taker, *under)
        ran = functools.partial(asyncio.sleep, 0, "ran")
        assert service._run_threadsafe(ran).result(10) == "ran"
    finally:
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


def _next_call_after_a_recovery(
    service: BrowserTabService, tab_id: str, at: str, secs: float
) -> Any:
    """What the next call's script returns when the hung call ends during its recovery.

    The watchdog's recovery runs for real.  The hung call ends, and the
    next one starts a script that spins for *secs*, at the write of the
    recovery's request *at* to the driver when the watchdog's lock is
    free there (the task thread would get through), else after it.
    """
    loop, rec = service._loop, service._pages[tab_id]
    assert loop is not None
    hung = _Watchdog(3600, service.interrupt, (tab_id,))
    end = f"const end = Date.now() + {secs * 1000}"
    script = f"(() => {{ {end}; while (Date.now() < end); return 'ran'; }})()"
    next_call: list[asyncio.Task[Any]] = []

    def roll_over(local: dict[str, Any]) -> bool:
        params = local["message"].get("params") or {}
        named = at in (params.get("method"), params.get("expression"))
        if named and not next_call and not hung._deciding.locked():
            hung.end()
            next_call.append(asyncio.Task(rec.page.evaluate(script), loop=loop, eager_start=True))
            time.sleep(0.5)  # the script spins by the time the recovery's request is written
        return False

    async def script_result() -> Any:
        hung.end()
        return await (next_call[0] if next_call else rec.page.evaluate(script))

    with Pause(PipeTransport.send.__code__, roll_over):
        asyncio.run_coroutine_threadsafe(service._interrupt(tab_id, hung.act), loop).result(60)
    return asyncio.run_coroutine_threadsafe(script_result(), loop).result(60)


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_a_recovery_reaches_neither_the_script_nor_the_page_of_the_next_call(
    daemon: Any, page_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watchdog's leave and the recovery's send are one step (audit L11 F11).

    ``_interrupt`` asked the watchdog, then sent ``Runtime.terminateExecution``:
    the hung call could end and the next one start in between, and the
    terminate killed that call's script.  A recovery whose probe then
    went unanswered (the page was busy with the next call) closed the
    page under it.
    """
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    assert service.open_for_user(page_server) is True
    tab_id = _events(printer, "openBrowserTab")[0]["tab_id"]
    probe_secs = _INTERRUPT_PROBE_TIMEOUT + 1
    assert _next_call_after_a_recovery(service, tab_id, "Runtime.terminateExecution", 1.5) == "ran"
    assert _next_call_after_a_recovery(service, tab_id, "1", probe_secs) == "ran"
    assert not _events(printer, "closeBrowserTab", tab_id=tab_id)
    assert service._pages[tab_id].closing is False

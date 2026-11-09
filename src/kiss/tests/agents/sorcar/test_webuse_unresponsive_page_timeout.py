# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: web tools must not hang forever on an unresponsive page.

A live task froze inside ``get_page_content(text_only=True)`` right after
``go_to_url`` had failed with ``Page.goto: Timeout 30000ms exceeded``.
The page's main thread was stuck, and Playwright's ``Page.title()`` has
no timeout: it waits for the frame's execution context to answer, which
a stuck renderer never does. The tool call never returned and the task
sat idle for good.

The test serves a real page whose script spins forever right after
``DOMContentLoaded`` and drives a real headless Chromium through
``WebUseTool`` from a worker thread; every tool call must come back
(with an error string) well before the worker-thread join deadline.
"""

import functools
import http.server
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from types import CodeType
from typing import Any

import pytest

from kiss.agents.sorcar.web_use_tool import (
    _CDP_CALL_TIMEOUT_SECS,
    _DRIVER_STOP_SECS,
    WebUseTool,
    _cdp_call,
    _stop_playwright,
)
from kiss.core.processes import pid_alive
from kiss.core.tool_interrupt import ToolCallInterrupted, inject_async_exception
from kiss.tests.conftest import posix_only
from kiss.tests.server.pause import Pause

_STUCK_PAGE = (
    b"<html><head><title>stuck page</title></head><body><p>hello</p>"
    b"<script>setTimeout(function () { while (true) {} }, 0);</script>"
    b"</body></html>"
)
_OK_PAGE = (
    b"<html><head><title>fine page</title></head><body><p>hello</p>"
    b'<a href="/stuck">go</a></body></html>'
)
# A top-level SVG document has an <svg> root, not <html>: the title must
# be read through the document root, not a hard-coded "html" selector.
_SVG_PAGE = b'<svg xmlns="http://www.w3.org/2000/svg"><title>svg title</title></svg>'

# The page is fully responsive except that reading document.title runs a
# never-returning getter.  A locator-based title read is only bounded
# during element resolution, so it hangs here; the whole-operation
# deadline of wait_for_function must turn this into a TimeoutError.
_HOSTILE_TITLE_PAGE = (
    b"<html><head><title>real</title></head><body><p>hostile</p><script>"
    b'Object.defineProperty(document, "title", {get() { while (true) {} }});'
    b"</script></body></html>"
)
# Healthy pages whose event handlers wedge the renderer in response to
# our own raw input: the pre-input liveness probe passes, then the
# timeout-less keyboard/mouse call blocks until the input watchdog kills
# Chromium and the pending call raises.
_KEYWEDGE_PAGE = (
    b"<html><head><title>keywedge</title></head><body><p>k</p><script>"
    b'addEventListener("keydown", function () { while (true) {} });'
    b"</script></body></html>"
)
_INPUTWEDGE_PAGE = (
    b"<html><head><title>inputwedge</title></head><body>"
    b'<input aria-label="box"><script>'
    b'addEventListener("keydown", function () { while (true) {} });'
    b"</script></body></html>"
)
_WHEELWEDGE_PAGE = (
    b"<html><head><title>wheelwedge</title></head><body><p>w</p><script>"
    b'addEventListener("mousemove", function () { while (true) {} });'
    b"</script></body></html>"
)

# Each bounded read gives up after 10 s (``_PAGE_READ_TIMEOUT_MS``); the
# join deadline leaves generous slack for browser start-up on slow CI.
_JOIN_TIMEOUT_S = 120


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        if self.path.startswith("/stuck"):
            body, ctype = _STUCK_PAGE, "text/html"
        elif self.path.startswith("/svg"):
            body, ctype = _SVG_PAGE, "image/svg+xml"
        elif self.path.startswith("/hostile-title"):
            body, ctype = _HOSTILE_TITLE_PAGE, "text/html"
        elif self.path.startswith("/keywedge"):
            body, ctype = _KEYWEDGE_PAGE, "text/html"
        elif self.path.startswith("/inputwedge"):
            body, ctype = _INPUTWEDGE_PAGE, "text/html"
        elif self.path.startswith("/wheelwedge"):
            body, ctype = _WHEELWEDGE_PAGE, "text/html"
        else:
            body, ctype = _OK_PAGE, "text/html"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server API
        pass


def _serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _drive(
    tool: WebUseTool,
    base: str,
    out: dict[str, object],
    calls: Callable[[WebUseTool, str, dict[str, object]], None] | None = None,
) -> None:
    """Run the tool calls and close the tool on one thread.

    Playwright's sync API is bound to the thread that started it, so the
    orderly ``close()`` must happen here too; the test thread only falls
    back to a cross-thread close when this worker never finishes.
    """
    try:
        (calls or _drive_calls)(tool, base, out)
    finally:
        tool.close()


def _drive_calls(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    out["ok_tree"] = tool.go_to_url(f"{base}/ok")
    out["ok_text"] = tool.get_page_content(text_only=True)
    # The aria snapshot of an SVG document fails (it has no <body>; a
    # pre-existing limit), but its title must still be read via the root.
    tool.go_to_url(f"{base}/svg")
    out["svg_tabs"] = tool.go_to_url("tab:list")
    t0 = time.monotonic()
    out["stuck_nav"] = tool.go_to_url(f"{base}/stuck")
    out["stuck_nav_s"] = time.monotonic() - t0
    t0 = time.monotonic()
    out["stuck_text"] = tool.get_page_content(text_only=True)
    out["stuck_text_s"] = time.monotonic() - t0
    out["tabs"] = tool.go_to_url("tab:list")
    # Interaction tools: keyboard/mouse/count calls have no Playwright
    # timeout of their own, so each must be guarded by the liveness probe.
    # click(1) resolves the link remembered from the healthy page's tree.
    out["click"] = tool.click(1)
    out["press"] = tool.press_key("End")
    out["scroll"] = tool.scroll("down", 1)
    out["done"] = True


def test_tools_return_errors_instead_of_hanging_on_stuck_page(tmp_path):
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    out: dict[str, object] = {}
    worker = threading.Thread(target=_drive, args=(tool, base, out), daemon=True)
    worker.start()
    worker.join(_JOIN_TIMEOUT_S)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"

        # Sanity: a responsive page is read through the same bounded path.
        assert str(out["ok_tree"]).startswith("Page: fine page\n")
        assert '[1] link "go"' in str(out["ok_tree"])
        assert str(out["ok_text"]).startswith("Page: fine page\n")
        assert "hello" in str(out["ok_text"])
        assert f"[0] svg title - {base}/svg (active)" in str(out["svg_tabs"])

        # The stuck page: navigation reports an error (its read timed out)
        # and get_page_content returns an error string instead of hanging.
        assert str(out["stuck_nav"]).startswith("Error navigating to ")
        assert "Timeout" in str(out["stuck_nav"])
        assert str(out["stuck_text"]).startswith("Error getting page content: ")
        assert "Timeout" in str(out["stuck_text"])
        assert float(out["stuck_nav_s"]) < 60  # type: ignore[arg-type]
        assert float(out["stuck_text_s"]) < 60  # type: ignore[arg-type]

        # Listing tabs still works while the active tab is unresponsive.
        tabs = str(out["tabs"])
        assert tabs.startswith("Open tabs (")
        assert "(unresponsive)" in tabs
        assert f"{base}/stuck (active)" in tabs

        # Interaction tools fail fast with the liveness probe's timeout.
        assert str(out["click"]).startswith("Error clicking element 1: ")
        assert "Timeout" in str(out["click"])
        assert str(out["press"]).startswith("Error pressing key 'End': ")
        assert "Timeout" in str(out["press"])
        assert str(out["scroll"]).startswith("Error scrolling down: ")
        assert "Timeout" in str(out["scroll"])
    finally:
        if worker.is_alive():
            # The worker is wedged inside Playwright: reap Chromium from
            # here (the tool kills the recorded browser PID directly).
            tool.close()
        srv.shutdown()
        srv.server_close()


def _drive_hostile_title(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    """Hostile ``document.title`` getter: every read is a bounded error."""
    out["ok_tree"] = tool.go_to_url(f"{base}/ok")
    t0 = time.monotonic()
    out["hostile_nav"] = tool.go_to_url(f"{base}/hostile-title")
    out["hostile_nav_s"] = time.monotonic() - t0
    t0 = time.monotonic()
    out["hostile_text"] = tool.get_page_content(text_only=True)
    out["hostile_text_s"] = time.monotonic() - t0
    out["hostile_tabs"] = tool.go_to_url("tab:list")
    # Evaluating the hostile getter left the renderer spinning forever
    # (the while(true) never exits), so the tab is permanently wedged —
    # the agent's documented escape hatch is close_browser(), after
    # which the next call relaunches a fresh browser.
    out["closed"] = tool.close_browser()
    out["recovered"] = tool.go_to_url(f"{base}/ok")
    out["done"] = True


def test_hostile_title_getter_times_out_instead_of_hanging(tmp_path):
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    out: dict[str, object] = {}
    worker = threading.Thread(
        target=_drive, args=(tool, base, out), kwargs={"calls": _drive_hostile_title},
        daemon=True,
    )
    worker.start()
    worker.join(_JOIN_TIMEOUT_S)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"
        assert str(out["ok_tree"]).startswith("Page: fine page\n")
        assert str(out["hostile_nav"]).startswith("Error navigating to ")
        assert "Timeout" in str(out["hostile_nav"])
        assert str(out["hostile_text"]).startswith("Error getting page content: ")
        assert "Timeout" in str(out["hostile_text"])
        assert float(out["hostile_nav_s"]) < 60  # type: ignore[arg-type]
        assert float(out["hostile_text_s"]) < 60  # type: ignore[arg-type]
        tabs = str(out["hostile_tabs"])
        assert tabs.startswith("Open tabs (")
        assert f"(unresponsive) - {base}/hostile-title (active)" in tabs
        assert str(out["closed"]).startswith("Browser closed")
        assert str(out["recovered"]).startswith("Page: fine page\n")
    finally:
        if worker.is_alive():
            tool.close()
        srv.shutdown()
        srv.server_close()


def _drive_input_wedge(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    """Handlers that wedge the renderer on our input: watchdog unblocks the call.

    Each wedged raw input call is unblocked by the input watchdog killing
    Chromium; the next ``go_to_url`` must transparently relaunch a fresh
    browser, proving the tool recovers after every kill.
    """
    out["key_tree"] = tool.go_to_url(f"{base}/keywedge")
    t0 = time.monotonic()
    out["press"] = tool.press_key("End")
    out["press_s"] = time.monotonic() - t0
    out["input_tree"] = tool.go_to_url(f"{base}/inputwedge")
    t0 = time.monotonic()
    out["type"] = tool.type_text(1, "x")
    out["type_s"] = time.monotonic() - t0
    out["wheel_tree"] = tool.go_to_url(f"{base}/wheelwedge")
    t0 = time.monotonic()
    out["scroll"] = tool.scroll("down", 1)
    out["scroll_s"] = time.monotonic() - t0
    out["recovered"] = tool.go_to_url(f"{base}/ok")
    out["done"] = True


def test_input_wedging_handlers_are_unblocked_by_watchdog(tmp_path):
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    # A tool that has not launched a browser has no recorded PID: the
    # guard arms the driver-kill fallback, cancelled on exit like any other.
    tool._input_hang_watchdog(lambda: None)
    out: dict[str, object] = {}
    worker = threading.Thread(
        target=_drive, args=(tool, base, out), kwargs={"calls": _drive_input_wedge},
        daemon=True,
    )
    worker.start()
    worker.join(_JOIN_TIMEOUT_S + 60)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"
        assert str(out["key_tree"]).startswith("Page: keywedge\n")
        assert str(out["press"]).startswith("Error pressing key 'End': ")
        assert float(out["press_s"]) < 60  # type: ignore[arg-type]
        assert str(out["input_tree"]).startswith("Page: inputwedge\n")
        assert '[1] textbox "box"' in str(out["input_tree"])
        assert str(out["type"]).startswith("Error typing into element 1: ")
        assert float(out["type_s"]) < 60  # type: ignore[arg-type]
        assert str(out["wheel_tree"]).startswith("Page: wheelwedge\n")
        assert str(out["scroll"]).startswith("Error scrolling down: ")
        assert float(out["scroll_s"]) < 60  # type: ignore[arg-type]
        assert str(out["recovered"]).startswith("Page: fine page\n")
    finally:
        if worker.is_alive():
            tool.close()
        srv.shutdown()
        srv.server_close()


def _drive_input_wedge_without_pid(tool: WebUseTool, base: str, out: dict[str, object]) -> None:
    """The pid capture timed out at launch (a tolerated failure): raw input still has a deadline.

    With no pid to signal, the watchdog kills the tool's own driver
    instead; the Chromium it launched exits with it, and the next call
    relaunches both.  The capture's failure (a CDP round trip that the
    browser does not answer right after launch) cannot be provoked on
    a healthy Chromium; the state it leaves behind is set up instead.
    """
    out["key_tree"] = tool.go_to_url(f"{base}/keywedge")
    tool._browser_pid = tool._browser_identity = None
    out["pid"] = tool._browser_pid
    t0 = time.monotonic()
    out["press"] = tool.press_key("End")
    out["press_s"] = time.monotonic() - t0
    out["recovered"] = tool.go_to_url(f"{base}/ok")
    out["done"] = True


def test_input_watchdog_without_a_browser_pid_kills_the_driver(tmp_path):
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    out: dict[str, object] = {}
    worker = threading.Thread(
        target=_drive, args=(tool, base, out), kwargs={"calls": _drive_input_wedge_without_pid},
        daemon=True,
    )
    worker.start()
    worker.join(_JOIN_TIMEOUT_S)
    try:
        assert out.get("done") is True, f"tool call hung; progress so far: {sorted(out)}"
        assert str(out["key_tree"]).startswith("Page: keywedge\n")
        assert out["pid"] is None
        assert str(out["press"]).startswith("Error pressing key 'End': ")
        assert float(out["press_s"]) < 60  # type: ignore[arg-type]
        assert str(out["recovered"]).startswith("Page: fine page\n")
    finally:
        if worker.is_alive():
            tool.close()
        srv.shutdown()
        srv.server_close()


def _driver_pid(tool: WebUseTool) -> int:
    """The pid of the Node driver behind *tool*'s Playwright."""
    return int(tool._playwright._impl_obj._connection._transport._proc.pid)


class _FrozenDriver:
    """Freeze *tool*'s Node driver (``SIGSTOP``) for the block; resume it on exit.

    A driver that no longer answers is what a stalled ``node run-driver``
    looks like to the tool: alive, its pipes open, every request
    unanswered.  Playwright's own ``timeout`` arguments are enforced by
    that driver, so only a deadline on the Python side can end a read.
    *at* names a Playwright ``_impl`` coroutine function: the freeze then
    lands as that coroutine starts (a :class:`Pause` hook acting at its
    entry, once: the hook is told of its resumptions too), so the read
    it belongs to is the one that must time out, after the reads before
    it answered; without *at* the driver is frozen before the block.  A
    safety timer resumes the driver after *resume_after* seconds for an
    unbounded call.  The tool kills a driver that does not even
    acknowledge the cancel, so the resume on exit tolerates a pid that
    is gone.
    """

    def __init__(self, tool: WebUseTool, at: Any = None, resume_after: float = 100.0) -> None:
        self._pid = _driver_pid(tool)
        self._frozen = False
        self._pause = None if at is None else Pause(at.__code__, self._freeze_at_entry)
        self._resume = threading.Timer(resume_after, self._resume_driver)
        self._resume.daemon = True

    def _freeze_at_entry(self, _locals: dict[str, Any]) -> bool:
        if not self._frozen:
            self._frozen = True
            os.kill(self._pid, signal.SIGSTOP)
        return False

    def _resume_driver(self) -> None:
        try:
            os.kill(self._pid, signal.SIGCONT)
        except OSError:
            pass

    def __enter__(self) -> None:
        self._resume.start()
        if self._pause is None:
            os.kill(self._pid, signal.SIGSTOP)
        else:
            self._pause.__enter__()

    def __exit__(self, *exc: object) -> None:
        if self._pause is not None:
            self._pause.__exit__(*exc)
        self._resume.cancel()
        self._resume_driver()


def _bounded_error(call: Callable[[], str], prefix: str, timeout_ms: int, limit: float) -> str:
    """Run *call* against the frozen driver: its deadline's error, within *limit* seconds."""
    t0 = time.monotonic()
    result = call()
    secs = time.monotonic() - t0
    assert secs < limit, f"a read waited {secs:.0f}s for the stalled driver: {result!r}"
    assert result.startswith(prefix), result
    assert f"Timeout {timeout_ms}ms exceeded" in result, result
    return result


@posix_only("the driver is frozen with SIGSTOP")
def test_reads_return_errors_when_the_driver_stalls(tmp_path):
    """Every page read is bounded on the Python side when the driver stalls.

    ``wait_for_function`` (the title read, the pre-input liveness probe
    and ``wait_for``'s predicate), ``aria_snapshot`` (the tree) and
    ``inner_text`` (the text-only body) pass their timeout to the driver,
    which a stalled driver never enforces: each read waited for good.
    Each must now return the tool's error string after its own deadline.
    A driver that does not even acknowledge the cancel is killed and
    discarded: the next call relaunches, so the page is read again.
    """
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    try:
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        page_impl = type(tool._page._impl_obj)
        locator_impl = type(tool._page.locator("body")._impl_obj)
        content = "Error getting page content: "
        text_only = functools.partial(tool.get_page_content, text_only=True)
        with _FrozenDriver(tool, at=page_impl.inner_text):  # after the title read answered
            _bounded_error(text_only, content, 10000, 30)
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        with _FrozenDriver(tool, at=locator_impl.aria_snapshot):
            _bounded_error(tool.get_page_content, content, 10000, 30)
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        with _FrozenDriver(tool):  # the pre-input liveness probe
            _bounded_error(lambda: tool.press_key("End"), "Error pressing key 'End': ", 10000, 30)
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        with _FrozenDriver(tool):  # the predicate wait, with its own timeout
            waiting = "Error waiting for text 'never': "
            _bounded_error(lambda: tool.wait_for("never", timeout=1), waiting, 1000, 15)
        recovered = tool.go_to_url(f"{base}/ok")
        assert recovered.startswith("Page: fine page\n"), recovered
        assert "hello" in tool.get_page_content(text_only=True)
    finally:
        tool.close()
        srv.shutdown()
        srv.server_close()


class _StopWhileTimerStarts:
    """Inject a Stop into the thread as its ``threading.Thread.start`` returns.

    A ``sys.monitoring`` hook on the real ``Thread.start`` (nothing is
    replaced; the tool id is this module's own, ``Pause`` holds 3): the
    ``ToolCallInterrupted`` the daemon's Stop injects then lands once
    the watchdog's thread exists and before the guarded operation runs,
    the window in which a context-manager guard loses its ``__exit__``.
    *arming*, another code object, is hooked instead of ``Thread.start``:
    the event loop's ``call_later``, whose return is the same window
    for a loop timer.  Only the thread that entered the block is stopped.
    """

    _TOOL = 4
    _RETURN = sys.monitoring.events.PY_RETURN

    def __init__(self, arming: CodeType = threading.Thread.start.__code__) -> None:
        self._arming = arming
        self._thread = threading.get_ident()

    def __enter__(self) -> None:
        sys.monitoring.use_tool_id(self._TOOL, "kiss.tests stop while a timer starts")
        sys.monitoring.register_callback(self._TOOL, self._RETURN, self._inject)
        sys.monitoring.set_local_events(self._TOOL, self._arming, self._RETURN)

    def __exit__(self, *exc: object) -> None:
        sys.monitoring.set_local_events(self._TOOL, self._arming, 0)
        sys.monitoring.register_callback(self._TOOL, self._RETURN, None)
        sys.monitoring.free_tool_id(self._TOOL)

    def _inject(self, code: Any, offset: int, retval: Any) -> None:
        if threading.get_ident() == self._thread:
            inject_async_exception(self._thread, ToolCallInterrupted)


def test_stop_while_a_watchdog_starts_leaves_it_disarmed(tmp_path):
    """A Stop landing as a watchdog timer starts must not leave it armed.

    The input watchdog kills Chromium (or the driver) and the driver-stop
    watchdog kills the driver when their guarded call outlives the
    deadline.  A ``ToolCallInterrupted`` injected while the timer was
    being started unwound without the cancel: the timer then fired under
    whatever ran next and killed its browser or driver.  The operation is
    never run; the browser and the driver must both survive their
    deadlines and keep serving the tool.
    """
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    try:
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        browser_pid = tool._browser_pid
        assert browser_pid is not None
        playwright = tool._playwright
        driver_pid = _driver_pid(tool)
        ran = False

        def operation() -> None:
            nonlocal ran
            ran = True

        with _StopWhileTimerStarts(), pytest.raises(ToolCallInterrupted):
            tool._input_hang_watchdog(operation, 0.3)
        assert not ran
        time.sleep(1.0)
        assert pid_alive(browser_pid), "the input watchdog fired after the Stop"
        assert tool.get_page_content(text_only=True).startswith("Page: fine page\n")

        with _StopWhileTimerStarts(), pytest.raises(ToolCallInterrupted):
            _stop_playwright(playwright)
        time.sleep(_DRIVER_STOP_SECS + 1.0)
        assert tool._playwright is playwright
        assert pid_alive(driver_pid), "the driver-stop watchdog fired after the Stop"
        assert tool.get_page_content(text_only=True).startswith("Page: fine page\n")

        # Review R8: the kill timer of a CDP call is a loop timer, armed
        # by ``loop.call_later``; a Stop landing as it returns (the timer
        # registered, its handle not yet held) must leave it inert.  The
        # loop turns again under the next read, past the timer's
        # deadline: that read must succeed and the driver outlive it.
        page = tool._page
        with (
            _StopWhileTimerStarts(type(page._loop).call_later.__code__),
            pytest.raises(ToolCallInterrupted),
        ):
            _cdp_call(page, page._impl_obj.evaluate("1"), timeout_secs=0.1)
        time.sleep(0.1 + _CDP_CALL_TIMEOUT_SECS + 0.5)
        assert tool.get_page_content(text_only=True).startswith("Page: fine page\n")
        assert tool._playwright is playwright
        assert pid_alive(driver_pid), "the cancelled-call kill timer fired after the Stop"
    finally:
        tool.close()
        srv.shutdown()
        srv.server_close()

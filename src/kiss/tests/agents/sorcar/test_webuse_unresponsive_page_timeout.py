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

import contextlib
import functools
import http.server
import importlib
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from types import CodeType
from typing import Any

import pytest

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.web_use_tool import (
    _CDP_CALL_TIMEOUT_SECS,
    _DRIVER_STOP_SECS,
    WebUseTool,
    _cdp_call,
    _kill_driver,
    _Session,
    _stop_playwright,
    _wait_pid_exit,
    _watchdog_kill,
)
from kiss.core.processes import pid_alive, signal_unreaped_child
from kiss.core.tool_interrupt import ToolCallInterrupted, inject_async_exception
from kiss.server.browser_tab import BrowserTabService, _log_future_error
from kiss.tests.agents.sorcar.test_web_use_sweep1030 import _UserChrome
from kiss.tests.conftest import posix_only
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.pause import Pause, Pauses

_STUCK_PAGE = (
    b"<html><head><title>stuck page</title></head><body><p>hello</p>"
    b"<script>setTimeout(function () { while (true) {} }, 0);</script>"
    b"</body></html>"
)
_OK_PAGE = (
    b"<html><head><title>fine page</title></head><body><p>hello</p>"
    b'<a href="/stuck">go</a></body></html>'
)
# A healthy page that, told to through ``data-busy`` on its root, turns busy
# for 3.5 s as its title is read: past the first deadline of the tree's read.
_BUSY_PAGE = (
    b"<html><head><title>-</title><script>"
    b'Object.defineProperty(document, "title", {get() {'
    b" if (document.documentElement.dataset.busy) {"
    b" delete document.documentElement.dataset.busy;"
    b" setTimeout(function () { const end = Date.now() + 3500; while (Date.now() < end) {} }, 0); }"
    b' return "busy page"; }});'
    b'</script></head><body><a href="/ok">back</a></body></html>'
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

# ``/held`` reports that its request arrived and is answered once let go:
# a call the late-watchdog test keeps in flight.
_HELD_ARRIVED = threading.Event()
_HELD_GO = threading.Event()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        if self.path.startswith("/held"):
            _HELD_ARRIVED.set()
            _HELD_GO.wait(60)
        if self.path.startswith("/stuck"):
            body, ctype = _STUCK_PAGE, "text/html"
        elif self.path.startswith("/busy"):
            body, ctype = _BUSY_PAGE, "text/html"
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
    # A page that turns busy for longer than the first deadline of the tree's
    # read (2 s, its wait for iframes) is read all the same, with the usual one.
    tool.go_to_url(f"{base}/busy")
    tool.evaluate_script("() => { document.documentElement.dataset.busy = '1' }")
    out["busy_tree"] = tool.get_page_content()
    tool.go_to_url(f"{base}/ok")
    # The page wedges (for 12 s) after it was read: the lookup of an id's
    # element has no Playwright timeout either and is guarded by the probe.
    tool.evaluate_script(
        "() => { setTimeout(() => { const end = Date.now() + 12000;"
        " while (Date.now() < end) {} }, 0) }"
    )
    out["wedged_click"] = tool.click(1)
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
    # Interaction tools: keyboard/mouse calls have no Playwright timeout
    # of their own, so each must be guarded by the liveness probe.  No
    # tree of this page was read: an id of the page before it is refused.
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
        assert str(out["busy_tree"]).startswith("Page: busy page\n")
        assert '[1] link "back"' in str(out["busy_tree"])
        assert f"[0] svg title - {base}/svg (active)" in str(out["svg_tabs"])
        assert str(out["wedged_click"]).startswith("Error clicking element 1: ")
        assert "Timeout" in str(out["wedged_click"])

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

        # Interaction tools fail fast with the liveness probe's timeout; an
        # id of a tree read before the reads that failed, at once.
        assert str(out["click"]).startswith("Error clicking element 1: ")
        assert "get_page_content()" in str(out["click"])
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
    it answered; *when* picks the entry by its arguments (the first
    without it); without *at* the driver is frozen before the block.  A
    safety timer resumes the driver *resume_after* seconds into the
    freeze for an unbounded call.  The tool kills a driver that does not even
    acknowledge the cancel, so the resume on exit tolerates a pid that
    is gone.  *also* are further :class:`Pause` hooks, installed with
    the one of *at* (the helper holds a single monitoring tool id).
    """

    def __init__(
        self,
        tool: WebUseTool,
        at: Any = None,
        resume_after: float = 100.0,
        also: tuple[Pause, ...] = (),
        when: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self._pid = _driver_pid(tool)
        self._frozen = False
        self._when = when
        self._pause = (
            None if at is None else Pauses(Pause(at.__code__, self._freeze_at_entry), *also)
        )
        self._resume = threading.Timer(resume_after, self._resume_driver)
        self._resume.daemon = True

    def _freeze(self) -> None:
        os.kill(self._pid, signal.SIGSTOP)
        self._resume.start()

    def _freeze_at_entry(self, entry: dict[str, Any]) -> bool:
        if not self._frozen and (self._when is None or self._when(entry)):
            self._frozen = True
            self._freeze()
        return False

    def _resume_driver(self) -> None:
        try:
            os.kill(self._pid, signal.SIGCONT)
        except OSError:
            pass

    def __enter__(self) -> None:
        if self._pause is None:
            self._freeze()
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
    The tree's snapshot is asked for with a short deadline first (2 s):
    a stall there is that read's error, not a busy page's, which would
    be given the usual deadline.
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
        with _FrozenDriver(tool, at=locator_impl.aria_snapshot):  # the tree
            _bounded_error(tool.get_page_content, content, 2000, 30)
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


def _add_button(tool: WebUseTool, name: str) -> None:
    """Put a button called *name* first on the page; a press of it is logged in the title."""
    tool.evaluate_script(
        "() => document.body.insertAdjacentHTML('afterbegin',"
        f" '<button onclick=\"document.title += \\' {name}\\'\">{name}</button>')"
    )


def _refused_untouched(tool: WebUseTool) -> None:
    """Id 1 is refused, to be read again, and nothing on the page was pressed."""
    result = tool.click(1)
    assert tool.evaluate_script("() => document.title") == "fine page", result
    assert result.startswith("Error clicking element 1: "), result
    assert "get_page_content()" in result, result


@posix_only("the driver is frozen with SIGSTOP")
def test_a_failed_read_of_the_tree_leaves_no_ids(tmp_path):
    """A read of the tree that fails leaves no ids: the page is to be read again.

    An id is the element reference of the snapshot it was read from, and
    a snapshot numbers the references of the document in the tab anew
    (two documents both number from ``e2``).  With the driver stalled
    for the read of the document the tab went to, and back before it is
    given up on, that document was numbered though no tree of it was
    returned: an id of the tab's first document, left in force, pressed
    the button the unseen tree had put first.
    """
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    try:
        tool.go_to_url("about:blank")
        tool.evaluate_script("() => { document.body.innerHTML = '<button>old</button>' }")
        assert '[1] button "old"' in tool.get_page_content()
        tool.evaluate_script(f"() => {{ location.href = '{base}/ok' }}")
        deadline = time.monotonic() + 10
        while tool.evaluate_script("() => document.title") != "fine page":
            assert time.monotonic() < deadline, "the tab did not navigate"
            time.sleep(0.1)
        _add_button(tool, "other")
        driver = _driver_pid(tool)
        snapshot = type(tool._page.locator("body")._impl_obj).aria_snapshot
        # Stalled for 6.5 s: past the snapshot's deadline (5 s), short of the kill (8 s).
        with _FrozenDriver(tool, at=snapshot, resume_after=6.5):
            _bounded_error(tool.get_page_content, "Error getting page content: ", 2000, 30)
        _refused_untouched(tool)
        assert _driver_pid(tool) == driver, "the driver was killed: nothing was tested"
    finally:
        tool.close()
        srv.shutdown()
        srv.server_close()


def _restore_storage(tool: WebUseTool, base: str) -> str:
    """Carry a localStorage entry into the page (a navigation, then the write)."""
    tool._restore_session(_Session(f"{base}/ok", [], origin=base, local_storage=[["k", "v"]]))
    return "restored" if tool._carried_storage[0] else "Error restoring: nothing was written"


def _is_final_click(entry: dict[str, Any]) -> bool:
    """Whether a ``Locator.click`` is the click itself, not the trial before it."""
    return not entry["trial"]


def _stalled_call(tool: WebUseTool, base: str, stalled: str, shot: str) -> tuple[Any, ...]:
    """The ``_FrozenDriver`` arguments *at* and *when*, the error prefix and the call of a case."""
    page = tool._page
    page_impl, locator_impl = type(page._impl_obj), type(page.locator("body")._impl_obj)
    connect = type(tool._playwright.chromium._impl_obj).connect_over_cdp
    call = functools.partial
    key, press = "Error pressing key 'End': ", call(tool.press_key, "End")
    click, attach = "Error clicking element 1: ", "Error attaching to Chrome at "
    cases: dict[str, tuple[Any, ...]] = {
        # frozen at the key press itself, after the preflight read
        "press_key": (type(page.keyboard._impl_obj).press, None, key, press),
        "go_to_url": (None, None, "Error navigating to ", call(tool.go_to_url, f"{base}/ok")),
        "screenshot": (None, None, "Error taking screenshot: ", call(tool.screenshot, shot)),
        # after the glide and the trial: the click, and the hover, themselves
        "click": (locator_impl.click, _is_final_click, click, call(tool.click, 1)),
        "hover": (locator_impl.hover, None, click, call(tool.click, 1, "hover")),
        "pause": (page_impl.wait_for_timeout, None, key, press),  # the one after the key press
        "attach": (connect, None, attach, call(tool.attach_to_chrome, "http://127.0.0.1:9")),
        "restore": (page_impl.goto, None, "Error restoring: ", call(_restore_storage, tool, base)),
    }
    return cases[stalled]


@posix_only("the driver is frozen with SIGSTOP")
@pytest.mark.parametrize(
    "stalled",
    ["press_key", "go_to_url", "screenshot", "click", "hover", "pause", "attach", "restore"],
)
def test_input_and_navigation_end_when_the_driver_stalls(tmp_path, stalled):
    """Raw input, navigation and every other driver request end when the driver stalls.

    Chromium's death under the input watchdog reaches the pending key
    press only through the driver, and ``goto``'s timeout is the driver's
    own: with the driver stalled, either call waited until something
    resumed it.  The driver is now killed a grace after each deadline:
    the call returns its error and the next one relaunches both.  (The
    Browser tab's page whose tab id is unknown takes the same failsafe;
    reaching it needs a popup the daemon has not registered yet.)  So
    do the requests whose only deadline was the driver's: a screenshot,
    the click and the hover after the glide, the pause after an input,
    the connect to a Chrome and the navigation that restores
    localStorage (the same call as the fallback navigations).
    """
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    try:
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        driver = _driver_pid(tool)
        at, when, prefix, call = _stalled_call(tool, base, stalled, str(tmp_path / "shot.png"))
        t0 = time.monotonic()
        with _FrozenDriver(tool, at=at, resume_after=60.0, when=when):
            result = call()
        secs = time.monotonic() - t0
        assert result.startswith(prefix) and secs < 50, f"{secs:.0f}s: {result[:200]!r}"
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        assert _driver_pid(tool) != driver
    finally:
        tool.close()
        srv.shutdown()
        srv.server_close()


def _freeze_first_driver(frozen: list[int], entry: dict[str, Any]) -> bool:
    """A :class:`Pause` hook on the transport's read loop: SIGSTOP the first driver started."""
    if not frozen:
        frozen.append(entry["self"]._proc.pid)
        os.kill(frozen[0], signal.SIGSTOP)
    return False


def _resume(pids: list[int]) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGCONT)


@posix_only("the driver is frozen with SIGSTOP")
def test_a_driver_that_stalls_while_starting_fails_the_start(tmp_path):
    """A new driver that never answers its handshake is killed, and the call says so.

    ``start()`` returns once the driver it spawned has answered: with
    that driver stalled from its first moment, the first tool call
    waited until something resumed it.
    """
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    transport = importlib.import_module(f"{web_stealth.playwright_package()}._impl._transport")
    frozen: list[int] = []
    freeze = functools.partial(_freeze_first_driver, frozen)
    resume = threading.Timer(60.0, _resume, args=(frozen,))
    resume.daemon = True
    try:
        resume.start()
        t0 = time.monotonic()
        with Pause(transport.PipeTransport.run.__code__, freeze):
            result = tool.go_to_url(f"{base}/ok")
        secs = time.monotonic() - t0
        assert frozen and secs < 50, f"{secs:.0f}s: {result[:200]!r}"
        assert result.startswith("Error navigating to "), result[:200]
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
    finally:
        resume.cancel()
        _resume(frozen)
        tool.close()
        srv.shutdown()
        srv.server_close()


def _stop_first_driver(started: list[int], entry: dict[str, Any]) -> bool:
    """A :class:`Pause` hook on the transport's read loop: the Stop, once the first driver runs."""
    if not started:
        started.append(entry["self"]._proc.pid)
        raise ToolCallInterrupted
    return False


@posix_only("the driver is this process's child, reaped before the call returns")
def test_stop_while_the_driver_starts_ends_that_driver(tmp_path):
    """A Stop landing while a new driver starts takes that driver with it.

    No Playwright object exists before ``start()`` returns: the driver
    the Stop left running was owned by nothing, and no ``close()``
    ended it.
    """
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    transport = importlib.import_module(f"{web_stealth.playwright_package()}._impl._transport")
    started: list[int] = []
    stop = functools.partial(_stop_first_driver, started)
    try:
        with Pause(transport.PipeTransport.run.__code__, stop), pytest.raises(ToolCallInterrupted):
            tool.go_to_url("http://127.0.0.1:9/")
        assert started and not pid_alive(started[0]), "the driver outlived the Stop"
    finally:
        tool.close()
        for pid in started:  # a driver that did outlive it is still this process's child
            signal_unreaped_child(pid, signal.SIGKILL)


def _resume_then_let_go(late: Pause, driver: int, wait_until_over: Callable[[], Any]) -> None:
    """Resume *driver* once its watchdog is held at *late*; let the watchdog go under ``/held``.

    The other thread of the late-watchdog test: the call the watchdog
    guards ends once the driver answers again, and the watchdog is let
    go only when the next call's request has reached the server, which
    answers once the watchdog is over (*wait_until_over*).
    """
    if late.paused.wait(60):
        os.kill(driver, signal.SIGCONT)
        if _HELD_ARRIVED.wait(60):
            late.release()
            wait_until_over()
    _HELD_GO.set()


# Per kind of watchdog: where its recovery starts, and the name it logs for the call.
_WATCHDOGS: dict[str, tuple[Any, str | None]] = {
    "failsafe": (_kill_driver, "Raw browser input"),
    "navigation": (_kill_driver, "Navigation"),
    "no_pid": (_kill_driver, "Raw browser input"),
    "browser": (_watchdog_kill, "Raw browser input"),
    "attached": (_kill_driver, "Raw input to the attached Chrome"),
    "close": (_kill_driver, "Graceful browser close"),
    "live": (BrowserTabService._interrupt, None),
}
# A script that is still running while the server holds its request.
_HELD_SCRIPT = (
    "() => { const x = new XMLHttpRequest(); x.open('GET', '/held', false); x.send();"
    " return x.status; }"
)


@posix_only("the driver is frozen with SIGSTOP")
@pytest.mark.parametrize("kind", list(_WATCHDOGS))
def test_a_watchdog_that_fired_as_its_call_ended_spares_the_next_call(tmp_path, kind):
    """A watchdog whose call ended while it fired leaves the next call alone.

    Reviews R1 and R2: ``cancel()`` cannot stop a timer whose thread is
    already in its body, and the Browser tab's recovery runs later
    still, on the daemon's loop.  The driver stalls until the watchdog
    of the call (a key press, a navigation, a close) is held where its
    recovery starts, then answers again, so the call ends.  The
    watchdog is let go while the next call waits for ``/held``, and is
    over (its thread joined, the daemon's recovery done) before that is
    answered: it used to kill the driver or Chromium under the next
    call, or stop its script.
    """
    srv, base = _serve()
    svc = BrowserTabService(MemoryPrinter(), tmp_path / "tab") if kind == "live" else None
    chrome = _UserChrome(tmp_path / "chrome", f"{base}/ok") if kind == "attached" else None
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True, live_browser=svc)
    _HELD_ARRIVED.clear()
    _HELD_GO.clear()
    try:
        if chrome is not None:
            assert tool.attach_to_chrome(chrome.url).startswith("Page: fine page\n")
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        if svc is not None:
            tool.show_browser()
            assert tool._live and tool._live_tab_id() is not None
        driver, browser = _driver_pid(tool), tool._browser_pid
        if kind in ("no_pid", "close"):  # the pid capture at launch timed out: tolerated
            tool._browser_pid = tool._browser_identity = None
        call: Callable[[], str] = functools.partial(tool.press_key, "End")
        at = type(tool._page.keyboard._impl_obj).press
        if kind == "navigation":
            at, call = type(tool._page._impl_obj).goto, functools.partial(tool.go_to_url, base)
        elif kind == "close":
            at, call = type(tool._context._impl_obj).close, tool.close_browser
        held, name = _WATCHDOGS[kind]
        fired: list[threading.Thread] = []
        recovered = threading.Event()

        def its_recovery_starts(entry: dict[str, Any]) -> bool:
            fired.append(threading.current_thread())  # the last one is the one held
            return bool(entry.get("operation", name) == name)

        def wait_until_over() -> None:
            if kind == "live":  # its thread is the daemon's loop: the recovery's future is done
                recovered.wait(30)
            else:
                fired[-1].join(30)

        late = Pause(held.__code__, its_recovery_starts)
        done = Pause(_log_future_error.__code__, lambda entry: bool(recovered.set()))
        other = threading.Thread(
            target=_resume_then_let_go, args=(late, driver, wait_until_over), daemon=True
        )
        other.start()
        with _FrozenDriver(tool, at=at, also=(late, done)):
            call()
            assert late.paused.is_set(), "the call ended before its watchdog fired"
            if kind == "live":
                after = tool.evaluate_script(_HELD_SCRIPT)
            else:
                after = tool.go_to_url(f"{base}/held")
            other.join(30)
        assert after.startswith("200" if kind == "live" else "Page: fine page\n"), after
        assert recovered.is_set() if kind == "live" else not fired[-1].is_alive()
        assert _driver_pid(tool) == driver and pid_alive(driver)
        if kind == "browser":
            assert browser is not None and pid_alive(browser), "the late watchdog killed Chromium"
    finally:
        _HELD_GO.set()
        tool.close()
        if svc is not None:
            svc.shutdown()
        if chrome is not None:
            chrome.stop()
        srv.shutdown()
        srv.server_close()


def _drain_once_returned(
    blocked: threading.Event, returned: threading.Event, pipe: int, size: int, out: dict[str, bool]
) -> None:
    """Empty the log pipe once the navigation has returned, or 15 s after it blocked; say which."""
    if blocked.wait(60):
        out["returned_while_blocked"] = returned.wait(15)
    os.read(pipe, size)


@posix_only("the driver is frozen with SIGSTOP")
def test_a_finished_navigation_does_not_wait_for_its_watchdogs_log_message(tmp_path):
    """A navigation that has ended returns while its watchdog is blocked in the log.

    Review R1 of the second round: the watchdog wrote its message
    holding the lock that the navigation's end takes, so a log sink
    nobody drains (a full pipe behind a real handler) held the finished
    call for good.  The driver stalls until the watchdog's message
    enters that sink, then answers: the navigation must return its page
    while the watchdog's thread is still in the write, and the
    watchdog, once the pipe is drained, must leave the driver alone.
    """
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    reader, writer = os.pipe()
    os.set_blocking(writer, False)
    size = 0
    with contextlib.suppress(BlockingIOError):
        while True:
            size += os.write(writer, b"x" * 4096)
    os.set_blocking(writer, True)
    sink = logging.StreamHandler(os.fdopen(writer, "w"))
    sink.setLevel(logging.WARNING)
    logging.getLogger(WebUseTool.__module__).addHandler(sink)
    writing: list[threading.Thread] = []
    blocked, returned = threading.Event(), threading.Event()
    out: dict[str, bool] = {}
    try:
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        driver = _driver_pid(tool)

        def message_enters_the_sink(entry: dict[str, Any]) -> bool:
            if entry["self"] is sink and not writing:
                writing.append(threading.current_thread())
                os.kill(driver, signal.SIGCONT)
                blocked.set()
            return False

        entering = Pause(logging.StreamHandler.emit.__code__, message_enters_the_sink)
        drainer = threading.Thread(
            target=_drain_once_returned, args=(blocked, returned, reader, size, out), daemon=True
        )
        drainer.start()
        with _FrozenDriver(tool, at=type(tool._page._impl_obj).goto, also=(entering,)):
            result = tool.go_to_url(f"{base}/ok")
            still_writing = bool(writing) and writing[0].is_alive()
            returned.set()
            drainer.join(30)
            writing[0].join(30)
        assert out.get("returned_while_blocked") and still_writing, "the call waited for the log"
        assert result.startswith("Page: fine page\n"), result
        assert not writing[0].is_alive(), "the watchdog did not finish once the pipe was drained"
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        assert _driver_pid(tool) == driver and pid_alive(driver)
    finally:
        logging.getLogger(WebUseTool.__module__).removeHandler(sink)
        sink.stream.close()
        os.close(reader)
        tool.close()
        srv.shutdown()
        srv.server_close()


def _stop_in(thread: int, _entry: dict[str, Any]) -> bool:
    """A :class:`Pause` hook: raise the Stop in *thread* at the entry of the hooked function."""
    if threading.get_ident() == thread:
        raise ToolCallInterrupted
    return False


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


@posix_only("the driver is frozen with SIGSTOP")
def test_stop_while_discarding_a_stalled_driver_loses_neither_process(tmp_path):
    """A Stop in the discard's wait for Chromium's exit leaves no process behind.

    The discard had forgotten the driver and the browser's pid before it
    waited: the Stop skipped the driver's stop, no later ``close()`` could
    find either, and both ran on.
    """
    srv, base = _serve()
    tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
    try:
        assert tool.go_to_url(f"{base}/ok").startswith("Page: fine page\n")
        driver, browser = _driver_pid(tool), tool._browser_pid
        assert browser is not None
        stop = Pause(_wait_pid_exit.__code__, functools.partial(_stop_in, threading.get_ident()))
        with _FrozenDriver(tool), stop, pytest.raises(ToolCallInterrupted):
            tool._discard_dead_driver(close_live_tab=False, why="did not answer")
    finally:
        tool.close()
        srv.shutdown()
        srv.server_close()
    assert _wait_pid_exit(driver, 10.0), "the driver outlived the interrupted discard and close()"
    assert _wait_pid_exit(browser, 10.0), "Chromium outlived the interrupted discard and close()"

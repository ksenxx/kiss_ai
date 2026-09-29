# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: ``show_browser`` moves the session into the streamed Browser tab.

Under the ``kiss-web`` daemon the agent's ``WebUseTool`` is handed the
daemon's :class:`BrowserTabService`.  ``show_browser()`` must then open
the page in a tab that every surface streams (instead of a window on the
daemon machine), keep driving that very page through the other web tools
while the user watches, and carry cookies between the headless browser
and the tab's persistent profile.  These tests run a REAL service (its
own Chromium and event loop) and a REAL ``WebUseTool`` (a second CDP
client on that Chromium); no mocks.

Branches that cannot be reached without doubles: the headed
``_mask_headless_user_agent`` no-op needs a display; the "page closed
while being scanned" skip in ``_find_live_page``, the tab clean-up when
``connect_over_cdp`` itself fails in ``_attach_live``, and the
``except`` arms of ``_live_connected``/``_live_tab_id``/``_detach_live``
need the browser or the page to vanish between two CDP round trips; the
localStorage restore failure needs the page's origin to be unreachable
at exactly that moment.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import web_use_tool as wut
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.server.browser_tab import BrowserTabService
from kiss.tests.server._memory_printer import MemoryPrinter

pytestmark = pytest.mark.skipif(
    not (Path.home() / ".cache" / "ms-playwright").is_dir(),
    reason="Playwright browsers not installed",
)

SESSION_COOKIE = "sid=only-in-memory"
TAB_COOKIE = "tab=set-in-the-browser-tab"
PERSISTENT_COOKIE = "keep=on-disk"


class _Handler(BaseHTTPRequestHandler):
    """``/login`` sets a session cookie, ``/tab-login`` another, ``/cookies``
    echoes them, ``/ua`` echoes the User-Agent, ``/form`` has a textbox
    and a ``target=_blank`` link, ``/flip`` is a page the first time and a
    redirect to the same server under the ``localhost`` origin afterwards,
    anything else is inert."""

    flip_served = False

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Answer according to the path."""
        if self.path == "/flip" and _Handler.flip_served:
            port = self.headers.get("Host", "").rsplit(":", 1)[-1]
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{port}/inert")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/flip":
            _Handler.flip_served = True
        body, cookie = "a page", ""
        if self.path == "/login":
            body, cookie = "logged in", f"{SESSION_COOKIE}; Path=/"
        elif self.path == "/tab-login":
            body, cookie = "tab logged in", f"{TAB_COOKIE}; Path=/"
        elif self.path == "/cookies":
            body = self.headers.get("Cookie", "(none)")
        elif self.path == "/ua":
            body = self.headers.get("User-Agent", "")
        elif self.path == "/form":
            body = (
                "<title>Form</title><input aria-label='Name' id='name'>"
                "<a href='/inert' target='_blank'>open</a>"
            )
        elif self.path == "/persist-login":
            body, cookie = "kept", f"{PERSISTENT_COOKIE}; Path=/; Max-Age=3600"
        elif self.path == "/logout":
            body, cookie = "logged out", "keep=; Path=/; Max-Age=0"
        elif self.path == "/hang":
            body = (
                "<title>Hang</title><input id='h' autofocus>"
                "<script>document.getElementById('h').addEventListener('keydown',"
                " () => { while (true) {} });</script>"
            )
        elif self.path == "/hang-again":
            # Stopping the handler is not enough: the task it queued hangs too.
            body = (
                "<title>Hang</title><input id='h' autofocus>"
                "<script>document.getElementById('h').addEventListener('keydown',"
                " () => { setTimeout(() => { while (true) {} }, 0); while (true) {} });"
                "</script>"
            )
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Keep the test output quiet."""


@pytest.fixture
def server() -> Any:
    _Handler.flip_served = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def service(tmp_path: Path) -> Any:
    printer = MemoryPrinter()
    svc = BrowserTabService(printer, tmp_path / "tab-profile")
    yield svc, printer
    svc.shutdown()


@pytest.fixture
def tool(tmp_path: Path, service: Any) -> Any:
    web = WebUseTool(user_data_dir=str(tmp_path / "agent-profile"), live_browser=service[0])
    yield web
    web.close()


def _wait(pred: Callable[[], Any], what: str, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = pred()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _events(printer: MemoryPrinter, kind: str, **match: Any) -> list[dict[str, Any]]:
    return [
        e
        for e in list(printer.emitted)
        if e["type"] == kind and all(e.get(k) == v for k, v in match.items())
    ]


def _in_service(svc: BrowserTabService, tab_id: str, expression: str) -> Any:
    """Evaluate *expression* on the tab's page through the DAEMON's client."""
    rec = svc._pages[tab_id]
    assert svc._loop is not None
    return asyncio.run_coroutine_threadsafe(rec.page.evaluate(expression), svc._loop).result(10)


def test_show_browser_opens_the_page_in_a_focused_tab_on_every_surface(tool, service, server):
    """The headless page reopens as a Browser tab that every surface switches to,
    with its session cookie, and the tool keeps driving that same page."""
    svc, printer = service
    tool.go_to_url(f"{server}/login")
    tool.go_to_url(f"{server}/form")
    own_pid = tool._browser_pid
    assert own_pid is not None

    tree = tool.show_browser()

    # Announced to everyone, then focused on every surface (the agent wants
    # the user's attention): a broadcast focused event, no connId.
    focused = _events(printer, "openBrowserTab", focus=True)
    assert len(focused) == 1 and focused[0]["tabId"] == "" and "connId" not in focused[0]
    assert focused[0]["popup"] is False
    tab_id = focused[0]["tab_id"]
    assert tool._live and tool._live_tab == tab_id and tool._headless
    # The tool's own Chromium is gone; the daemon's browser is what it drives.
    assert tool._browser_pid is None
    assert tree.startswith("Page: Form") and f"{server}/form" in tree
    assert svc._pages[tab_id].page.url == f"{server}/form"

    # Every web tool now acts on the page the user sees.
    tool.type_text(1, "Ada")
    assert _in_service(svc, tab_id, "document.getElementById('name').value") == "Ada"
    shot_path = Path(svc._profile_dir).parent / "live.png"
    assert tool.screenshot(file_path=str(shot_path)).startswith("Screenshot saved")
    assert shot_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # The headless session's cookie travelled into the tab's profile.
    tool.go_to_url(f"{server}/cookies")
    assert SESSION_COOKIE in tool.get_page_content(text_only=True)
    assert svc._pages[tab_id].page.url == f"{server}/cookies"
    _wait(lambda: _events(printer, "browserState", tab_id=tab_id, url=f"{server}/cookies"), "state")

    assert tool.show_browser() == "Browser is already visible."


def test_user_closing_the_tab_gets_a_fresh_tab_on_the_next_call(tool, service, server):
    """A tab the user closed is replaced, over the same connection, on the next call."""
    svc, printer = service
    svc.open(f"{server}/inert", "user")  # the user's own tab keeps the browser running
    _wait(lambda: _events(printer, "openBrowserTab"), "user tab")
    tool.show_browser()
    first = tool._live_tab
    connection = tool._browser
    svc.close(first)
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=first), "closeBrowserTab")
    assert not tool._is_alive()

    tree = tool.go_to_url(f"{server}/inert")
    assert tree.startswith("Page:")
    assert tool._live_tab != first and tool._live_tab in svc._pages
    assert tool._browser is connection


def test_browser_exit_after_the_last_tab_closes_is_survived(tool, service, server):
    """Closing the agent's only tab exits the daemon browser; the next call
    relaunches it and reconnects (the old connection is dead)."""
    svc, printer = service
    tool.show_browser()
    first = tool._live_tab
    connection = tool._browser
    svc.close(first)
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=first), "closeBrowserTab")
    _wait(lambda: svc._context is None, "browser exited")

    assert tool.go_to_url(f"{server}/inert").startswith("Page:")
    assert tool._live_tab in svc._pages and tool._browser is not connection


def test_target_blank_popup_is_followed(tool, service, server):
    """A link the agent clicks that opens a new page moves the tool to it (as headless does)."""
    svc, _printer = service
    tool.show_browser()
    tool.go_to_url(f"{server}/form")
    tool.click(2)
    _wait(lambda: tool._page.url == f"{server}/inert", "popup adopted")
    # The popup is a real tab on every surface too.
    _wait(lambda: len(svc._pages) == 2, "popup registered")


def test_back_to_headless_closes_the_tab_and_carries_cookies_back(tool, service, server):
    """``show_browser(False)`` closes the agent's tab everywhere and continues
    headless on the same page, with the cookies set while in the tab."""
    svc, printer = service
    tool.go_to_url(f"{server}/inert")
    tool.show_browser()
    tab_id = tool._live_tab
    tool.go_to_url(f"{server}/tab-login")  # "the user logged in" while watching

    tree = tool.show_browser(visible=False)

    assert tree.startswith("Page:") and tool._page.url == f"{server}/tab-login"
    assert not tool._live and tool._live_tab is None and tool._headless
    assert tool._browser_pid is not None  # the tool's own Chromium again
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "closeBrowserTab")
    assert tab_id not in svc._pages
    tool.go_to_url(f"{server}/cookies")
    assert TAB_COOKIE in tool.get_page_content(text_only=True)
    assert tool.show_browser(visible=False) == "Browser is already headless."


def test_task_end_leaves_the_tab_open_for_the_user(tool, service, server):
    """``close()`` (end of a task) only disconnects: the page stays on every surface."""
    svc, printer = service
    tool.go_to_url(f"{server}/inert")
    tool.show_browser()
    tab_id = tool._live_tab

    assert tool.close() == "Browser closed."

    assert tool._browser is None and tool._page is None and tool._live_tab is None
    assert tab_id in svc._pages and not svc._pages[tab_id].page.is_closed()
    assert not _events(printer, "closeBrowserTab", tab_id=tab_id)
    # Cookies live in the tab's persistent profile, not in the tool.
    assert _in_service(svc, tab_id, "location.href") == f"{server}/inert"


def test_close_browser_closes_the_tab_and_the_next_call_reopens_one(tool, service, server):
    """The explicit ``close_browser`` tool closes the agent's tab everywhere;
    browsing stays in the Browser tab afterwards."""
    svc, printer = service
    tool.show_browser()
    tab_id = tool._live_tab

    assert tool.close_browser().startswith("Browser closed.")

    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "closeBrowserTab")
    assert tool._live and tool._live_tab is None and tool._browser is None
    tool.go_to_url(f"{server}/inert")
    assert tool._live_tab is not None and tool._live_tab != tab_id
    assert tool._live_tab in svc._pages


def test_service_shutdown_while_attached_is_reported_and_recoverable(tool, service, server):
    """The daemon browser exiting (the service shut down) drops the connection;
    the tool reports the error and ``show_browser(False)`` goes back to headless."""
    svc, _printer = service
    tool.show_browser()
    svc.shutdown()
    assert not tool._is_alive()

    err = tool.go_to_url(f"{server}/inert")
    assert err.startswith("Error") and "shut down" in err
    tree = tool.show_browser(visible=False)
    assert tree == "Browser is now headless." and tool._is_alive() and not tool._live


def test_show_browser_reports_an_unavailable_service_and_stays_usable(tmp_path, service, server):
    """When the Browser tab cannot be opened the tool says so and keeps browsing headless."""
    svc, _printer = service
    svc.shutdown()
    web = WebUseTool(user_data_dir=str(tmp_path / "p"), live_browser=svc)
    try:
        web.go_to_url(f"{server}/inert")
        err = web.show_browser()
        assert err.startswith("Error making the browser visible:") and "shut down" in err
        assert not web._live and web._headless
        assert web.go_to_url(f"{server}/inert").startswith("Page:")
    finally:
        web.close()


def test_show_browser_without_a_page_reports_the_tab(tool, service):
    """Switching before any navigation opens a blank tab and says where the browser is."""
    svc, _printer = service
    assert tool.show_browser() == "Browser is now visible in the Browser tab."
    assert tool._is_alive() and tool._live_tab in svc._pages


def test_find_live_page_gives_up_on_an_unknown_target(tool, service, monkeypatch):
    """A target id no page carries is reported after the deadline (rescans in between)."""
    tool.show_browser()
    monkeypatch.setattr(wut, "_LIVE_PAGE_TIMEOUT", 0.5)
    with pytest.raises(RuntimeError, match="did not show up"):
        tool._find_live_page("no-such-target")


def test_tab_browser_hides_its_headless_user_agent(tool, service, server):
    """The daemon's browser (headless on a server) must not announce ``HeadlessChrome``
    to sites, or the user would meet bot challenges in every streamed tab."""
    svc, _printer = service
    tool.show_browser()
    tool.go_to_url(f"{server}/ua")
    assert "HeadlessChrome" not in tool.get_page_content(text_only=True)
    assert "HeadlessChrome" not in _in_service(svc, tool._live_tab, "navigator.userAgent")


def test_agent_tab_cdp_url_points_at_the_browser(service):
    """``open_for_agent`` returns the DevTools endpoint recorded by the browser."""
    svc, printer = service
    tab = svc.open_for_agent()
    assert tab.tab_id in svc._pages and tab.target_id
    assert tab.cdp_url.startswith("http://127.0.0.1:")
    assert _events(printer, "openBrowserTab", tab_id=tab.tab_id, focus=True)
    assert svc.tab_for_target(tab.target_id) == tab.tab_id
    assert svc.tab_for_target("no-such-target") is None
    svc.interrupt("browser__999")  # unknown tab: a no-op, never an error
    svc.interrupt(tab.tab_id)  # a responsive page just keeps running
    time.sleep(0.5)
    assert not svc._pages[tab.tab_id].page.is_closed()


def test_agent_hands_its_live_browser_to_the_web_tool(service):
    """The ``show_browser`` tool the agent registers reaches the daemon's service
    (``run(live_browser=...)`` stores it; ``_get_tools`` passes it on)."""
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    svc, printer = service
    agent = ChatSorcarAgent("live-browser-wiring-test")
    agent._live_browser = svc  # what run(live_browser=svc) records
    try:
        tools = {t.__name__: t for t in agent._get_tools()}
        assert tools["show_browser"]() == "Browser is now visible in the Browser tab."
        assert _events(printer, "openBrowserTab", focus=True)
        assert agent.web_use_tool is not None and agent.web_use_tool._live_tab in svc._pages
    finally:
        if agent.web_use_tool is not None:
            agent.web_use_tool.close()


def test_browser_that_cannot_launch_is_reported_by_show_browser(
    tmp_path, service, server, monkeypatch
):
    """A launch failure surfaces as the tool's error and browsing continues headless."""
    svc, printer = service
    fake = tmp_path / "not-a-browser"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("KISS_BROWSER", str(fake))
    monkeypatch.setenv("KISS_HEADLESS", "1")
    web = WebUseTool(user_data_dir=str(tmp_path / "p"), live_browser=svc)
    try:
        web.go_to_url(f"{server}/inert")
        err = web.show_browser()
        assert err.startswith("Error making the browser visible: Cannot open a Browser tab:")
        assert not web._live and not _events(printer, "openBrowserTab")
        assert web.go_to_url(f"{server}/inert").startswith("Page:")
    finally:
        web.close()


def test_following_a_popup_moves_tab_ownership(tool, service, server):
    """After following a ``target=_blank`` link the popup is the tool's page:
    ``close_browser`` closes THAT tab, and a user closing it is noticed."""
    svc, printer = service
    tool.show_browser()
    first = tool._live_tab
    tool.go_to_url(f"{server}/form")
    tool.click(2)
    _wait(lambda: tool._page.url == f"{server}/inert", "popup adopted")
    _wait(lambda: len(svc._pages) == 2, "popup registered")
    popup = next(t for t in svc._pages if t != first)
    assert tool._live_tab_id() == popup  # what the hang watchdog would interrupt

    tool.close_browser()

    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=popup), "popup closed")
    assert first in svc._pages and not svc._pages[first].page.is_closed()


def test_user_navigation_in_the_tab_is_what_comes_back_headless(tool, service, server):
    """The page the USER navigated to in the tab (not the tool's last ``go_to_url``)
    is the one reopened headless."""
    svc, printer = service
    tool.go_to_url(f"{server}/inert")
    tool.show_browser()
    tab_id = tool._live_tab
    svc.navigate(tab_id, "go", f"{server}/form")
    _wait(lambda: _events(printer, "browserState", tab_id=tab_id, title="Form"), "user nav")

    tree = tool.show_browser(visible=False)

    assert tree.startswith("Page: Form") and tool._page.url == f"{server}/form"


def test_local_storage_of_the_page_travels_both_ways(tool, service, server):
    """Token-based logins live in localStorage; the page origin's entries move
    into the tab and back, following the latest state each way."""
    svc, _printer = service
    tool.go_to_url(f"{server}/inert")
    tool._page.evaluate("localStorage.setItem('token', 'agent-login')")

    tool.show_browser()
    assert _in_service(svc, tool._live_tab, "localStorage.getItem('token')") == "agent-login"
    _in_service(svc, tool._live_tab, "localStorage.setItem('token', 'user-login')")

    tool.show_browser(visible=False)
    assert tool._page.evaluate("localStorage.getItem('token')") == "user-login"


def test_a_cookie_deleted_while_live_does_not_come_back_headless(tool, service, server):
    """A persistent cookie carried into the tab and deleted there (logout) must
    not be resurrected from the headless profile's disk copy; the tab's other
    cookies are untouched."""
    svc, _printer = service
    tool.go_to_url(f"{server}/persist-login")
    tool.show_browser()
    tool.go_to_url(f"{server}/tab-login")  # a cookie of the tab's own
    tool.go_to_url(f"{server}/logout")
    tool.go_to_url(f"{server}/cookies")
    assert PERSISTENT_COOKIE not in tool.get_page_content(text_only=True)

    tool.show_browser(visible=False)
    tool.go_to_url(f"{server}/cookies")
    served = tool.get_page_content(text_only=True)
    assert PERSISTENT_COOKIE not in served and TAB_COOKIE in served
    # Back in the tab: the cookie stays gone there too, its own cookie remains.
    tool.show_browser()
    tool.go_to_url(f"{server}/cookies")
    served = tool.get_page_content(text_only=True)
    assert PERSISTENT_COOKIE not in served and TAB_COOKIE in served


def test_hung_input_in_the_tab_is_interrupted_without_touching_the_browser(
    tool, service, server
):
    """A key handler spinning forever must not wedge the tool: the daemon stops
    the script (or closes just that tab); the user's browser keeps running."""
    svc, _printer = service
    tool.show_browser()
    tool.go_to_url(f"{server}/hang")
    started = time.monotonic()
    result = tool.press_key("a")
    assert time.monotonic() - started < wut._INPUT_WATCHDOG_SECS + 20
    assert isinstance(result, str)
    assert svc._context is not None
    assert tool.go_to_url(f"{server}/inert").startswith("Page:")


def test_page_that_stays_wedged_after_interrupt_is_closed_alone(tool, service, server):
    """When stopping the script does not free the page, only that tab is closed;
    the user's other tab and the browser survive, and the tool recovers."""
    svc, printer = service
    svc.open(f"{server}/inert", "user")
    _wait(lambda: _events(printer, "openBrowserTab"), "user tab")
    tool.show_browser()
    tab_id = tool._live_tab
    tool.go_to_url(f"{server}/hang-again")
    started = time.monotonic()
    result = tool.press_key("a")
    assert time.monotonic() - started < wut._INPUT_WATCHDOG_SECS + 30
    assert isinstance(result, str)
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "hung tab closed")
    assert svc._context is not None and len(svc._pages) == 1
    assert tool.go_to_url(f"{server}/inert").startswith("Page:")


def test_local_storage_is_not_written_to_a_redirect_destination(tool, service, server):
    """A token set on one origin must not be handed to the origin a redirect lands on."""
    svc, _printer = service
    tool.go_to_url(f"{server}/flip")  # served once; reopening it redirects to localhost
    tool._page.evaluate("localStorage.setItem('token', 'SECRET')")

    tree = tool.show_browser()

    assert tool._page.url.startswith("http://localhost:")
    assert tree.startswith("Page:")
    assert _in_service(svc, tool._live_tab, "localStorage.getItem('token')") is None

    # The skipped write is not a transfer: back on the original origin the
    # token's absence in the tab must not delete it from the headless profile.
    tool.go_to_url(f"{server}/inert")
    tool.show_browser(visible=False)
    assert tool._page.evaluate("localStorage.getItem('token')") == "SECRET"


def test_local_storage_removed_while_live_stays_removed(tool, service, server):
    """A logout that deletes a localStorage key in the tab survives the switch back."""
    svc, _printer = service
    tool.go_to_url(f"{server}/inert")
    tool._page.evaluate("localStorage.setItem('token', 'original-login')")
    tool.show_browser()
    assert _in_service(svc, tool._live_tab, "localStorage.getItem('token')") == "original-login"
    _in_service(svc, tool._live_tab, "localStorage.removeItem('token')")

    tool.show_browser(visible=False)

    assert tool._page.evaluate("localStorage.getItem('token')") is None


def test_switching_after_the_page_was_closed_deletes_nothing(tool, service, server):
    """``close_browser`` then ``show_browser(False)`` has no session to capture;
    that must not be mistaken for "every carried cookie was deleted"."""
    tool.go_to_url(f"{server}/persist-login")
    tool.show_browser()
    tool.close_browser()

    assert tool.show_browser(visible=False) == "Browser is now headless."

    tool.go_to_url(f"{server}/cookies")
    assert PERSISTENT_COOKIE in tool.get_page_content(text_only=True)

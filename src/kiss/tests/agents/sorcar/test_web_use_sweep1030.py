# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-30, slice S4: browser-tool findings in web_use_tool.py, end to end.

A REAL stealth Chromium driven by ``WebUseTool``, a REAL Chromium the
"user" started with ``--remote-debugging-port`` (driven over its own
DevTools HTTP endpoints, with no second Playwright client on it), a REAL
daemon ``BrowserTabService`` and a REAL Chrome-bridge extension peer over
a loopback WebSocket; no mocks, patches or test doubles.

Branches not covered here, and why (unreachable without test doubles):

* ``_kill_driver``'s ``except`` arm: the driver would have to exit on its
  own, or Playwright's private attribute chain change, at the very
  instant the watchdog fires on a call it is blocking.
* ``evaluate_script``'s "no CDP session" return: attaching the session
  would have to fail while the page still answered the reads before it.
"""

from __future__ import annotations

import http.server
import json
import re
import socket
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import web_use_tool as wut
from kiss.agents.sorcar.chrome_bridge import CONFIG_FILE_NAME, ChromeBridge, extension_dir
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.models.model import _tool_summary_and_params
from kiss.core.processes import kill_process_tree, popen_process_group
from kiss.server.browser_tab import BrowserTabService
from kiss.server.standalone_app import playwright_chromium
from kiss.tests.agents.sorcar.test_chrome_bridge import ExtensionPeer
from kiss.tests.server._memory_printer import MemoryPrinter

pytestmark = pytest.mark.skipif(
    not (Path.home() / ".cache" / "ms-playwright").is_dir(),
    reason="Playwright browsers not installed",
)

# How long a page script keeps the renderer busy: longer than the 3 s
# bound of a CDP round trip, so a Runtime.enable sent meanwhile times out.
_SPIN_MS = 5000

_PAGES = {
    "/page-a": "<title>page-a</title><p>first tab</p>",
    "/page-b": "<title>page-b</title><p>second tab</p>",
    "/keyhang": (
        "<title>keyhang</title><p>keys spin</p>"
        "<script>addEventListener('keydown', () => { while (true) {} });</script>"
    ),
    # V8 formats a logged error once for its own message store, and once
    # more for every DevTools client with the Runtime domain enabled: the
    # title says how often the error's name was turned into a string.
    "/probe": (
        "<title>probe</title><script>"
        "let n = 0; const e = new Error('probe');"
        "e.name = { toString() { n++; return 'Error'; } };"
        "console.log(e); document.title = 'n=' + n;"
        "</script>"
    ),
    "/opener": "<title>opener</title><p>opener</p>",
    "/busy": (
        "<title>busy</title><p>busy</p><script>console.log('before');"
        "function spin(ms) { setTimeout(() => {"
        " const end = Date.now() + ms; while (Date.now() < end) {} }, 0); }"
        "</script>"
    ),
    "/details": (
        "<title>details</title>"
        "<details><summary>More info</summary><p>Order confirmed</p>Loose text</details>"
        "<details open><summary>Open one</summary><p>Shown text</p></details>"
    ),
    "/slot": (
        "<title>slot</title><slot>Saved</slot><div>x</div><slot><span>Slotted span</span></slot>"
    ),
}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        body = _PAGES.get(self.path, "<title>missing</title>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server API
        pass


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port: int = sock.getsockname()[1]
    sock.close()
    return port


def _poll(probe: Callable[[], Any], timeout: float = 20.0) -> Any:
    """Call *probe* until it returns a truthy value or *timeout* passes; return the last value."""
    deadline = time.monotonic() + timeout
    while True:
        value = probe()
        if value or time.monotonic() > deadline:
            return value
        time.sleep(0.2)


class _UserChrome:
    """A Chromium the user started with remote debugging, driven over its HTTP endpoints.

    Every tab lives in the browser's default context, like the tabs of a
    real Chrome window; ``chrome://`` tabs can be opened too.
    """

    def __init__(self, profile: Path, first_url: str) -> None:
        # Asked on a thread of its own: the lookup starts a sync Playwright,
        # which must not share a thread with the tool's.
        with ThreadPoolExecutor(1) as pool:
            executable = pool.submit(playwright_chromium).result(timeout=120)
        if executable is None:
            pytest.skip("Playwright's Chromium is not installed")
        self.url = f"http://127.0.0.1:{_free_port()}"
        self.proc = popen_process_group(
            [
                str(executable),
                "--headless=new",
                "--no-sandbox",
                "--no-first-run",
                "--no-default-browser-check",
                f"--remote-debugging-port={self.url.rsplit(':', 1)[1]}",
                f"--user-data-dir={profile}",
                first_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert _poll(self._answers), "the user's Chrome did not start"

    def _get(self, path: str, method: str = "GET") -> str:
        request = urllib.request.Request(self.url + path, method=method)
        with urllib.request.urlopen(request, timeout=10) as reply:
            return str(reply.read().decode())

    def _answers(self) -> bool:
        try:
            return bool(self._get("/json/version"))
        except OSError:
            return False

    def tabs(self) -> list[dict[str, Any]]:
        return [t for t in json.loads(self._get("/json/list")) if t["type"] == "page"]

    def open_tab(self, url: str, title: str) -> str:
        """Open *url* in a new tab, wait until it shows *title*, and return its target id."""
        target_id = str(json.loads(self._get(f"/json/new?{url}", "PUT"))["id"])
        assert _poll(lambda: any(t["title"] == title for t in self.tabs())), title
        return target_id

    def close_tab(self, target_id: str) -> None:
        self._get(f"/json/close/{target_id}")

    def stop(self) -> None:
        """Kill the browser with its whole process group, once (never a reaped leader's pid)."""
        if self.proc.returncode is None:
            kill_process_tree(self.proc)
            self.proc.wait(timeout=30)


@pytest.fixture
def server() -> Any:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def web(tmp_path: Path) -> Any:
    tool = WebUseTool(user_data_dir=str(tmp_path / "agent"), headless=True)
    yield tool
    tool.close()


@pytest.fixture
def user_chrome(tmp_path: Path, server: str) -> Any:
    chrome = _UserChrome(tmp_path / "user-chrome", f"{server}/page-a")
    assert _poll(lambda: any(t["title"] == "page-a" for t in chrome.tabs()))
    yield chrome
    chrome.stop()


@pytest.fixture
def bridge_state() -> Any:
    """No bridge singleton and no bridge config file before or after the test."""
    ChromeBridge.shutdown()
    (extension_dir() / CONFIG_FILE_NAME).unlink(missing_ok=True)
    yield
    ChromeBridge.shutdown()
    (extension_dir() / CONFIG_FILE_NAME).unlink(missing_ok=True)


def _tab_index(tool: WebUseTool, title: str) -> int:
    match = re.search(rf"\[(\d+)\] {title} - ", tool.go_to_url("tab:list"))
    assert match, title
    return int(match.group(1))


def _guidance(tool: WebUseTool) -> str:
    """get_page_content() when it says how to reconnect, else empty."""
    content = tool.get_page_content()
    return content if "attach_to_chrome" in content else ""


def _page_shown(tool: WebUseTool) -> str:
    """get_page_content() when it shows a page (not an error), else empty."""
    content = tool.get_page_content()
    return content if content.startswith("Page: ") else ""


# -- attach_to_chrome --------------------------------------------------------


def _press_into_a_spinning_handler(tool: WebUseTool, chrome_url: str, out: dict[str, Any]) -> None:
    """Attach, press a key whose handler never returns, then read the page (worker thread)."""
    try:
        tool.attach_to_chrome(chrome_url)
        out["attached"] = tool.go_to_url(f"tab:{_tab_index(tool, 'keyhang')}")
        started = time.monotonic()
        out["press"] = tool.press_key("a")
        out["press_secs"] = time.monotonic() - started
        out["after"] = tool.get_page_content()
    finally:
        tool.close()
        out["done"] = True


def test_hung_key_handler_in_an_attached_chrome_is_cut_off_without_touching_it(
    tmp_path: Path, server: str, user_chrome: _UserChrome
) -> None:
    """The raw-input watchdog also guards an attached Chrome: it ends the hung
    call by killing the tool's own driver, never the user's browser, and the
    next call says how to reconnect."""
    user_chrome.open_tab(f"{server}/keyhang", "keyhang")
    tool = WebUseTool(user_data_dir=str(tmp_path / "agent"), headless=True)
    out: dict[str, Any] = {}
    worker = threading.Thread(
        target=_press_into_a_spinning_handler, args=(tool, user_chrome.url, out), daemon=True
    )
    worker.start()
    try:
        worker.join(wut._INPUT_WATCHDOG_SECS + 45)
        assert out.get("done"), f"press_key hung; progress so far: {sorted(out)}"
        assert str(out["attached"]).startswith("Page: keyhang\n")
        assert str(out["press"]).startswith("Error pressing key 'a': ")
        assert out["press_secs"] < wut._INPUT_WATCHDOG_SECS + 20
        assert "disconnected" in out["after"] and "attach_to_chrome" in out["after"]
        assert user_chrome.proc.poll() is None
        assert f"{server}/keyhang" in [t["url"] for t in user_chrome.tabs()]
    finally:
        user_chrome.stop()
        worker.join(30)


def test_a_crashed_attached_tab_is_not_reported_as_a_lost_chrome(
    tmp_path: Path, user_chrome: _UserChrome
) -> None:
    tool = WebUseTool(user_data_dir=str(tmp_path / "agent"), headless=True)
    try:
        assert tool.attach_to_chrome(user_chrome.url).startswith("Page: page-a\n")
        tool.go_to_url("chrome://crash")
        content = _poll(lambda: _guidance(tool))
        assert content.startswith("Error getting page content: The attached tab crashed."), content
        assert "disconnected" not in content
        assert user_chrome.proc.poll() is None
    finally:
        tool.close()


def test_show_browser_from_an_attached_chrome_carries_the_tab_being_driven(
    tmp_path: Path, server: str, user_chrome: _UserChrome
) -> None:
    service = BrowserTabService(MemoryPrinter(), tmp_path / "tab-profile")
    tool = WebUseTool(user_data_dir=str(tmp_path / "agent"), live_browser=service)
    try:
        assert tool.attach_to_chrome(user_chrome.url).startswith("Page: page-a\n")
        # The user opens another tab, the newest one, while the agent
        # keeps driving page-a.
        user_chrome.open_tab(f"{server}/page-b", "page-b")
        assert _poll(lambda: "] page-b - " in tool.go_to_url("tab:list"))
        # The page the agent drives moves into the Browser tab.
        assert tool.show_browser().startswith("Page: page-a\n")
    finally:
        tool.close()
        service.shutdown()


def test_closing_the_driven_tab_never_moves_the_agent_onto_chrome_ui(
    tmp_path: Path, server: str, user_chrome: _UserChrome
) -> None:
    tab_b = user_chrome.open_tab(f"{server}/page-b", "page-b")
    tool = WebUseTool(user_data_dir=str(tmp_path / "agent"), headless=True)
    try:
        tool.attach_to_chrome(user_chrome.url)
        assert tool.go_to_url(f"tab:{_tab_index(tool, 'page-b')}").startswith("Page: page-b\n")
        # Opened after the attach, Chrome's own page is the newest tab the
        # tool knows of when the user closes the driven one.
        user_chrome.open_tab("chrome://version", "About Version")
        assert _poll(lambda: "] About Version - " in tool.go_to_url("tab:list"))
        user_chrome.close_tab(tab_b)
        content = _poll(lambda: _page_shown(tool))
        assert content.startswith("Page: page-a\n"), content
    finally:
        tool.close()


# -- console capture -----------------------------------------------------------


def test_stealth_pages_see_no_runtime_domain_until_console_messages_are_listed(
    web: WebUseTool, server: str
) -> None:
    assert web.go_to_url(f"{server}/probe").startswith("Page: n=1\n")
    # evaluate_script needs no Runtime domain either.
    assert web.evaluate_script("() => document.title") == "n=1"
    assert web.go_to_url(f"{server}/probe").startswith("Page: n=1\n")
    # The first listing starts capture and replays what this document logged.
    assert web.list_console_messages().count("[log] Error: probe") == 1
    # From then on every document of the page is captured.
    web.go_to_url(f"{server}/probe")
    listing = _poll(lambda: web.list_console_messages().count("[log] Error: probe") == 2)
    assert listing, web.list_console_messages()


def test_console_capture_retried_on_a_busy_page_records_each_message_once(
    web: WebUseTool, server: str
) -> None:
    """A Runtime.enable that times out on a busy renderer still completes later:
    retrying it must not attach a second session whose listeners record
    every message again."""
    web.go_to_url(f"{server}/opener")
    assert web.evaluate_script("() => { window.popup = window.open('/busy'); return 1; }") == "1"
    assert _poll(lambda: web.evaluate_script("() => typeof window.popup.spin") == "function"), (
        "the popup did not load"
    )
    index = _tab_index(web, "busy")
    # The popup shares the opener's renderer: adopt it while that is busy.
    assert web.evaluate_script(f"() => {{ window.popup.spin({_SPIN_MS}); return 1; }}") == "1"
    web.go_to_url(f"tab:{index}")
    # List while the popup's renderer is busy: Runtime.enable times out.
    assert web.evaluate_script(f"() => {{ spin({_SPIN_MS}); return 1; }}") == "1"
    web.list_console_messages()
    assert web.evaluate_script("() => { console.log('hello'); return 1; }") == "1"
    listing = web.list_console_messages()
    assert listing.count("[log] before") == 1, listing
    assert listing.count("[log] hello") == 1, listing


# -- evaluate_script and wait_for ----------------------------------------------


def test_evaluate_script_whose_last_line_ends_in_a_comment(web: WebUseTool, server: str) -> None:
    web.go_to_url(f"{server}/page-a")
    assert web.evaluate_script("document.title // the title") == "page-a"
    assert web.evaluate_script("() => 40 + 2 // the answer") == "42"


def test_wait_for_treats_the_body_of_a_closed_details_as_hidden(
    web: WebUseTool, server: str
) -> None:
    web.go_to_url(f"{server}/details")
    assert web.wait_for("Order confirmed", timeout=1).startswith("Timed out after 1s")
    assert web.wait_for("Loose text", timeout=1).startswith("Timed out after 1s")
    assert web.wait_for("Order confirmed", timeout=1, disappear=True).startswith("Page: details")
    assert web.wait_for("More info", timeout=1).startswith("Page: details")
    assert web.wait_for("Shown text", timeout=1).startswith("Page: details")
    web.evaluate_script("() => { document.querySelector('details').open = true; return 1; }")
    assert web.wait_for("Order confirmed", timeout=5).startswith("Page: details")
    assert web.wait_for("Loose text", timeout=5).startswith("Page: details")


def test_wait_for_reads_a_slot_outside_any_shadow_tree(web: WebUseTool, server: str) -> None:
    web.go_to_url(f"{server}/slot")
    assert web.wait_for("Saved", timeout=1).startswith("Page: slot")
    assert web.wait_for("Slotted span", timeout=1).startswith("Page: slot")
    assert web.wait_for("Saved", timeout=1, disappear=True).startswith("Timed out after 1s")


# -- use_my_chrome ---------------------------------------------------------------


def _answer_late(command: dict[str, Any]) -> Any:
    """The extension's reply to list_tabs, a moment after the command arrived."""
    time.sleep(0.3)
    return [{"id": 1, "title": "Inbox", "url": "https://mail.example/", "active": True}]


def test_use_my_chrome_never_waits_less_than_a_second(bridge_state: None) -> None:
    """A zero or negative timeout must not time the command out at once (and
    drop the extension's connection) while the user's Chrome still runs it."""
    bridge = ChromeBridge.instance()
    peer = ExtensionPeer(bridge.port, bridge.token, _answer_late)
    try:
        for timeout in (0, -5):
            out = WebUseTool().use_my_chrome("list_tabs", timeout=timeout)
            assert "id=1 [active] Inbox — https://mail.example/" in out, out
            assert bridge.extension_connected
    finally:
        peer.close()


def test_use_my_chrome_description_names_no_stock_brand_path(bridge_state: None) -> None:
    """The tool description goes to the model verbatim; the extension's real
    directory and name come from the error returned while none is connected."""
    _summary, description, _params = _tool_summary_and_params(
        WebUseTool().use_my_chrome, "use_my_chrome"
    )
    assert "~/.kiss" not in description
    assert "KISS Sorcar" not in description
    out = WebUseTool().use_my_chrome("list_tabs")
    assert str(extension_dir()) in out, out

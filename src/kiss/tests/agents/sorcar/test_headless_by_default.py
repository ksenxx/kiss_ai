# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``WebUseTool`` browses headless by default.

A visible Chromium window steals the user's focus and cannot run on a
machine without a display, so every page visit now happens in a headless
browser.  These tests drive a REAL Chromium (no mocks) and check the three
properties that make headless browsing usable:

1. ``headless`` is the default, for the tool and for the agent that owns it.
2. Screenshots still capture the rendered page, at full viewport
   resolution, and really contain the page's pixels.
3. The headless ``HeadlessChrome`` user-agent token — which many sites
   answer with a bot challenge instead of content — reaches neither the
   server nor page JavaScript.

``show_browser`` is exercised through its relaunch path, which restores the
page the tool was on.
"""

from __future__ import annotations

import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool

_RED_PAGE = "data:text/html,<body style='margin:0;background:%23ff0000'>red</body>"
_BLUE_PAGE = "data:text/html,<body style='margin:0;background:%230000ff'>blue</body>"


def _png_size(path: Path) -> tuple[int, int]:
    """Return the (width, height) recorded in a PNG file's IHDR chunk.

    Args:
        path: PNG file to inspect.

    Returns:
        Pixel width and height.
    """
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    return struct.unpack(">II", data[16:24])


SESSION_COOKIE = "sid=only-in-memory"


class _ProbeHandler(BaseHTTPRequestHandler):
    """Serve the three probe pages the tests below navigate to.

    ``/ua`` echoes the request's ``User-Agent``, ``/login`` sets a
    session-only cookie, ``/cookies`` echoes the request's ``Cookie``
    header, and any other path is an inert page.
    """

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Answer the request according to its path."""
        body = "a page"
        cookie = ""
        if self.path == "/ua":
            body = self.headers.get("User-Agent", "")
        elif self.path == "/login":
            body = "logged in"
            cookie = f"{SESSION_COOKIE}; Path=/"
        elif self.path == "/cookies":
            body = self.headers.get("Cookie", "(none)")
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default stderr request log.

        Args:
            format: Printf-style template the base class would have logged.
            args: Values for *format*.
        """


@pytest.fixture
def server():
    """Yield the base URL of a local server serving the probe pages."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ProbeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def tool(tmp_path):
    """Yield a default-configured WebUseTool with a throwaway profile."""
    web = WebUseTool(user_data_dir=str(tmp_path / "profile"), work_dir=str(tmp_path))
    try:
        yield web
    finally:
        web.close()


def test_headless_is_the_default():
    """Constructing the tool without arguments must select headless."""
    web = WebUseTool(user_data_dir=None)
    try:
        assert web._headless is True
    finally:
        web.close()


def test_agent_browser_is_headless():
    """The browser the agent hands to the model must be headless."""
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

    agent = ChatSorcarAgent("headless-default-test")
    try:
        agent._get_tools()
        assert agent.web_use_tool is not None
        assert agent.web_use_tool._headless is True
        assert "show_browser" in {t.__name__ for t in agent.web_use_tool.get_tools()}
    finally:
        if agent.web_use_tool is not None:
            agent.web_use_tool.close()


def test_screenshot_in_headless_browser_captures_the_page(tool, tmp_path):
    """A headless screenshot is a full-resolution PNG of the rendered page."""
    tool.go_to_url(_RED_PAGE)
    red = tmp_path / "shots" / "red.png"
    assert tool.screenshot(str(red)) == f"Screenshot saved to {red}"
    assert red.is_file()

    width, height = _png_size(red)
    scale = tool._context_args()["device_scale_factor"]
    assert (width, height) == (tool.viewport[0] * scale, tool.viewport[1] * scale)

    # A blank/stub capture would be byte-identical for both pages; real
    # rendering makes a red page differ from a blue one.
    tool.go_to_url(_BLUE_PAGE)
    blue = tmp_path / "shots" / "blue.png"
    tool.screenshot(str(blue))
    assert blue.read_bytes() != red.read_bytes()


def test_headless_user_agent_is_not_advertised(tool, server):
    """Neither the server nor page JavaScript may see "HeadlessChrome"."""
    tree = tool.go_to_url(f"{server}/ua")
    assert "Error" not in tree.splitlines()[0]

    served = tool.get_page_content(text_only=True)
    assert "HeadlessChrome" not in served
    assert "Chrome/" in served
    assert "HeadlessChrome" not in tool._page.evaluate("navigator.userAgent")


def test_show_browser_relaunches_and_restores_the_page(tool):
    """Switching visibility restarts the browser on the same page."""
    tool.go_to_url(_RED_PAGE)
    assert tool.show_browser(visible=False) == "Browser is already headless."

    # Pretend the session had been made visible, then switch back: the
    # browser must be torn down, relaunched headless, and re-navigated.
    old_pid = tool._browser_pid
    tool._headless = False
    tree = tool.show_browser(visible=False)
    assert tool._headless is True
    assert tool._browser_pid not in (None, old_pid)
    assert tree.startswith("Page:")
    assert tool._page.url.startswith("data:text/html,")


def test_show_browser_keeps_the_session_cookie(tool, server):
    """A session-only cookie must survive the switch to a visible window.

    A login or bot check is usually mid-flight when the human is called
    in, and its state lives in a cookie that dies with the browser
    process, so it has to be handed to the relaunched browser.
    """
    tool.go_to_url(f"{server}/login")
    tool.go_to_url(f"{server}/inert")

    tool._headless = False
    tool.show_browser(visible=False)

    tool.go_to_url(f"{server}/cookies")
    assert SESSION_COOKIE in tool.get_page_content(text_only=True)


def test_show_browser_keeps_the_tab_the_user_opened(tool, server):
    """The tab opened last in the visible window is the one carried over."""
    tool.go_to_url(f"{server}/inert")
    opened_by_user = tool._context.new_page()
    opened_by_user.goto(f"{server}/cookies")

    tool._headless = False
    tool.show_browser(visible=False)

    assert tool._page.url == f"{server}/cookies"


def test_show_browser_without_an_open_page_reports_state(tmp_path):
    """Switching visibility before any navigation just launches the browser."""
    web = WebUseTool(user_data_dir=str(tmp_path / "profile"))
    try:
        assert web.show_browser(visible=False) == "Browser is now headless."
        assert web._headless is True
        assert web._is_alive()
    finally:
        web.close()

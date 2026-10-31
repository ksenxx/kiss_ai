# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing while ``show_browser`` attaches closes the Browser tab it opened.

``WebUseTool._attach_live`` asks the daemon's :class:`BrowserTabService`
for a tab (every surface shows it at once), then connects over CDP and
adopts the tab's page, which can take seconds.  A Stop is delivered as
:class:`ToolCallInterrupted`, a ``BaseException``; the clean-up arm
caught only ``Exception``, so a Stop landing in that window left a blank
tab open on every surface.  The test runs a REAL service and a REAL
tool, and lands the Stop the way the daemon does (an exception injected
into the tool's thread) at the entry of the CDP connection check, through
the :class:`Pause` trace hook.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import web_stealth
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.tool_interrupt import ToolCallInterrupted, inject_async_exception
from kiss.server.browser_tab import BrowserTabService
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.pause import Pause

pytestmark = pytest.mark.skipif(
    not (Path.home() / ".cache" / "ms-playwright").is_dir(),
    reason="Playwright browsers not installed",
)


class _Handler(BaseHTTPRequestHandler):
    """Serve one small page for every path."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Answer with an inert page."""
        body = b"<html><head><title>Inert</title></head><body>a page</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep the test output quiet."""


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def service(tmp_path: Path) -> Iterator[tuple[BrowserTabService, MemoryPrinter]]:
    printer = MemoryPrinter()
    svc = BrowserTabService(printer, tmp_path / "tab-profile")
    yield svc, printer
    svc.shutdown()
    web_stealth.stop_virtual_display()


@pytest.fixture
def tool(tmp_path: Path, service: tuple[BrowserTabService, MemoryPrinter]) -> Iterator[WebUseTool]:
    web = WebUseTool(user_data_dir=str(tmp_path / "agent-profile"), live_browser=service[0])
    yield web
    web.close()


def _wait(pred: Callable[[], Any], what: str, timeout: float = 15) -> Any:
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


def _stop_once(fired: threading.Event, _locals: dict[str, Any]) -> bool:
    """Pause matcher: on the first entry, deliver a Stop to the entering thread."""
    if not fired.is_set():
        fired.set()
        inject_async_exception(threading.get_ident(), ToolCallInterrupted)
    return False


def test_stop_while_attaching_closes_the_tab_it_opened(
    tool: WebUseTool, service: tuple[BrowserTabService, MemoryPrinter], server: str
) -> None:
    svc, printer = service
    fired = threading.Event()
    with (
        Pause(WebUseTool._live_connected.__code__, functools.partial(_stop_once, fired)),
        pytest.raises(ToolCallInterrupted),
    ):
        tool.show_browser()

    assert fired.is_set()
    # Announced, then focused: two events for the one tab.
    (tab_id,) = {e["tab_id"] for e in _events(printer, "openBrowserTab")}
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "the blank tab closed")
    assert tab_id not in svc._pages
    assert tool._live_tab != tab_id
    # The next call opens a fresh tab and drives it.
    assert tool.go_to_url(f"{server}/inert").startswith("Page:")
    assert tool._live_tab in svc._pages and tool._live_tab != tab_id

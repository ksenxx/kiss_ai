"""``go_to_url`` retries a navigation only for ``net::ERR_NETWORK_CHANGED``.

Chromium fails every in-flight request with ``net::ERR_NETWORK_CHANGED``
when a host interface comes or goes (Wi-Fi hand-over, VPN connect, a
container's veth link); the browser UI retries that by itself. Before
the retry, ``WebUseTool.go_to_url`` handed the agent
``Error navigating to <url>: Page.goto: net::ERR_NETWORK_CHANGED``
(seen in the full-suite run of 2026-09-30 for a live site) and the
agent had to spend a step re-issuing the call.

Branch coverage note: the retry branch itself (an error whose text
contains ``net::ERR_NETWORK_CHANGED``) is only raised by Chromium's own
network stack in response to a real interface change on the host, which
needs ``CAP_NET_ADMIN``. Neither Playwright's ``route.abort(<code>)`` nor
CDP's ``Fetch.failRequest`` offers that error code, and starting or
removing Docker networks/containers during an in-flight loopback request
did not trigger it in headless Chromium (tried 2026-09-30). The branch is
therefore documented rather than driven by a test double; the tests below
cover what is reachable: a successful navigation and a non-network-change
failure, which must surface after exactly one attempt.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = b"<html><head><title>landed</title></head><body><h1>landed</h1></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def base_url() -> Iterator[str]:
    """Serve one static page on an ephemeral loopback port."""
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join(timeout=10)


@pytest.fixture
def tool() -> Iterator[WebUseTool]:
    """A headless tool with a throw-away profile."""
    web = WebUseTool(headless=True, user_data_dir=None)
    try:
        yield web
    finally:
        web.close()


def test_a_navigation_that_succeeds_is_issued_once(tool: WebUseTool, base_url: str) -> None:
    requests: list[str] = []

    def count(route) -> None:
        requests.append(route.request.url)
        route.continue_()

    tool.go_to_url("about:blank")
    tool._context.route(f"{base_url}/**", count)
    tree = tool.go_to_url(f"{base_url}/page")
    assert tree.startswith("Page: landed"), tree[:200]
    assert requests == [f"{base_url}/page"]


def test_a_failure_that_is_not_a_network_change_is_not_retried(
    tool: WebUseTool, base_url: str
) -> None:
    attempts: list[str] = []

    def refuse(route) -> None:
        attempts.append(route.request.url)
        route.abort("connectionrefused")

    tool.go_to_url("about:blank")
    tool._context.route(f"{base_url}/**", refuse)
    result = tool.go_to_url(f"{base_url}/down")
    assert result.startswith(f"Error navigating to {base_url}/down: "), result
    assert "net::ERR_CONNECTION_REFUSED" in result, result
    assert attempts == [f"{base_url}/down"], attempts

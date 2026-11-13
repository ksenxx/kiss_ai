# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The start page (``media/appStart.html``) keeps polling a daemon that accepts but never answers.

The page probes the daemon every half second and reschedules the probe
from the fetch's rejection handler.  A daemon whose listening socket is
bound but whose request loop is not serving yet accepts the connection
and never answers: the fetch neither fulfils nor rejects, so without a
timeout on the probe the dots froze and the counter went stale until
Chromium's own network timeout (minutes) finally rejected it.

The page it hands over to on macOS (``media/appWindow.html``, served by
the daemon) opens the ``?url=`` it is given.  Any site can link to that
page, so it opens a page of the daemon only: a ``javascript:`` URL would
run in the daemon's origin and any other URL would leave it.
"""

from __future__ import annotations

import html
import http.server
import socket
import threading
import urllib.parse

from kiss.core.brand import BRAND
from kiss.server import standalone_app
from kiss.tests.conftest import find_free_port


class _Answer(http.server.BaseHTTPRequestHandler):
    """A stand-in daemon answering every request with an empty page."""

    def do_GET(self) -> None:  # noqa: N802
        """Answer ``200`` with an empty HTML body (a 204 would abort the navigation)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output clean."""


def _hold_connections(
    listener: socket.socket, held: list[socket.socket], stop: threading.Event
) -> None:
    """Accept every connection on *listener* and keep it open without answering.

    The accept is bounded by the listener's timeout so the loop notices
    *stop*: closing or shutting down a listening socket does not wake a
    blocked ``accept()`` on macOS or Windows (ENOTCONN / WSAENOTCONN).
    """
    while not stop.is_set():
        try:
            conn, _addr = listener.accept()
        except TimeoutError:
            continue
        except OSError:
            return
        held.append(conn)


def test_a_silent_daemon_does_not_freeze_the_start_page() -> None:
    """A probe the daemon accepts but never answers is given up and polling continues."""
    from playwright.sync_api import sync_playwright

    port = find_free_port()
    daemon = f"http://127.0.0.1:{port}"
    url = standalone_app.start_page_url(f"{daemon}/landing", daemon)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen()
    listener.settimeout(0.1)
    held: list[socket.socket] = []
    stop = threading.Event()
    acceptor = threading.Thread(
        target=_hold_connections, args=(listener, held, stop), daemon=True
    )
    acceptor.start()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url)
            assert page.title() == BRAND["product_name"]
            # The first probe is accepted and never answered; a bounded
            # probe gives up and the catch branch paints the dots.
            page.wait_for_function(
                "document.getElementById('dots').textContent !== ''",
                timeout=6000, polling=100,
            )
            assert page.url == url, "the page moved on without an answer"
            assert held, "the silent daemon never saw the probe"
            # The acceptor must be out of accept() before the port is
            # handed to the answering daemon below.
            stop.set()
            acceptor.join(timeout=5)
            assert not acceptor.is_alive(), "the acceptor kept the port"
            listener.close()
            for conn in held:
                conn.close()
            httpd = http.server.HTTPServer(("127.0.0.1", port), _Answer)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            try:
                page.wait_for_url(f"{daemon}/landing", timeout=5000)
            finally:
                httpd.shutdown()
        finally:
            browser.close()


class _Site(http.server.BaseHTTPRequestHandler):
    """A stand-in daemon that records which of its other pages are requested.

    It serves the packaged ``appWindow.html``, answers that no popup is
    open, and at ``/link?to=<url>`` a page linking to *url* in the window
    the app lives in; ``requested`` lists ``(Host, path)`` of the rest.
    """

    requested: list[tuple[str, str]] = []

    def do_GET(self) -> None:  # noqa: N802
        """Serve the three pages above, an empty page for anything else."""
        path, _, query = self.path.partition("?")
        body = b""
        if path == standalone_app.APP_WINDOW_PAGE:
            body = (standalone_app.MEDIA_DIR / "appWindow.html").read_bytes()
        elif path == standalone_app.APP_WINDOW_API:
            body = b'{"open": false}'
        elif path == "/link":
            to = html.escape(urllib.parse.parse_qs(query)["to"][0])
            body = f'<a target="kiss-app-window" href="{to}">open</a>'.encode()
        else:
            self.requested.append((self.headers["Host"], self.path))
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output clean."""


def test_the_app_window_page_opens_only_the_daemons_own_pages() -> None:
    """``appWindow.html`` opens a ``?url=`` on its own origin and nothing else."""
    from playwright.sync_api import sync_playwright

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    daemon = f"127.0.0.1:{httpd.server_address[1]}"
    elsewhere = f"localhost:{httpd.server_address[1]}"  # the same site, another origin
    window_page = f"http://{daemon}{standalone_app.APP_WINDOW_PAGE}?url="
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            with page.expect_popup() as popup:
                page.goto(standalone_app.app_window_url(f"http://{daemon}/app"))
            popup.value.wait_for_url(f"http://{daemon}/app")
            assert _Site.requested == [(daemon, "/app")]
            # A link on another site into the app's window: its script
            # must not run as the daemon's page.
            script = urllib.parse.quote("javascript:void(fetch('/stolen'))")
            link = urllib.parse.quote(window_page + script)
            page.goto(f"http://{elsewhere}/link?to={link}")
            with page.expect_popup() as popup:
                page.click("a")
            popup.value.wait_for_load_state("networkidle")
            # Another site, also when spelled as a path of this one.
            for away in (f"http://{elsewhere}/away", f"/\\{elsewhere}/away"):
                page.goto(window_page + urllib.parse.quote(away))
                page.wait_for_load_state("networkidle")
            assert _Site.requested == [(daemon, "/app")]
        finally:
            browser.close()
            httpd.shutdown()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real Chromium: a failed xterm.js load leaves no stylesheet behind.

``media/terminalTab.js`` downloads xterm.js's stylesheet and its two
UMD bundles from jsDelivr when the first terminal opens.  When the
bundles do not initialise (the CDN, or a proxy in front of it, answers
with a page that is not xterm.js), the tab shows its load note and the
next show of the tab downloads again.  The stylesheet used to go into
the page before the bundles ran, so every failed attempt left a copy
behind (the proxy's page parsed as CSS); it now goes in once the
bundles initialised: failed attempts add none, the attempt that
succeeds adds exactly one, and the terminal it opens is styled.

Chromium resolves cdn.jsdelivr.net to a local HTTPS server
(``--host-resolver-rules``) that plays the CDN: first a block page,
then the real xterm.js files, downloaded from jsDelivr once for the
module.  The test page itself comes from the same server, so the
CDN requests stay inside the loopback address space.
"""

from __future__ import annotations

import re
import ssl
import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import certifi
import pytest
from playwright.sync_api import Page, sync_playwright

from kiss.server.web_server import _generate_self_signed_cert

_TERMINAL_JS = (
    Path(__file__).resolve().parents[2] / "webapp" / "media" / "terminalTab.js"
).read_text(encoding="utf-8")

_BLOCK_PAGE = b"<!DOCTYPE html><title>Blocked</title><p>This site is blocked.</p>"
_BLANK_PAGE = b"<!DOCTYPE html><html><head></head><body></body></html>"


def _cdn_urls() -> list[str]:
    """The three URLs terminalTab.js downloads xterm.js from."""
    xterm = re.search(r"const XTERM_BASE = '([^']+)'", _TERMINAL_JS)
    fit = re.search(r"const FIT_BASE = '([^']+)'", _TERMINAL_JS)
    assert xterm and fit
    return [
        xterm.group(1) + "/css/xterm.css",
        xterm.group(1) + "/lib/xterm.js",
        fit.group(1) + "/lib/addon-fit.js",
    ]


def _download(url: str) -> bytes:
    """*url*'s bytes from jsDelivr, three attempts in all.

    Verified against certifi's roots, as the daemon's own HTTPS client
    (httpx) does: a Windows VM's certificate store may lack the CDN's
    root until Windows has fetched it, and OpenSSL does not fetch it.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30, context=context) as response:
                return bytes(response.read())
        except OSError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


class _Cdn(ThreadingHTTPServer):
    """The stand-in for cdn.jsdelivr.net: the block page while
    ``broken``, else the real files; ``/`` is the blank test page.
    ``requested`` lists the CDN paths asked for (appended from the
    handler threads; a list append is atomic)."""

    files: dict[str, bytes]
    requested: list[str]
    broken = True


class _CdnHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802  (http.server's name)
        cdn = self.server
        assert isinstance(cdn, _Cdn)
        if self.path == "/":
            body, ctype = _BLANK_PAGE, "text/html"
        else:
            cdn.requested.append(self.path)
            if cdn.broken:
                body, ctype = _BLOCK_PAGE, "text/html"
            else:
                body = cdn.files[self.path]
                ctype = "text/css" if self.path.endswith(".css") else "text/javascript"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output free of request lines."""


@pytest.fixture(scope="module")
def cdn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Cdn]:
    """The local CDN over TLS, serving on a free loopback port."""
    files = {urlsplit(url).path: _download(url) for url in _cdn_urls()}
    tls = tmp_path_factory.mktemp("cdn-tls")
    _generate_self_signed_cert(tls / "cert.pem", tls / "key.pem")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tls / "cert.pem", tls / "key.pem")
    server = _Cdn(("127.0.0.1", 0), _CdnHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.files = files
    server.requested = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def page(cdn: _Cdn) -> Iterator[Page]:
    """A Chromium page whose cdn.jsdelivr.net is the local CDN."""
    port = cdn.server_address[1]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            f"--host-resolver-rules=MAP cdn.jsdelivr.net 127.0.0.1:{port}",
        ])
        page = browser.new_page(ignore_https_errors=True)
        page.goto(f"https://localhost:{port}/")
        page.add_script_tag(content=_TERMINAL_JS)
        yield page
        browser.close()


_OPEN_TERMINAL_JS = """() => {
  window.__sent = [];
  const view = TerminalTabView.create('tab-1', {send: m => window.__sent.push(m)});
  view.el.style.cssText = 'width: 640px; height: 320px';
  document.body.appendChild(view.el);
  window.__view = view;
  view.setVisible(true);
}"""

# Switch to another tab and back: the retry the load note asks for.
_SHOW_AGAIN_JS = """() => {
  document.querySelector('.terminal-load-error').dataset.seen = '1';
  window.__view.setVisible(false);
  window.__view.setVisible(true);
}"""

_HEAD_STYLES_JS = "[...document.head.querySelectorAll('style')].map(s => s.textContent)"


def test_failed_xterm_loads_leave_no_stylesheet_and_the_retry_adds_one(
    cdn: _Cdn, page: Page,
) -> None:
    page.evaluate(_OPEN_TERMINAL_JS)
    page.wait_for_selector(".terminal-load-error")
    assert len(cdn.requested) == 3
    assert page.evaluate(_HEAD_STYLES_JS) == []

    page.evaluate(_SHOW_AGAIN_JS)
    page.wait_for_selector(".terminal-load-error:not([data-seen])")
    assert len(cdn.requested) == 6
    assert page.evaluate(_HEAD_STYLES_JS) == []

    cdn.broken = False
    page.evaluate(_SHOW_AGAIN_JS)
    page.wait_for_selector(".terminal-view .xterm", timeout=30000)
    assert sorted(cdn.requested[6:]) == sorted(cdn.files)
    css = cdn.files[urlsplit(_cdn_urls()[0]).path].decode("utf-8")
    assert page.evaluate(_HEAD_STYLES_JS) == [css]
    assert page.evaluate(
        "getComputedStyle(document.querySelector('.terminal-view .xterm')).position",
    ) == "relative"
    assert page.evaluate("window.__sent.map(m => m.type)") == ["terminalOpen"]

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: the mobile Chats drawer's refresh button is not under the x.

Below the 900px breakpoint the history panel is a drawer whose close
button (``#sidebar-close``) sits absolutely in the panel's top-right
corner.  The Chats header row puts the app refresh button
(``#chats-refresh-btn``) at its right end, so without a reserved
margin the x lands on top of it and every tap on "refresh" closes the
drawer instead.

The real page (``_build_html``, ``main.css`` + ``remote-codex.css``)
is rendered in headless Chromium at several phone widths; the test
checks that the two boxes are disjoint, that ``elementFromPoint`` at
the refresh button's centre is the refresh button itself, and that a
real click there fires the refresh handler and not the close one.  On
desktop, where the x is hidden, the row's reserved space goes away.
"""

from __future__ import annotations

import functools
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from kiss.server.web_server import MEDIA_DIR, _build_html
from kiss.tests.conftest import goto_retrying_network_change

MOBILE_WIDTHS = (320, 390, 600, 899)

# main.js only reveals #app once the websocket handshake succeeds; a
# static server never completes it, so force the layout visible (see
# test_remote_desktop_layout.py).  The page reload the refresh button
# triggers is replaced by a counter so the click can be observed.
_PREPARE_JS = """
() => {
  const style = document.createElement('style');
  style.textContent = `
    #app { display: flex !important; }
    #kiss-server-loading { display: none !important; }
    #auth-modal { display: none !important; }
  `;
  document.head.appendChild(style);
  window.__clicks = { refresh: 0, close: 0 };
  document.getElementById('chats-refresh-btn').addEventListener('click', (ev) => {
    ev.stopImmediatePropagation();
    ev.preventDefault();
    window.__clicks.refresh++;
  }, true);
  document.getElementById('sidebar-close').addEventListener(
    'click', () => { window.__clicks.close++; }, true);
  document.getElementById('sidebar').classList.add('open');
}
"""

_GEOMETRY_JS = """
() => {
  const rect = (id) => {
    const r = document.getElementById(id).getBoundingClientRect();
    return { left: r.left, top: r.top, right: r.right, bottom: r.bottom,
             cx: r.left + r.width / 2, cy: r.top + r.height / 2 };
  };
  const refresh = rect('chats-refresh-btn');
  const close = rect('sidebar-close');
  const hit = document.elementFromPoint(refresh.cx, refresh.cy);
  const closeStyle = getComputedStyle(document.getElementById('sidebar-close'));
  const hdrStyle = getComputedStyle(document.querySelector('.sidebar-hdr.chats-hdr'));
  return {
    refresh, close,
    closeVisible: closeStyle.display !== 'none',
    hitId: hit ? (hit.closest('button') || hit).id : null,
    hdrPaddingRight: hdrStyle.paddingRight,
  };
}
"""


def _serve(directory: Path) -> Iterator[str]:
    """Serve ``directory`` over HTTP on an ephemeral port; yield its URL."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def remote_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve the real remote HTML page plus the real media assets."""
    root = tmp_path_factory.mktemp("remote-webapp")
    (root / "index.html").write_text(_build_html(), encoding="utf-8")
    (root / "media").symlink_to(MEDIA_DIR, target_is_directory=True)
    yield from _serve(root)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """Launch one headless Chromium for the whole module."""
    with sync_playwright() as pw:
        chromium = pw.chromium.launch()
        try:
            yield chromium
        finally:
            chromium.close()


def _open_page(browser: Browser, url: str, width: int) -> Page:
    """Open the remote page at ``width`` with the history drawer open."""
    page = browser.new_page(viewport={"width": width, "height": 800})
    goto_retrying_network_change(page, url)
    page.wait_for_selector("#chats-refresh-btn", state="attached")
    page.evaluate(_PREPARE_JS)
    page.wait_for_selector("#chats-refresh-btn", state="visible")
    return page


def _disjoint(a: dict[str, float], b: dict[str, float]) -> bool:
    """True when the two boxes share no point."""
    return (
        a["right"] <= b["left"]
        or b["right"] <= a["left"]
        or a["bottom"] <= b["top"]
        or b["bottom"] <= a["top"]
    )


@pytest.mark.parametrize("width", MOBILE_WIDTHS)
def test_mobile_refresh_button_is_clear_of_the_close_button(
    browser: Browser, remote_url: str, width: int
) -> None:
    """On a phone the refresh button sits left of the x and takes the tap."""
    page = _open_page(browser, remote_url, width)
    try:
        page.wait_for_function("() => !document.body.classList.contains('remote-desktop')")
        geo = page.evaluate(_GEOMETRY_JS)
        assert geo["closeVisible"], geo
        assert _disjoint(geo["refresh"], geo["close"]), geo
        assert geo["refresh"]["right"] <= geo["close"]["left"], geo
        assert geo["hitId"] == "chats-refresh-btn", geo

        page.mouse.click(geo["refresh"]["cx"], geo["refresh"]["cy"])
        assert page.evaluate("() => window.__clicks") == {
            "refresh": 1,
            "close": 0,
        }
    finally:
        page.close()


def test_desktop_row_reserves_no_space_for_the_hidden_close(
    browser: Browser, remote_url: str
) -> None:
    """Docked desktop panel: the x is gone, so the row keeps no gap for it."""
    page = _open_page(browser, remote_url, 1280)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        geo = page.evaluate(_GEOMETRY_JS)
        assert not geo["closeVisible"], geo
        assert geo["hdrPaddingRight"] == "0px", geo
        assert geo["hitId"] == "chats-refresh-btn", geo
    finally:
        page.close()

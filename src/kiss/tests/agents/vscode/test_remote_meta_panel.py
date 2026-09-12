# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the docked task-info panel of the remote webapp.

Desktop mode (viewport >= 900px) docks a task-info panel on the RIGHT
of the chat: the twin of the left task-history panel.  It lists the
task meta information — tokens, cost, steps, time, machine name,
workdir, max budget — as a bulleted list and REPLACES the
``#tab-status-bar`` strip that used to render those values at the top
of the chat panel.

These tests render the REAL page returned by
:func:`kiss.server.web_server._build_html` (same ``media/chat.html``
template, same ``main.css`` + ``remote-codex.css``, same
``remote-chat`` body class) in headless Chromium and measure the
resulting geometry and live DOM mirroring.

Covered behavior:

* The panel is docked at the right edge and takes one fifth of the
  browser window, and ``#app`` clears it.
* The seven meta values render as bullet items (a real ``<ul>`` with
  ``list-style-type: disc``) with the Tokens / Cost / Steps / Time /
  Machine / Workdir / Max budget labels.
* Whatever the app writes into the status-bar spans (tokens, cost,
  steps, the running timer text and its color, the machine name) is
  mirrored live into the panel.
* The top status bar is hidden on desktop; below the 900px breakpoint
  the panel disappears and the status bar comes back.

No mocks, patches or fakes: a real HTTP server serves the real assets
to a real browser.
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

# Widths at or above the 900px desktop breakpoint.
DESKTOP_WIDTHS = (900, 1280, 1920)

# The docked panels' share of the browser window (--sidebar-default-w
# and --meta-panel-w are both 20vw in remote-codex.css).
PANEL_FRACTION = 0.2

# main.js only reveals #app once the websocket handshake succeeds,
# which never happens against a static server; an !important rule
# beats the inline re-hiding (see test_remote_desktop_layout.py).
_PREPARE_JS = """
() => {
  const style = document.createElement('style');
  style.textContent = `
    #app { display: flex !important; }
    #kiss-server-loading { display: none !important; }
    #auth-modal { display: none !important; }
  `;
  document.head.appendChild(style);
}
"""

_META_LAYOUT_JS = """
() => {
  const panel = document.getElementById('meta-panel');
  const app = document.getElementById('app');
  const rect = panel.getBoundingClientRect();
  return {
    display: getComputedStyle(panel).display,
    left: rect.left,
    right: rect.right,
    width: rect.width,
    appRight: app.getBoundingClientRect().right,
    statusBarHidden:
      document.getElementById('tab-status-bar').offsetParent === null,
  };
}
"""

_META_LIST_JS = """
() => {
  const list = document.getElementById('meta-list');
  const items = [...list.querySelectorAll('li.meta-item')];
  return {
    tag: list.tagName,
    listStyle: getComputedStyle(list).listStyleType,
    displays: items.map(li => getComputedStyle(li).display),
    labels: items.map(
      li => li.querySelector('.meta-label').textContent.trim(),
    ),
    values: items.map(
      li => li.querySelector('.meta-value').textContent.trim(),
    ),
  };
}
"""

_WRITE_STATUS_JS = """
() => {
  document.getElementById('status-tokens').textContent = 'Tokens: 12,345';
  document.getElementById('status-budget').textContent = 'Cost: $1.23';
  document.getElementById('status-steps').textContent = 'Steps: 7';
  const time = document.getElementById('status-text');
  time.textContent = 'Running 5s';
  time.style.color = 'rgb(255, 0, 0)';
  document.getElementById('status-machine').textContent = 'build-host-42';
}
"""

_READ_MIRROR_JS = """
() => ({
  tokens: document.getElementById('meta-tokens').textContent,
  cost: document.getElementById('meta-cost').textContent,
  steps: document.getElementById('meta-steps').textContent,
  time: document.getElementById('meta-time').textContent,
  timeColor: getComputedStyle(
    document.getElementById('meta-time'),
  ).color,
  machine: document.getElementById('meta-machine').textContent,
})
"""


def _serve(directory: Path) -> Iterator[str]:
    """Serve ``directory`` over HTTP on an ephemeral port; yield its URL."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(directory),
    )
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


def _open_desktop_page(browser: Browser, url: str, width: int) -> Page:
    """Open the remote page at ``width`` in desktop mode."""
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.goto(url)
    page.wait_for_selector("body.remote-desktop", state="attached")
    page.evaluate(_PREPARE_JS)
    page.wait_for_selector("#meta-panel", state="visible")
    return page


@pytest.mark.parametrize("width", DESKTOP_WIDTHS)
def test_meta_panel_docked_right_at_one_fifth(
    browser: Browser,
    remote_url: str,
    width: int,
) -> None:
    """The task-info panel hugs the right edge at window width / 5 and
    the chat column (#app) stops where the panel begins."""
    page = _open_desktop_page(browser, remote_url, width)
    try:
        layout = page.evaluate(_META_LAYOUT_JS)
        assert layout["display"] != "none"
        assert layout["width"] == pytest.approx(
            width * PANEL_FRACTION, abs=1,
        ), (
            f"the task-info panel must take one fifth of the {width}px "
            f"window, got {layout['width']}px"
        )
        assert layout["right"] == pytest.approx(width, abs=1), layout
        assert layout["appRight"] == pytest.approx(
            layout["left"], abs=1,
        ), "#app must clear the docked task-info panel"
    finally:
        page.close()


def test_meta_panel_replaces_the_top_status_bar(
    browser: Browser,
    remote_url: str,
) -> None:
    """Desktop hides #tab-status-bar; the panel carries the values."""
    page = _open_desktop_page(browser, remote_url, 1280)
    try:
        layout = page.evaluate(_META_LAYOUT_JS)
        assert layout["statusBarHidden"] is True, (
            "the meta values must move into the task-info panel INSTEAD "
            "of rendering at the top of the chat panel"
        )
    finally:
        page.close()


def test_meta_values_render_as_a_bulleted_list(
    browser: Browser,
    remote_url: str,
) -> None:
    """The panel lists Tokens / Cost / Steps / Time / Machine /
    Workdir / Max budget as real ``<ul>`` bullet items."""
    page = _open_desktop_page(browser, remote_url, 1280)
    try:
        listing = page.evaluate(_META_LIST_JS)
        assert listing["tag"] == "UL"
        assert listing["listStyle"] == "disc", (
            "the meta items must render as a bulleted list, got "
            f"list-style-type: {listing['listStyle']}"
        )
        assert listing["displays"] == ["list-item"] * 7, listing
        assert listing["labels"] == [
            "Tokens:",
            "Cost:",
            "Steps:",
            "Time:",
            "Machine:",
            "Workdir:",
            "Max budget:",
        ], listing
        # Before any task ran the numeric values show the em-dash
        # placeholder and the time mirrors the "Ready" status.
        assert listing["values"][:3] == ["\u2014"] * 3, listing
        assert listing["values"][3] == "Ready", listing
    finally:
        page.close()


def test_status_values_mirror_live_into_the_panel(
    browser: Browser,
    remote_url: str,
) -> None:
    """Writes to the status-bar spans (as the app does while a task
    runs) appear in the panel's bullets, with labels stripped and the
    timer color carried over."""
    page = _open_desktop_page(browser, remote_url, 1280)
    try:
        page.evaluate(_WRITE_STATUS_JS)
        page.wait_for_function(
            "() => document.getElementById('meta-tokens')"
            ".textContent === '12,345'"
        )
        mirrored = page.evaluate(_READ_MIRROR_JS)
        assert mirrored["tokens"] == "12,345", mirrored
        assert mirrored["cost"] == "$1.23", mirrored
        assert mirrored["steps"] == "7", mirrored
        assert mirrored["time"] == "Running 5s", mirrored
        assert mirrored["timeColor"] == "rgb(255, 0, 0)", (
            "the running/done color of the timer must carry over: "
            f"{mirrored}"
        )
        assert mirrored["machine"] == "build-host-42", mirrored
    finally:
        page.close()


def test_mobile_keeps_the_status_bar_and_hides_the_panel(
    browser: Browser,
    remote_url: str,
) -> None:
    """Below the 900px breakpoint the panel is gone and the status bar
    at the top of the chat comes back."""
    page = _open_desktop_page(browser, remote_url, 1280)
    try:
        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_function(
            "() => !document.body.classList.contains('remote-desktop')"
        )
        assert page.locator("#meta-panel").is_hidden()
        assert page.locator("#tab-status-bar").is_visible()
    finally:
        page.close()

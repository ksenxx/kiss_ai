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
* The meta values render as bullet items (a real ``<ul>`` with
  ``list-style-type: disc``): the live status values (Tokens / Cost /
  Steps / Time / Machine / Workdir / Max budget) followed by the
  task's own settings (Date / Base model / Worktree mode / Parallel
  mode / Chat id / Task id / Parent task, the last hidden until a
  parent id arrives).
* Whatever the app writes into the status-bar spans (tokens, cost,
  steps, the running timer text and its color, the machine name) is
  mirrored live into the panel.
* The top status bar is hidden at EVERY width — below the 900px
  breakpoint the panel becomes a right-slide drawer toggled by a
  button at the tab bar's right edge (so it steals no chat space and
  stays clear of the composer's button row), dismissed by its close
  button, the backdrop or Escape.

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
    """The panel lists the live status values (Tokens / Cost / Steps /
    Time / Machine / Workdir / Max budget) followed by the task's own
    settings (Date / Base model / Worktree mode / Parallel mode / Chat
    id / Task id / Parent task) as real ``<ul>`` bullet items.  The
    Parent-task row starts hidden until a parent id arrives."""
    page = _open_desktop_page(browser, remote_url, 1280)
    try:
        listing = page.evaluate(_META_LIST_JS)
        assert listing["tag"] == "UL"
        assert listing["listStyle"] == "disc", (
            "the meta items must render as a bulleted list, got "
            f"list-style-type: {listing['listStyle']}"
        )
        # Every row bar the initially-hidden Parent-task row renders as
        # a bullet; the hidden row collapses to display:none.
        assert listing["displays"] == ["list-item"] * 13 + ["none"], listing
        assert listing["labels"] == [
            "Tokens:",
            "Cost:",
            "Steps:",
            "Time:",
            "Machine:",
            "Workdir:",
            "Max budget:",
            "Date:",
            "Base model:",
            "Worktree mode:",
            "Parallel mode:",
            "Chat id:",
            "Task id:",
            "Parent task:",
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


_MOBILE_WIDTH = 420

_MOBILE_LAYOUT_JS = """
() => {
  const vw = window.innerWidth;
  const panel = document.getElementById('meta-panel');
  const rect = panel.getBoundingClientRect();
  const btn = document.getElementById('meta-drawer-btn');
  const btnRect = btn.getBoundingClientRect();
  const tabBar = document.getElementById('tab-bar').getBoundingClientRect();
  const inputArea = document
    .getElementById('input-area')
    .getBoundingClientRect();
  const app = document.getElementById('app').getBoundingClientRect();
  return {
    open: panel.classList.contains('open'),
    inert: panel.hasAttribute('inert'),
    position: getComputedStyle(panel).position,
    panelLeft: rect.left,
    panelRight: rect.right,
    panelWidth: rect.width,
    viewport: vw,
    statusBarHidden:
      document.getElementById('tab-status-bar').offsetParent === null,
    btnVisible: btn.offsetParent !== null,
    btnExpanded: btn.getAttribute('aria-expanded'),
    btnInTabBar:
      btnRect.top >= tabBar.top - 1 && btnRect.bottom <= tabBar.bottom + 1,
    btnAboveComposer: btnRect.bottom <= inputArea.top,
    appWidth: app.width,
    overlayOpen: document
      .getElementById('meta-overlay')
      .classList.contains('open'),
    focusedId: document.activeElement && document.activeElement.id,
  };
}
"""


def _open_mobile_page(browser: Browser, url: str) -> Page:
    """Open the remote page below the 900px desktop breakpoint."""
    page = browser.new_page(
        viewport={"width": _MOBILE_WIDTH, "height": 900},
    )
    page.goto(url)
    page.wait_for_selector("body.remote-chat", state="attached")
    page.evaluate(_PREPARE_JS)
    page.wait_for_function(
        "() => !document.body.classList.contains('remote-desktop')"
    )
    return page


def test_mobile_hides_status_bar_and_parks_the_drawer_offscreen(
    browser: Browser,
    remote_url: str,
) -> None:
    """Below the 900px breakpoint the status bar STAYS hidden (the
    drawer carries its values) and the panel waits off-screen right,
    inert, behind a toggle that lives in the tab-bar row — above the
    chat transcript and clear of the composer's button row — so it
    never shrinks the chat area."""
    page = _open_mobile_page(browser, remote_url)
    try:
        layout = page.evaluate(_MOBILE_LAYOUT_JS)
        assert layout["statusBarHidden"] is True, (
            "the top status bar must stay hidden on mobile; the drawer "
            f"shows those values instead: {layout}"
        )
        assert layout["position"] == "fixed", layout
        assert layout["panelLeft"] >= layout["viewport"] - 1, (
            f"the closed drawer must sit off-screen right: {layout}"
        )
        assert layout["open"] is False and layout["inert"] is True, (
            f"the closed drawer must be inert (out of tab order): {layout}"
        )
        assert layout["btnVisible"] is True, layout
        assert layout["btnExpanded"] == "false", layout
        assert layout["btnInTabBar"] is True, (
            f"the toggle must ride the existing tab-bar row: {layout}"
        )
        assert layout["btnAboveComposer"] is True, (
            f"the toggle must not crowd the composer buttons: {layout}"
        )
        assert layout["appWidth"] == pytest.approx(
            layout["viewport"], abs=1,
        ), f"the chat column must keep the full width: {layout}"
    finally:
        page.close()


def test_mobile_drawer_slides_in_from_the_right_and_dismisses(
    browser: Browser,
    remote_url: str,
) -> None:
    """The tab-bar button slides the drawer in over the chat (backdrop
    up, focus on the close button); close button, Escape and backdrop
    each dismiss it, handing focus back to the toggle."""
    page = _open_mobile_page(browser, remote_url)
    try:
        page.click("#meta-drawer-btn")
        page.wait_for_function(
            "() => document.getElementById('meta-panel')"
            f".getBoundingClientRect().right <= {_MOBILE_WIDTH} + 1"
        )
        layout = page.evaluate(_MOBILE_LAYOUT_JS)
        assert layout["open"] is True and layout["inert"] is False, layout
        assert layout["panelRight"] == pytest.approx(
            _MOBILE_WIDTH, abs=1,
        ), f"the open drawer must hug the right edge: {layout}"
        assert layout["panelWidth"] < _MOBILE_WIDTH, (
            f"the drawer overlays, not replaces, the chat: {layout}"
        )
        assert layout["btnExpanded"] == "true", layout
        assert layout["overlayOpen"] is True, layout
        assert layout["focusedId"] == "meta-close", layout

        # Dismissal 1: the drawer's own close button.
        page.click("#meta-close")
        page.wait_for_function(
            "() => !document.getElementById('meta-panel')"
            ".classList.contains('open')"
        )
        layout = page.evaluate(_MOBILE_LAYOUT_JS)
        assert layout["inert"] is True, layout
        assert layout["overlayOpen"] is False, layout
        assert layout["focusedId"] == "meta-drawer-btn", (
            f"closing must hand focus back to the toggle: {layout}"
        )

        # Dismissal 2: Escape.
        page.click("#meta-drawer-btn")
        page.wait_for_function(
            "() => document.getElementById('meta-panel')"
            ".classList.contains('open')"
        )
        page.keyboard.press("Escape")
        page.wait_for_function(
            "() => !document.getElementById('meta-panel')"
            ".classList.contains('open')"
        )

        # Dismissal 3: the backdrop.
        page.click("#meta-drawer-btn")
        page.wait_for_function(
            "() => document.getElementById('meta-panel')"
            ".classList.contains('open')"
        )
        page.click("#meta-overlay", position={"x": 5, "y": 450})
        page.wait_for_function(
            "() => !document.getElementById('meta-panel')"
            ".classList.contains('open')"
        )
    finally:
        page.close()

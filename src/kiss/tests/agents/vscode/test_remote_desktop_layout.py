# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end layout tests for the remote webapp in desktop mode.

These tests render the REAL page returned by
:func:`kiss.server.web_server._build_html` (same ``media/chat.html``
template, same ``main.css`` + ``remote-codex.css``, same
``remote-chat`` body class) in headless Chromium and measure the
resulting geometry.  A JSDOM test cannot cover any of this because
JSDOM has no layout engine.

Covered behavior:

* The task-history panel's default width is one fifth of the browser
  window, and a resizer double-click restores that fraction.
* The docked history panel is permanent on desktop: the burger button
  (``#menu-btn``) and the drawer close button (``#sidebar-close``) are
  hidden, and both come back below the 900px (mobile) breakpoint.
* The panel resizer may collapse the panel to a 10px sliver and is
  limited on the right only by the chat's minimum width.
* The settings panel stays narrow — like the VS Code extension's
  sidebar — instead of covering 90% of a wide desktop window.

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

# Widths at or above the 900px desktop breakpoint that the remote
# webapp uses to dock the history panel.
DESKTOP_WIDTHS = (900, 1000, 1280, 1440, 1920)

# --sidebar-min-w / --chat-min-w from remote-codex.css: the sliver a
# drag may collapse the history panel down to, and the chat width the
# panel may never eat into.
SLIVER_W = 10
MIN_CHAT_W = 360

# The docked panel's default share of the browser window
# (--sidebar-default-w: 20vw, mirrored by sidebarDefaultW in main.js).
PANEL_FRACTION = 0.2

# main.js only reveals #app once the websocket handshake succeeds,
# which never happens against a static server: the shim keeps
# retrying and re-hiding #app behind the "server is starting" veil.
# An !important stylesheet rule beats those inline styles, so the
# layout stays stable while every real main.js handler and every real
# CSS rule still applies.  Then expand the collapsible Filters
# section so the toggle chips get laid out.
_PREPARE_JS = """
() => {
  const style = document.createElement('style');
  style.textContent = `
    #app { display: flex !important; }
    #kiss-server-loading { display: none !important; }
    #auth-modal { display: none !important; }
  `;
  document.head.appendChild(style);
  document.getElementById('history-filters-toggle').click();
}
"""

_LAYOUT_JS = """
() => {
  const sidebar = document.getElementById('sidebar');
  const app = document.getElementById('app');
  const sb = sidebar.getBoundingClientRect();
  const ar = app.getBoundingClientRect();
  return {
    open: sidebar.classList.contains('open'),
    sidebarRight: sb.right,
    sidebarWidth: sb.width,
    appLeft: ar.left,
    appWidth: ar.width,
  };
}
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


def _drag_resizer_to(page: Page, x: float) -> None:
    """Drag the history panel's resize handle to viewport position ``x``."""
    box = page.locator("#sidebar-resizer").bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 200)
    page.mouse.down()
    page.mouse.move(x, box["y"] + 200, steps=8)
    page.mouse.up()


def _open_desktop_page(browser: Browser, url: str, width: int) -> Page:
    """Open the remote page at ``width`` and expand the history filters."""
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.goto(url)
    page.wait_for_selector("body.remote-desktop", state="attached")
    page.evaluate(_PREPARE_JS)
    page.wait_for_selector(".history-filter-chips .hf-chip", state="visible")
    return page


@pytest.mark.parametrize("width", DESKTOP_WIDTHS)
def test_history_panel_takes_one_fifth_of_the_window(
    browser: Browser,
    remote_url: str,
    width: int,
) -> None:
    """The docked panel's default width is exactly window width / 5."""
    page = _open_desktop_page(browser, remote_url, width)
    try:
        layout = page.evaluate(_LAYOUT_JS)
        assert layout["open"] is True
        assert layout["sidebarWidth"] == pytest.approx(
            width * PANEL_FRACTION, abs=1,
        ), (
            f"the history panel must take one fifth of the {width}px "
            f"window, got {layout['sidebarWidth']}px"
        )
        assert layout["appLeft"] == pytest.approx(
            layout["sidebarWidth"], abs=1,
        ), layout
    finally:
        page.close()


def test_resizer_collapses_to_a_sliver_and_dblclick_restores(
    browser: Browser,
    remote_url: str,
) -> None:
    """A drag may collapse the panel to the 10px sliver; a double-click
    on the resizer restores the one-fifth default."""
    page = _open_desktop_page(browser, remote_url, 1440)
    try:
        _drag_resizer_to(page, 700)
        widened = page.evaluate(_LAYOUT_JS)
        assert widened["sidebarWidth"] == pytest.approx(700, abs=2), widened
        _drag_resizer_to(page, 0)
        narrowed = page.evaluate(_LAYOUT_JS)
        assert narrowed["sidebarWidth"] == pytest.approx(SLIVER_W, abs=2), (
            "dragging fully left must collapse the panel to the "
            f"{SLIVER_W}px sliver, got {narrowed['sidebarWidth']}px"
        )
        page.locator("#sidebar-resizer").dblclick()
        restored = page.evaluate(_LAYOUT_JS)
        assert restored["sidebarWidth"] == pytest.approx(
            1440 * PANEL_FRACTION, abs=1,
        ), (
            "double-click must restore the one-fifth default width, "
            f"got {restored['sidebarWidth']}px"
        )
    finally:
        page.close()


def test_widened_panel_never_crushes_the_chat(
    browser: Browser,
    remote_url: str,
) -> None:
    """A panel widened on a big window keeps the chat usable when the
    window shrinks to the desktop breakpoint."""
    page = _open_desktop_page(browser, remote_url, 1920)
    try:
        _drag_resizer_to(page, 1900)
        wide = page.evaluate(_LAYOUT_JS)
        # The ACTUAL chat column (between the two docked panels) keeps
        # its minimum: the drag cap must reserve the right-hand
        # task-info panel's fifth of the window too.
        assert wide["appWidth"] >= MIN_CHAT_W - 1, wide

        page.set_viewport_size({"width": 900, "height": 900})
        page.wait_for_timeout(400)
        narrow = page.evaluate(_LAYOUT_JS)
        assert narrow["appWidth"] >= MIN_CHAT_W - 1, (
            "after shrinking the window the chat must keep at least "
            f"{MIN_CHAT_W}px, but it is {narrow['appWidth']}px wide "
            f"(panel {narrow['sidebarWidth']}px of 900px)"
        )
    finally:
        page.close()


@pytest.mark.parametrize("width", (1000, 1440))
def test_burger_hidden_and_panel_permanently_docked(
    browser: Browser,
    remote_url: str,
    width: int,
) -> None:
    """Desktop has no burger and no close button: the panel is fixed."""
    page = _open_desktop_page(browser, remote_url, width)
    try:
        docked = page.evaluate(_LAYOUT_JS)
        assert docked["open"] is True
        assert docked["sidebarRight"] > 0, docked
        assert docked["appLeft"] == pytest.approx(docked["sidebarWidth"]), docked

        assert page.locator("#menu-btn").is_hidden(), (
            "the burger only toggled the docked panel; on desktop the "
            "panel is permanent so the button must be gone"
        )
        assert page.locator("#sidebar-close").is_hidden(), (
            "without the burger there would be no way to reopen a panel "
            "dismissed by the drawer close button, so it must be gone too"
        )

        # Below the 900px breakpoint the panel is a drawer again, so
        # both controls come back.
        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_selector("#menu-btn", state="visible")
        page.wait_for_function(
            "() => !document.body.classList.contains('remote-desktop')"
        )
    finally:
        page.close()


def test_panel_can_grow_beyond_the_old_default_cap(
    browser: Browser,
    remote_url: str,
) -> None:
    """A drag may widen the panel past 820px; only MIN_CHAT_W limits it."""
    page = _open_desktop_page(browser, remote_url, 1920)
    try:
        _drag_resizer_to(page, 1000)
        grown = page.evaluate(_LAYOUT_JS)
        assert grown["sidebarWidth"] == pytest.approx(1000, abs=2), (
            "the resizer used to clamp at --sidebar-max-w (820px); a "
            f"drag to 1000px left the panel at {grown['sidebarWidth']}px"
        )
        _drag_resizer_to(page, 1910)
        maxed = page.evaluate(_LAYOUT_JS)
        # Fully right stops where the chat keeps MIN_CHAT_W between the
        # history panel and the right task-info panel (a fifth of the
        # window): 1920 * 0.8 - 360.
        assert maxed["sidebarWidth"] == pytest.approx(
            1920 * (1 - PANEL_FRACTION) - MIN_CHAT_W, abs=2
        ), (
            "dragging fully right must stop where the chat keeps its "
            f"minimum width, got {maxed['sidebarWidth']}px"
        )
        assert maxed["appWidth"] >= MIN_CHAT_W - 1, maxed
    finally:
        page.close()


@pytest.mark.parametrize("width", DESKTOP_WIDTHS)
def test_settings_panel_is_narrow_on_desktop(
    browser: Browser,
    remote_url: str,
    width: int,
) -> None:
    """The settings drawer keeps the VS Code sidebar's narrow width."""
    page = _open_desktop_page(browser, remote_url, width)
    try:
        panel_width = page.eval_on_selector(
            "#settings-panel",
            "el => el.getBoundingClientRect().width",
        )
        # --settings-panel-w is 620px: wide enough for the "Tips",
        # "Git Commit", "Update", "Reset Server" and "Update Models"
        # buttons to sit on one line (see remote-codex.css), yet still
        # a narrow drawer rather than the 90vw mobile sheet.
        assert panel_width <= 620, (
            f"settings panel is {panel_width}px wide at viewport {width}px; "
            "it must stay narrow like the VS Code extension sidebar"
        )
        assert panel_width >= 300, panel_width
    finally:
        page.close()

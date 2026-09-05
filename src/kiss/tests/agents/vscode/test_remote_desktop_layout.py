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

* The task-history panel is wide enough that the five status/scope
  toggle buttons inside the collapsible ``Filters`` section all sit on
  a single line.
* The burger button (``#menu-btn``) really shows and hides the docked
  history panel: the panel slides off screen and the chat area
  reclaims the freed horizontal space.
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

# --sidebar-min-w / --chat-min-w from remote-codex.css: the narrowest
# the history panel may get (all filter toggles on one line) and the
# chat width the panel may never eat into.
MIN_PANEL_W = 520
MIN_CHAT_W = 360

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

_FILTER_CHIP_ROWS_JS = """
() => {
  const chips = [...document.querySelectorAll('.history-filter-chips .hf-chip')];
  const rows = new Set(chips.map(c => Math.round(c.getBoundingClientRect().top)));
  return {count: chips.length, rows: rows.size};
}
"""

_LAYOUT_JS = """
() => {
  const sidebar = document.getElementById('sidebar');
  const app = document.getElementById('app');
  const sb = sidebar.getBoundingClientRect();
  return {
    open: sidebar.classList.contains('open'),
    sidebarRight: sb.right,
    sidebarWidth: sb.width,
    appLeft: app.getBoundingClientRect().left,
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
def test_filter_toggles_fit_on_one_line(
    browser: Browser,
    remote_url: str,
    width: int,
) -> None:
    """Every filter toggle button shares a single row of the panel."""
    page = _open_desktop_page(browser, remote_url, width)
    try:
        chips = page.evaluate(_FILTER_CHIP_ROWS_JS)
        assert chips["count"] == 5, chips
        assert chips["rows"] == 1, (
            f"filter toggles wrapped into {chips['rows']} rows "
            f"at viewport width {width}"
        )
    finally:
        page.close()


def test_narrowest_resizable_panel_still_fits_the_toggles(
    browser: Browser,
    remote_url: str,
) -> None:
    """Dragging the resizer fully left keeps the toggles on one line."""
    page = _open_desktop_page(browser, remote_url, 1440)
    try:
        _drag_resizer_to(page, 900)
        widened = page.evaluate(_LAYOUT_JS)
        assert widened["sidebarWidth"] > MIN_PANEL_W, widened
        _drag_resizer_to(page, 0)
        narrowed = page.evaluate(_LAYOUT_JS)
        assert narrowed["sidebarWidth"] < widened["sidebarWidth"], (
            "the drag must actually have narrowed the panel, it is still "
            f"{narrowed['sidebarWidth']}px wide"
        )
        assert narrowed["sidebarWidth"] == pytest.approx(MIN_PANEL_W), (
            "dragging fully left must stop at the one-line filter width, "
            f"got {narrowed['sidebarWidth']}px"
        )
        chips = page.evaluate(_FILTER_CHIP_ROWS_JS)
        assert chips["rows"] == 1, (
            "shrinking the panel to its minimum width must not wrap the "
            f"filter toggles, got {chips['rows']} rows"
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
        assert 1920 - wide["sidebarWidth"] >= MIN_CHAT_W, wide

        page.set_viewport_size({"width": 900, "height": 900})
        page.wait_for_timeout(400)
        narrow = page.evaluate(_LAYOUT_JS)
        assert narrow["sidebarWidth"] >= MIN_PANEL_W, narrow
        assert 900 - narrow["sidebarWidth"] >= MIN_CHAT_W, (
            "after shrinking the window the chat must keep at least "
            f"{MIN_CHAT_W}px, but the panel is {narrow['sidebarWidth']}px "
            "of 900px"
        )
        chips = page.evaluate(_FILTER_CHIP_ROWS_JS)
        assert chips["rows"] == 1, chips
    finally:
        page.close()


@pytest.mark.parametrize("width", (1000, 1440))
def test_burger_button_hides_and_shows_the_history_panel(
    browser: Browser,
    remote_url: str,
    width: int,
) -> None:
    """#menu-btn slides the docked panel out of view and back."""
    page = _open_desktop_page(browser, remote_url, width)
    try:
        docked = page.evaluate(_LAYOUT_JS)
        assert docked["open"] is True
        assert docked["sidebarRight"] > 0, docked
        assert docked["appLeft"] == pytest.approx(docked["sidebarWidth"]), docked

        page.click("#menu-btn")
        page.wait_for_function("() => !document.getElementById('sidebar')"
                              ".classList.contains('open')")
        page.wait_for_timeout(400)
        hidden = page.evaluate(_LAYOUT_JS)
        assert hidden["open"] is False
        assert hidden["sidebarRight"] <= 0, (
            "burger click must move the history panel off screen, "
            f"its right edge is still at {hidden['sidebarRight']}px"
        )
        assert hidden["appLeft"] == 0, (
            "chat area must reclaim the space freed by the hidden panel, "
            f"left offset is still {hidden['appLeft']}px"
        )

        page.click("#menu-btn")
        page.wait_for_function("() => document.getElementById('sidebar')"
                              ".classList.contains('open')")
        page.wait_for_timeout(400)
        reshown = page.evaluate(_LAYOUT_JS)
        assert reshown["open"] is True
        assert reshown["sidebarRight"] == pytest.approx(docked["sidebarRight"])
        assert reshown["appLeft"] == pytest.approx(docked["appLeft"])
        assert (
            page.eval_on_selector(
                "#sidebar-overlay",
                "el => el.classList.contains('open')",
            )
            is False
        )
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

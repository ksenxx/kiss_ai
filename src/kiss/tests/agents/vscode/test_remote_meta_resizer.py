# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for resizing the docked task-info panel.

Desktop mode (viewport >= 900px) docks the task-info panel on the
RIGHT of the chat.  Like the history panel on the left it carries a
drag handle — ``#meta-resizer``, riding the panel's LEFT edge — that
resizes it: a drag may collapse the panel to the 10px sliver or widen
it until the chat column between the two panels hits its 360px
minimum, a double-click restores the one-fifth default, and the chosen
width persists across reloads via localStorage.

These tests render the REAL page returned by
:func:`kiss.server.web_server._build_html` (same ``media/chat.html``
template, same ``main.css`` + ``remote-codex.css``, same
``remote-chat`` body class) in headless Chromium, drag with a real
mouse and measure the resulting geometry.

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

# --sidebar-min-w / --chat-min-w from remote-codex.css: the sliver a
# drag may collapse either docked panel down to, and the chat width
# the panels may never eat into.
SLIVER_W = 10
MIN_CHAT_W = 360

# The docked panels' default share of the browser window
# (--sidebar-default-w and --meta-panel-w are both 20vw).
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

_GEOMETRY_JS = """
() => {
  const panel = document.getElementById('meta-panel');
  const sidebar = document.getElementById('sidebar');
  const app = document.getElementById('app');
  const resizer = document.getElementById('meta-resizer');
  const pr = panel.getBoundingClientRect();
  const rr = resizer.getBoundingClientRect();
  return {
    panelLeft: pr.left,
    panelRight: pr.right,
    panelWidth: pr.width,
    resizerVisible: resizer.offsetParent !== null,
    resizerLeft: rr.left,
    resizerCursor: getComputedStyle(resizer).cursor,
    appWidth: app.getBoundingClientRect().width,
    appRight: app.getBoundingClientRect().right,
    sidebarWidth: sidebar.getBoundingClientRect().width,
    stored: localStorage.getItem('kiss-meta-w'),
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


def _open_desktop_page(browser: Browser, url: str, width: int) -> Page:
    """Open the remote page at ``width`` in desktop mode."""
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.goto(url)
    page.wait_for_selector("body.remote-desktop", state="attached")
    page.evaluate(_PREPARE_JS)
    page.wait_for_selector("#meta-panel", state="visible")
    return page


def _drag_meta_resizer_to(page: Page, x: float) -> None:
    """Drag the task-info panel's resize handle to viewport ``x``."""
    box = page.locator("#meta-resizer").bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 200)
    page.mouse.down()
    page.mouse.move(x, box["y"] + 200, steps=8)
    page.mouse.up()


def test_resizer_rides_the_panels_left_edge(
    browser: Browser,
    remote_url: str,
) -> None:
    """The handle is visible on desktop, hugs the panel's left edge and
    shows the col-resize cursor."""
    page = _open_desktop_page(browser, remote_url, 1440)
    try:
        geo = page.evaluate(_GEOMETRY_JS)
        assert geo["resizerVisible"] is True
        assert geo["resizerCursor"] == "col-resize", geo
        assert geo["resizerLeft"] == pytest.approx(geo["panelLeft"], abs=2), (
            "the handle must ride the panel's LEFT edge (facing the "
            f"chat): {geo}"
        )
    finally:
        page.close()


def test_drag_left_widens_the_panel_and_app_follows(
    browser: Browser,
    remote_url: str,
) -> None:
    """Dragging the handle toward the chat widens the panel; the chat
    column's right edge tracks the panel's new left edge."""
    page = _open_desktop_page(browser, remote_url, 1440)
    try:
        _drag_meta_resizer_to(page, 1440 - 500)
        geo = page.evaluate(_GEOMETRY_JS)
        assert geo["panelWidth"] == pytest.approx(500, abs=2), geo
        assert geo["panelRight"] == pytest.approx(1440, abs=1), geo
        assert geo["appRight"] == pytest.approx(geo["panelLeft"], abs=1), (
            f"#app must clear the widened panel: {geo}"
        )
        assert geo["stored"] == "500", (
            f"the drag must persist the width to localStorage: {geo}"
        )
    finally:
        page.close()


def test_drag_right_collapses_to_a_sliver_and_dblclick_restores(
    browser: Browser,
    remote_url: str,
) -> None:
    """A drag to the window's right edge collapses the panel to the
    10px sliver; a double-click restores the one-fifth default and
    clears the persisted width."""
    page = _open_desktop_page(browser, remote_url, 1440)
    try:
        _drag_meta_resizer_to(page, 1440)
        sliver = page.evaluate(_GEOMETRY_JS)
        assert sliver["panelWidth"] == pytest.approx(SLIVER_W, abs=2), (
            "dragging fully right must collapse the panel to the "
            f"{SLIVER_W}px sliver, got {sliver['panelWidth']}px"
        )
        page.locator("#meta-resizer").dblclick()
        restored = page.evaluate(_GEOMETRY_JS)
        assert restored["panelWidth"] == pytest.approx(
            1440 * PANEL_FRACTION, abs=1,
        ), (
            "double-click must restore the one-fifth default width, "
            f"got {restored['panelWidth']}px"
        )
        assert restored["stored"] is None, (
            f"double-click must clear the persisted width: {restored}"
        )
    finally:
        page.close()


def test_drag_left_stops_where_the_chat_keeps_its_minimum(
    browser: Browser,
    remote_url: str,
) -> None:
    """A drag across the whole window stops where the chat column —
    between the history panel and the task-info panel — keeps 360px."""
    page = _open_desktop_page(browser, remote_url, 1920)
    try:
        _drag_meta_resizer_to(page, 0)
        geo = page.evaluate(_GEOMETRY_JS)
        assert geo["panelWidth"] == pytest.approx(
            1920 - MIN_CHAT_W - geo["sidebarWidth"], abs=2,
        ), (
            "dragging fully left must stop where the chat keeps its "
            f"minimum beside the history panel: {geo}"
        )
        assert geo["appWidth"] >= MIN_CHAT_W - 1, geo
    finally:
        page.close()


def test_both_panels_widened_cannot_crush_the_chat(
    browser: Browser,
    remote_url: str,
) -> None:
    """Widening the history panel first must shrink how far the
    task-info panel may grow: after both drags the chat keeps 360px."""
    page = _open_desktop_page(browser, remote_url, 1920)
    try:
        # Widen the history panel as far as it goes ...
        box = page.locator("#sidebar-resizer").bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 200)
        page.mouse.down()
        page.mouse.move(1900, box["y"] + 200, steps=8)
        page.mouse.up()
        # ... then try to widen the task-info panel across the window.
        _drag_meta_resizer_to(page, 0)
        geo = page.evaluate(_GEOMETRY_JS)
        assert geo["appWidth"] >= MIN_CHAT_W - 1, (
            "after widening BOTH panels the chat column must keep "
            f"{MIN_CHAT_W}px: {geo}"
        )
        assert geo["sidebarWidth"] + geo["panelWidth"] <= (
            1920 - MIN_CHAT_W + 2
        ), geo
    finally:
        page.close()


def test_width_persists_across_reloads(
    browser: Browser,
    remote_url: str,
) -> None:
    """The dragged width is restored from localStorage on the next
    load of the page."""
    page = _open_desktop_page(browser, remote_url, 1440)
    try:
        _drag_meta_resizer_to(page, 1440 - 450)
        assert page.evaluate(_GEOMETRY_JS)["stored"] == "450"
        page.goto(page.url)
        page.wait_for_selector("body.remote-desktop", state="attached")
        page.evaluate(_PREPARE_JS)
        page.wait_for_selector("#meta-panel", state="visible")
        geo = page.evaluate(_GEOMETRY_JS)
        assert geo["panelWidth"] == pytest.approx(450, abs=2), (
            f"the persisted width must be restored on reload: {geo}"
        )
    finally:
        page.close()


def test_sliver_keeps_a_grabbable_handle(
    browser: Browser,
    remote_url: str,
) -> None:
    """Collapsed to 10px the panel's padding shrinks with it, so the
    handle still covers the sliver and a drag can re-widen the panel."""
    page = _open_desktop_page(browser, remote_url, 1440)
    try:
        _drag_meta_resizer_to(page, 1440)
        assert page.evaluate(_GEOMETRY_JS)["panelWidth"] == pytest.approx(
            SLIVER_W, abs=2,
        )
        _drag_meta_resizer_to(page, 1440 - 300)
        geo = page.evaluate(_GEOMETRY_JS)
        assert geo["panelWidth"] == pytest.approx(300, abs=2), (
            f"the sliver must still be resizable back out: {geo}"
        )
    finally:
        page.close()


def test_handle_stays_grabbable_when_the_panel_content_scrolls(
    browser: Browser,
    remote_url: str,
) -> None:
    """Long meta values on a short window overflow the panel; the list
    scrolls INSIDE the panel (like the history panel's #history-list)
    so the drag handle keeps covering the panel's whole edge and stays
    under the mouse at any scroll position."""
    page = _open_desktop_page(browser, remote_url, 1000)
    try:
        page.set_viewport_size({"width": 1000, "height": 300})
        page.wait_for_selector("body.remote-desktop", state="attached")
        # Fill the bullets with realistic long values and collapse the
        # panel so overflow-wrap:anywhere stacks them far past the
        # viewport height.
        page.evaluate(
            """
            () => {
              const long = 'ksen-vm-32.c.r2eg-441800.internal (x86_64)';
              for (const id of ['meta-tokens', 'meta-cost', 'meta-steps',
                                'meta-time', 'meta-machine']) {
                document.getElementById(id).textContent = long + ' ' + long;
              }
            }
            """
        )
        _drag_meta_resizer_to(page, 1000 - 60)
        state = page.evaluate(
            """
            () => {
              const panel = document.getElementById('meta-panel');
              const list = document.getElementById('meta-list');
              list.scrollTop = list.scrollHeight;
              panel.scrollTop = panel.scrollHeight;
              const pr = panel.getBoundingClientRect();
              const rr = document
                .getElementById('meta-resizer')
                .getBoundingClientRect();
              const probe = document.elementFromPoint(
                pr.left + 3,
                window.innerHeight - 20,
              );
              return {
                overflows: list.scrollHeight > list.clientHeight,
                panelScrolled: panel.scrollTop,
                resizerTop: rr.top,
                resizerBottom: rr.bottom,
                probeId: probe ? probe.id : null,
              };
            }
            """
        )
        assert state["overflows"] is True, (
            f"the long values must overflow the collapsed panel: {state}"
        )
        assert state["panelScrolled"] == 0, (
            "the PANEL itself must not scroll (the list scrolls inside "
            f"it), or the handle scrolls away with the content: {state}"
        )
        assert state["resizerTop"] == pytest.approx(0, abs=2), state
        assert state["resizerBottom"] == pytest.approx(300, abs=2), (
            f"the handle must keep covering the panel's whole edge: {state}"
        )
        assert state["probeId"] == "meta-resizer", (
            "with the list scrolled to the bottom the mouse must still "
            f"land on the drag handle at the panel's edge: {state}"
        )
        # And the handle really works there: drag the panel back out.
        _drag_meta_resizer_to(page, 1000 - 250)
        geo = page.evaluate(_GEOMETRY_JS)
        assert geo["panelWidth"] == pytest.approx(250, abs=2), geo
    finally:
        page.close()


def test_mobile_hides_the_resizer(
    browser: Browser,
    remote_url: str,
) -> None:
    """Below the 900px breakpoint the panel and its handle are gone."""
    page = _open_desktop_page(browser, remote_url, 1280)
    try:
        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_function(
            "() => !document.body.classList.contains('remote-desktop')"
        )
        assert page.locator("#meta-resizer").is_hidden()
        assert page.locator("#meta-panel").is_hidden()
    finally:
        page.close()

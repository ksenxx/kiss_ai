# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   test parameters of the same name)
"""The content context menu takes the chat page's theme in both of its hosts.

``media/contentContextMenu.js`` styles its menu with the page's design
tokens (``var(--menu-bg,...)`` and so on), so in the chat page the menu
follows the theme.  The sandboxed preview iframe (an opaque origin)
inherits none of them, and the page it shows may use the same names for
its own purposes, so the bootstrap markup carries the values the tokens
have on the host page at that moment.  A token the host does not define
falls back to the dark palette's value in both hosts.  Only a browser
resolves ``var()`` and renders the iframe, so the real script is inlined
into a page served to Chromium from a tiny HTTP server on 127.0.0.1.

The values travel as data of the bootstrap script: one that holds
``</script>`` must not end it.  A preview that is already open follows a
later change of the page's appearance: the production page (a real
``RemoteAccessServer``) sends its frames the menu's stylesheet again when
the user switches the theme.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import FrameLocator, Locator, Page, expect, sync_playwright

from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_editing import _uncover_pane
from kiss.tests.webapp.test_content_tab_file_links import (
    _inject_file_link,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_result_ls_listing_file_links import _open_page

MENU_JS = Path(__file__).resolve().parents[2] / "webapp" / "media" / "contentContextMenu.js"

# The host defines five of the menu's tokens and leaves the rest (the
# shadow, the shortcut's colour, the spacing) to their fallbacks.  The
# previewed page sets the same names to other values on its own root,
# and posts itself a stylesheet for the menu the way the host does: the
# menu listens to the host alone.
PAGE = b"""<!DOCTYPE html><html><head><style>
body { --menu-bg: rgb(250, 250, 250); --fg: rgb(11, 11, 11); --border: rgb(1, 2, 3);
       --hover: rgb(4, 5, 6); --radius-lg: 9px; }
</style></head><body>
<p id="text">Host text</p>
<iframe id="preview" sandbox="allow-scripts"></iframe>
<script>/*MENU_JS*/</script>
<script>
window.__menu = ContentContextMenu.installContentContextMenu(document);
document.getElementById('preview').srcdoc =
  '<style>:root { --menu-bg: red; --fg: red; --border: red; --hover: red;' +
  ' --radius-lg: 30px; --faint: red; }</style><body><p id="text">Preview text</p>' +
  ContentContextMenu.contentContextMenuBootstrapHtml().trim() +
  '<script>postMessage({kissMenuCss: "#sorcar-content-context-menu{background:red}"}, "*")<' +
  '/script></body>';
</script>
</body></html>"""

# A token value that closes the bootstrap's <script> and opens one of its
# own (a quoted font family, U+2028 included), and a second preview fed
# while the host's --font-ui holds it; resolves once that preview loaded.
_HOSTILE_FONT = "\"X\u2028</script><script>document.body.dataset.injected='1'</script>\""
_ADD_PREVIEW_JS = """font => new Promise(resolve => {
  document.body.style.setProperty('--font-ui', font);
  const frame = document.createElement('iframe');
  frame.id = 'hostile';
  frame.setAttribute('sandbox', 'allow-scripts');
  frame.onload = resolve;
  frame.srcdoc = '<body><p id="text">Preview text</p>' +
    ContentContextMenu.contentContextMenuBootstrapHtml() + '</body>';
  document.body.append(frame);
})"""

# The preview frame of the content tab on screen, in the production page.
_FRAME = "#content-tab-area .content-tab-view:not([style*='display: none']) .content-html-frame"

# The colours the production page's own tokens give the menu's surface.
_PAGE_MENU_COLOURS_JS = """() => {
  const probe = document.createElement('div');
  probe.style.cssText = 'background: var(--menu-bg); color: var(--fg)';
  document.body.appendChild(probe);
  const style = getComputedStyle(probe);
  const colours = [style.backgroundColor, style.color];
  probe.remove();
  return colours;
}"""

# What the open menu looks like: its surface, its focused row (opening
# focuses the first enabled item) and a shortcut.
LOOK = """menu => {
  const style = getComputedStyle(menu);
  return {
    background: style.backgroundColor,
    color: style.color,
    border: style.borderTopColor,
    radius: style.borderTopLeftRadius,
    shadow: style.boxShadow,
    font: style.fontFamily,
    row: getComputedStyle(menu.querySelector('.sorcar-ctx-item:focus')).backgroundColor,
    key: getComputedStyle(menu.querySelector('.sorcar-ctx-key')).color,
  };
}"""

HOST_LOOK = {
    "background": "rgb(250, 250, 250)",
    "color": "rgb(11, 11, 11)",
    "border": "rgb(1, 2, 3)",
    "radius": "9px",
    "shadow": "rgba(0, 0, 0, 0.45) 0px 12px 40px 0px",
    "font": 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    "row": "rgb(4, 5, 6)",
    "key": "rgb(137, 135, 129)",
}


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve the fixture page with the production context-menu script inlined."""

    def do_GET(self) -> None:  # noqa: N802  (http.server API)
        """Answer every path with the page."""
        body = PAGE.replace(b"/*MENU_JS*/", MENU_JS.read_bytes())
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output free of request logs."""


def _open_menu(scope: Page | FrameLocator) -> Locator:
    """Right-click the text of *scope*'s document and return its menu."""
    scope.locator("#text").click(button="right", position={"x": 4, "y": 4})
    return scope.locator("#sorcar-content-context-menu")


def _close_menu(page: Page, menu: Locator) -> None:
    """Close *menu* with Escape: left open, the host's menu lies over the
    preview's text."""
    page.keyboard.press("Escape")
    menu.wait_for(state="detached")


def _look(page: Page, scope: Page | FrameLocator) -> dict[str, str]:
    """The look of the menu a right-click opens in *scope*'s document."""
    menu = _open_menu(scope)
    look: dict[str, str] = menu.evaluate(LOOK)
    _close_menu(page, menu)
    return look


@pytest.fixture
def host_page() -> Iterator[Page]:
    """The fixture page in Chromium; its scripts must not have thrown."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            chromium = p.chromium.launch()
            try:
                page = chromium.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                origin = f"http://127.0.0.1:{server.server_address[1]}"
                goto_retrying_network_change(page, origin + "/", wait_until="load")
                assert page.evaluate("() => !!window.__menu"), errors
                yield page
                assert errors == []
            finally:
                chromium.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    assert not thread.is_alive(), "fixture HTTP server failed to stop"


@pytest.mark.timeout(180)
def test_menu_takes_the_host_page_theme_in_the_page_and_in_the_preview(host_page: Page) -> None:
    """The menu shows the host's token values in the host document and in the
    sandboxed preview, whose own values for the same names (and a stylesheet
    it posts itself) do not reach it; in the host it follows a later change
    of a token."""
    assert _look(host_page, host_page) == HOST_LOOK
    assert _look(host_page, host_page.frame_locator("#preview")) == HOST_LOOK

    host_page.evaluate("() => document.body.style.setProperty('--menu-bg', 'rgb(7, 8, 9)')")
    assert _look(host_page, host_page)["background"] == "rgb(7, 8, 9)"


@pytest.mark.timeout(180)
def test_a_token_value_cannot_end_the_preview_script(host_page: Page) -> None:
    """A token value that holds ``</script><script>...`` reaches the preview
    as text: the script it carries does not run, and the preview's menu
    shows the value like the host's."""
    host_page.evaluate(_ADD_PREVIEW_JS, _HOSTILE_FONT)
    preview = host_page.frame_locator("#hostile")
    assert preview.locator("body").get_attribute("data-injected") is None
    look = _look(host_page, preview)
    assert "dataset.injected" in look["font"], look
    assert look == _look(host_page, host_page)


@pytest.mark.timeout(180)
def test_an_open_preview_follows_the_page_theme(browser, harness) -> None:
    """In the production page the menu of a preview that is already open
    takes the new palette when the user switches the theme: on the first
    switch the menu has not been opened yet, on the second it has."""
    context, page, _sent = _open_page(browser, harness)
    try:
        path = Path(harness.work_dir) / "menu_theme.html"
        path.write_text(
            '<!DOCTYPE html><html><body><p id="text">Preview text</p></body></html>',
            encoding="utf-8",
        )
        _inject_file_link(page, str(path), "lnk-menu-theme")
        page.click("#lnk-menu-theme")
        page.wait_for_selector(_FRAME, timeout=30000)
        _uncover_pane(page)
        palettes: list[str] = []
        for _ in range(2):
            page.evaluate("() => document.getElementById('theme-btn').click()")
            background, colour = page.evaluate(_PAGE_MENU_COLOURS_JS)
            menu = _open_menu(page.frame_locator(_FRAME))
            expect(menu).to_have_css("background-color", background)
            expect(menu).to_have_css("color", colour)
            _close_menu(page, menu)
            palettes.append(background)
        assert palettes[0] != palettes[1], palettes
    finally:
        context.close()

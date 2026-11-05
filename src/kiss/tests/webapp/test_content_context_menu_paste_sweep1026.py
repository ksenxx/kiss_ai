# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Paste and Copy from the content context menu on the native selection of a
real Chromium.

``number`` / ``email`` inputs expose no selection API (``selectionStart``
is null and ``setRangeText`` throws), but they do keep a native selection:
after Select All, typing or a native paste replaces the whole value.  The
menu's Paste used to append the clipboard text instead (``12`` + Select
All + Paste ``34`` gave ``1234``), so it now goes through
``document.execCommand('insertText')``, which honours the native selection
and fires the field's ``input`` event itself.  Copy was disabled for those
inputs, whose selection only the document ``Selection`` reports, and on a
page it copied the selected ranges' source text (``Range.toString()``):
hidden elements and inline ``<script>`` source that the native Copy, like
the user, never sees.  jsdom has no ``execCommand`` and renders nothing,
so this can only be shown in a browser: the real
``media/contentContextMenu.js`` is inlined into a page served to Chromium
from a tiny HTTP server on 127.0.0.1 (a secure context, so the async
clipboard works once the permission is granted).  Inlined, the page is a
single request, which :func:`goto_retrying_network_change` retries, and
``load`` means both scripts ran, so the menu handle is checked at once,
with the page's errors as the message.  Under a full-suite load the page
once loaded without the handle (2026-10-29), and a 30 s wait for it, with
the script fetched as a second request, gave no cause.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from playwright.sync_api import Page, Position, sync_playwright

from kiss.tests.conftest import goto_retrying_network_change

MENU_JS = Path(__file__).resolve().parents[2] / "webapp" / "media" / "contentContextMenu.js"

PAGE = b"""<!DOCTYPE html><html><body>
<main><span id="a">VISIBLE_A</span><span hidden>HIDDEN_ATTRIBUTE</span><span \
style="display:none">HIDDEN_DISPLAY</span><script>window.__inline = 'NOT_VISIBLE_SOURCE';\
</script><span id="b">VISIBLE_B</span><script>void 0</script></main>
<input id="num" type="number" value="12">
<input id="mail" type="email" value="a@b">
<input id="txt" type="text" value="ab">
<script>/*MENU_JS*/</script>
<script>
window.__inputs = [];
for (const el of document.querySelectorAll('input')) {
  el.addEventListener('input', () => window.__inputs.push(el.id + '=' + el.value));
}
window.__menu = ContentContextMenu.installContentContextMenu(document);
</script>
</body></html>"""


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


def _menu(page: Page, selector: str, action: str) -> None:
    """Right-click *selector* and choose the menu item *action*."""
    page.click(selector, button="right")
    page.click(f'#sorcar-content-context-menu .sorcar-ctx-item[data-action="{action}"]')


def _paste(page: Page, selector: str, text: str) -> str:
    """Put *text* on the clipboard, menu-Paste it into *selector*, return the value."""
    page.evaluate("t => navigator.clipboard.writeText(t)", text)
    page.evaluate("() => { window.__inputs.length = 0; }")
    _menu(page, selector, "paste")
    page.wait_for_function("() => window.__inputs.length > 0", timeout=10000)
    return str(page.evaluate(f"document.getElementById('{selector[1:]}').value"))


def _copy(page: Page, selector: str, position: Position | None = None) -> str:
    """Right-click *selector* (at *position* in it), menu-Copy, return the clipboard."""
    page.evaluate("() => navigator.clipboard.writeText('<before>')")
    page.click(selector, button="right", position=position)
    item = page.locator('#sorcar-content-context-menu .sorcar-ctx-item[data-action="copy"]')
    assert item.get_attribute("aria-disabled") is None, "Copy is disabled"
    item.click()
    page.wait_for_function(
        "async () => (await navigator.clipboard.readText()) !== '<before>'", timeout=10000
    )
    return str(page.evaluate("() => navigator.clipboard.readText()"))


@contextmanager
def _page() -> Iterator[Page]:
    """Serve the fixture page and yield a Chromium page on it, the clipboard allowed."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context()
                context.grant_permissions(["clipboard-read", "clipboard-write"], origin=origin)
                page = context.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                goto_retrying_network_change(page, origin + "/", wait_until="load")
                # The scripts are inline, so "load" means they ran: no handle is a script error.
                assert page.evaluate("() => !!window.__menu"), errors
                yield page
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    assert not thread.is_alive(), "fixture HTTP server failed to stop"


@pytest.mark.timeout(180)
def test_paste_replaces_the_native_selection_of_number_and_email_inputs() -> None:
    """Select All then Paste replaces the value; without a selection the
    text is inserted at the caret; each paste fires exactly one ``input``."""
    with _page() as page:
        _menu(page, "#num", "select-all")
        assert _paste(page, "#num", "34") == "34"
        assert page.evaluate("window.__inputs") == ["num=34"]

        _menu(page, "#mail", "select-all")
        assert _paste(page, "#mail", "x@y") == "x@y"
        assert page.evaluate("window.__inputs") == ["mail=x@y"]

        # No selection: inserted at the caret (the right-click that
        # opens the menu lands right of the text, so the caret is
        # at the end, exactly as with the native menu).
        page.click("#num")
        page.keyboard.press("End")
        assert _paste(page, "#num", "5") == "345"
        assert page.evaluate("window.__inputs") == ["num=345"]

        # A text input keeps its selection-API path (caret after the paste).
        _menu(page, "#txt", "select-all")
        assert _paste(page, "#txt", "X") == "X"
        assert page.evaluate("document.getElementById('txt').selectionStart") == 1
        assert page.evaluate("window.__inputs") == ["txt=X"]


@pytest.mark.timeout(180)
def test_copy_reads_the_native_selection_of_number_and_email_inputs() -> None:
    """Copy is offered for, and copies, the selected part of a ``number`` /
    ``email`` input's value: a Select All and a partial selection alike."""
    with _page() as page:
        page.click("#num")
        page.keyboard.press("ControlOrMeta+a")
        assert _copy(page, "#num") == "12"
        page.click("#mail")
        page.keyboard.press("Home")
        page.keyboard.press("Shift+ArrowRight")
        page.keyboard.press("Shift+ArrowRight")
        # Right-clicked on the selected "a@": a right click elsewhere in
        # the field moves the caret there, as the native menu's does.
        assert _copy(page, "#mail", position={"x": 6, "y": 10}) == "a@"


@pytest.mark.timeout(180)
def test_copy_puts_the_rendered_selection_on_the_clipboard() -> None:
    """Copy after a native Select All copies the selection as rendered, the
    hidden elements and the inline script source left out like the native
    Copy does; the menu's own Select All leaves them out the same way, and
    so does text the page hides between Select All and Copy."""
    with _page() as page:
        page.click("#a")
        page.keyboard.press("ControlOrMeta+a")
        shown = str(page.evaluate("() => getSelection().toString()")).strip()
        assert "VISIBLE_A" in shown and "VISIBLE_B" in shown and "HIDDEN" not in shown
        assert _copy(page, "#a") == shown
        _menu(page, "#a", "select-all")
        # Chromium on Windows puts CRLF line breaks on the clipboard.
        assert _copy(page, "#a").replace("\r\n", "\n") == "VISIBLE_A\nVISIBLE_B"
        # The second stretch (one text node, between two scripts) is
        # hidden after Select All: Copy leaves it out, as the native Copy
        # would.
        _menu(page, "#a", "select-all")
        page.evaluate("() => { document.getElementById('b').style.display = 'none'; }")
        assert _copy(page, "#a") == "VISIBLE_A"

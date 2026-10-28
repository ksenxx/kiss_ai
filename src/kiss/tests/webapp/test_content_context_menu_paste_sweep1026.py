# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Paste from the content context menu replaces the native selection of a
``number`` / ``email`` input in a real Chromium.

Those input types expose no selection API (``selectionStart`` is null and
``setRangeText`` throws), but they do keep a native selection: after
Select All, typing or a native paste replaces the whole value.  The menu's
Paste used to append the clipboard text instead (``12`` + Select All +
Paste ``34`` gave ``1234``), so it now goes through
``document.execCommand('insertText')``, which honours the native selection
and fires the field's ``input`` event itself.  jsdom has no
``execCommand``, so this can only be shown in a browser: the real
``media/contentContextMenu.js`` is served into Chromium from a tiny HTTP
server on 127.0.0.1 (a secure context, so the async clipboard works once
the permission is granted).
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright

from kiss.tests.conftest import goto_retrying_network_change

MENU_JS = Path(__file__).resolve().parents[2] / "webapp" / "media" / "contentContextMenu.js"

PAGE = b"""<!DOCTYPE html><html><body>
<input id="num" type="number" value="12">
<input id="mail" type="email" value="a@b">
<input id="txt" type="text" value="ab">
<script src="/contentContextMenu.js"></script>
<script>
window.__inputs = [];
for (const el of document.querySelectorAll('input')) {
  el.addEventListener('input', () => window.__inputs.push(el.id + '=' + el.value));
}
window.__menu = ContentContextMenu.installContentContextMenu(document);
</script>
</body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve the fixture page and the production context-menu script."""

    def do_GET(self) -> None:  # noqa: N802  (http.server API)
        """Answer ``/`` with the page and ``/contentContextMenu.js`` with the script."""
        if self.path.startswith("/contentContextMenu.js"):
            body, ctype = MENU_JS.read_bytes(), "text/javascript"
        else:
            body, ctype = PAGE, "text/html"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
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


@pytest.mark.timeout(180)
def test_paste_replaces_the_native_selection_of_number_and_email_inputs() -> None:
    """Select All then Paste replaces the value; without a selection the
    text is inserted at the caret; each paste fires exactly one ``input``."""
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
                goto_retrying_network_change(page, origin + "/", wait_until="load")
                page.wait_for_function("() => !!window.__menu")

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
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    assert not thread.is_alive(), "fixture HTTP server failed to stop"

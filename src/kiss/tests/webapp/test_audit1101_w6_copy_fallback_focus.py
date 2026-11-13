# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The hidden-textarea copy fallback hands focus and the selection back.

Audit 2026-11-01 W6 F4: when the async clipboard API is missing or
refuses, ``PanelCopy.fallbackCopyText`` and the content context menu's
``copyViaTextarea`` copy through a temporary textarea they select and
remove.  Chromium focuses the textarea when it is selected, so removing
it left focus on ``<body>`` and dropped the document selection: typing
after a copy from the composer went nowhere, and a copy from a preview's
contenteditable lost its highlighted text.  Both helpers now restore the
control focused when the fallback ran (into an open shadow root too, the
tips sheet lives in one) with its selection offsets, and the document
selection.  jsdom's ``select()`` does not move focus, so this needs a
browser: the real ``panelCopy.js`` and ``contentContextMenu.js`` are
inlined into a page served from 127.0.0.1 (the inline scripts make it one
request, which :func:`goto_retrying_network_change` retries).
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright

from kiss.tests.conftest import goto_retrying_network_change

MEDIA = Path(__file__).resolve().parents[2] / "webapp" / "media"

PAGE = b"""<!DOCTYPE html><html><body>
<textarea id="composer">Keep typing here</textarea>
<input id="other" value="elsewhere">
<div id="edit" contenteditable="true">Editable words</div>
<div id="host"></div>
<script>/*PANEL_JS*/</script>
<script>/*MENU_JS*/</script>
<script>
const host = document.getElementById('host');
host.attachShadow({mode: 'open'}).innerHTML = '<input id="inner" value="shadow field">';
window.__menu = ContentContextMenu.installContentContextMenu(document);
// The async clipboard refuses every write (as it does for an unfocused
// document or an opaque-origin preview), so every copy takes the
// execCommand fallback; reads go to the real clipboard to check what
// the fallback put there.  A write is held until the test releases it
// when window.__hold is set.
const real = navigator.clipboard;
window.__release = null;
Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {
  writeText: () => new Promise((_resolve, reject) => {
    if (window.__hold) window.__release = reject;
    else reject(new Error('refused'));
  }),
  readText: () => real.readText(),
}});
window.__focus = () => {
  const el = document.activeElement;
  const inner = el && el.shadowRoot && el.shadowRoot.activeElement;
  return (inner || el || document.body).id || (el || document.body).tagName;
};
</script>
</body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve the fixture page with the two production scripts inlined."""

    def do_GET(self) -> None:  # noqa: N802  (http.server API)
        """Answer every path with the page."""
        body = PAGE.replace(b"/*PANEL_JS*/", (MEDIA / "panelCopy.js").read_bytes())
        body = body.replace(b"/*MENU_JS*/", (MEDIA / "contentContextMenu.js").read_bytes())
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


@pytest.mark.timeout(180)
def test_copy_fallback_restores_focus_and_selection() -> None:
    """After a fallback copy the focused control, its selection offsets and
    the document selection are as before, the copy is on the clipboard, and
    a focus the user moved while the write was pending is left alone."""
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
                assert page.evaluate("() => !!window.__menu"), errors

                # A copy from the composer: focus, the selected range and the
                # clipboard text; typing then replaces the selection.
                result = page.evaluate(
                    """async () => {
                      const field = document.getElementById('composer');
                      field.focus();
                      field.setSelectionRange(5, 11);
                      const ok = await PanelCopy.copyText('Copied text');
                      return {ok, focus: window.__focus(),
                              start: field.selectionStart, end: field.selectionEnd,
                              clipboard: await navigator.clipboard.readText()};
                    }"""
                )
                assert result == {
                    "ok": True,
                    "focus": "composer",
                    "start": 5,
                    "end": 11,
                    "clipboard": "Copied text",
                }
                page.keyboard.type("HELLO")
                assert page.locator("#composer").input_value() == "Keep HELLO here"

                # A field inside an open shadow root (the tips sheet) too.
                result = page.evaluate(
                    """async () => {
                      const root = document.getElementById('host').shadowRoot;
                      const inner = root.getElementById('inner');
                      inner.focus();
                      inner.setSelectionRange(0, 6);
                      const ok = await PanelCopy.copyText('from tips');
                      return {ok, focus: window.__focus(),
                              start: inner.selectionStart, end: inner.selectionEnd,
                              clipboard: await navigator.clipboard.readText()};
                    }"""
                )
                assert result == {
                    "ok": True,
                    "focus": "inner",
                    "start": 0,
                    "end": 6,
                    "clipboard": "from tips",
                }

                # The context menu's Copy of a contenteditable's Select All:
                # the highlighted words stay highlighted and focused.
                _menu(page, "#edit", "select-all")
                _menu(page, "#edit", "copy")
                page.wait_for_function("() => document.activeElement !== document.body")
                result = page.evaluate(
                    """async () => ({focus: window.__focus(),
                                     selected: getSelection().toString(),
                                     clipboard: await navigator.clipboard.readText()})"""
                )
                assert result == {
                    "focus": "edit",
                    "selected": "Editable words",
                    "clipboard": "Editable words",
                }

                # A focus the user moved while the write was pending is kept:
                # the fallback restores the focus of its own moment.
                result = page.evaluate(
                    """async () => {
                      window.__hold = true;
                      document.getElementById('composer').focus();
                      const copy = PanelCopy.copyText('late');
                      document.getElementById('other').focus();
                      window.__hold = false;
                      window.__release(new Error('refused late'));
                      const ok = await copy;
                      return {ok, focus: window.__focus(),
                              clipboard: await navigator.clipboard.readText()};
                    }"""
                )
                assert result == {"ok": True, "focus": "other", "clipboard": "late"}

                # The textarea copy itself, which the menu exports for the
                # page's other copy controls (A/W1-F8).
                result = page.evaluate(
                    """async () => [ContentContextMenu.copyViaTextarea(document, 'direct'),
                                    await navigator.clipboard.readText()]"""
                )
                assert result == [True, "direct"]
                assert errors == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    assert not thread.is_alive(), "fixture HTTP server failed to stop"

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Select All, then Copy, puts on the clipboard what the selection covers now.

Audit 2026-11-01 W6 F5, round 2: the content context menu's Select All
selects the visible text as one live ``Range``, and Copy reads that Range
afresh.  The first fix re-read the whole scope instead, so a text node a
preview's script appended after the selection's end was copied although
the selection (its highlight, ``getSelection().toString()``) did not
cover it.  Now the Range Select All installed is kept and read: text
inserted inside it is copied, text added past its end is not, as for
Chromium's own selection, whose text the copy equals.  (Audit 2026-11-07
A23: it used to be one Range per stretch between hidden sources, of which
Chromium keeps the first, so the keyboard Copy got the first stretch
alone and the menu Copy joined the stretches with a line break the page
does not render.)

jsdom's live Ranges agree with Chromium's here, but the parent document
copies through ``PanelCopy`` while the sandboxed preview iframe runs the
menu's serialized bundle (``contentContextMenuBootstrapHtml``), whose
``execCommand`` copy and function list only a browser exercises: both are
driven in real Chromium from a page served on 127.0.0.1 (the inline
scripts make it one request, which :func:`goto_retrying_network_change`
retries).
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest
from playwright.sync_api import FrameLocator, Page, sync_playwright

from kiss.tests.conftest import goto_retrying_network_change

MEDIA = Path(__file__).resolve().parents[2] / "webapp" / "media"

# No whitespace text node after the editable: the selection ends right
# after it (the editable taken in whole, as the native Select All takes
# it), so a node appended to the body lands past the selection's end.
CONTENT = (
    b'<p id="one">One</p><script>var hidden = 1;</script>'
    b'<div id="edit" contenteditable="true">SELECTED</div>'
)

PAGE = (
    b"<!DOCTYPE html><html><body>"
    b'<div id="content-tab-area">' + CONTENT + b"</div>"
    b'<iframe id="preview" sandbox="allow-scripts"></iframe>'
    b"<script>/*PANEL_JS*/</script>"
    b"<script>/*MENU_JS*/</script>"
    b"<script>"
    b"window.__menu = ContentContextMenu.installContentContextMenu(document);"
    b"document.getElementById('preview').srcdoc = '<body>' +"
    b"  ContentContextMenu.contentContextMenuBootstrapHtml().trim() +"
    b"  /*CONTENT*/ + '</body>';"
    b"</script>"
    b"</body></html>"
)

# The page's script edits the selected text after Select All: an insertion
# inside the editable, a node appended to the body (past the selection's
# end) and data appended to *one* (the element the locator found), inside.
EDIT = """one => {
  const doc = one.ownerDocument;
  const edit = doc.getElementById('edit');
  edit.firstChild.insertData(3, '-in-');
  doc.body.appendChild(doc.createTextNode(' APPENDED'));
  one.firstChild.appendData(' more');
  return doc.defaultView.getSelection().toString();
}"""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve the fixture page with the two production scripts inlined."""

    def do_GET(self) -> None:  # noqa: N802  (http.server API)
        """Answer every path with the page."""
        body = PAGE.replace(b"/*PANEL_JS*/", (MEDIA / "panelCopy.js").read_bytes())
        body = body.replace(b"/*MENU_JS*/", (MEDIA / "contentContextMenu.js").read_bytes())
        # The iframe's own copy of the content, as a JS string literal
        # (`<\/script>` keeps the HTML parser from ending the inline script).
        literal = b"'" + CONTENT.replace(b"</script>", b"<\\/script>") + b"'"
        body = body.replace(b"/*CONTENT*/", literal)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output free of request logs."""


def _menu(scope: Page | FrameLocator, selector: str, action: str) -> None:
    """Right-click the text of *selector* in *scope* and choose the menu item *action*.

    On the text, not the element's centre: a right-click in a block's
    empty space collapses the selection, as it does for the user.
    """
    scope.locator(selector).click(button="right", position={"x": 4, "y": 4})
    scope.locator(f'#sorcar-content-context-menu .sorcar-ctx-item[data-action="{action}"]').click()


def _select_edit_copy(page: Page, scope: Page | FrameLocator) -> str:
    """Select All over *scope*'s document, edit it, Copy, and return the selection text."""
    _menu(scope, "#one", "select-all")
    selected = scope.locator("#one").evaluate(EDIT)
    _menu(scope, "#one", "copy")
    page.wait_for_function(
        "async () => (await navigator.clipboard.readText()).includes('SEL-in-ECTED')"
    )
    return str(selected)


def _clipboard(page: Page) -> str:
    """Return the clipboard text; Windows hands the system clipboard CRLF line ends."""
    return str(page.evaluate("() => navigator.clipboard.readText()")).replace("\r\n", "\n")


@pytest.mark.timeout(180)
def test_select_all_copy_reads_the_live_ranges() -> None:
    """In the parent document and in the sandboxed preview iframe, Copy after
    Select All puts on the clipboard what the live Range covers now, as the
    selection renders it: the insertions inside, not the addition past its
    end."""
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

                # The parent document: its copy goes through PanelCopy.
                # The copy is the selection's own text, as the native
                # Copy (Ctrl+C) gives it: it ends with the line break of
                # the block it ends after.
                assert _select_edit_copy(page, page) == "One more\n\nSEL-in-ECTED\n"
                assert _clipboard(page) == "One more\n\nSEL-in-ECTED\n"

                # The preview iframe: the serialized bundle, its own
                # execCommand copy (the async clipboard refuses there).
                page.evaluate("() => navigator.clipboard.writeText('cleared')")
                selected = _select_edit_copy(page, page.frame_locator("#preview"))
                assert selected == "One more\n\nSEL-in-ECTED\n"
                assert _clipboard(page) == "One more\n\nSEL-in-ECTED\n"
                assert errors == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    assert not thread.is_alive(), "fixture HTTP server failed to stop"

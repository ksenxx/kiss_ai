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
from collections.abc import Callable, Iterator
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
<div id="ce" contenteditable="true"></div>
<pre id="pre">    indented()  </pre>
<div id="view"><input id="url" type="text" value="https://original.test/"></div>
<textarea id="composer">NEW CHAT DRAFT</textarea>
<div id="cases"></div>
<script>/*MENU_JS*/</script>
<script>
window.__inputs = [];
window.__value = el => (el.value === undefined ? el.textContent : el.value);
for (const el of document.querySelectorAll('input, [contenteditable]')) {
  el.addEventListener('input', () => window.__inputs.push(el.id + '=' + window.__value(el)));
}
window.__menu = ContentContextMenu.installContentContextMenu(document, {
  scope: target => target.closest('main, pre, .case'),
});
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
    return _value(page, selector)


def _value(page: Page, selector: str) -> str:
    """The value of the field *selector* (the text of a contenteditable)."""
    return str(page.evaluate(f"() => window.__value(document.querySelector('{selector}'))"))


def _clipboard_after(page: Page, action: Callable[[], None]) -> str:
    """Run *action* with a sentinel on the clipboard and return what replaced it.

    Line breaks come back as ``\\n``: the Windows clipboard holds text
    with ``\\r\\n`` (Chromium converts on the way in, for a ``writeText``
    and a Ctrl+C alike, and reads it back as is), and what the tests
    compare is which text was copied, not the platform's line ending.
    """
    page.evaluate("() => navigator.clipboard.writeText('<before>')")
    action()
    page.wait_for_function(
        "async () => (await navigator.clipboard.readText()) !== '<before>'", timeout=10000
    )
    return str(page.evaluate("() => navigator.clipboard.readText()")).replace("\r\n", "\n")


def _copy(page: Page, selector: str, position: Position | None = None) -> str:
    """Right-click *selector* (at *position* in it), menu-Copy, return the clipboard."""

    def copy() -> None:
        page.click(selector, button="right", position=position)
        item = page.locator('#sorcar-content-context-menu .sorcar-ctx-item[data-action="copy"]')
        assert item.get_attribute("aria-disabled") is None, "Copy is disabled"
        item.click()

    return _clipboard_after(page, copy)


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
def test_paste_is_undone_by_the_native_undo() -> None:
    """A menu Paste is one step of the field's own undo history, in a text
    input and in a contenteditable alike: Ctrl+Z takes the paste back and
    leaves what was typed before it (it used to write the value directly,
    which the input could not undo and the contenteditable undid past)."""
    with _page() as page:
        for selector, before in (("#txt", "ab"), ("#ce", "")):
            assert _value(page, selector) == before
            page.click(selector)
            page.keyboard.press("ControlOrMeta+a")
            page.keyboard.type("A")
            assert _value(page, selector) == "A"
            assert _paste(page, selector, "B") == "AB"
            page.keyboard.press("ControlOrMeta+z")
            assert _value(page, selector) == "A", selector
            page.keyboard.press("ControlOrMeta+z")
            assert _value(page, selector) == before, selector


@pytest.mark.timeout(180)
def test_copy_puts_the_rendered_selection_on_the_clipboard() -> None:
    """Copy after a native Select All copies the selection as rendered, the
    hidden elements and the inline script source left out like the native
    Copy does; the menu's own Select All leaves them out the same way, and
    so does text the page hides between Select All and Copy.  The menu's
    Select All is one selection over the visible text of the scope the
    page names (``<main>`` here), the one the native Copy (Ctrl+C) reads
    too: the inline hidden elements between VISIBLE_A and VISIBLE_B do not
    split it into two, nor break the line."""
    with _page() as page:
        page.click("#a")
        page.keyboard.press("ControlOrMeta+a")
        shown = str(page.evaluate("() => getSelection().toString()"))
        assert "VISIBLE_A" in shown and "VISIBLE_B" in shown and "HIDDEN" not in shown
        assert _copy(page, "#a") == shown
        _menu(page, "#a", "select-all")
        assert page.evaluate("() => getSelection().toString()") == "VISIBLE_AVISIBLE_B"
        assert _copy(page, "#a") == "VISIBLE_AVISIBLE_B"
        _menu(page, "#a", "select-all")
        native = _clipboard_after(page, lambda: page.keyboard.press("ControlOrMeta+c"))
        assert native == "VISIBLE_AVISIBLE_B"
        # The second stretch (one text node, between two scripts) is
        # hidden after Select All: Copy leaves it out, as the native Copy
        # would.
        _menu(page, "#a", "select-all")
        page.evaluate("() => { document.getElementById('b').style.display = 'none'; }")
        assert _copy(page, "#a") == "VISIBLE_A"


@pytest.mark.timeout(180)
def test_paste_asked_for_in_a_hidden_field_does_not_land_in_the_focused_control() -> None:
    """A paste asked for in one content tab stays out of another (W1 F1):
    while the clipboard read is pending the user switches tabs, which
    hides the field's view (``display:none``, still mounted) and focuses a
    chat composer.  ``focus()`` on the hidden field fails silently, so the
    ``insertText`` command used to write the clipboard text into the
    composer.  The paste is dropped instead, the composer and the focus
    left alone; one whose field is shown and focused again lands.

    The clipboard is the real one.  The tab switch is a one-shot
    ``click`` listener on the Paste item, added after the menu's own
    (the item's first listener), so it runs in the same click dispatch,
    once the menu has started the real ``readText`` and before that
    read can settle: the browser answers it from a later task.  The
    test's own ``readText`` afterwards is answered behind the menu's,
    so once it resolves the menu's read has settled and run on."""
    with _page() as page:
        page.evaluate("t => navigator.clipboard.writeText(t)", "CLIPBOARD FOR OLD BROWSER TAB")
        page.evaluate("() => { window.__inputs.length = 0; }")
        page.click("#url", button="right")
        item = '#sorcar-content-context-menu .sorcar-ctx-item[data-action="paste"]'
        page.evaluate(
            """item => document.querySelector(item).addEventListener('click', () => {
              document.getElementById('view').style.display = 'none';
              const composer = document.getElementById('composer');
              composer.focus();
              composer.setSelectionRange(composer.value.length, composer.value.length);
            }, {once: true})""",
            item,
        )
        page.click(item)
        assert page.evaluate("() => navigator.clipboard.readText()") == (
            "CLIPBOARD FOR OLD BROWSER TAB"
        )
        page.wait_for_timeout(100)
        assert _value(page, "#composer") == "NEW CHAT DRAFT"
        assert _value(page, "#url") == "https://original.test/"
        assert page.evaluate("document.activeElement.id") == "composer"
        assert page.evaluate("window.__inputs") == []
        # Shown again and focused by the right-click: the paste lands.
        page.evaluate("() => { document.getElementById('view').style.display = ''; }")
        pasted = _paste(page, "#url", "X")
        assert pasted.replace("X", "", 1) == "https://original.test/" and "X" in pasted
        assert page.evaluate("window.__inputs") == ["url=" + pasted]


# The real clipboard, each read answered only when the test calls
# ``window.__release()``: the wait of a browser that asks the user first.
# No real read lasts across the test's input: headless Chromium refuses an
# ungranted one at once and answers a granted one before the next key.
_HOLD_READS = """() => {
  const real = navigator.clipboard;
  Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {
    readText: () => new Promise(resolve => {
      window.__release = async () => resolve(await real.readText());
    }),
  }});
}"""


@pytest.mark.timeout(180)
def test_paste_answered_after_the_native_selection_moved_does_not_land() -> None:
    """A paste the clipboard answers late stays out of a ``number`` /
    ``email`` input whose selection moved meanwhile (B/W1-F1).  Those
    inputs do not tell their selection (the offsets are null), so the text
    landed at the new caret: a key or a pointer pressed since counts as a
    move, and so does other selected text.  A paste chosen with the
    keyboard is not undone by that very key and lands at the caret.  An
    input in a contenteditable host is the field it is: the same there,
    and no Paste on a read-only one (the host made it pass for writable)."""
    moves: dict[str, Callable[[Page, str], object]] = {
        "key": lambda page, field: page.keyboard.press("Home"),
        "pointer": lambda page, field: page.click(field, position={"x": 4, "y": 8}),
        "script": lambda page, field: page.locator(field).evaluate("el => el.select()"),
    }
    nested = '<input id="nested" type="number" value="12"><input id="ro" value="abc" readonly>'
    with _page() as page:
        page.evaluate("() => navigator.clipboard.writeText('7')")
        page.evaluate(_HOLD_READS)
        page.evaluate("html => { document.getElementById('ce').innerHTML = html; }", nested)
        for field, value in (("#num", "12"), ("#mail", "a@b"), ("#nested", "12")):
            for name, move in moves.items():
                page.click(field)
                page.keyboard.press("End")
                _menu(page, field, "paste")
                move(page, field)
                page.evaluate("() => window.__release()")
                assert _value(page, field) == value, (field, name)
            page.click(field)
            page.keyboard.press("End")
            page.click(field, button="right")
            page.keyboard.press("Enter")
            page.evaluate("() => window.__release()")
            assert _value(page, field) == value + "7", field
        page.click("#ro", button="right")
        rows = page.locator("#sorcar-content-context-menu .sorcar-ctx-item")
        assert rows.evaluate_all("rows => rows.map(row => row.dataset.action)") == [
            "copy",
            "select-all",
        ]


@pytest.mark.timeout(180)
def test_select_all_then_copy_puts_the_native_selection_text_on_the_clipboard() -> None:
    """The menu's Select All then Copy copies the selection as Chromium
    renders it (``getSelection().toString()``, what Ctrl+C copies): the
    content of a closed ``<details>``, a ``user-select:none`` span, a
    textarea's text and a select's options are not in the selection, a
    text transform shows, and a ``<pre>`` next to a block is one line
    break apart.  Copy used to replace that text with its own reading of
    the selected nodes' source (W1 F2), which had all of them wrong."""
    cases = {
        "closed details": (
            "<p>Alpha</p><details><summary>Label</summary><p>SECRET</p></details><p>Omega</p>",
            "SECRET",
        ),
        "user-select none": (
            '<p>Alpha <span style="user-select:none">UNSELECTABLE</span> Omega</p>',
            "UNSELECTABLE",
        ),
        "text transform": ('<p style="text-transform:uppercase">Mixed case</p>', "Mixed"),
        "textarea": ("<p>Alpha</p><textarea>INITIAL</textarea><p>Omega</p>", "INITIAL"),
        "select": (
            "<p>Alpha</p><select><option>Chosen</option><option>UNCHOSEN</option></select>"
            "<p>Omega</p>",
            "UNCHOSEN",
        ),
        "pre then block": ("<pre>Alpha\n</pre><div>Beta</div>", "Alpha\n\nBeta"),
    }
    with _page() as page:
        for name, (html, absent) in cases.items():
            page.evaluate(
                "h => { document.getElementById('cases').innerHTML = "
                "'<div class=\"case\"><p>Lead</p>' + h + '</div>'; }",
                html,
            )
            _menu(page, ".case > p", "select-all")
            native = str(page.evaluate("() => getSelection().toString()"))
            assert native.startswith("Lead") and absent not in native, (name, native)
            assert _copy(page, ".case > p") == native, name


@pytest.mark.timeout(180)
def test_copy_keeps_the_selected_whitespace() -> None:
    """Copy puts the selection on the clipboard as rendered, its leading
    and trailing whitespace included: the indentation of a selected code
    line is part of it (it used to be trimmed off)."""
    with _page() as page:
        page.evaluate("() => getSelection().selectAllChildren(document.getElementById('pre'))")
        shown = str(page.evaluate("() => getSelection().toString()"))
        assert shown == "    indented()  "
        assert _copy(page, "#pre") == shown
        _menu(page, "#pre", "select-all")
        assert page.evaluate("() => getSelection().toString()") == shown
        assert _copy(page, "#pre") == shown

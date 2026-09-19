# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   test parameters of the same name)
"""End-to-end tests: the Edit source toggle of .md / .html content tabs.

A ``.md`` or ``.html`` file opened in the remote webapp renders as its
PREVIEW (converted markdown / the page itself) in a sandboxed iframe,
like VS Code's markdown preview.  When the daemon reported a version
stamp (the file is strictly UTF-8, so it can be saved back), the Save
bar above the preview carries an "Edit source" toggle: it flips to the
same editable Monaco surface every other text file gets, and back —
the preview re-renders from the edited text, unsaved edits and the
dirty dot survive the round trip, and Save / Ctrl+S writes the SOURCE
(never the rendered HTML) to disk.  A finished-task report, a file
that is not valid UTF-8, and a directory named like ``foo.md`` never
get the toggle.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` and the real
``cdn.jsdelivr.net`` Monaco bundle — no mocks.  When the CDN is
unreachable the editor cannot exist, so the editing tests skip (one
test blocks the CDN on purpose to pin the fallback behavior).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.tests.agents.vscode.test_content_tab_editing import (
    _dismiss_toasts,
    _wait_for_disk,
)
from kiss.tests.agents.vscode.test_content_tab_file_links import (
    _inject_file_link,
    _open_page,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)

_MD_SOURCE = "# Alpha Title\n\nhello *world* from markdown\n"
_HTML_SOURCE = (
    "<!DOCTYPE html><html><head><title>t</title></head>"
    "<body><h1>Page Heading</h1><p>body text</p></body></html>\n"
)
_VIEW = "#content-tab-area .content-tab-view:not([style*='display: none']) "
_FRAME = _VIEW + ".content-html-frame"
_MONACO = _VIEW + ".monaco-editor"
_FALLBACK = _VIEW + ".content-code-fallback"
_SAVE_BAR = _VIEW + ".content-save-bar"
_SAVE_BTN = _VIEW + ".content-save-btn"
_MODE_BTN = _VIEW + ".content-mode-btn"
_STATUS = _VIEW + ".content-save-status"
_PREVIEW = _VIEW + ".content-preview-holder"
_SOURCE_HOLDER = _VIEW + ".content-monaco-holder"
_DIRTY_TAB = ".chat-tab.content-tab.content-dirty"


def _fresh_file(harness, name: str, text: str) -> Path:
    path = Path(harness.work_dir) / name
    # Byte-exact fixture: the daemon saves the editor text without
    # newline translation, and the assertions count LF bytes, so the
    # file must not pick up CRLF from Windows text-mode writes.
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _open_preview(page, path: str, link_id: str) -> None:
    """Click a link to *path* and wait for its preview iframe."""
    _inject_file_link(page, path, link_id)
    page.click("#" + link_id)
    page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
    page.wait_for_selector(_FRAME, timeout=30000)


def _enter_source_mode(page) -> None:
    """Click Edit source and wait for the Monaco editor (skip when the
    CDN fallback rendered instead: no editor, nothing to edit)."""
    _dismiss_toasts(page)
    page.click(_MODE_BTN)
    page.wait_for_selector(_MONACO + ", " + _FALLBACK, timeout=30000)
    if page.locator(_FALLBACK).count() > 0:
        pytest.skip("Monaco CDN unreachable: no editor to test")


def _type_at_end(page, text: str) -> None:
    page.click(_MONACO + " .view-lines")
    page.keyboard.press("Control+End")
    page.keyboard.type(text)


def _wait_editor_contains(page, needle: str) -> None:
    page.wait_for_function(
        """(needle) => {
             const el = document.querySelector('#content-tab-area');
             return el &&
               el.innerText.replace(/\\u00a0/g, ' ').includes(needle);
           }""",
        arg=needle,
        timeout=30000,
    )


class TestMarkdownHtmlEditSource:
    """Browser E2E: preview <-> Edit source toggle with a save path."""

    def test_markdown_opens_as_preview_with_edit_toggle(
        self, browser, harness,
    ) -> None:
        """A .md file renders CONVERTED (an <h1>, not '# ' text) with
        the Save bar + Edit source toggle above; the editor surface is
        not mounted until the toggle is used."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "toggle_view.md", _MD_SOURCE)
            _open_preview(page, str(path), "lnk-p1")
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="Alpha Title").wait_for(timeout=30000)
            assert page.locator(_MODE_BTN).inner_text() == "Edit source"
            assert page.locator(_SAVE_BTN).is_disabled()
            path_el = page.locator(_VIEW + ".content-save-path")
            assert path_el.get_attribute("title") == str(path)
            # The Monaco surface exists but is empty and hidden until
            # the first Edit source click.
            assert page.locator(_SOURCE_HOLDER).count() == 1
            assert not page.locator(_SOURCE_HOLDER).is_visible()
            assert page.locator(_MONACO).count() == 0
            assert page.locator(_DIRTY_TAB).count() == 0
        finally:
            context.close()

    def test_edit_source_saves_markdown_source_to_disk(
        self, browser, harness,
    ) -> None:
        """Edit source shows the RAW markdown; typing marks the tab
        dirty and Ctrl+S writes the source (not rendered HTML) back."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "toggle_save.md", _MD_SOURCE)
            _open_preview(page, str(path), "lnk-p2")
            _enter_source_mode(page)
            _wait_editor_contains(page, "# Alpha Title")
            assert page.locator(_MODE_BTN).inner_text() == "Preview"
            assert not page.locator(_PREVIEW).is_visible()
            _type_at_end(page, "\nappended_line")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            assert page.locator(_SAVE_BTN).is_enabled()
            page.keyboard.press("Control+s")
            _wait_for_disk(path, "appended_line")
            assert path.read_text() == _MD_SOURCE + "\nappended_line"
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            assert page.locator(_DIRTY_TAB).count() == 0
            saves = [f for f in sent if f.get("type") == "saveFile"]
            assert len(saves) == 1
            assert saves[0]["path"] == str(path)
            assert saves[0]["content"] == _MD_SOURCE + "\nappended_line"
            assert saves[0]["version"].endswith(f":{len(_MD_SOURCE)}")
        finally:
            context.close()

    def test_preview_shows_unsaved_edits_and_saves_from_preview(
        self, browser, harness,
    ) -> None:
        """Toggling back to Preview re-renders it from the EDITED text;
        the dirty dot survives and the Save button works from there."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "toggle_dirty.md", _MD_SOURCE)
            _open_preview(page, str(path), "lnk-p3")
            _enter_source_mode(page)
            _wait_editor_contains(page, "# Alpha Title")
            _type_at_end(page, "\n## Added Section\n")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            _dismiss_toasts(page)
            page.click(_MODE_BTN)  # back to preview
            frame = page.frame_locator(_FRAME)
            frame.locator("h2", has_text="Added Section").wait_for(
                timeout=30000,
            )
            # The edits are on screen but NOT on disk, and the tab is
            # still dirty; the (hidden) editor keeps the text alive.
            assert path.read_text() == _MD_SOURCE
            assert page.locator(_DIRTY_TAB).count() == 1
            assert page.locator(_MODE_BTN).inner_text() == "Edit source"
            _dismiss_toasts(page)
            page.click(_SAVE_BTN)
            _wait_for_disk(path, "## Added Section")
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            assert page.locator(_DIRTY_TAB).count() == 0
            # Toggling forth once more shows the same edited source
            # (the editor was hidden, never rebuilt from disk).
            _dismiss_toasts(page)
            page.click(_MODE_BTN)
            _wait_editor_contains(page, "## Added Section")
        finally:
            context.close()

    def test_html_file_edit_source_and_save(self, browser, harness) -> None:
        """An .html file previews as the page itself; Edit source shows
        the raw markup and saves it byte-for-byte."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "toggle_page.html", _HTML_SOURCE)
            _open_preview(page, str(path), "lnk-p4")
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="Page Heading").wait_for(
                timeout=30000,
            )
            _enter_source_mode(page)
            _wait_editor_contains(page, "<h1>Page Heading</h1>")
            _type_at_end(page, "<!-- edited -->")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            page.keyboard.press("Control+s")
            _wait_for_disk(path, "<!-- edited -->")
            assert path.read_text() == _HTML_SOURCE + "<!-- edited -->"
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            # Preview now shows the saved page again (rebuilt from the
            # editor's text, which equals the disk).
            _dismiss_toasts(page)
            page.click(_MODE_BTN)
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="Page Heading").wait_for(
                timeout=30000,
            )
        finally:
            context.close()

    def test_conflict_reload_returns_to_source_mode(
        self, browser, harness,
    ) -> None:
        """A save conflict's "Reload from disk" re-renders the tab —
        landing back in SOURCE mode with the disk's text, clean."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "toggle_conflict.md", _MD_SOURCE)
            _open_preview(page, str(path), "lnk-p5")
            _enter_source_mode(page)
            _wait_editor_contains(page, "# Alpha Title")
            path.write_text("# Rewritten By Agent\n")
            _type_at_end(page, "\nmine")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            page.keyboard.press("Control+s")
            page.wait_for_selector(
                ".kiss-notification-action:has-text('Reload from disk')",
                timeout=10000,
            )
            assert path.read_text() == "# Rewritten By Agent\n"
            page.click(
                ".kiss-notification-action:has-text('Reload from disk')",
            )
            _wait_editor_contains(page, "# Rewritten By Agent")
            assert page.locator(_MODE_BTN).inner_text() == "Preview"
            assert page.locator(_SOURCE_HOLDER).is_visible()
            assert not page.locator(_PREVIEW).is_visible()
            assert page.locator(_DIRTY_TAB).count() == 0
            # And the preview of the reloaded text renders on demand.
            _dismiss_toasts(page)
            page.click(_MODE_BTN)
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="Rewritten By Agent").wait_for(
                timeout=30000,
            )
        finally:
            context.close()

    # Monaco virtualizes its DOM: innerText holds only the lines near
    # the scroll position, so seeing one marker and NOT the other
    # proves the editor really jumped (same technique as the code
    # line-jump tests in test_content_tab_file_links.py).
    _JUMPED_JS = """([seen, gone]) => {
         const area = document.getElementById('content-tab-area');
         if (!area) return false;
         const text = area.innerText.replace(/\\u00a0/g, ' ');
         return text.includes(seen) && !text.includes(gone);
       }"""

    def test_md_line_link_reveals_in_source_editor(
        self, browser, harness,
    ) -> None:
        """A ``notes.md:250`` link never scrolls the PREVIEW, but the
        line reveal stays pending and fires when Edit source first
        shows the editor; a later ``:5`` link on the (now dirty) tab
        jumps the visible editor too, keeping the edits."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = Path(harness.work_dir) / "long_notes.md"
            source = "".join(f"marker-{n:03d} line\n" for n in range(1, 301))
            path.write_text(source)
            _open_preview(page, str(path) + ":250", "lnk-ln1")
            opens = [f for f in sent if f.get("type") == "openFile"]
            assert opens[-1]["line"] == 250
            assert opens[-1]["path"] == str(path)
            _enter_source_mode(page)
            page.wait_for_function(
                self._JUMPED_JS, arg=["marker-250", "marker-001"],
                timeout=30000,
            )
            _type_at_end(page, " EDIT")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            # Back to the chat, then a NEW :5 link to the dirty file:
            # the tab comes forward in source mode, jumped to line 5,
            # with the unsaved edit intact.
            page.click(".chat-tab:not(.content-tab) .chat-tab-label")
            page.wait_for_selector("#task-input", state="visible")
            _inject_file_link(page, str(path) + ":5", "lnk-ln2")
            page.click("#lnk-ln2")
            page.wait_for_function(
                self._JUMPED_JS, arg=["marker-005", "marker-250"],
                timeout=30000,
            )
            assert page.locator(".chat-tab.content-tab").count() == 1
            assert page.locator(_DIRTY_TAB).count() == 1
            assert path.read_text() == source
        finally:
            context.close()

    def test_ctrl_s_inside_preview_iframe_saves(
        self, browser, harness,
    ) -> None:
        """Ctrl+S pressed while focus sits INSIDE the preview iframe
        (a separate sandboxed document the parent's key listener never
        sees) still saves: the bridge shipped into an editable preview
        forwards the shortcut to the parent."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "toggle_framesave.md", _MD_SOURCE)
            _open_preview(page, str(path), "lnk-fs")
            _enter_source_mode(page)
            _wait_editor_contains(page, "# Alpha Title")
            _type_at_end(page, "\nframe_saved_line")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            _dismiss_toasts(page)
            page.click(_MODE_BTN)  # back to preview
            frame = page.frame_locator(_FRAME)
            frame.locator("body").wait_for(timeout=30000)
            frame.locator("body").click()  # focus moves into the iframe
            page.keyboard.press("Control+s")
            _wait_for_disk(path, "frame_saved_line")
            assert path.read_text() == _MD_SOURCE + "\nframe_saved_line"
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            assert page.locator(_DIRTY_TAB).count() == 0
            saves = [f for f in sent if f.get("type") == "saveFile"]
            assert len(saves) == 1
            assert saves[0]["content"] == _MD_SOURCE + "\nframe_saved_line"
            assert saves[0]["force"] is False
        finally:
            context.close()

    _HOSTILE_HTML = (
        "<!DOCTYPE html><html><head><script>\n"
        "// Poison every configurable KeyboardEvent accessor so that\n"
        "// ANY real keypress reads as a plain Ctrl+S; the bridge must\n"
        "// see through this via its captured native getters.\n"
        "try {\n"
        "  var KP = KeyboardEvent.prototype;\n"
        "  Object.defineProperty(KP, 'ctrlKey', {get: () => true});\n"
        "  Object.defineProperty(KP, 'metaKey', {get: () => false});\n"
        "  Object.defineProperty(KP, 'altKey', {get: () => false});\n"
        "  Object.defineProperty(KP, 'shiftKey', {get: () => false});\n"
        "  Object.defineProperty(KP, 'key', {get: () => 's'});\n"
        "} catch (_e) {}\n"
        "window.addEventListener('message', function (e) {\n"
        "  // steal the save port if it is ever visible to page code\n"
        "  if (e.ports && e.ports[0]) {\n"
        "    try { e.ports[0].postMessage('save'); } catch (_e) {}\n"
        "  }\n"
        "});\n"
        "</script></head><body><h1>Hostile Page</h1><script>\n"
        "setInterval(function () {\n"
        "  try { parent.postMessage({kissPreviewSaveKey: true}, '*'); }\n"
        "  catch (_e) {}\n"
        "  try { parent.postMessage({kissPreviewSavePort: true}, '*'); }\n"
        "  catch (_e) {}\n"
        "  try {\n"
        "    window.dispatchEvent(new MessageEvent('message',\n"
        "      {data: {kissPreviewSavePort: true}, source: window.parent,\n"
        "       ports: (function () {\n"
        "         try { return [new MessageChannel().port2]; }\n"
        "         catch (_e) { return []; }\n"
        "       })()}));\n"
        "  } catch (_e) {}\n"
        "  try {\n"
        "    document.dispatchEvent(new KeyboardEvent('keydown',\n"
        "      {key: 's', ctrlKey: true, bubbles: true}));\n"
        "  } catch (_e) {}\n"
        "}, 50);\n"
        "</script></body></html>\n"
    )

    def test_hostile_preview_page_cannot_forge_a_save(
        self, browser, harness,
    ) -> None:
        """Scripts of the PREVIEWED page run inside the same sandboxed
        iframe as the Ctrl+S bridge, so they try everything: posting
        the retired kissPreviewSaveKey shape, spoofing the port
        delivery (synthetic events are not trusted), synthesizing
        Ctrl+S keydowns (also not trusted), eavesdropping for the
        port (the bridge hides its delivery event), and poisoning the
        KeyboardEvent.prototype getters so a real plain keypress reads
        as Ctrl+S (the bridge uses captured native getters). None of
        it may save; a REAL Ctrl+S inside the same hostile preview
        still must."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "hostile.html", self._HOSTILE_HTML)
            _open_preview(page, str(path), "lnk-hp")
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="Hostile Page").wait_for(
                timeout=30000,
            )
            _enter_source_mode(page)
            _wait_editor_contains(page, "Hostile Page")
            _type_at_end(page, "<!-- victim edit -->")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            _dismiss_toasts(page)
            page.click(_MODE_BTN)  # preview: the hostile scripts run
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="Hostile Page").wait_for(
                timeout=30000,
            )
            # ~24 forgery rounds fire during this window.
            page.wait_for_timeout(1200)
            assert not any(f.get("type") == "saveFile" for f in sent)
            assert page.locator(_DIRTY_TAB).count() == 1
            assert path.read_text() == self._HOSTILE_HTML
            # A REAL but plain keypress must not save either, even
            # though the page poisoned every KeyboardEvent.prototype
            # accessor to make it read as Ctrl+S: the bridge consults
            # the native getters it captured before page scripts ran.
            frame.locator("body").click()
            page.keyboard.press("x")
            page.wait_for_timeout(600)
            assert not any(f.get("type") == "saveFile" for f in sent)
            assert page.locator(_DIRTY_TAB).count() == 1
            assert path.read_text() == self._HOSTILE_HTML
            # The genuine shortcut still works from inside the very
            # same hostile document.
            page.keyboard.press("Control+s")
            _wait_for_disk(path, "victim edit")
            assert (
                path.read_text() == self._HOSTILE_HTML + "<!-- victim edit -->"
            )
            # _wait_for_disk never pumps Playwright's event loop; this
            # wait does, flushing the framesent callback into `sent`.
            page.wait_for_function(
                "() => document.querySelectorAll('.chat-tab.content-tab"
                ".content-dirty').length === 0",
                timeout=10000,
            )
            saves = [f for f in sent if f.get("type") == "saveFile"]
            assert len(saves) == 1
            assert saves[0]["force"] is False
        finally:
            context.close()

    def test_report_tab_gets_no_toggle(self, browser, harness) -> None:
        """A finished-task report's content is CONVERTED HTML, so it
        must never be editable — even if a (buggy or malicious) event
        carried a version stamp, isReport wins.  The synthetic event is
        injected through the same window `message` dispatch the
        WebSocket shim uses for real frames."""
        context, page, sent = _open_page(browser, harness)
        try:
            page.evaluate(
                """() => window.dispatchEvent(new MessageEvent('message',
                     {data: {type: 'fileContent',
                             path: '/tmp/reports/r.md', name: 'r.md',
                             content: '<h1>Report Body</h1>',
                             isReport: true, version: '1:1'}}))""",
            )
            page.wait_for_selector(_FRAME, timeout=30000)
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="Report Body").wait_for(
                timeout=30000,
            )
            assert page.locator(_MODE_BTN).count() == 0
            assert page.locator(_SAVE_BAR).count() == 0
            page.keyboard.press("Control+s")
            page.wait_for_timeout(300)
            assert not any(f.get("type") == "saveFile" for f in sent)
        finally:
            context.close()

    def test_invalid_utf8_markdown_stays_readonly_preview(
        self, browser, harness,
    ) -> None:
        """A .md file that is not valid UTF-8 previews with U+FFFD but
        gets no Save bar and no toggle: the daemon sent no version
        stamp, and saving replacement characters back would corrupt the
        bytes the decoder could not represent."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = Path(harness.work_dir) / "latin1_notes.md"
            path.write_bytes(b"# caf\xe9 notes\n")
            _open_preview(page, str(path), "lnk-p6")
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="caf").wait_for(timeout=30000)
            assert page.locator(_SAVE_BAR).count() == 0
            assert page.locator(_MODE_BTN).count() == 0
            assert page.locator(_PREVIEW).count() == 0
            assert path.read_bytes() == b"# caf\xe9 notes\n"
        finally:
            context.close()

    def test_directory_named_like_markdown_stays_plain(
        self, browser, harness,
    ) -> None:
        """A DIRECTORY named notes.md opens as a plain-text listing:
        no preview iframe, no Save bar, no toggle."""
        context, page, sent = _open_page(browser, harness)
        try:
            folder = Path(harness.work_dir) / "listing_dir.md"
            folder.mkdir(exist_ok=True)
            (folder / "inner.txt").write_text("inner\n")
            _inject_file_link(page, str(folder), "lnk-p7")
            page.click("#lnk-p7")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            page.wait_for_selector(
                _MONACO + ", " + _FALLBACK, timeout=30000,
            )
            _wait_editor_contains(page, "inner.txt")
            assert page.locator(_FRAME).count() == 0
            assert page.locator(_SAVE_BAR).count() == 0
            assert page.locator(_MODE_BTN).count() == 0
        finally:
            context.close()

    def test_cdn_fallback_keeps_toggle_but_cannot_save(
        self, browser, harness,
    ) -> None:
        """With the Monaco CDN unreachable, Edit source falls back to
        the read-only <pre> viewer.  The bar STAYS (unlike a plain code
        tab, the toggle back to Preview lives there), the Save button
        stays disabled, and both directions keep working."""
        context = browser.new_context(ignore_https_errors=True)
        context.route(
            "https://cdn.jsdelivr.net/**", lambda route: route.abort(),
        )
        page = context.new_page()
        sent: list[dict] = []
        page.on(
            "websocket",
            lambda ws: ws.on(
                "framesent",
                lambda payload: sent.append(__import__("json").loads(payload)),
            ),
        )
        try:
            page.goto(harness.base_url + "/")
            page.wait_for_selector(
                "#task-input", state="visible", timeout=30000,
            )
            page.wait_for_selector(".chat-tab", timeout=30000)
            path = _fresh_file(harness, "toggle_fb.md", _MD_SOURCE)
            _open_preview(page, str(path), "lnk-p8")
            _dismiss_toasts(page)
            page.click(_MODE_BTN)
            page.wait_for_selector(_FALLBACK, timeout=30000)
            _wait_editor_contains(page, "# Alpha Title")
            assert page.locator(_SAVE_BAR).count() == 1
            assert page.locator(_SAVE_BTN).is_disabled()
            page.keyboard.press("Control+s")
            page.wait_for_timeout(300)
            assert not any(f.get("type") == "saveFile" for f in sent)
            _dismiss_toasts(page)
            page.click(_MODE_BTN)  # back to preview
            frame = page.frame_locator(_FRAME)
            frame.locator("h1", has_text="Alpha Title").wait_for(
                timeout=30000,
            )
            assert path.read_text() == _MD_SOURCE
        finally:
            context.close()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   test parameters of the same name)
"""End-to-end tests: editing and saving a file in a remote-webapp content tab.

A file opened from a chat link renders in a full (editable) Monaco
editor.  Typing marks the tab dirty (a ``●`` in the tab strip, the
Save button enabled), Ctrl/Cmd+S or the Save button sends ``saveFile``
over the WebSocket, :meth:`RemoteAccessServer._handle_save_file`
rewrites the file and replies ``fileSaved``, and the tab settles back
to clean.  A file that changed on disk in the meantime is not
overwritten silently: the daemon answers ``conflict`` and the client
offers Overwrite / Reload.  Directory listings stay read-only.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` and the real
``cdn.jsdelivr.net`` Monaco bundle — no mocks.  When the CDN is
unreachable the editor cannot exist, so those tests skip.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from kiss.tests.agents.vscode.test_content_tab_file_links import (
    _inject_file_link,
    _open_page,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)

_SOURCE = "alpha = 1\nbeta = 2\n"
_EDITOR = "#content-tab-area .content-tab-view:not([style*='display: none']) "
_MONACO = _EDITOR + ".monaco-editor"
_SAVE_BTN = _EDITOR + ".content-save-btn"
_STATUS = _EDITOR + ".content-save-status"
_DIRTY_TAB = ".chat-tab.content-tab.content-dirty"


def _fresh_file(harness, name: str, text: str = _SOURCE) -> Path:
    path = Path(harness.work_dir) / name
    # Byte-exact fixture: the daemon saves the editor text without
    # newline translation, and the assertions count LF bytes, so the
    # file must not pick up CRLF from Windows text-mode writes.
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _open_editor(page, path: str, link_id: str) -> None:
    """Click a link to *path* and wait for its Monaco editor (skip when
    the CDN fallback rendered instead: no editor, nothing to edit)."""
    _inject_file_link(page, path, link_id)
    page.click("#" + link_id)
    page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
    page.wait_for_selector(
        _MONACO + ", " + _EDITOR + ".content-code-fallback", timeout=30000,
    )
    if page.locator(_EDITOR + ".content-code-fallback").count() > 0:
        pytest.skip("Monaco CDN unreachable: no editor to test")
    # The editor is created asynchronously; wait for its text.
    page.wait_for_function(
        """() => {
             const el = document.querySelector('#content-tab-area');
             return el && el.innerText.replace(/\\u00a0/g, ' ').includes('alpha');
           }""",
        timeout=30000,
    )


def _type_at_end(page, text: str) -> None:
    page.click(_MONACO + " .view-lines")
    # Monaco binds "go to end of document" per platform: Ctrl+End on
    # Linux/Windows, Cmd+Down on macOS (Ctrl+End is unbound there, so
    # the text would land wherever the click put the cursor).
    page.keyboard.press("Meta+ArrowDown" if sys.platform == "darwin" else "Control+End")
    page.keyboard.type(text)


def _editor_text(page) -> str:
    return str(page.evaluate(
        "() => document.querySelector('#content-tab-area')"
        ".innerText.replace(/\\u00a0/g, ' ')",
    ))


def _dismiss_toasts(page) -> None:
    """Close every notification toast: the container sits over the top
    of the content area where the Save bar lives, and Playwright refuses
    to click through it (a real user would dismiss it the same way)."""
    for _ in range(10):
        closes = page.locator(".kiss-notification-close")
        if closes.count() == 0:
            return
        closes.first.click()
        page.wait_for_timeout(50)


def _click_save(page) -> None:
    _dismiss_toasts(page)
    page.click(_SAVE_BTN)


def _wait_for_disk(path: Path, needle: str, timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if needle in path.read_text():
                return
        except PermissionError:
            # Windows refuses to open a file for the instant the daemon's
            # atomic ``Path.replace`` swaps it in; poll again.
            pass
        time.sleep(0.05)
    raise AssertionError(f"{needle!r} never reached {path}")


class TestContentTabEditing:
    """Browser E2E: the content tab is a real editor with a save path."""

    def test_typing_marks_dirty_and_ctrl_s_saves_to_disk(
        self, browser, harness,
    ) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "edit_ctrl_s.py")
            _open_editor(page, str(path), "lnk-e1")
            assert page.locator(_SAVE_BTN).is_disabled()
            assert page.locator(_DIRTY_TAB).count() == 0
            path_el = page.locator(_EDITOR + ".content-save-path")
            assert path_el.get_attribute("title") == str(path)
            assert str(path) in path_el.inner_text()
            _type_at_end(page, "gamma = 3")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            assert page.locator(".chat-tab.content-tab .chat-tab-dirty").count() == 1
            assert page.locator(_SAVE_BTN).is_enabled()
            assert page.locator(_STATUS).inner_text() == "Unsaved changes"
            assert path.read_text() == _SOURCE
            page.keyboard.press("Control+s")
            _wait_for_disk(path, "gamma = 3")
            assert path.read_text() == _SOURCE + "gamma = 3"
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            assert page.locator(_DIRTY_TAB).count() == 0
            assert page.locator(_SAVE_BTN).is_disabled()
            saves = [f for f in sent if f.get("type") == "saveFile"]
            assert len(saves) == 1
            assert saves[0]["path"] == str(path)
            assert saves[0]["content"] == _SOURCE + "gamma = 3"
            st = path.stat()
            # The stamp sent is the one the file had when it was OPENED.
            assert saves[0]["version"].endswith(f":{len(_SOURCE)}")
            assert saves[0]["version"] != f"{st.st_mtime_ns}:{st.st_size}"
            assert saves[0]["force"] is False
            # token = "<content tab id>:<save sequence>"
            tab_id = page.get_attribute(".chat-tab.content-tab", "data-tab-id")
            assert saves[0]["token"].startswith(tab_id + ":")
            # The chat tab is untouched by all of this.
            assert not any(
                f.get("type") in ("closeTab", "newChat") for f in sent
            )
        finally:
            context.close()

    def test_save_button_saves_and_updates_version_for_next_save(
        self, browser, harness,
    ) -> None:
        """The Save button (the only way on a phone) saves; a second
        edit + save works because the client took the new stamp."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "edit_button.py")
            _open_editor(page, str(path), "lnk-e2")
            _type_at_end(page, "one = 1")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            _click_save(page)
            _wait_for_disk(path, "one = 1")
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            _type_at_end(page, "\ntwo = 2")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            _click_save(page)
            _wait_for_disk(path, "two = 2")
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            saves = [f for f in sent if f.get("type") == "saveFile"]
            assert len(saves) == 2
            # The second save carries the stamp the FIRST save's reply
            # reported (the file as written by save #1), not the stale
            # open-time stamp — otherwise it would have been a conflict.
            assert saves[1]["version"] != saves[0]["version"]
            assert saves[1]["version"].endswith(f":{len(_SOURCE + 'one = 1')}")
            assert path.read_text() == _SOURCE + "one = 1\ntwo = 2"
        finally:
            context.close()

    def test_undo_back_to_saved_text_clears_dirty(
        self, browser, harness,
    ) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "edit_undo.py")
            _open_editor(page, str(path), "lnk-e3")
            _type_at_end(page, "zzz")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            # Undo is Monaco's own keybinding, and Monaco follows the
            # platform: Ctrl+Z on Linux/Windows, Cmd+Z on macOS.
            page.keyboard.press("ControlOrMeta+z")
            page.wait_for_function(
                "sel => document.querySelector(sel) === null",
                arg=_DIRTY_TAB, timeout=10000,
            )
            assert page.locator(_SAVE_BTN).is_disabled()
            assert page.locator(_STATUS).inner_text() == ""
            # Ctrl+S on a clean tab sends nothing.
            page.keyboard.press("Control+s")
            page.wait_for_timeout(500)
            assert not any(f.get("type") == "saveFile" for f in sent)
        finally:
            context.close()

    def test_directory_listing_stays_read_only(self, browser, harness) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            folder = harness.work_dir / "ro_dir"
            folder.mkdir(exist_ok=True)
            (folder / "alpha.txt").write_text("alpha\n")
            _inject_file_link(page, str(folder), "lnk-dir")
            page.click("#lnk-dir")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            page.wait_for_selector(
                _MONACO + ", " + _EDITOR + ".content-code-fallback",
                timeout=30000,
            )
            if page.locator(_EDITOR + ".content-code-fallback").count() > 0:
                pytest.skip("Monaco CDN unreachable: no editor to test")
            page.wait_for_function(
                """() => document.querySelector('#content-tab-area')
                         .innerText.includes('alpha.txt')""",
                timeout=30000,
            )
            assert page.locator(_EDITOR + ".content-save-bar").count() == 0
            before = _editor_text(page)
            _type_at_end(page, "TYPED")
            page.wait_for_timeout(300)
            assert "TYPED" not in _editor_text(page)
            assert _editor_text(page).replace("\n", "") == before.replace(
                "\n", "",
            ) or "Cannot edit" in _editor_text(page)
            assert page.locator(_DIRTY_TAB).count() == 0
            page.keyboard.press("Control+s")
            page.wait_for_timeout(300)
            assert not any(f.get("type") == "saveFile" for f in sent)
        finally:
            context.close()

    def test_reclicking_link_keeps_unsaved_edits(self, browser, harness) -> None:
        """Opening the same file again while its tab is dirty brings the
        tab forward but does NOT reload the text over the edits."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "edit_reclick.py")
            _open_editor(page, str(path), "lnk-e5")
            _type_at_end(page, "kept = True")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            # Back to the chat, then click the link again.
            page.click(".chat-tab:not(.content-tab) .chat-tab-label")
            page.wait_for_selector("#task-input", state="visible")
            page.click("#lnk-e5")
            page.wait_for_selector("#content-tab-area", state="visible")
            page.wait_for_timeout(1000)
            assert page.locator(".chat-tab.content-tab").count() == 1
            assert "kept = True" in _editor_text(page)
            assert page.locator(_DIRTY_TAB).count() == 1
            assert path.read_text() == _SOURCE
        finally:
            context.close()

    def _make_conflict(self, page, harness, name: str, link_id: str) -> Path:
        """Open *name*, change it on disk behind the editor's back, type
        an edit and press Ctrl+S; returns the path.  The conflict
        notification is left on screen for the caller."""
        path = _fresh_file(harness, name)
        _open_editor(page, str(path), link_id)
        path.write_text("rewritten_by_agent = True\n")
        _type_at_end(page, "mine = 1")
        page.wait_for_selector(_DIRTY_TAB, timeout=10000)
        page.keyboard.press("Control+s")
        page.wait_for_selector(
            ".kiss-notification-action:has-text('Overwrite')", timeout=10000,
        )
        assert page.locator(".kiss-notification-warning").count() == 1
        assert path.read_text() == "rewritten_by_agent = True\n"
        assert "changed on disk" in page.locator(_STATUS).inner_text()
        assert page.locator(_DIRTY_TAB).count() == 1
        return path

    def test_conflict_overwrite_writes_the_edits(self, browser, harness) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            path = self._make_conflict(page, harness, "conflict_ow.py", "lnk-c1")
            page.click(".kiss-notification-action:has-text('Overwrite')")
            _wait_for_disk(path, "mine = 1")
            assert path.read_text() == _SOURCE + "mine = 1"
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            assert page.locator(_DIRTY_TAB).count() == 0
            saves = [f for f in sent if f.get("type") == "saveFile"]
            assert [s["force"] for s in saves] == [False, True]
        finally:
            context.close()

    def test_conflict_reload_drops_the_edits(self, browser, harness) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            path = self._make_conflict(page, harness, "conflict_rl.py", "lnk-c2")
            page.click(".kiss-notification-action:has-text('Reload from disk')")
            page.wait_for_function(
                """() => {
                     const t = document.querySelector('#content-tab-area')
                       .innerText.replace(/\\u00a0/g, ' ');
                     return t.includes('rewritten_by_agent') && !t.includes('mine = 1');
                   }""",
                timeout=30000,
            )
            assert page.locator(_DIRTY_TAB).count() == 0
            assert page.locator(".chat-tab.content-tab").count() == 1
            assert path.read_text() == "rewritten_by_agent = True\n"
            # The reloaded tab saves normally again (fresh mtime).
            _type_at_end(page, "after_reload = 1")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            page.keyboard.press("Control+s")
            _wait_for_disk(path, "after_reload = 1")
            saves = [f for f in sent if f.get("type") == "saveFile"]
            assert saves[-1]["force"] is False
        finally:
            context.close()

    def test_closing_dirty_tab_asks_first(self, browser, harness) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "edit_close.py")
            _open_editor(page, str(path), "lnk-e8")
            _type_at_end(page, "unsaved = 1")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            dialogs: list[str] = []

            def _dismiss(dialog) -> None:
                dialogs.append(dialog.message)
                dialog.dismiss()

            page.once("dialog", _dismiss)
            page.click(".chat-tab.content-tab .chat-tab-close")
            page.wait_for_timeout(300)
            assert dialogs and "unsaved changes" in dialogs[0]
            assert "edit_close.py" in dialogs[0]
            assert page.locator(".chat-tab.content-tab").count() == 1
            assert "unsaved = 1" in _editor_text(page)

            page.once("dialog", lambda d: d.accept())
            page.click(".chat-tab.content-tab .chat-tab-close")
            page.wait_for_function(
                "() => document.querySelectorAll('.chat-tab.content-tab').length === 0",
                timeout=10000,
            )
            assert page.locator("#task-input").is_visible()
            assert path.read_text() == _SOURCE
            assert not any(f.get("type") == "closeTab" for f in sent)
        finally:
            context.close()

    def test_clean_tab_closes_without_asking(self, browser, harness) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "edit_clean_close.py")
            _open_editor(page, str(path), "lnk-e9")
            page.on("dialog", lambda d: (_ for _ in ()).throw(
                AssertionError("no dialog expected for a clean tab"),
            ))
            page.click(".chat-tab.content-tab .chat-tab-close")
            page.wait_for_function(
                "() => document.querySelectorAll('.chat-tab.content-tab').length === 0",
                timeout=10000,
            )
        finally:
            context.close()

    def test_stale_reply_for_an_earlier_save_is_ignored(
        self, browser, harness,
    ) -> None:
        """Every save carries its own token. A success reply for an
        EARLIER request (one that timed out, with a newer save since)
        must not mark the newer, unsaved text as saved. The stale
        reply is injected through the same window `message` event the
        WebSocket shim dispatches for real frames."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "edit_tokens.py")
            _open_editor(page, str(path), "lnk-tok")
            _type_at_end(page, "first = 1")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            page.keyboard.press("Control+s")
            _wait_for_disk(path, "first = 1")
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            first = [f for f in sent if f.get("type") == "saveFile"][0]
            tab_id = page.get_attribute(".chat-tab.content-tab", "data-tab-id")
            assert first["token"].startswith(tab_id + ":")
            _type_at_end(page, "\nsecond = 2")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            # A straggler acknowledging the FIRST save arrives now.
            page.evaluate(
                """([token, p]) => window.dispatchEvent(new MessageEvent(
                     'message', {data: {type: 'fileSaved', ok: true,
                       path: p, name: 'edit_tokens.py', token: token,
                       version: 'stale:0'}}))""",
                [first["token"], str(path)],
            )
            page.wait_for_timeout(300)
            assert page.locator(_DIRTY_TAB).count() == 1
            assert page.locator(_STATUS).inner_text() == "Unsaved changes"
            assert page.locator(_SAVE_BTN).is_enabled()
            # The real second save uses a new token and still succeeds
            # (the stale stamp above was not taken either).
            page.keyboard.press("Control+s")
            _wait_for_disk(path, "second = 2")
            page.wait_for_function(
                "sel => document.querySelector(sel).innerText === 'Saved'",
                arg=_STATUS, timeout=10000,
            )
            saves = [f for f in sent if f.get("type") == "saveFile"]
            assert len(saves) == 2
            assert saves[1]["token"] != saves[0]["token"]
            assert saves[1]["token"].startswith(tab_id + ":")
            assert saves[1]["version"] != "stale:0"
        finally:
            context.close()

    def test_invalid_utf8_file_opens_read_only(self, browser, harness) -> None:
        """A file that is not valid UTF-8 shows with U+FFFD but cannot
        be edited or saved: the daemon sends no version stamp, so the
        client keeps the read-only viewer and mounts no Save bar."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = harness.work_dir / "latin1.py"
            path.write_bytes(b"alpha = 'caf\xe9'\n")
            _inject_file_link(page, str(path), "lnk-l1")
            page.click("#lnk-l1")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            page.wait_for_selector(
                _MONACO + ", " + _EDITOR + ".content-code-fallback",
                timeout=30000,
            )
            if page.locator(_EDITOR + ".content-code-fallback").count() > 0:
                pytest.skip("Monaco CDN unreachable: no editor to test")
            page.wait_for_function(
                """() => document.querySelector('#content-tab-area')
                         .innerText.includes('alpha')""",
                timeout=30000,
            )
            assert page.locator(_EDITOR + ".content-save-bar").count() == 0
            _type_at_end(page, "TYPED")
            page.wait_for_timeout(300)
            assert "TYPED" not in _editor_text(page)
            assert page.locator(_DIRTY_TAB).count() == 0
            page.keyboard.press("Control+s")
            page.wait_for_timeout(300)
            assert not any(f.get("type") == "saveFile" for f in sent)
            assert path.read_bytes() == b"alpha = 'caf\xe9'\n"
        finally:
            context.close()

    def test_fallback_viewer_has_no_save_bar(self, browser, harness) -> None:
        """With the Monaco CDN unreachable the pre/code fallback is a
        viewer only: no Save bar, and Ctrl+S sends nothing."""
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
            page.wait_for_selector("#task-input", state="visible", timeout=30000)
            page.wait_for_selector(".chat-tab", timeout=30000)
            path = _fresh_file(harness, "edit_fallback.py")
            _inject_file_link(page, str(path), "lnk-fb")
            page.click("#lnk-fb")
            page.wait_for_selector(
                "#content-tab-area .content-code-fallback", timeout=30000,
            )
            assert page.locator("#content-tab-area .content-save-bar").count() == 0
            page.keyboard.press("Control+s")
            page.wait_for_timeout(300)
            assert not any(f.get("type") == "saveFile" for f in sent)
        finally:
            context.close()

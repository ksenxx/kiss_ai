# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   test parameters of the same name)
"""End-to-end tests: the menu bar of a remote-webapp content tab's editor.

A file opened in the remote webapp is edited in a Monaco editor.  Monaco
alone only offers a short right-click menu (unreachable on a phone), so
the tab carries a VS Code-style menu bar — File / Edit / Selection /
View / Go — in a row of its own directly above the editor.  Each item
runs a real Monaco action or command, and the menus work with a mouse,
a touchscreen and the keyboard.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` and the real
``cdn.jsdelivr.net`` Monaco bundle — no mocks.  When the CDN is
unreachable the editor cannot exist, so those tests skip (one test
blocks the CDN on purpose to pin the fallback behavior).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from kiss.tests.agents.vscode.test_content_tab_editing import (
    _dismiss_toasts,
    _fresh_file,
    _open_editor,
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

_VIEW = "#content-tab-area .content-tab-view:not([style*='display: none']) "
_MONACO = _VIEW + ".monaco-editor"
_MENUBAR = _VIEW + ".content-menubar"
_DROPDOWN = ".content-menu-dropdown"
_FALLBACK = _VIEW + ".content-code-fallback"
_MODE_BTN = _VIEW + ".content-mode-btn"
_DIRTY_TAB = ".chat-tab.content-tab.content-dirty"


def _menu_btn(page, label: str):
    return page.locator(_MENUBAR + " .content-menu-btn", has_text=label)


def _item(page, label: str):
    return page.locator(
        _DROPDOWN + " .content-menu-item",
        has=page.locator(".content-menu-label", has_text=label),
    )


def _choose(page, menu: str, label: str) -> None:
    """Open *menu* in the menu bar and click its *label* item."""
    _dismiss_toasts(page)
    _menu_btn(page, menu).click()
    _item(page, label).first.click()
    assert page.locator(_DROPDOWN).count() == 0


def _model_text(page) -> str:
    return str(page.evaluate(
        "() => window.monaco.editor.getEditors()"
        ".find(e => e.getDomNode().offsetParent).getModel().getValue()",
    ))


def _labels(page) -> list[str]:
    return cast(
        list[str],
        page.locator(_DROPDOWN + " .content-menu-label").all_inner_texts(),
    )


def _disabled(page) -> list[str]:
    return cast(list[str], page.locator(
        _DROPDOWN + " .content-menu-item.disabled .content-menu-label",
    ).all_inner_texts())


def _focused_label(page) -> str:
    return str(page.evaluate(
        "() => { const a = document.activeElement;"
        " const l = a && a.querySelector && a.querySelector("
        "'.content-menu-label'); return l ? l.textContent : a.textContent; }",
    ))


class TestContentTabMenuBar:
    """Browser E2E: the editor's menus open, close and run commands."""

    def test_menu_bar_opens_switches_and_closes(self, browser, harness) -> None:
        """The five menus sit in a row between the Save bar and the
        editor; a click opens one, a second click or Escape or a click
        elsewhere closes it, and pointing at a sibling while one is
        open switches menus."""
        context, page, _sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "menus_basic.py")
            _open_editor(page, str(path), "lnk-m1")
            bar = page.locator(_MENUBAR)
            assert page.locator(_VIEW + ".content-save-bar .content-menubar").count() == 0
            assert page.locator(
                _VIEW + ".content-save-bar + .content-menubar"
                " + .content-monaco-holder",
            ).count() == 1
            assert bar.locator(".content-menu-btn").all_inner_texts() == [
                "File", "Edit", "Selection", "View", "Go",
            ]
            _dismiss_toasts(page)
            _menu_btn(page, "Edit").click()
            assert _labels(page)[:2] == ["Undo", "Redo"]
            assert _menu_btn(page, "Edit").get_attribute("aria-expanded") == "true"
            keys = page.locator(_DROPDOWN + " .content-menu-key").all_inner_texts()
            assert keys[0] in ("Ctrl+Z", "\u2318Z")
            _menu_btn(page, "Go").hover()
            assert page.locator(_DROPDOWN).count() == 1
            assert _labels(page)[0] == "Go to Line/Column\u2026"
            _menu_btn(page, "Go").click()
            assert page.locator(_DROPDOWN).count() == 0
            assert _menu_btn(page, "Go").get_attribute("aria-expanded") == "false"
            _menu_btn(page, "View").click()
            page.keyboard.press("Escape")
            assert page.locator(_DROPDOWN).count() == 0
            _menu_btn(page, "Selection").click()
            page.click(_MONACO + " .view-lines")
            assert page.locator(_DROPDOWN).count() == 0
            _menu_btn(page, "File").click()
            page.set_viewport_size({"width": 1000, "height": 700})
            page.wait_for_function(
                "() => !document.querySelector('.content-menu-dropdown')",
                timeout=5000,
            )
            # Hovering with no menu open opens nothing.
            _menu_btn(page, "Edit").hover()
            assert page.locator(_DROPDOWN).count() == 0
        finally:
            context.close()

    def test_edit_commands_change_text_and_file_save_writes_it(
        self, browser, harness,
    ) -> None:
        """Move Line Down edits the buffer (the tab turns dirty), Undo
        reverts it, Redo re-applies it and File > Save writes it."""
        context, page, sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "menus_edit.py")
            _open_editor(page, str(path), "lnk-m2")
            page.click(_MONACO + " .view-lines .view-line >> nth=0")
            _dismiss_toasts(page)
            _menu_btn(page, "File").click()
            assert _disabled(page) == ["Save"]
            page.keyboard.press("Escape")
            _choose(page, "Selection", "Move Line Down")
            assert _model_text(page) == "beta = 2\nalpha = 1\n"
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            _choose(page, "Edit", "Undo")
            assert _model_text(page) == "alpha = 1\nbeta = 2\n"
            page.wait_for_function(
                "() => !document.querySelector('.chat-tab.content-dirty')",
                timeout=10000,
            )
            _choose(page, "Edit", "Redo")
            assert _model_text(page) == "beta = 2\nalpha = 1\n"
            _choose(page, "File", "Save")
            _wait_for_disk(path, "beta = 2\nalpha = 1")
            assert path.read_text() == "beta = 2\nalpha = 1\n"
            assert [f["type"] for f in sent].count("saveFile") == 1
        finally:
            context.close()

    def test_copy_and_paste_through_the_menu(self, browser, harness) -> None:
        """Select All + Copy puts the buffer on the clipboard; Paste at
        the end of the document inserts it again."""
        context, page, _sent = _open_page(browser, harness)
        try:
            context.grant_permissions(
                ["clipboard-read", "clipboard-write"],
                origin=harness.base_url,
            )
            path = _fresh_file(harness, "menus_clip.py")
            _open_editor(page, str(path), "lnk-m3")
            page.click(_MONACO + " .view-lines")
            _choose(page, "Selection", "Select All")
            _choose(page, "Edit", "Copy")
            clip = page.evaluate("() => navigator.clipboard.readText()")
            assert clip == "alpha = 1\nbeta = 2\n"
            page.keyboard.press("Control+End")
            _choose(page, "Edit", "Paste")
            page.wait_for_function(
                "() => window.monaco.editor.getEditors()[0].getModel()"
                ".getValue().split('alpha').length === 3",
                timeout=10000,
            )
            assert _model_text(page) == "alpha = 1\nbeta = 2\n" * 2
        finally:
            context.close()

    def test_find_go_to_line_and_view_toggles(self, browser, harness) -> None:
        """Find opens Monaco's find widget, Go to Line its quick input,
        and the View toggles flip (and show) the editor option."""
        context, page, _sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "menus_view.py")
            _open_editor(page, str(path), "lnk-m4")
            _choose(page, "Edit", "Find")
            page.wait_for_selector(
                _MONACO + " .find-widget.visible", timeout=10000,
            )
            page.keyboard.press("Escape")
            _choose(page, "Go", "Go to Line/Column")
            page.wait_for_selector(
                ".quick-input-widget:not([style*='display: none'])",
                timeout=10000,
            )
            page.keyboard.press("Escape")
            _dismiss_toasts(page)
            _menu_btn(page, "View").click()
            wrap = _item(page, "Word Wrap")
            assert wrap.get_attribute("role") == "menuitemcheckbox"
            assert wrap.get_attribute("aria-checked") == "false"
            wrap.click()
            _menu_btn(page, "View").click()
            assert _item(page, "Word Wrap").get_attribute("aria-checked") == "true"
            assert _item(page, "Word Wrap").locator(
                ".content-menu-check",
            ).inner_text() == "\u2713"
            _item(page, "Minimap").click()
            page.wait_for_selector(_MONACO + " .minimap", timeout=10000)
            _menu_btn(page, "View").click()
            assert _item(page, "Minimap").get_attribute("aria-checked") == "true"
            # A plain .py file has no symbol provider: Go to Symbol is
            # offered but disabled, as Monaco reports it unsupported.
            page.keyboard.press("Escape")
            _menu_btn(page, "Go").click()
            assert "Go to Symbol\u2026" in _disabled(page)
        finally:
            context.close()

    def test_close_editor_closes_the_tab(self, browser, harness) -> None:
        context, page, _sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "menus_close.py")
            _open_editor(page, str(path), "lnk-m5")
            _choose(page, "File", "Close Editor")
            page.wait_for_function(
                "() => !document.querySelector('.chat-tab.content-tab')",
                timeout=10000,
            )
        finally:
            context.close()

    def test_read_only_listing_gets_menus_without_edits(
        self, browser, harness,
    ) -> None:
        """A directory listing has no Save bar: the menus get their own
        row, and every text-changing item is disabled."""
        context, page, _sent = _open_page(browser, harness)
        try:
            folder = Path(harness.work_dir) / "menus_dir"
            folder.mkdir(exist_ok=True)
            (folder / "inner.txt").write_text("inner\n")
            _inject_file_link(page, str(folder), "lnk-m6")
            page.click("#lnk-m6")
            page.wait_for_selector(_MONACO + ", " + _FALLBACK, timeout=30000)
            if page.locator(_FALLBACK).count() > 0:
                pytest.skip("Monaco CDN unreachable: no editor to test")
            page.wait_for_selector(_MENUBAR)
            assert page.locator(_VIEW + ".content-save-bar").count() == 0
            _dismiss_toasts(page)
            _menu_btn(page, "Edit").click()
            disabled = _disabled(page)
            for label in ("Undo", "Redo", "Cut", "Paste", "Replace"):
                assert label in disabled
            assert "Copy" not in disabled
            assert "Find" not in disabled
            page.keyboard.press("Escape")
            _menu_btn(page, "File").click()
            assert _disabled(page) == ["Save"]
        finally:
            context.close()

    def test_markdown_menus_follow_the_source_surface(
        self, browser, harness,
    ) -> None:
        """A .md tab shows its preview first: no menus until Edit
        source mounts the editor, hidden again with the source, and an
        open menu closes when the surface flips under it (keyboard
        activation of the toggle fires no mousedown)."""
        context, page, _sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "menus_doc.md", "# Title\n\nbody\n")
            _inject_file_link(page, str(path), "lnk-m7")
            page.click("#lnk-m7")
            page.wait_for_selector(_VIEW + ".content-html-frame", timeout=30000)
            assert page.locator(_MENUBAR).count() == 0
            _dismiss_toasts(page)
            page.click(_MODE_BTN)
            page.wait_for_selector(_MONACO + ", " + _FALLBACK, timeout=30000)
            if page.locator(_FALLBACK).count() > 0:
                pytest.skip("Monaco CDN unreachable: no editor to test")
            page.wait_for_selector(_MENUBAR, state="visible")
            _menu_btn(page, "Edit").click()
            page.focus(_MODE_BTN)
            page.keyboard.press("Enter")
            assert page.locator(_DROPDOWN).count() == 0
            assert not page.locator(_MENUBAR).is_visible()
            page.focus(_MODE_BTN)
            page.keyboard.press("Enter")
            page.wait_for_selector(_MENUBAR, state="visible")
        finally:
            context.close()

    def test_open_menu_closes_with_its_tab(self, browser, harness) -> None:
        """Closing the tab from the keyboard (no mousedown) while its
        menu is open removes the dropdown with the editor."""
        context, page, _sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "menus_dispose.py")
            _open_editor(page, str(path), "lnk-m8")
            _dismiss_toasts(page)
            _menu_btn(page, "Edit").click()
            page.evaluate(
                "() => document.querySelector("
                "'.chat-tab.content-tab .chat-tab-close').click()",
            )
            page.wait_for_function(
                "() => !document.querySelector('.chat-tab.content-tab')",
                timeout=10000,
            )
            assert page.locator(_DROPDOWN).count() == 0
        finally:
            context.close()

    def test_keyboard_navigation(self, browser, harness) -> None:
        """Enter on a menu button focuses the first item; Up / Down /
        Home / End move between enabled items (wrapping), Left / Right
        open the neighbouring menu, Enter runs the focused item, Escape
        returns focus to the button and Tab closes the menu."""
        context, page, _sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "menus_keys.py")
            _open_editor(page, str(path), "lnk-k1")
            _dismiss_toasts(page)
            page.focus(_MENUBAR + " .content-menu-btn >> nth=1")  # Edit
            page.keyboard.press("Enter")
            assert _focused_label(page) == "Undo"
            page.keyboard.press("ArrowDown")
            assert _focused_label(page) == "Redo"
            page.keyboard.press("End")
            last = _focused_label(page)
            # Format Document is disabled (no formatter): End lands on
            # the last ENABLED item.
            assert last == "Toggle Block Comment"
            page.keyboard.press("ArrowDown")
            assert _focused_label(page) == "Undo"
            page.keyboard.press("ArrowUp")
            assert _focused_label(page) == last
            page.keyboard.press("Home")
            assert _focused_label(page) == "Undo"
            page.keyboard.press("ArrowLeft")
            assert _labels(page)[0] == "Save"
            assert _focused_label(page) == "Close Editor"
            page.keyboard.press("ArrowLeft")  # wraps to Go
            assert _focused_label(page) == "Go to Line/Column\u2026"
            page.keyboard.press("ArrowRight")  # wraps to File
            page.keyboard.press("ArrowRight")
            page.keyboard.press("ArrowRight")  # Selection
            assert _focused_label(page) == "Select All"
            page.keyboard.press("x")  # an unbound key does nothing
            assert page.locator(_DROPDOWN).count() == 1
            page.keyboard.press("Enter")
            assert page.locator(_DROPDOWN).count() == 0
            selected = page.evaluate(
                "() => { const e = window.monaco.editor.getEditors()[0];"
                " return e.getModel().getValueInRange(e.getSelection()); }",
            )
            assert selected == "alpha = 1\nbeta = 2\n"
            page.focus(_MENUBAR + " .content-menu-btn >> nth=3")  # View
            page.keyboard.press("ArrowDown")
            assert _focused_label(page) == "Command Palette\u2026"
            page.keyboard.press("Escape")
            assert page.locator(_DROPDOWN).count() == 0
            assert _focused_label(page) == "View"
            page.keyboard.press("Space")
            assert page.locator(_DROPDOWN).count() == 1
            page.keyboard.press("Tab")
            assert page.locator(_DROPDOWN).count() == 0
            assert _focused_label(page) == "Go"
            page.focus(_MENUBAR + " .content-menu-btn >> nth=3")  # View
            page.keyboard.press("Enter")
            page.keyboard.press("Shift+Tab")
            assert page.locator(_DROPDOWN).count() == 0
            assert _focused_label(page) == "Selection"
        finally:
            context.close()

    def test_tapping_between_menus_on_a_touchscreen(
        self, browser, harness,
    ) -> None:
        """A tap on another menu opens it (a tap's compatibility
        pointer events must not open-then-close it), and on a phone
        the Save button stays on screen next to the status."""
        context = browser.new_context(
            ignore_https_errors=True,
            viewport={"width": 375, "height": 740},
            is_mobile=True,
            has_touch=True,
        )
        page = context.new_page()
        try:
            page.goto(harness.base_url + "/")
            page.wait_for_selector("#task-input", state="visible", timeout=30000)
            page.wait_for_selector(".chat-tab", timeout=30000)
            path = _fresh_file(harness, "menus_touch.py")
            _open_editor(page, str(path), "lnk-t1")
            _dismiss_toasts(page)
            _menu_btn(page, "Edit").tap()
            assert _labels(page)[0] == "Undo"
            _menu_btn(page, "View").tap()
            assert page.locator(_DROPDOWN).count() == 1
            assert _labels(page)[0] == "Command Palette\u2026"
            _item(page, "Word Wrap").tap()
            page.click(_MONACO + " .view-lines")
            page.keyboard.press("Control+End")
            page.keyboard.type("gamma = 3")
            page.wait_for_selector(_DIRTY_TAB, timeout=10000)
            box = page.locator(_VIEW + ".content-save-btn").bounding_box()
            assert box is not None
            assert box["x"] + box["width"] <= 375
        finally:
            context.close()

    def test_switching_tabs_closes_an_open_menu(self, browser, harness) -> None:
        """A keyboard tab switch (no mousedown) hides the editor; its
        dropdown must go too, or its items would edit a hidden file."""
        context, page, _sent = _open_page(browser, harness)
        try:
            path = _fresh_file(harness, "menus_switch.py")
            _open_editor(page, str(path), "lnk-s1")
            _dismiss_toasts(page)
            _menu_btn(page, "Edit").click()
            page.evaluate(
                "() => document.querySelector("
                "'.chat-tab:not(.content-tab)').click()",
            )
            page.wait_for_selector("#content-tab-area", state="hidden")
            assert page.locator(_DROPDOWN).count() == 0
            page.evaluate(
                "() => document.querySelector('.chat-tab.content-tab').click()",
            )
            page.wait_for_selector(_MENUBAR, state="visible")
        finally:
            context.close()

    def test_cdn_fallback_viewer_has_no_menus(self, browser, harness) -> None:
        """Without Monaco there are no editor commands to offer."""
        context = browser.new_context(ignore_https_errors=True)
        context.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
        page = context.new_page()
        try:
            page.goto(harness.base_url + "/")
            page.wait_for_selector("#task-input", state="visible", timeout=30000)
            page.wait_for_selector(".chat-tab", timeout=30000)
            path = _fresh_file(harness, "menus_fb.py")
            _inject_file_link(page, str(path), "lnk-m9")
            page.click("#lnk-m9")
            page.wait_for_selector(_FALLBACK, timeout=30000)
            assert page.locator(".content-menubar").count() == 0
        finally:
            context.close()

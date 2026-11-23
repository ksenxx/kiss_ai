# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and reused as parameters)
"""E2E tests: the strip under the Chats list (``#sidebar-footer``) and
the one under the task-info panel (``#meta-footer``).

A REAL browser (Playwright Chromium) drives a REAL
:class:`RemoteAccessServer`, on the desktop layout and as a phone:

* the Chats strip stays at the bottom of the panel while the list above
  it scrolls, the user name is at the left (10px further in on a
  phone, clear of the screen's rounded corner), the buttons are in
  order and end at the right, and a panel too narrow for them wraps
  them instead of clipping them;
* the task-info strip lies along the bottom of its panel in the same
  way, its buttons start at the left with Settings, wrap in a narrow
  panel too, and the buttons of both strips have one size; on a phone
  a strip's space under its buttons is a fifth of the desktop's 8px;
* the font buttons move the root font size, which every CSS size is a
  rem or em of, and the size survives a reload; the Monaco editors (a
  file's and a diff's, open or opened later; the jsdom suite has no
  diff editor) with their own widgets (Find, the command palette, the
  right-click menu; the jsdom suite has no Monaco), the xterm.js
  terminal, the menu inside a preview and the text of a rendered
  Markdown file take it, the page of an ``.html`` file does not;
* on a phone, Settings and Terminal leave the task info screen for
  what they open.

Monaco and xterm.js come from the real ``cdn.jsdelivr.net``; when it is
unreachable the test that needs them skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import expect

from kiss.tests.conftest import reload_retrying_network_change
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_editing import (
    _dismiss_toasts,
    _fresh_file,
    _open_editor,
    _uncover_pane,
)
from kiss.tests.webapp.test_content_tab_file_links import _inject_file_link
from kiss.tests.webapp.test_mobile_swipe_hint import PHONE_UA, _open_phone_page
from kiss.tests.webapp.test_mobile_swipe_screens import (
    _seed_chat,
    _swipe_left,
    _swipe_right,
    _touch_session,
    _wait_screen,
)
from kiss.tests.webapp.test_sidebar_menus import _show_panel
from kiss.tests.webapp.test_workspace_sections import (
    _open_page,
    _show_section,
    browser,  # noqa: F401  (module fixture used by param name)
)

_ORDER = [
    "theme-btn", "contrast-btn", "theme-auto-btn",
    "font-step -1", "font-step 1",
]
_INFO_ORDER = [
    "settings-btn", "terminal-btn", "browser-btn", "autocommit-btn", "create-pr-btn",
]

# The boxes [left, top, right, bottom] of the Chats panel, the strip,
# the name and the list, each button as [id, left, top, right, bottom]
# (a font button has no id: "font-step " and its step stand for it),
# and by how much the list is longer than its box; then the task-info
# panel, its strip and that strip's buttons.  Of each strip also the
# [left, right] of the room inside its padding, and of the name the
# left edge of its text.
_LAYOUT_JS = """() => {
  const box = el => {
    const r = el.getBoundingClientRect();
    return [r.left, r.top, r.right, r.bottom].map(Math.round);
  };
  const room = el => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return [r.left + parseFloat(cs.paddingLeft), r.right - parseFloat(cs.paddingRight)]
      .map(Math.round);
  };
  const textLeft = el => {
    const range = document.createRange();
    range.selectNodeContents(el);
    return Math.round(range.getBoundingClientRect().left);
  };
  const byId = id => document.getElementById(id);
  const buttons = id =>
    Array.from(byId(id).querySelectorAll('.sidebar-tool-btn'))
      .map(b => [b.id || 'font-step ' + b.dataset.fontStep, ...box(b)]);
  const list = byId('history-list');
  return {
    sidebar: box(byId('sidebar')),
    footer: box(byId('sidebar-footer')),
    footerRoom: room(byId('sidebar-footer')),
    user: box(byId('sidebar-user')),
    userText: textLeft(byId('sidebar-user')),
    list: box(list),
    overflow: list.scrollHeight - list.clientHeight,
    buttons: buttons('sidebar-footer'),
    info: box(byId('meta-panel')),
    infoFooter: box(byId('meta-footer')),
    infoRoom: room(byId('meta-footer')),
    infoButtons: buttons('meta-footer'),
  };
}"""

_SET_NAME_JS = "name => { document.getElementById('sidebar-user').textContent = name; }"
_ROOT_PX = "getComputedStyle(document.documentElement).fontSize"
# The space under the buttons of the Chats strip and the task-info strip.
_STRIP_BOTTOM_PADS = """() => ['sidebar-footer', 'meta-footer'].map(
  id => getComputedStyle(document.getElementById(id)).paddingBottom)"""
_NAME_PX = (
    "parseFloat(getComputedStyle(document.getElementById('sidebar-user')).fontSize)"
)
# The font size of every Monaco editor of the page (a diff editor is
# two of them), as its distance from Monaco's default.
_MONACO_STEPS_JS = """() => monaco.editor.getEditors().map(
  e => e.getOption(monaco.editor.EditorOption.fontSize)
    - monaco.editor.EditorOptions.fontSize.defaultValue)"""
# The font sizes (px) of the text Monaco's own widgets show: what is
# typed into Find, its count of matches, and the command palette (its
# input and its first command).
_MONACO_WIDGETS_PX_JS = """() => [
  '.find-widget.visible .monaco-findInput textarea',
  '.find-widget.visible .matchesCount',
  '.quick-input-widget .monaco-inputbox input',
  '.quick-input-widget .monaco-list-row .label-name',
].map(selector => {
  const el = document.querySelector('#content-tab-area ' + selector);
  return el && Math.round(parseFloat(getComputedStyle(el).fontSize) * 100) / 100;
})"""
# The font size (px) of the right-click menu Monaco has open, which it
# builds in a shadow tree; null while there is none.
_MONACO_MENU_PX_JS = """() => {
  const host = document.querySelector('#content-tab-area .shadow-root-host');
  const item = host && host.shadowRoot.querySelector('.monaco-menu .action-label');
  return item && parseFloat(getComputedStyle(item).fontSize);
}"""
_VIEW = "#content-tab-area .content-tab-view:not([style*='display: none'])"
_XTERM_ROWS = ".terminal-tab-view .xterm-rows"
_XTERM_PX = f"getComputedStyle(document.querySelector('{_XTERM_ROWS}')).fontSize"
_XTERM_ROW_COUNT = f"document.querySelector('{_XTERM_ROWS}').children.length"
_FRAME = _VIEW + " .content-html-frame"


@pytest.fixture(scope="module")
def chats(harness) -> None:
    """Forty finished chats: more than the Chats list shows at once."""
    for i in range(40):
        _seed_chat(harness, f"strip-chat-{i}", f"Strip chat {i}")


def _in_order_and_inside(buttons: list, strip: list, order: list, size: int) -> None:
    """*buttons* read in *order*, each *size* px square and inside the
    box of its *strip*."""
    left, top, right, bottom = strip
    in_reading_order = sorted(buttons, key=lambda b: (b[2], b[1]))
    assert [b[0] for b in in_reading_order] == order
    for _name, b_left, b_top, b_right, b_bottom in buttons:
        assert (b_right - b_left, b_bottom - b_top) == (size, size)
        assert left <= b_left and b_right <= right
        assert top <= b_top and b_bottom <= bottom


def _strip(page, size: int, name_inset: int = 0) -> dict:
    """The layout of both strips, having checked that the Chats strip
    lies along the bottom of its panel under the list, with the name at
    its left (its text *name_inset* px in from the strip's padding) and
    the *size* px buttons in order up to its right end, none of them
    clipped, and that the task-info strip lies along the bottom of its
    panel with its buttons of that same size in order from its left end;
    ``rows`` is the number of rows the Chats buttons take, ``infoRows``
    the number the task-info buttons take."""
    lay: dict = page.evaluate(_LAYOUT_JS)
    height = page.viewport_size["height"]
    assert lay["footer"][3] == lay["sidebar"][3] == height
    assert lay["list"][3] <= lay["footer"][1], "the list ends where the strip begins"
    lay["rows"] = len({b[2] for b in lay["buttons"]})
    _in_order_and_inside(lay["buttons"], lay["footer"], _ORDER, size)
    assert all(lay["user"][0] <= b[1] for b in lay["buttons"]), "the name is at the left"
    assert lay["userText"] == lay["footerRoom"][0] + name_inset
    assert lay["buttons"][-1][3] == lay["footerRoom"][1], "the buttons end at the right"
    assert lay["infoFooter"][3] == lay["info"][3] == height
    lay["infoRows"] = len({b[2] for b in lay["infoButtons"]})
    _in_order_and_inside(lay["infoButtons"], lay["infoFooter"], _INFO_ORDER, size)
    assert lay["infoButtons"][0][1] == lay["infoRoom"][0], "the buttons start at the left"
    assert lay["infoButtons"][-1][3] < lay["infoRoom"][1], "and leave the right end free"
    return lay


def _wait_chats(page) -> None:
    """Wait for the seeded chats and the user name."""
    page.wait_for_selector("#history-list .history-chat-group", timeout=30000)
    expect(page.locator("#sidebar-user")).not_to_be_empty(timeout=30000)


def _monaco_menu_px(page, text: str) -> float:
    """The font size (px) of the menu a right-click on the Monaco text
    at selector *text* opens."""
    page.click(text, button="right")
    px: float = page.wait_for_function(_MONACO_MENU_PX_JS, timeout=10000).json_value()
    page.keyboard.press("Escape")
    return px


def _wait_terminal(page) -> None:
    """Wait for the terminal on screen (skip when xterm.js, which the
    first terminal downloads from the CDN, could not be loaded)."""
    page.wait_for_selector(_XTERM_ROWS + ", .terminal-load-error", timeout=60000)
    if page.locator(".terminal-load-error").count():
        pytest.skip("xterm.js CDN unreachable: no terminal to size")


def test_strip_stays_under_the_scrolling_list_on_the_desktop(browser, harness, chats):
    context, page, _frames = _open_page(browser, harness, width=1920)
    try:
        _wait_chats(page)
        # A wide panel holds the name and the buttons in one row.
        lay = _strip(page, 32)
        assert page.evaluate(_STRIP_BOTTOM_PADS) == ["8px", "8px"]
        assert lay["overflow"] > 0, "the list is longer than the panel"
        assert (lay["rows"], lay["infoRows"]) == (1, 1)
        first = lay["buttons"][0]
        assert lay["user"][2] <= first[1] and lay["user"][1] < first[4]
        page.evaluate("document.getElementById('history-list').scrollTop = 1e6")
        assert page.evaluate("document.getElementById('history-list').scrollTop") > 0
        assert _strip(page, 32)["footer"] == lay["footer"], "the strip did not scroll"

        # A docked panel is a fifth of the window.  At 1366px a short
        # name still shares the row with the five buttons (168px) and a
        # long one has the row above them; at 940px (156px of room) the
        # buttons of both strips wrap.
        page.set_viewport_size({"width": 1366, "height": 900})
        beside = {}
        for name in ("ab", "a-user-with-a-long-name"):
            page.evaluate(_SET_NAME_JS, name)
            lay = _strip(page, 32)
            assert (lay["rows"], lay["infoRows"]) == (1, 1)
            beside[name] = lay["user"][2] <= lay["buttons"][0][1]
            assert beside[name] or lay["user"][3] <= lay["buttons"][0][2], "beside or above"
        assert beside == {"ab": True, "a-user-with-a-long-name": False}
        page.set_viewport_size({"width": 940, "height": 900})
        lay = _strip(page, 32)
        assert (lay["rows"], lay["infoRows"]) == (2, 2)

        # A button names what its press does next.
        page.click("#theme-btn")
        expect(page.locator("#theme-btn")).to_have_attribute("aria-label", "Light mode")
        assert not page.evaluate("document.body.classList.contains('light-theme')")
    finally:
        context.close()


def test_font_buttons_resize_the_page_the_editors_and_the_terminal(browser, harness):
    context, page, _frames = _open_page(browser, harness)
    try:
        inc = page.locator('[data-font-step="1"]')
        dec = page.locator('[data-font-step="-1"]')
        assert page.evaluate(_ROOT_PX) == "13px"
        name_px = page.evaluate(_NAME_PX)
        inc.click()
        assert page.evaluate(_ROOT_PX) == "14px"
        assert page.evaluate(_NAME_PX) > name_px, "the page's text grew"

        # A diff and a file opened now start one step above Monaco's
        # default; open, they follow the next step.
        _show_section(page, "meta-scm")
        commit = page.locator(f"#scm-graph .scm-commit[data-scm-sha='{harness.shas['second']}']")
        commit.click()
        commit.locator("xpath=..").locator(".scm-commit-files .scm-row").nth(1).click()
        fallback = "#content-tab-area .content-code-fallback"
        page.wait_for_selector(f"#content-tab-area .monaco-diff-editor, {fallback}", timeout=30000)
        if page.locator(fallback).count():
            pytest.skip("Monaco CDN unreachable: no editor to size")
        modified = _VIEW + " .editor.modified .view-lines:not(.line-delete)"
        assert _monaco_menu_px(page, modified) == 14
        _open_editor(page, str(_fresh_file(harness, "strip_font.py")), "lnk-strip")
        assert page.evaluate(_MONACO_STEPS_JS) == [1, 1, 1]
        inc.click()
        assert page.evaluate(_MONACO_STEPS_JS) == [2, 2, 2]

        # Monaco's own widgets, which its stylesheet sizes in px, have
        # the page's sizes: 15px by now, 12/13 of it for the count.
        assert _monaco_menu_px(page, _VIEW + " .view-lines") == 15
        # Monaco's Find is Cmd+F on macOS, where Ctrl+F moves the cursor.
        page.keyboard.press("ControlOrMeta+f")
        page.keyboard.type("alpha")
        count = page.locator("#content-tab-area .find-widget.visible .matchesCount")
        expect(count).to_contain_text(" of ", timeout=10000)
        page.keyboard.press("F1")
        page.wait_for_selector(_VIEW + " .quick-input-widget .monaco-list-row", timeout=10000)
        assert page.evaluate(_MONACO_WIDGETS_PX_JS) == [15, 13.85, 15, 15]
        dec.click()
        assert page.evaluate(_MONACO_WIDGETS_PX_JS) == [14, 12.92, 14, 14]
        inc.click()
        page.keyboard.press("Escape")

        # A terminal starts at the page's size and is refitted when it
        # changes: smaller cells, more rows.
        _show_panel(page)
        page.click("#terminal-btn")
        _wait_terminal(page)
        assert page.evaluate(_XTERM_PX) == "15px"
        rows = page.evaluate(_XTERM_ROW_COUNT)
        dec.click()
        assert page.evaluate(_XTERM_PX) == "14px"
        page.wait_for_function(f"{_XTERM_ROW_COUNT} > {rows}", timeout=10000)

        # The menu inside a preview has the page's size, also after a
        # step taken while the preview is open.
        preview = Path(harness.work_dir) / "strip_preview.html"
        preview.write_text('<!DOCTYPE html><p id="text">Preview text</p>', encoding="utf-8")
        _inject_file_link(page, str(preview), "lnk-strip-preview")
        page.click("#lnk-strip-preview")
        page.wait_for_selector(_FRAME, timeout=30000)
        _uncover_pane(page)
        frame = page.frame_locator(_FRAME)
        menu = frame.locator("#sorcar-content-context-menu")
        for expected in ("14px", "13px"):
            frame.locator("#text").click(button="right", position={"x": 4, "y": 4})
            expect(menu).to_have_css("font-size", expected)
            # The page of an .html file keeps its own sizes.
            expect(frame.locator("#text")).to_have_css("font-size", "16px")
            page.keyboard.press("Escape")
            menu.wait_for(state="detached")
            dec.click()
        # 12px by now; back at 13px the editors have Monaco's default.
        assert page.evaluate(_MONACO_STEPS_JS) == [-1, -1, -1]
        inc.click()
        assert page.evaluate(_MONACO_STEPS_JS) == [0, 0, 0]

        # A rendered Markdown file opens at the text size (the browser's
        # 16px moved by as many px as the page's 13px) and follows a
        # step taken while it is open; its code stays smaller than its
        # text, as at the default size.
        inc.click()
        notes = Path(harness.work_dir) / "strip_notes.md"
        notes.write_text("Notes text and `code`.\n", encoding="utf-8")
        _inject_file_link(page, str(notes), "lnk-strip-notes")
        page.click("#lnk-strip-notes")
        text = page.frame_locator(_FRAME).locator("p", has_text="Notes text")
        for text_px, code_px in (("17px", "13.8125px"), ("16px", "13px")):
            expect(text).to_have_css("font-size", text_px, timeout=30000)
            expect(text.locator("code")).to_have_css("font-size", code_px)
            dec.click()
        expect(text).to_have_css("font-size", "15px")
        inc.click()

        # A button is disabled at its limit; the size survives a reload.
        for _ in range(3):
            dec.click()
        assert page.evaluate(_ROOT_PX) == "10px"
        expect(dec).to_be_disabled()
        expect(inc).to_be_enabled()
        reload_retrying_network_change(page)
        page.wait_for_selector("#task-input", state="visible", timeout=30000)
        assert page.evaluate(_ROOT_PX) == "10px"
        expect(dec).to_be_disabled()
    finally:
        context.close()


def test_strip_on_a_phone(browser, harness, chats):
    context, page, _frames = _open_phone_page(browser, harness, PHONE_UA)
    try:
        _dismiss_toasts(page)
        cdp = _touch_session(page)
        # Each swipe rests before it lifts: a tap follows it.
        _swipe_right(cdp, rest=True)
        _wait_screen(page, "chats")
        _wait_chats(page)

        # One row of 36px touch targets beside the name, under the list.
        # The name starts 10px in from the strip's padding, 26px from
        # the screen's edge: clear of an iPhone's rounded corner.
        lay = _strip(page, 36, name_inset=10)
        assert page.evaluate(_STRIP_BOTTOM_PADS) == ["1.6px", "1.6px"]
        assert lay["userText"] - lay["sidebar"][0] == 26, lay
        assert lay["overflow"] > 0, "the list is longer than the screen"
        assert (lay["rows"], lay["infoRows"]) == (1, 1)
        first = lay["buttons"][0]
        assert lay["user"][2] <= first[1] and lay["user"][1] < first[4]
        page.evaluate("document.getElementById('history-list').scrollTop = 1e6")
        assert _strip(page, 36, name_inset=10)["footer"] == lay["footer"], (
            "the strip did not scroll"
        )
        page.tap('[data-font-step="1"]')
        assert page.evaluate(_ROOT_PX) == "14px"

        # Settings, the first button of the task-info screen's strip,
        # leaves that screen for its sheet.  Closing the sheet focuses
        # the New chat button, not the composer, whose keyboard would
        # cover half the screen.
        _swipe_left(cdp, rest=True)
        _wait_screen(page, "chat")
        _swipe_left(cdp, rest=True)
        _wait_screen(page, "info")
        expect(page.locator("#settings-btn")).to_be_in_viewport()
        page.tap("#settings-btn")
        _wait_screen(page, "chat")
        expect(page.locator("#settings-panel")).to_be_visible()
        expect(page.locator("#meta-panel")).not_to_be_in_viewport()
        page.tap("#settings-panel-close")
        expect(page.locator("#settings-panel")).to_be_hidden()
        expect(page.locator("#new-chat-btn")).to_be_focused()

        # Terminal leaves the task-info screen for its tab.
        _swipe_left(cdp, rest=True)
        _wait_screen(page, "info")
        expect(page.locator("#terminal-btn")).to_be_in_viewport()
        page.tap("#terminal-btn")
        _wait_screen(page, "file")
        _wait_terminal(page)
        expect(page.locator(_XTERM_ROWS)).to_be_visible()
        assert page.evaluate(_XTERM_PX) == "14px"
    finally:
        context.close()

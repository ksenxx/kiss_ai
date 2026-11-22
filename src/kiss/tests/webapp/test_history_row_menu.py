# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The buttons on the line of a Chats-panel row, in a real Chromium: a
header's fold button and the "…" button with the menu it opens.

A chat panel's header shows one line of text that ends in a fold
button and a "…" button, a task panel one line that ends in the "…"
button alone; the row's other buttons, its "... ago" label and its
machine are the lines of a menu that the "…" button opens as a popover
(``makeSidebarRowMenu`` in ``media/main.js``).  jsdom lays nothing out
and has no Popover API, so where the buttons sit and what the page
does while a menu is open is tested here:

* a header's fold button sits immediately left of its "…" button in a
  box of the same size, after the ellipsized title, and folds the
  chat's tasks without opening a chat, at a desktop's and a phone's
  width;
* a task panel is one line at both widths: it has no fold button and
  nothing hangs under its text;
* one menu is open at a time, and its own button or a press elsewhere
  closes it without opening a chat;
* the arrow keys, Home and End walk its lines and Enter chooses one;
  Tab closes the menu and moves on; Escape closes the menu alone and
  gives the keyboard back to the button;
* copy and the delete question answer inside the menu;
* an open menu holds a changed list back until it closes;
* the menu opens above a row that is too near the window's bottom;
* it closes when its list scrolls or the window changes size;
* no tooltip shows over it.

What a row looks like with its menu closed and what the menu lists is
pinned by ``test_codex_task_panel_style.py``; that the menu closes
with the phone layout's Chats drawer, by ``test_mobile_swipe_screens.py``.

A browser without the Popover API (Firefox before 125, Safari before
17) cannot open a menu.  There ``main.js`` asks for no open menu (the
``POPOVERS`` check, the path every jsdom suite takes) and ``main.css``
leaves the "…" buttons and the menus out
(``@supports not selector(:popover-open)``), so a row is its line of
text, and a header keeps its fold button.  Every browser Playwright
installs has popovers, so the last test puts the rules of those
stylesheet blocks in force itself; no such browser was run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect, sync_playwright

from kiss.tests.webapp.test_history_failed_red_cross import (
    _open_history_page,
    _post_history,
    _sample_sessions,
)

_CONTEXT_MENU_JS = (
    Path(__file__).resolve().parents[2] / "webapp" / "media" / "contentContextMenu.js"
)

_FOLD = ":scope > .sidebar-item-collapse"
_MORE = ":scope > .sidebar-item-more"
_MENU = ":scope > .sidebar-item-actions"

# How the fold button and the "…" button of a chat's header sit on the
# line of its title.
_LINE_BUTTONS_JS = """row => {
  const box = el => {
    const r = el.getBoundingClientRect();
    return {left: r.left, right: r.right, middle: (r.top + r.bottom) / 2,
            width: r.width, height: r.height};
  };
  const fold = row.querySelector(':scope > .sidebar-item-collapse');
  const text = row.querySelector(':scope > .history-chat-title');
  return {
    foldThenMore: fold.nextElementSibling.matches('.sidebar-item-more'),
    fold: box(fold),
    more: box(fold.nextElementSibling),
    text: box(text),
    textIsCut: text.scrollWidth > text.clientWidth,
    textOverflow: getComputedStyle(text).textOverflow,
  };
}"""

# A task panel: its fold buttons, its children by their first class,
# how its "…" button sits on the line of its text, and how much taller
# the panel is than that line with the panel's own padding and border.
_TASK_LINE_JS = """row => {
  const text = row.querySelector(':scope > .sidebar-item-text');
  const line = text.getBoundingClientRect();
  const more = row.querySelector(':scope > .sidebar-item-more').getBoundingClientRect();
  const style = getComputedStyle(row);
  const frame = ['paddingTop', 'paddingBottom', 'borderTopWidth', 'borderBottomWidth']
    .reduce((sum, side) => sum + parseFloat(style[side]), 0);
  return {
    foldButtons: row.querySelectorAll('.sidebar-item-collapse').length,
    children: Array.from(row.children, child => child.classList[0]),
    textLines: line.height / parseFloat(getComputedStyle(text).lineHeight),
    textIsCut: text.scrollWidth > text.clientWidth,
    textOverflow: getComputedStyle(text).textOverflow,
    moreAfterText: more.left - line.right,
    moreOffTextMiddle: (more.top + more.bottom - line.top - line.bottom) / 2,
    underTheLine: row.getBoundingClientRect().height - line.height - frame,
  };
}"""

# Put in force, at their place in the sheet, the rules main.css keeps for
# a browser without popovers; returns how many blocks of them there are.
_APPLY_NO_POPOVER_RULES_JS = """() => {
  let blocks = 0;
  for (const sheet of document.styleSheets) {
    for (let i = sheet.cssRules.length - 1; i >= 0; i--) {
      const rule = sheet.cssRules[i];
      if (!(rule instanceof CSSSupportsRule)) continue;
      if (!/not\\s+selector\\(:popover-open\\)/.test(rule.conditionText)) continue;
      blocks++;
      Array.from(rule.cssRules).forEach((inner, k) => {
        sheet.insertRule(inner.cssText, i + 1 + k);
      });
    }
  }
  return blocks;
}"""

_A_MENU_IS_OPEN_JS = (
    "() => !!document.querySelector('#history-list .sidebar-item-actions:popover-open')"
)
_SIDEBAR_IS_OPEN_JS = "() => document.getElementById('sidebar').classList.contains('open')"
_TOOLTIP_SHOWS_JS = (
    "() => document.getElementById('custom-tooltip').classList.contains('visible')"
)

# Whether the keyboard is on a control after the "…" button *more*, and in no menu.
_KEYBOARD_MOVED_ON_JS = """more => {
  const focused = document.activeElement;
  return focused !== document.body && !focused.closest('.sidebar-item-actions')
    && !!(more.compareDocumentPosition(focused) & Node.DOCUMENT_POSITION_FOLLOWING);
}"""

# Where the open menu of a row hangs: from its "…" button, and in the window.
_MENU_PLACE_JS = """row => {
  const more = row.querySelector(':scope > .sidebar-item-more').getBoundingClientRect();
  const menu = row.querySelector(':scope > .sidebar-item-actions').getBoundingClientRect();
  return {
    below: menu.top - more.bottom,
    above: more.top - menu.bottom,
    rightEdges: menu.right - more.right,
    top: menu.top,
    left: menu.left,
  };
}"""

# The chat whose header is the last one wholly inside the list's box.
_LOWEST_HEADER_IN_VIEW_JS = """() => {
  const bottom = document.getElementById('history-list').getBoundingClientRect().bottom;
  const inView = [...document.querySelectorAll('#history-list .history-chat-header')]
    .filter(header => header.getBoundingClientRect().bottom <= bottom);
  return inView[inView.length - 1].parentElement.dataset.chatId;
}"""


@pytest.fixture(scope="module")
def _browser():
    """One headless Chromium for every test of the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _open_page(browser, sessions: list[dict]):
    """The Chats drawer of a phone-sized page listing *sessions*, every
    chat unfolded; returns ``(context, page)``."""
    context, page = _open_history_page(browser)
    # The keyboard model the page's menus share; chat.html loads it
    # before main.js.
    page.add_script_tag(path=str(_CONTEXT_MENU_JS))
    _post_history(page, sessions)
    return context, page


def _open_long_task_page(browser, width: int, height: int):
    """A page of *width* by *height* listing the sample chats, every
    chat unfolded and the text of the finished one far too long for a
    line of the list; returns ``(context, page)``."""
    long_text = "a task whose text is far too long for one line of the Chats list " * 12
    sessions = [
        {**s, "title": long_text, "preview": long_text} if s["id"] == "chat-ok" else s
        for s in _sample_sessions()
    ]
    context, page = _open_history_page(browser, width, height)
    # The desktop layout shows the list at once and has asked for it:
    # the rows come as the answer to that request.
    asked = page.evaluate(
        "window.__postedMessages.filter(m => m.type === 'getHistory').map(m => m.generation)"
    )
    _post_history(page, sessions, generation=asked[-1] if asked else 0)
    return context, page


def _post_page(page: Page, sessions: list[dict]) -> None:
    """The daemon's next ``history`` page arrives."""
    page.evaluate(
        "event => window.__post(event)",
        {"type": "history", "sessions": sessions, "offset": 0, "generation": 0},
    )


def _finished_chats(count: int) -> list[dict]:
    """*count* finished one-task chats, the newest first: more than the list's box holds."""
    done = _sample_sessions()[2]
    return [
        {
            **done,
            "id": f"chat-{n}",
            "task_id": 2000 + n,
            "title": f"task {n}",
            "preview": f"task {n}",
            "timestamp": done["timestamp"] - n,
        }
        for n in range(count)
    ]


def _header(page: Page, chat_id: str) -> Locator:
    """The header of the chat panel of *chat_id*."""
    return page.locator(
        f'#history-list .history-chat-group[data-chat-id="{chat_id}"] > .history-chat-header'
    )


def _task(page: Page, chat_id: str) -> Locator:
    """The task panel of the one task of *chat_id*."""
    return page.locator(
        f'#history-list .history-chat-group[data-chat-id="{chat_id}"] .running-item'
    )


def _is_open(row: Locator) -> bool:
    """Whether the menu of *row* is open."""
    return bool(row.locator(_MENU).evaluate("menu => menu.matches(':popover-open')"))


@pytest.mark.parametrize(("width", "height"), [(1280, 800), (390, 844)])
def test_fold_button_sits_left_of_the_menu_button_and_folds(_browser, width, height) -> None:
    """On a chat's header the fold button comes right before the "…"
    button, on the line of the title and in a box of the same size; a
    title too long for the line is cut with an ellipsis before the
    two.  Both come into sight with the pointer on the header.  A click
    on the fold button folds or unfolds the chat's tasks, renames the
    button, with the tooltip the click brought up, and opens no chat."""
    context, page = _open_long_task_page(_browser, width, height)
    try:
        posted = page.evaluate("window.__postedMessages.length")
        header, task = _header(page, "chat-ok"), _task(page, "chat-ok")
        place = header.evaluate(_LINE_BUTTONS_JS)
        fold, more, text = place["fold"], place["more"], place["text"]
        assert place["foldThenMore"], place
        assert fold["right"] <= more["left"] + 0.5, place
        assert fold["width"] == pytest.approx(more["width"], abs=0.5), place
        assert fold["height"] == pytest.approx(more["height"], abs=0.5), place
        assert fold["middle"] == pytest.approx(more["middle"], abs=0.5), place
        assert fold["middle"] == pytest.approx(text["middle"], abs=1), place
        assert place["textIsCut"] and place["textOverflow"] == "ellipsis", place
        assert text["right"] <= fold["left"] + 0.5, place
        header.hover()
        expect(header.locator(_FOLD)).to_have_css("opacity", "1")
        expect(header.locator(_MORE)).to_have_css("opacity", "1")

        tooltip = page.locator("#custom-tooltip.visible")
        header.locator(_FOLD).click()
        expect(task).to_be_hidden()
        expect(header.locator(_FOLD)).to_have_attribute("aria-label", "Show the chat's tasks")
        expect(tooltip).to_have_text("Show details")
        header.locator(_FOLD).click()
        expect(task).to_be_visible()
        expect(header.locator(_FOLD)).to_have_attribute("aria-label", "Hide the chat's tasks")
        expect(tooltip).to_have_text("Hide details")
        assert page.evaluate("window.__postedMessages.length") == posted, "a fold opens no chat"
    finally:
        context.close()


@pytest.mark.parametrize(("width", "height"), [(1280, 800), (390, 844)])
def test_task_panel_is_one_line_with_no_fold_button(_browser, width, height) -> None:
    """A task panel is its status icon, one line of text and the "…"
    button with its closed menu: it has no fold button and nothing
    hangs under the text, so the panel is as tall as that line.  A text
    too long for the line is cut with an ellipsis before the "…"
    button, which comes into sight with the pointer on the panel."""
    context, page = _open_long_task_page(_browser, width, height)
    try:
        task = _task(page, "chat-ok")
        line = task.evaluate(_TASK_LINE_JS)
        assert line["foldButtons"] == 0, line
        assert line["children"] == [
            "sidebar-item-completed",
            "sidebar-item-text",
            "sidebar-item-more",
            "sidebar-item-actions",
        ], line
        assert line["textLines"] == pytest.approx(1, abs=0.1), line
        assert line["textIsCut"] and line["textOverflow"] == "ellipsis", line
        assert line["moreAfterText"] >= -0.5, line
        assert line["moreOffTextMiddle"] == pytest.approx(0, abs=1), line
        assert line["underTheLine"] == pytest.approx(0, abs=0.5), line
        expect(task.locator(_MENU)).to_be_hidden()
        task.hover()
        expect(task.locator(_MORE)).to_have_css("opacity", "1")
    finally:
        context.close()


def test_one_menu_at_a_time_and_a_press_elsewhere_closes_it(_browser) -> None:
    """A row's "…" button opens its menu and closes the menu of another
    row; the button of the open menu closes it, and so does a press
    anywhere else.  None of it opens the row's chat."""
    context, page = _open_page(_browser, _sample_sessions())
    try:
        posted = page.evaluate("window.__postedMessages.length")
        failed, done = _header(page, "chat-fail"), _header(page, "chat-ok")
        failed.locator(_MORE).click()
        assert _is_open(failed)
        done.locator(_MORE).click()
        assert _is_open(done) and not _is_open(failed)
        done.locator(_MORE).click()
        assert not _is_open(done)
        done.locator(_MORE).click()
        assert _is_open(done)
        page.locator("#history-list .history-day-sep").first.click()
        assert not _is_open(done)
        assert page.evaluate("window.__postedMessages.length") == posted
    finally:
        context.close()


def test_keys_walk_the_menu_and_escape_closes_it_alone(_browser) -> None:
    """Enter on the "…" button opens the menu with the keyboard on its
    first line; Down, Up, Home and End walk the lines round and Enter
    chooses the focused one.  Tab closes the menu and moves on to the
    control after the button.  Escape closes the menu, gives the
    keyboard back to the button and leaves the Chats drawer open: the
    next Escape closes that."""
    context, page = _open_page(_browser, _sample_sessions())
    try:
        header = _header(page, "chat-ok")
        labels = header.locator(f"{_MENU} > button").evaluate_all(
            "buttons => buttons.map(button => button.getAttribute('aria-label'))"
        )
        assert len(labels) == 4, labels
        on_first_line = "() => document.activeElement.matches('.history-chat-favorite')"
        on_more_button = (
            "() => document.activeElement.matches('.history-chat-header > .sidebar-item-more')"
        )
        header.locator(_MORE).focus()
        page.keyboard.press("Enter")
        page.wait_for_function(on_first_line)
        for key, label in [
            ("ArrowDown", labels[1]),
            ("End", labels[3]),
            ("ArrowDown", labels[0]),
            ("ArrowUp", labels[3]),
            ("Home", labels[0]),
        ]:
            page.keyboard.press(key)
            assert page.evaluate("document.activeElement.getAttribute('aria-label')") == label, key
        # The first line stars the chat.
        page.keyboard.press("Enter")
        assert page.evaluate(
            "window.__postedMessages.filter(m => m.type === 'setChatFavorite').map(m => m.chatId)"
        ) == ["chat-ok"]
        assert not _is_open(header)
        page.wait_for_function(on_more_button, timeout=5000)

        page.keyboard.press("Enter")
        page.wait_for_function(on_first_line)
        page.keyboard.press("Tab")
        assert not _is_open(header)
        assert header.locator(_MORE).evaluate(_KEYBOARD_MOVED_ON_JS)

        header.locator(_MORE).focus()
        page.keyboard.press("Enter")
        page.wait_for_function(on_first_line)
        page.keyboard.press("Escape")
        assert not _is_open(header)
        assert page.evaluate(on_more_button)
        assert page.evaluate(_SIDEBAR_IS_OPEN_JS)
        page.keyboard.press("Escape")
        assert not page.evaluate(_SIDEBAR_IS_OPEN_JS)
    finally:
        context.close()


def test_copy_and_delete_answer_inside_the_menu(_browser) -> None:
    """Copy shows its tick and the trash asks its Delete / Cancel
    question in the menu, which stays open for both.  Escape answers
    an open question (Cancel) before it closes the menu, and a
    question left open goes with its menu.  After Delete, the list the
    daemon sends without the chat lands although the menu is open."""
    context, page = _open_page(_browser, _sample_sessions())
    try:
        task = _task(page, "chat-ok")
        copy = task.locator(f"{_MENU} > .sidebar-item-copy")
        task.locator(_MORE).click()
        copy.click()
        expect(copy).to_have_class(re.compile(r"\bcopied\b"))
        assert _is_open(task)

        header = _header(page, "chat-ok")
        trash = header.locator(f"{_MENU} > .sidebar-item-delete")
        question = header.locator(f"{_MENU} > .sidebar-item-confirm")
        header.locator(_MORE).click()
        assert not _is_open(task)
        trash.click()
        expect(question).to_be_visible()
        expect(trash).to_be_hidden()
        page.keyboard.press("Escape")
        expect(question).to_be_hidden()
        expect(trash).to_be_visible()
        assert _is_open(header)

        trash.click()
        page.locator("#history-list .history-day-sep").first.click()
        assert not _is_open(header)
        header.locator(_MORE).click()
        expect(question).to_be_hidden()
        expect(trash).to_be_visible()

        trash.click()
        question.locator(".sidebar-confirm-yes").click()
        assert page.evaluate(
            "window.__postedMessages.filter(m => m.type === 'deleteChat').map(m => m.chatId)"
        ) == ["chat-ok"]
        assert _is_open(header)
        _post_page(page, [s for s in _sample_sessions() if s["id"] != "chat-ok"])
        expect(header).to_have_count(0)
        assert not page.evaluate(_A_MENU_IS_OPEN_JS)
    finally:
        context.close()


def test_open_menu_holds_a_changed_list_back_until_it_closes(_browser) -> None:
    """A changed list that arrives while a menu is open would take the
    menu from under the pointer.  It waits, for longer than the page
    waits for the release of a stale press (5 s), and lands when the
    menu closes, the keyboard still on the row's "…" button."""
    context, page = _open_page(_browser, _sample_sessions())
    try:
        header = _header(page, "chat-ok")
        title = header.locator(".history-chat-title")
        header.locator(_MORE).click()
        _post_page(
            page,
            [
                {**s, "title": "renamed elsewhere", "preview": "renamed elsewhere"}
                if s["id"] == "chat-ok"
                else s
                for s in _sample_sessions()
            ],
        )
        page.wait_for_timeout(5500)
        expect(title).to_have_text("successful task")
        assert _is_open(header)
        page.keyboard.press("Escape")
        expect(title).to_have_text("renamed elsewhere")
        assert header.locator(_MORE).evaluate("more => more === document.activeElement")
    finally:
        context.close()


def test_menu_opens_above_a_row_too_near_the_window_bottom(_browser) -> None:
    """The menu hangs 4px under its button, their right edges in line;
    where the window ends too soon below the row, it stands 4px above
    the button instead, inside the window."""
    context, page = _open_page(_browser, _finished_chats(40))
    try:
        high = _header(page, "chat-0")
        high.locator(_MORE).click()
        place = high.evaluate(_MENU_PLACE_JS)
        assert place["below"] == pytest.approx(4, abs=0.5), place
        assert place["rightEdges"] == pytest.approx(0, abs=0.5), place

        low = _header(page, page.evaluate(_LOWEST_HEADER_IN_VIEW_JS))
        low.locator(_MORE).click()
        assert _is_open(low) and not _is_open(high)
        place = low.evaluate(_MENU_PLACE_JS)
        assert place["above"] == pytest.approx(4, abs=0.5), place
        assert place["rightEdges"] == pytest.approx(0, abs=0.5), place
        assert place["top"] >= 8 and place["left"] >= 8, place
    finally:
        context.close()


def test_menu_closes_when_its_list_scrolls_or_the_window_resizes(_browser) -> None:
    """A menu hangs from its row: it closes when the list scrolls the
    row away and when the window changes size."""
    context, page = _open_page(_browser, _finished_chats(40))
    try:
        header = _header(page, "chat-0")
        header.locator(_MORE).click()
        assert _is_open(header)
        box = page.locator("#history-list").bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + 40, box["y"] + box["height"] - 40)
        page.mouse.wheel(0, 200)
        page.wait_for_function(f"() => !({_A_MENU_IS_OPEN_JS})()", timeout=5000)
        assert page.evaluate("document.getElementById('history-list').scrollTop") > 0

        header = _header(page, page.evaluate(_LOWEST_HEADER_IN_VIEW_JS))
        header.locator(_MORE).click()
        assert _is_open(header)
        page.set_viewport_size({"width": 470, "height": 880})
        page.wait_for_function(f"() => !({_A_MENU_IS_OPEN_JS})()", timeout=5000)
    finally:
        context.close()


def test_no_tooltip_shows_over_an_open_menu(_browser) -> None:
    """The row's tooltip goes when its menu opens and none comes while
    the menu is open, neither for the line the keyboard lands on nor
    for the line under the pointer: the menu names its own lines.
    The keyboard's return to the "…" button as the menu closes brings
    no tooltip either."""
    context, page = _open_page(_browser, _sample_sessions())
    try:
        task = _task(page, "chat-ok")
        task.locator(".sidebar-item-text").hover()
        page.wait_for_function(_TOOLTIP_SHOWS_JS, timeout=5000)
        task.locator(_MORE).click()
        page.wait_for_function(
            "() => document.activeElement.matches('.sidebar-item-actions > .sidebar-item-favorite')"
        )
        assert not page.evaluate(_TOOLTIP_SHOWS_JS)
        task.locator(f"{_MENU} > .sidebar-item-delete").hover()
        # A tooltip comes 400 ms after the pointer.
        page.wait_for_timeout(700)
        assert not page.evaluate(_TOOLTIP_SHOWS_JS)
        # The pointer off every row, Escape closes the menu.
        page.mouse.move(5, 5)
        page.keyboard.press("Escape")
        assert not _is_open(task)
        assert task.locator(_MORE).evaluate("more => more === document.activeElement")
        page.wait_for_function(f"() => !({_TOOLTIP_SHOWS_JS})()", timeout=5000)
    finally:
        context.close()


def test_fold_button_stays_where_the_browser_has_no_popovers(_browser) -> None:
    """Under the rules main.css keeps for a browser without popovers a
    row has no "…" button, and a header's fold button is still there
    and still folds."""
    context, page = _open_page(_browser, _sample_sessions())
    try:
        assert page.evaluate(_APPLY_NO_POPOVER_RULES_JS) >= 1
        header, task = _header(page, "chat-ok"), _task(page, "chat-ok")
        for row in (header, task):
            expect(row.locator(_MORE)).to_be_hidden()
        expect(header.locator(_FOLD)).to_be_visible()
        header.locator(_FOLD).click()
        expect(task).to_be_hidden()
        header.locator(_FOLD).click()
        expect(task).to_be_visible()
    finally:
        context.close()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pins in the Chats sidebar, against a real daemon in a real Chromium.

A chat is pinned from the "…" menu of its header and a task from the
menu of its row (``Pin chat`` / ``Pin task``, the line after the
star).  Pinned chats and the chats of pinned tasks are listed under a
"Pinned" separator ahead of the days.  The daemon keeps the pins in
its history database (``setChatPinned`` / ``setPinned``), so a browser
that has nothing cached and a reloaded page list them there too;
unpinned, a chat goes back under its day.

Real daemon (``RemoteAccessServer``) with its own history database,
real Chromium (Playwright), at a desktop's size and on a phone.  What
a press does before the daemon answers (the move at once, a refetch
that predates the write, the filters) is tested in jsdom:
``webapp/test/historyPinned.test.js``.
"""

from __future__ import annotations

import datetime

import pytest
from playwright.sync_api import BrowserContext, Locator, Page, expect, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from kiss.agents.sorcar import persistence
from kiss.tests.conftest import click_row_menu_item, goto_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import _ServerHarness
from kiss.tests.webapp.test_content_tab_file_links import _wait_ready

pytestmark = pytest.mark.timeout(180)

_TIMEOUT = 30000
_DAY = 86400.0
_FOLD = ":scope > .sidebar-item-collapse"
_MORE = ":scope > .sidebar-item-more"
_PINNED_SEP = '#history-list .history-day-sep[data-day="pinned"]'

# What the Chats list shows from top to bottom: its separators, and its
# chats with their tasks.
_LIST_JS = """() => Array.from(document.getElementById('history-list').children)
  .filter(el => el.style.display !== 'none')
  .map(el => el.classList.contains('history-day-sep')
    ? 'sep:' + el.textContent
    : 'chat:' + Array.from(el.querySelectorAll('.sidebar-item-text'))
        .map(text => text.textContent).join(','))"""

# The name and the pressed state of every pin button, in the list's order.
_PINS_JS = """() => Array.from(
  document.querySelectorAll('#history-list .sidebar-item-pin'),
).map(pin => pin.getAttribute('aria-label') + ':' + pin.getAttribute('aria-pressed'))"""

# The left and right edges of buttons of a chat's header, read in one
# turn of the page: the refetch that follows a pin rebuilds the rows,
# so a button measured on its own may be gone before it is measured.
_EDGES_JS = """([chatId, ...buttons]) => {
  const header = document.querySelector(
    `#history-list .history-chat-group[data-chat-id="${chatId}"] > .history-chat-header`);
  return header && buttons.map(button => {
    const box = header.querySelector(button).getBoundingClientRect();
    return {left: box.left, right: box.right};
  });
}"""


@pytest.fixture
def harness():
    """A real daemon with its own history database."""
    h = _ServerHarness()
    yield h
    h.stop()


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
        yield b
        b.close()


def _noon_today() -> float:
    """Noon of the local day, as the timestamp of a task launched today."""
    noon = datetime.datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    return noon.timestamp()


def _seed(task: str, chat_id: str, launched: float) -> str:
    """Persist a finished task of *chat_id* ("" starts a chat) launched
    at *launched*; returns its chat's id."""
    task_id, chat = persistence._add_task(task, chat_id)
    persistence._save_task_result("done", task_id=task_id)
    with persistence._rw_lock.write_lock():
        persistence._get_db().execute(
            "UPDATE task_history SET timestamp = ? WHERE id = ?", (launched, task_id)
        )
    return chat


def _open(browser, harness: _ServerHarness, **device) -> tuple[BrowserContext, Page]:
    """The daemon's page in a browser context of its own: nothing is cached."""
    context = browser.new_context(ignore_https_errors=True, **device)
    page = context.new_page()
    goto_retrying_network_change(page, harness.base_url + "/")
    _wait_ready(page)
    return context, page


def _expect_list(page: Page, expected: list[str]) -> None:
    """Wait until the Chats list shows *expected* (what ``_LIST_JS`` reads)."""
    try:
        page.wait_for_function(
            f"expected => JSON.stringify(({_LIST_JS})()) === JSON.stringify(expected)",
            arg=expected,
            timeout=_TIMEOUT,
        )
    except PlaywrightTimeoutError:
        assert page.evaluate(_LIST_JS) == expected
        raise


def _header(page: Page, chat_id: str) -> Locator:
    """The header of the chat panel of *chat_id*."""
    return page.locator(
        f'#history-list .history-chat-group[data-chat-id="{chat_id}"] > .history-chat-header'
    )


def _task(page: Page, text: str) -> Locator:
    """The task panel of the task *text*."""
    return page.locator("#history-list .running-item").filter(has_text=text)


def test_pins_made_in_the_menus_are_kept_by_the_daemon(browser, harness) -> None:
    """Pinning a chat from its header's menu and a task from its row's
    menu lists both chats under "Pinned", newest first, ahead of the
    days; the day left without a chat loses its separator.  A browser
    with nothing cached and the reloaded page show the same list and
    the same pressed pins: the task's pin does not pin its chat, nor
    the chat's pin its tasks.  Unpinned, both chats are back under
    their days, for another browser too."""
    today = _noon_today()
    chat_a = _seed("alpha one", "", today - 600)
    _seed("alpha two", chat_a, today)
    chat_b = _seed("beta one", "", today - _DAY)
    chat_c = _seed("gamma one", "", today - 3 * _DAY)
    desktop = {"viewport": {"width": 1280, "height": 800}}
    context, page = _open(browser, harness, **desktop)
    try:
        old_day = page.evaluate(
            "ts => new Date(ts * 1000).toLocaleDateString(undefined, "
            "{weekday: 'short', month: 'short', day: 'numeric'})",
            today - 3 * _DAY,
        )
        by_day = [
            "sep:Today",
            "chat:alpha two,alpha one",
            "sep:Yesterday",
            "chat:beta one",
            f"sep:{old_day}",
            "chat:gamma one",
        ]
        _expect_list(page, by_day)
        assert page.locator(_PINNED_SEP).count() == 0
        # The daemon's finished tasks carry the green tick.
        expect(
            _header(page, chat_a).locator(":scope > .history-chat-status.status-tick")
        ).to_be_visible()
        # Away from the pointer a header's two buttons are out of sight.
        page.mouse.move(900, 400)
        for button in (_FOLD, _MORE):
            expect(_header(page, chat_a).locator(button)).to_have_css("opacity", "0")

        click_row_menu_item(_header(page, chat_c), ".history-chat-pin")
        _expect_list(
            page,
            [
                "sep:Pinned",
                "chat:gamma one",
                "sep:Today",
                "chat:alpha two,alpha one",
                "sep:Yesterday",
                "chat:beta one",
            ],
        )
        _header(page, chat_b).locator(_FOLD).click()
        click_row_menu_item(_task(page, "beta one"), ".sidebar-item-pin")
        pinned = [
            "sep:Pinned",
            "chat:beta one",
            "chat:gamma one",
            "sep:Today",
            "chat:alpha two,alpha one",
        ]
        pins = [
            "Pin chat:false",
            "Unpin task:true",
            "Unpin chat:true",
            "Pin task:false",
            "Pin chat:false",
            "Pin task:false",
            "Pin task:false",
        ]
        _expect_list(page, pinned)
        expect(page.locator(_PINNED_SEP)).to_be_visible()
        assert page.evaluate(_PINS_JS) == pins

        other_context, other = _open(browser, harness, **desktop)
        try:
            _expect_list(other, pinned)
            assert other.evaluate(_PINS_JS) == pins
        finally:
            other_context.close()

        page.reload()
        _wait_ready(page)
        _expect_list(page, pinned)
        click_row_menu_item(_header(page, chat_c), ".history-chat-pin")
        _header(page, chat_b).locator(_FOLD).click()
        click_row_menu_item(_task(page, "beta one"), ".sidebar-item-pin")
        _expect_list(page, by_day)
        assert page.locator(_PINNED_SEP).count() == 0

        other_context, other = _open(browser, harness, **desktop)
        try:
            _expect_list(other, by_day)
            assert set(other.evaluate(_PINS_JS)) == {"Pin chat:false", "Pin task:false"}
        finally:
            other_context.close()
    finally:
        context.close()


def test_a_chat_is_pinned_on_a_phone(browser, harness) -> None:
    """In the phone's Chats drawer a chat pinned from its menu goes
    under "Pinned" as on a desktop.  A touch screen has no hover: the
    fold button and the "…" button of a chat's header show at rest,
    inside the screen."""
    today = _noon_today()
    chat_a = _seed("alpha one", "", today)
    chat_b = _seed("beta one", "", today - _DAY)
    phone = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True}
    context, page = _open(browser, harness, **phone)
    try:
        page.click("#menu-btn")
        _expect_list(page, ["sep:Today", "chat:alpha one", "sep:Yesterday", "chat:beta one"])
        click_row_menu_item(_header(page, chat_b), ".history-chat-pin")
        _expect_list(page, ["sep:Pinned", "chat:beta one", "sep:Today", "chat:alpha one"])
        expect(page.locator(_PINNED_SEP)).to_be_visible()
        untouched = _header(page, chat_a)
        for button in (_FOLD, _MORE):
            expect(untouched.locator(button)).to_have_css("opacity", "1")
        edges = page.wait_for_function(_EDGES_JS, arg=[chat_a, _FOLD, _MORE], timeout=_TIMEOUT)
        fold, more = edges.json_value()
        assert 0 < fold["left"] < fold["right"] <= more["left"] + 0.5, (fold, more)
        assert more["left"] < more["right"] <= 390, (fold, more)
    finally:
        context.close()

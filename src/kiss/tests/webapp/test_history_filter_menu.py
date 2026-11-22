# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Chats panel's Filter and group menu in a real Chromium.

The button beside the search box opens a menu with one row per filter
of the list and a Group by row; a row names its setting and the
setting's value and opens a submenu with the options (``chat.html``,
``main.js`` "The Filter and group menu").  The menu and its submenus
are native popovers, which jsdom does not have:
``historyFilterMenu.test.js`` covers what an option does to the
filters, and this file what needs a browser: where the menu and the
submenus open, how the mouse, a finger and the keyboard drive them,
what closes them, and how they look.

Every test starts a daemon of its own (a real ``RemoteAccessServer``
over a real task database), seeds its tasks through the persistence
layer and drives the page that daemon serves over its WebSocket.
"""

from __future__ import annotations

import functools
import json
import re
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from playwright.sync_api import Browser, BrowserContext, Locator, Page, expect, sync_playwright

from kiss.agents.sorcar import persistence
from kiss.core import user_config
from kiss.tests.conftest import goto_retrying_network_change, reload_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import _ServerHarness
from kiss.tests.webapp.test_mobile_swipe_screens import _swipe_right, _touch_session, _wait_screen

DAY = 86400.0
BUTTON = "#history-filter-btn"
MENU = "#history-filter-menu"
ROW = (MENU + ' > .fg-row[data-fg="{}"]').format
SUBMENU = "#fg-{}".format
TOOLTIP = "#custom-tooltip"

# The settings of the menu, top to bottom.  Machine is listed only once a
# remote machine is registered, which no test here does.
SETTINGS = ["status", "favorites", "cron", "date", "sea", "tag", "workdir", "group"]

# What the toolbar had before the menu: the flat-list switch, the folding
# Filters panel and its chips.
OLD_CONTROLS = (
    "#history-view-toggle",
    "#history-filters-panel",
    "#history-filters-toggle",
    "#history-filters-body",
    ".history-filter-bar",
    ".hf-chip",
)

PARSER = "Fix the flaky parser unit test"
SHOP = "Buy a birthday present for mom on Amazon"
DEPLOY = "Deploy the nightly build to staging"
CLEANUP = "Nightly cleanup of the temp folders"
REVIEW = "Review the pull request for the tokenizer"
REPORT = "Draft the quarterly report outline"
WIKI = "Migrate the old wiki pages"

# The chats of ``_seed_week`` as the list shows them, newest first: the cron
# chat ("cleanup") is hidden until Cron tasks is set to Shown.
CHATS = ["parser", "shop", "deploy", "planning", "wiki"]

PHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1"
)


class _Window(NamedTuple):
    """A browser window on the daemon's page and the JSON frames of its WebSocket."""

    page: Page
    sent: list[dict[str, Any]]
    received: list[dict[str, Any]]


def _record(frames: list[dict[str, Any]], payload: str | bytes) -> None:
    """Append the JSON frame *payload* to *frames*; a frame that is not JSON is skipped."""
    try:
        frames.append(json.loads(payload))
    except ValueError:
        pass


def _watch(window: _Window, socket: Any) -> None:
    """Record the frames *socket*, a WebSocket of the window's page, sends and receives."""
    socket.on("framesent", functools.partial(_record, window.sent))
    socket.on("framereceived", functools.partial(_record, window.received))


def _count(frames: list[dict[str, Any]], kind: str) -> int:
    """How many frames of type *kind* are in *frames*."""
    return sum(1 for frame in frames if frame.get("type") == kind)


def _after_page_timers(page: Page, ms: int) -> None:
    """Wait *ms* on the page's own clock.

    A timer the page had already set and that is due within *ms* has run
    when this returns, however slow the machine is.  ``wait_for_timeout``
    counts on the test's clock and promises no such thing.
    """
    page.evaluate("ms => new Promise(resolve => setTimeout(resolve, ms))", ms)


def _settle(window: _Window, focus_requests: int) -> None:
    """Wait until the page is up and the daemon's focus request of this load has run.

    The daemon answers a page's ``ready`` with ``focusInput``, which puts the
    keyboard into the composer at once and again 100 and 300 ms later.  A test
    that used the keyboard before that would have it taken away mid-walk, so
    the load counts as done once the page has run a 350 ms timer that was set
    after the request arrived.

    Args:
        window: The window whose page is loading.
        focus_requests: The ``focusInput`` frames the window had received
            before this load began.
    """
    page = window.page
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_function(
        "() => !!(window._testApi && window._testApi.getActiveTabId())", timeout=30000,
    )
    deadline = time.monotonic() + 30
    while _count(window.received, "focusInput") <= focus_requests:
        assert time.monotonic() < deadline, "the daemon did not answer `ready` with focusInput"
        page.wait_for_timeout(25)
    _after_page_timers(page, 350)


class _Windows:
    """Opens browser windows on a daemon's page and closes them after the test."""

    def __init__(self, browser: Browser, harness: _ServerHarness) -> None:
        self._browser = browser
        self._harness = harness
        self._contexts: list[BrowserContext] = []

    def open(self, *, phone: bool = False) -> _Window:
        """Open the daemon's page in a window of its own and wait until it has settled.

        The window uses US English and UTC, so a date box reads month, day,
        year and "today" is the UTC day.  A phone's one-time swipe hint counts
        as dismissed long ago: it would cover the top of the screen.

        Args:
            phone: False for a 1280 x 900 desktop window with a mouse, True for
                a 390 x 844 touch screen that says it is an iPhone.

        Returns:
            The page with the live lists of the frames it sent and received.
        """
        options: dict[str, Any] = {"viewport": {"width": 1280, "height": 900}}
        if phone:
            options = {
                "viewport": {"width": 390, "height": 844},
                "is_mobile": True,
                "has_touch": True,
                "user_agent": PHONE_UA,
            }
        context = self._browser.new_context(
            ignore_https_errors=True, locale="en-US", timezone_id="UTC", **options,
        )
        self._contexts.append(context)
        context.add_init_script("localStorage.setItem('kiss-swipe-hint-dismissed', '1')")
        window = _Window(context.new_page(), [], [])
        window.page.on("websocket", functools.partial(_watch, window))
        goto_retrying_network_change(window.page, self._harness.base_url + "/")
        _settle(window, 0)
        return window

    def close(self) -> None:
        """Close every window this object opened."""
        for context in self._contexts:
            context.close()


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """One headless Chromium for every test of the module."""
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch(headless=True)
        try:
            yield chromium
        finally:
            chromium.close()


@pytest.fixture
def harness() -> Iterator[_ServerHarness]:
    """A real daemon with a task database and a config file of its own."""
    server = _ServerHarness()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def windows(browser: Browser, harness: _ServerHarness) -> Iterator[_Windows]:
    """The windows a test opens on its daemon's page, closed before the daemon stops."""
    opened = _Windows(browser, harness)
    try:
        yield opened
    finally:
        opened.close()


def _seed(
    harness: _ServerHarness,
    title: str,
    chat: str,
    *,
    age: float,
    success: bool = True,
    favorite: bool = False,
    **extra: Any,
) -> None:
    """Persist a finished task as a run leaves it: its row, its result and its end time.

    The end time is what dates the row in the list and what makes the daemon
    derive the row's tags from *title*.

    Args:
        harness: The daemon whose database gets the task.
        title: The task text.
        chat: The id of the task's chat.
        age: How many seconds ago the task ended.
        success: False for a task that ended in an error.
        favorite: True to star the task.
        **extra: Further columns of the row (``is_cron``, ``sea``, ``work_dir``).
    """
    columns = {"work_dir": str(harness.work_dir), **extra}
    task_id, _ = persistence._add_task(title, chat, columns)
    persistence._save_task_result(
        "done" if success else "it broke", task_id=task_id, success=success,
    )
    ended = int((time.time() - age) * 1000)
    persistence._save_task_extra({"startTs": ended - 60000, "endTs": ended}, task_id)
    if favorite:
        persistence._set_task_favorite(task_id, True)


def _seed_week(harness: _ServerHarness) -> None:
    """Seed seven tasks in six chats: a starred one, a failed one, a cron one, old ones."""
    _seed(harness, PARSER, "parser", age=300)
    _seed(harness, SHOP, "shop", age=600, favorite=True)
    _seed(harness, DEPLOY, "deploy", age=900, success=False)
    _seed(harness, CLEANUP, "cleanup", age=1200, is_cron=True)
    _seed(harness, REVIEW, "planning", age=DAY)
    _seed(harness, REPORT, "planning", age=2 * DAY)
    _seed(harness, WIKI, "wiki", age=40 * DAY)


_LISTED_JS = """() => [...document.querySelectorAll(
  '#history-list > .history-chat-group, #history-list > .sidebar-item')]
  .filter(el => el.getClientRects().length > 0)
  .map(el => el.dataset.chatId || el.querySelector('.sidebar-item-text').textContent)"""


def _listed(page: Page) -> list[str]:
    """What the Chats list shows, top to bottom: chat ids, or task texts in the flat list."""
    listed: list[str] = page.evaluate(_LISTED_JS)
    return listed


def _expect_listed(page: Page, wanted: list[str]) -> None:
    """The Chats list comes to show exactly *wanted* (see ``_listed``)."""
    deadline = time.monotonic() + 10
    while _listed(page) != wanted and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert _listed(page) == wanted


_FOCUS_JS = """() => {
  const el = document.activeElement;
  if (!el || el === document.body) return 'body';
  if (el.id) return '#' + el.id;
  if (el.classList.contains('fg-row')) return 'row:' + el.dataset.fg;
  if (el.classList.contains('fg-option')) return 'option:' + el.textContent.trim();
  return el.tagName.toLowerCase() + '.' + el.className;
}"""


def _focus(page: Page) -> str:
    """Where the keyboard is: ``row:<setting>``, ``option:<label>``, ``#<id>`` or ``body``."""
    name: str = page.evaluate(_FOCUS_JS)
    return name


def _expect_focus(page: Page, wanted: str) -> None:
    """The keyboard comes to rest on *wanted* (see ``_focus``)."""
    deadline = time.monotonic() + 3
    while _focus(page) != wanted and time.monotonic() < deadline:
        page.wait_for_timeout(25)
    assert _focus(page) == wanted


_BOX_JS = """el => {
  const r = el.getBoundingClientRect();
  return {left: r.left, top: r.top, right: r.right, bottom: r.bottom,
          width: r.width, height: r.height};
}"""


def _box(page: Page, selector: str) -> dict[str, float]:
    """The box of the one element *selector* matches, in window coordinates."""
    box: dict[str, float] = page.locator(selector).evaluate(_BOX_JS)
    return box


def _off_screen(page: Page, selector: str) -> dict[str, float]:
    """By how many pixels *selector* sticks out of the window, per side that it does."""
    box = _box(page, selector)
    size = page.viewport_size
    assert size is not None
    over = {
        "left": -box["left"],
        "top": -box["top"],
        "right": box["right"] - size["width"],
        "bottom": box["bottom"] - size["height"],
    }
    return {side: by for side, by in over.items() if by > 0}


def _open_submenus(page: Page) -> list[str]:
    """The settings whose submenu is open."""
    keys: list[str] = page.locator(MENU + " .fg-submenu").evaluate_all(
        "subs => subs.filter(s => s.matches(':popover-open')).map(s => s.dataset.fg)",
    )
    return keys


def _expect_submenus_stay(page: Page, keys: list[str]) -> None:
    """Fail unless the open submenus are exactly *keys* for the next two seconds.

    Two seconds are many times the 150 ms a resting mouse takes to open a
    row's submenu, so a submenu the hover opens or closes shows here.
    """
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        assert _open_submenus(page) == keys
        page.wait_for_timeout(50)


def _option(page: Page, key: str, label: str) -> Locator:
    """The option *label* in the submenu of the setting *key*."""
    return page.locator(SUBMENU(key) + " > .fg-option").filter(
        has_text=re.compile(f"^{re.escape(label)}$"),
    )


def _ticked(page: Page, key: str) -> list[str]:
    """The labels of the ticked options of the setting *key*."""
    labels: list[str] = page.locator(
        SUBMENU(key) + ' > .fg-option[aria-checked="true"]',
    ).evaluate_all("options => options.map(option => option.textContent.trim())")
    return labels


def _value(page: Page, key: str) -> Locator:
    """The value the row of the setting *key* shows."""
    return page.locator(ROW(key) + " > .fg-row-value")


def open_filter_options(page: Page, key: str) -> Locator:
    """Open the Filter and group menu on the options of the setting *key*, as a user does.

    Clicks the button and then the setting's row.  Both menus stay open: a
    click on the button closes them.

    Args:
        page: A page of the web app with the Chats panel on view and the
            menu closed.
        key: The setting, as in the row's ``data-fg``: ``status``,
            ``favorites``, ``cron``, ``sea``, ``tag``, ``machine``,
            ``workdir`` or ``group``.

    Returns:
        The options of the setting, top to bottom.
    """
    page.click(BUTTON)
    page.click(ROW(key))
    return page.locator(SUBMENU(key) + " > .fg-option")


def pick_filter_option(page: Page, key: str, label: str) -> None:
    """Pick *label* for the setting *key* in the Filter and group menu and close the menu.

    The click ticks or unticks an option of Status and chooses an option of
    every other setting.  The other tests of the Chats panel reach a filter
    or the flat list through this.

    Args:
        page: A page of the web app with the Chats panel on view and the
            menu closed.
        key: The setting (see ``open_filter_options``).
        label: The option's text, for example ``"None"`` of ``group``.
    """
    open_filter_options(page, key).filter(has_text=re.compile(f"^{re.escape(label)}$")).click()
    page.click(BUTTON)
    expect(page.locator(MENU)).to_be_hidden()


def _today() -> date:
    """Today by the page's clock (UTC, see ``_Windows.open``)."""
    return datetime.now(UTC).date()


def _day_text(day: date) -> str:
    """*day* as the Date row writes it: "Nov 3", with the year when it is not this year."""
    text = f"{day:%b} {day.day}"
    return text if day.year == _today().year else f"{text}, {day.year}"


def _open_from_keyboard(page: Page) -> None:
    """Open the menu as a keyboard user does: Tab from the search box, then Enter.

    The click into the search box also parks the mouse there, off the menu:
    a mouse resting on a row would open that row's submenu.
    """
    page.click("#history-search")
    page.keyboard.press("Tab")
    _expect_focus(page, BUTTON)
    page.keyboard.press("Enter")
    expect(page.locator(MENU)).to_be_visible()
    _expect_focus(page, "row:status")


def _show_chats(page: Page) -> None:
    """Bring the phone's Chats screen in with a right swipe, as a thumb does.

    The thumb rests before it lifts: the tests tap next.
    """
    _swipe_right(_touch_session(page), rest=True)
    _wait_screen(page, "chats")


_TOOLBAR_JS = """() => {
  const button = document.getElementById('history-filter-btn');
  const before = button.previousElementSibling;
  return {
    row: button.parentElement.className,
    beforeHoldsSearch: !!before && before.contains(document.getElementById('history-search')),
  };
}"""


def test_the_button_beside_the_search_box_opens_the_menu_under_itself(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """The button is the search box's right-hand neighbour and its menu hangs under it.

    Checks, in a desktop window: the flat-list switch, the Filters panel and
    its chips are gone; the button follows the search box on the same row,
    centred on it, and is named by its tooltip; a click opens the menu 0 to
    8px under the button with their right edges in line, inside the window,
    with the eight settings and their default values, the keyboard on the
    first row and the tooltip gone; Escape closes it and hands the keyboard
    back to the button; a second click on the button closes the menu with
    the submenu that is open; a click elsewhere on the page closes it too
    and leaves the keyboard on nothing inside the closed menu.
    """
    _seed_week(harness)
    page = windows.open().page
    _expect_listed(page, CHATS)
    for selector in OLD_CONTROLS:
        assert page.locator(selector).count() == 0, selector

    button = page.locator(BUTTON)
    menu = page.locator(MENU)
    expect(button).to_be_visible()
    expect(button).to_have_attribute("aria-label", "Filter and group")
    expect(button).to_have_attribute("aria-haspopup", "menu")
    expect(button).to_have_attribute("aria-expanded", "false")
    expect(menu).to_be_hidden()
    toolbar = page.evaluate(_TOOLBAR_JS)
    assert toolbar == {"row": "history-search-row", "beforeHoldsSearch": True}
    search = _box(page, "#history-search")
    at = _box(page, BUTTON)
    assert 0 <= at["left"] - search["right"] <= 12, (search, at)
    assert abs((at["top"] + at["bottom"]) - (search["top"] + search["bottom"])) <= 2, (search, at)

    button.hover()
    expect(page.locator(TOOLTIP)).to_have_class(re.compile(r"\bvisible\b"))
    expect(page.locator(TOOLTIP)).to_have_text("Filter and group")

    button.click()
    expect(menu).to_be_visible()
    expect(button).to_have_attribute("aria-expanded", "true")
    expect(page.locator(TOOLTIP)).not_to_have_class(re.compile(r"\bvisible\b"))
    card = _box(page, MENU)
    assert 0 <= card["top"] - at["bottom"] <= 8, (at, card)
    assert abs(card["right"] - at["right"]) <= 1, (at, card)
    assert _off_screen(page, MENU) == {}
    rows = page.locator(MENU + " > .fg-row:visible")
    expect(rows.locator(".fg-row-name")).to_have_text(
        ["Status", "Favorites", "Cron tasks", "Date", "SEA", "Tag", "Work dir", "Group by"],
    )
    expect(rows.locator(".fg-row-value")).to_have_text(
        ["All", "All", "Hidden", "All", "All", "All", "All", "Project"],
    )
    _expect_focus(page, "row:status")

    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()
    expect(button).to_have_attribute("aria-expanded", "false")
    _expect_focus(page, BUTTON)

    button.click()
    page.click(ROW("status"))
    expect(page.locator(SUBMENU("status"))).to_be_visible()
    button.click()
    expect(menu).to_be_hidden()
    expect(page.locator(SUBMENU("status"))).to_be_hidden()
    expect(button).to_have_attribute("aria-expanded", "false")

    button.click()
    expect(menu).to_be_visible()
    page.mouse.click(700, 300)
    expect(menu).to_be_hidden()
    expect(button).to_have_attribute("aria-expanded", "false")
    _expect_focus(page, "body")


def test_the_button_is_as_tall_as_the_search_box(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """The button and the search box beside it have the same height."""
    page = windows.open().page
    expect(page.locator(BUTTON)).to_be_visible()
    assert _box(page, BUTTON)["height"] == pytest.approx(
        _box(page, "#history-search")["height"], abs=0.5,
    )


_FIRST_ITEM_TOP_JS = """sub => [...sub.children]
  .find(child => child.getClientRects().length > 0).getBoundingClientRect().top"""


def test_a_row_opens_its_submenu_beside_the_menu_by_hover_and_by_click(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """Resting the mouse on a row, or clicking it, shows that row's submenu at the menu's side.

    Checks, in a desktop window, for each of the eight settings: the mouse
    resting on the row opens its submenu and no other, the row says it is
    expanded, the submenu starts 0 to 8px right of the menu with its first
    item level with the row, inside the window.  Then: a mouse that only
    crosses the next row on its way into the open submenu opens nothing; a
    click on another row shows that row's submenu instead; a second click on
    the same row keeps it.
    """
    _seed_week(harness)
    page = windows.open().page
    _expect_listed(page, CHATS)
    page.click(BUTTON)
    expect(page.locator(MENU)).to_be_visible()
    card = _box(page, MENU)

    for key in SETTINGS:
        page.hover(ROW(key))
        expect(page.locator(SUBMENU(key))).to_be_visible()
        assert _open_submenus(page) == [key]
        expect(page.locator(ROW(key))).to_have_attribute("aria-expanded", "true")
        sub = _box(page, SUBMENU(key))
        row = _box(page, ROW(key))
        assert 0 <= sub["left"] - card["right"] <= 8, (key, card, sub)
        first_top = page.locator(SUBMENU(key)).evaluate(_FIRST_ITEM_TOP_JS)
        assert abs(first_top - row["top"]) <= 2, (key, row, first_top)
        assert _off_screen(page, SUBMENU(key)) == {}, key
    expect(page.locator(MENU + ' > .fg-row[aria-expanded="true"]')).to_have_count(1)

    # A hand crosses a row in well under 100 ms.  On a loaded machine the
    # two moves of the test can be further apart, which is a rest on the
    # row and not a crossing: such an attempt is made again.
    crossed = _box(page, ROW("favorites"))
    slow = 0
    while True:
        page.hover(ROW("status"))
        expect(page.locator(SUBMENU("status"))).to_be_visible()
        target = _box(page, SUBMENU("status") + " > .fg-option >> nth=2")
        started = time.monotonic()
        page.mouse.move(crossed["right"] - 20, (crossed["top"] + crossed["bottom"]) / 2)
        page.mouse.move(target["left"] + 20, (target["top"] + target["bottom"]) / 2)
        if time.monotonic() - started < 0.1:
            break
        slow += 1
        assert slow < 5, "five crossings of a row took 100 ms or more each"
    _after_page_timers(page, 400)
    assert _open_submenus(page) == ["status"]

    page.click(ROW("tag"))
    expect(page.locator(SUBMENU("tag"))).to_be_visible()
    expect(page.locator(SUBMENU("status"))).to_be_hidden()
    expect(page.locator(ROW("status"))).to_have_attribute("aria-expanded", "false")
    page.click(ROW("tag"))
    _after_page_timers(page, 300)
    assert _open_submenus(page) == ["tag"]


_OPTION_ROLES_JS = """() => Object.fromEntries(
  [...document.querySelectorAll('#history-filter-menu .fg-submenu')].map(sub => [
    sub.dataset.fg,
    [...new Set([...sub.querySelectorAll('.fg-option')].map(o => o.getAttribute('role')))],
  ]))"""


def test_picked_options_are_ticked_named_on_their_row_counted_and_applied(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """A pick ticks its option, shows on the row and the button, and filters the list.

    Checks, with real clicks in the open menu: the Status options are check
    boxes to assistive technology and those of every other list one-of-many;
    Status takes several ticks (unticking Succeeded leaves the failed chat,
    the row reads "Running, Errored", the button counts one filter and both
    menus stay open; unticking Errored too empties the list, which says so);
    Cron tasks is one of two (Shown lists the cron chat); Favorites is one of
    two (Only favorites leaves the starred chat, two filters are counted);
    undoing each pick restores the list, the values and the button's plain
    name.
    """
    _seed_week(harness)
    page = windows.open().page
    _expect_listed(page, CHATS)
    button = page.locator(BUTTON)
    page.click(BUTTON)

    page.click(ROW("status"))
    assert _ticked(page, "status") == ["Running", "Errored", "Succeeded"]
    roles = page.evaluate(_OPTION_ROLES_JS)
    assert roles.pop("status") == ["menuitemcheckbox"]
    assert roles.pop("date") == []
    assert set(roles) == {"favorites", "cron", "sea", "tag", "machine", "workdir", "group"}
    assert all(found == ["menuitemradio"] for found in roles.values()), roles
    _option(page, "status", "Succeeded").click()
    assert _ticked(page, "status") == ["Running", "Errored"]
    expect(_value(page, "status")).to_have_text("Running, Errored")
    expect(button).to_have_attribute("aria-label", "Filter and group (1 filter on)")
    expect(button).to_have_class(re.compile(r"\bis-filtered\b"))
    _expect_listed(page, ["deploy"])
    expect(page.locator(MENU)).to_be_visible()
    assert _open_submenus(page) == ["status"]
    _option(page, "status", "Errored").click()
    expect(_value(page, "status")).to_have_text("Running")
    _expect_listed(page, [])
    expect(page.locator("#history-list .sidebar-empty-filter")).to_have_text(
        "No tasks match the filter",
    )
    _option(page, "status", "Errored").click()
    _option(page, "status", "Succeeded").click()
    expect(_value(page, "status")).to_have_text("All")
    expect(button).to_have_attribute("aria-label", "Filter and group")
    expect(button).not_to_have_class(re.compile(r"\bis-filtered\b"))
    _expect_listed(page, CHATS)

    page.click(ROW("cron"))
    assert _ticked(page, "cron") == ["Hidden"]
    _option(page, "cron", "Shown").click()
    assert _ticked(page, "cron") == ["Shown"]
    expect(_value(page, "cron")).to_have_text("Shown")
    expect(button).to_have_attribute("aria-label", "Filter and group (1 filter on)")
    _expect_listed(page, ["parser", "shop", "deploy", "cleanup", "planning", "wiki"])

    page.click(ROW("favorites"))
    assert _ticked(page, "favorites") == ["All"]
    _option(page, "favorites", "Only favorites").click()
    assert _ticked(page, "favorites") == ["Only favorites"]
    expect(_value(page, "favorites")).to_have_text("Only favorites")
    expect(button).to_have_attribute("aria-label", "Filter and group (2 filters on)")
    _expect_listed(page, ["shop"])

    _option(page, "favorites", "All").click()
    page.click(ROW("cron"))
    _option(page, "cron", "Hidden").click()
    expect(button).to_have_attribute("aria-label", "Filter and group")
    _expect_listed(page, CHATS)


def _await_frame(page: Page, frames: list[dict[str, Any]], kind: str, **fields: Any) -> dict:
    """Wait for a frame of type *kind* that carries *fields*; returns the newest one."""
    deadline = time.monotonic() + 10
    while True:
        for frame in reversed(frames):
            if frame.get("type") == kind and all(frame.get(k) == v for k, v in fields.items()):
                return frame
        assert time.monotonic() < deadline, f"no {kind} frame with {fields}"
        page.wait_for_timeout(25)


def test_a_tag_picked_in_the_menu_is_applied_by_the_daemon(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """A Tag pick is sent to the daemon, and the list shows the page the daemon answers with.

    Fifty-two newer tasks fill the daemon's first page of fifty, so the one
    shopping task is not in the page at all until the daemon is asked for
    its tag.  Checks: the pick sends ``getHistory`` with ``tag: "shopping"``;
    the daemon's reply to that request holds exactly the shopping task; the
    list shows exactly that chat; the option is ticked, the row reads
    "shopping" and the button counts one filter.  Picking All asks without a
    tag and brings the first page back.
    """
    newest = [f"build-{number:02d}" for number in range(1, 53)]
    for number, chat in enumerate(newest, start=1):
        _seed(harness, f"Summarize the log of build {number}", chat, age=60 * number)
    _seed(harness, SHOP, "shop", age=3 * DAY)
    window = windows.open()
    page = window.page
    _expect_listed(page, newest[:50])
    assert page.locator('#history-list [data-chat-id="shop"]').count() == 0

    page.click(BUTTON)
    page.click(ROW("tag"))
    _option(page, "tag", "shopping").click()
    asked = _await_frame(page, window.sent, "getHistory", tag="shopping", offset=0)
    answer = _await_frame(page, window.received, "history", generation=asked["generation"])
    assert [session["title"] for session in answer["sessions"]] == [SHOP]
    _expect_listed(page, ["shop"])
    assert _ticked(page, "tag") == ["shopping"]
    expect(_value(page, "tag")).to_have_text("shopping")
    expect(page.locator(BUTTON)).to_have_attribute(
        "aria-label", "Filter and group (1 filter on)",
    )

    requests = _count(window.sent, "getHistory")
    _option(page, "tag", "All").click()
    _expect_listed(page, newest[:50])
    again = [frame for frame in window.sent if frame.get("type") == "getHistory"][requests:]
    assert [frame["tag"] for frame in again] == [""]
    expect(page.locator(BUTTON)).to_have_attribute("aria-label", "Filter and group")


_DATE_LINES_JS = """sub => {
  const box = el => {
    const r = el.getBoundingClientRect();
    return {left: r.left, right: r.right, width: r.width, middle: (r.top + r.bottom) / 2};
  };
  return [...sub.querySelectorAll('.history-filter-date-group')].map(line =>
    ['label', 'input', 'button'].map(part => box(line.querySelector(part))));
}"""


def test_typed_dates_filter_the_list_and_clear_dates_restores_it(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """Days typed into the From and To boxes bound the list; Clear dates lifts the bound.

    Checks: the date boxes take no room while the menu is closed; the Date
    submenu opens with the keyboard in the From box and "All" on its row
    (the range the daemon reported narrows nothing); From and To are two
    lines of a label, a date box and a calendar button, each line on one
    level inside the card and the two boxes under each other; typing the day
    ten days back, digit by digit, sets the box, keeps the keyboard in it and
    the menus open, hides the forty-day-old chat, puts the range on the row
    and counts one filter; Clear dates empties both boxes, restores the list
    and the row, and goes away with nothing left to clear; a day twenty days
    back typed into the To box leaves only the old chat and reads "Until ..."
    on the row; with the menu closed the bound still holds and the button
    still counts it, and the reopened submenu shows the day again.
    """
    _seed_week(harness)
    page = windows.open().page
    _expect_listed(page, CHATS)
    button = page.locator(BUTTON)
    clear = page.locator("#hf-date-clear")
    expect(page.locator("#hf-from")).to_be_hidden()
    expect(page.locator("#hf-to")).to_be_hidden()
    page.click(BUTTON)
    page.click(ROW("date"))
    expect(page.locator(SUBMENU("date"))).to_be_visible()
    _expect_focus(page, "#hf-from")
    expect(_value(page, "date")).to_have_text("All")

    expect(page.locator(SUBMENU("date") + " label")).to_have_text(["From", "To"])
    card = _box(page, SUBMENU("date"))
    lines = page.locator(SUBMENU("date")).evaluate(_DATE_LINES_JS)
    for label, box, calendar in lines:
        assert card["left"] < label["left"] < label["right"] <= box["left"], (card, lines)
        assert box["right"] <= calendar["left"] < calendar["right"] < card["right"], (card, lines)
        assert abs(label["middle"] - box["middle"]) <= 2, lines
        assert abs(calendar["middle"] - box["middle"]) <= 2, lines
    (_, from_box, _), (_, to_box, _) = lines
    assert from_box["middle"] < to_box["middle"], lines
    assert from_box["left"] == pytest.approx(to_box["left"], abs=0.5), lines
    assert from_box["width"] == pytest.approx(to_box["width"], abs=0.5), lines

    since = _today() - timedelta(days=10)
    page.keyboard.type(since.strftime("%m%d%Y"))
    expect(page.locator("#hf-from")).to_have_value(since.isoformat())
    until = date.fromisoformat(page.input_value("#hf-to"))
    _expect_listed(page, ["parser", "shop", "deploy", "planning"])
    expect(_value(page, "date")).to_have_text(f"{_day_text(since)} \u2013 {_day_text(until)}")
    expect(button).to_have_attribute("aria-label", "Filter and group (1 filter on)")
    assert _focus(page) == "#hf-from"
    assert _open_submenus(page) == ["date"]

    clear.click()
    expect(page.locator("#hf-from")).to_have_value("")
    expect(page.locator("#hf-to")).to_have_value("")
    expect(_value(page, "date")).to_have_text("All")
    expect(button).to_have_attribute("aria-label", "Filter and group")
    expect(clear).to_be_hidden()
    _expect_listed(page, CHATS)

    before = _today() - timedelta(days=20)
    page.click("#hf-to", position={"x": 6, "y": 10})
    page.keyboard.type(before.strftime("%m%d%Y"))
    expect(page.locator("#hf-to")).to_have_value(before.isoformat())
    _expect_listed(page, ["wiki"])
    expect(_value(page, "date")).to_have_text(f"Until {_day_text(before)}")

    page.click(BUTTON)
    expect(page.locator(MENU)).to_be_hidden()
    expect(page.locator("#hf-to")).to_be_hidden()
    _expect_listed(page, ["wiki"])
    expect(button).to_have_attribute("aria-label", "Filter and group (1 filter on)")
    page.click(BUTTON)
    expect(_value(page, "date")).to_have_text(f"Until {_day_text(before)}")
    page.click(ROW("date"))
    expect(page.locator("#hf-to")).to_be_visible()
    expect(page.locator("#hf-to")).to_have_value(before.isoformat())


def test_the_calendar_of_a_date_box_opens_over_the_menu_and_sets_the_day(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """The calendar button of a date box shows a month grid that works on top of the menu.

    The menu is in the browser's top layer, which would cover a calendar
    drawn anywhere else.  Checks: the To box's button opens the calendar
    inside the Date submenu, on the month of the To day, inside the window;
    a real click on a day twenty days back (after one step to the previous
    month when needed) sets the To box, closes the calendar, keeps the
    submenu open and leaves only the forty-day-old chat; Escape with a
    calendar open closes the calendar alone, and the next Escape the submenu.
    """
    _seed_week(harness)
    page = windows.open().page
    _expect_listed(page, CHATS)
    calendar = page.locator(SUBMENU("date") + " .kiss-datepicker")
    page.click(BUTTON)
    page.click(ROW("date"))
    until = date.fromisoformat(page.input_value("#hf-to"))
    page.click("#hf-to-btn")
    expect(calendar).to_be_visible()
    expect(calendar.locator(".dp-title")).to_have_text(f"{until:%B} {until.year}")
    assert _off_screen(page, SUBMENU("date") + " .kiss-datepicker") == {}

    before = _today() - timedelta(days=20)
    if (before.year, before.month) != (until.year, until.month):
        calendar.locator('.dp-nav[data-nav="prev"]').click()
    expect(calendar.locator(".dp-title")).to_have_text(f"{before:%B} {before.year}")
    calendar.locator(f'.dp-day[data-date="{before.isoformat()}"]').click()
    expect(page.locator("#hf-to")).to_have_value(before.isoformat())
    expect(calendar).to_have_count(0)
    assert _open_submenus(page) == ["date"]
    _expect_listed(page, ["wiki"])

    page.click("#hf-from-btn")
    expect(calendar).to_be_visible()
    # The calendar listens for keys from its next timer task on.  A person
    # cannot press Escape within that millisecond; the test could, so it
    # waits for a timer of its own, which runs after the calendar's.
    _after_page_timers(page, 0)
    page.keyboard.press("Escape")
    expect(calendar).to_have_count(0)
    assert _open_submenus(page) == ["date"]
    _expect_focus(page, "#hf-from-btn")
    page.keyboard.press("Escape")
    expect(page.locator(SUBMENU("date"))).to_be_hidden()
    expect(page.locator(MENU)).to_be_visible()
    _expect_focus(page, "row:date")


def test_group_by_none_is_the_flat_list_and_the_choice_survives_a_reload(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """Group by: None lists every task on a row of its own, Chat the chats; a reload keeps it.

    Checks: the default is Project (ticked, on the row; with no project it
    lists the five chat panels, as Chat does); None
    lists the six task texts newest first with no chat panel, is ticked and
    named on the row, and does not count as a filter on the button; after a
    reload the list is still flat and the menu still says None; Chat brings
    the chat panels back, and a second reload keeps them.
    """
    _seed_week(harness)
    window = windows.open()
    page = window.page
    tasks = [PARSER, SHOP, DEPLOY, REVIEW, REPORT, WIKI]
    groups = page.locator("#history-list .history-chat-group")
    button = page.locator(BUTTON)
    _expect_listed(page, CHATS)

    page.click(BUTTON)
    expect(_value(page, "group")).to_have_text("Project")
    page.click(ROW("group"))
    assert _ticked(page, "group") == ["Project"]
    _option(page, "group", "None").click()
    _expect_listed(page, tasks)
    expect(groups).to_have_count(0)
    assert _ticked(page, "group") == ["None"]
    expect(_value(page, "group")).to_have_text("None")
    expect(button).to_have_attribute("aria-label", "Filter and group")
    expect(button).not_to_have_class(re.compile(r"\bis-filtered\b"))

    focus_requests = _count(window.received, "focusInput")
    reload_retrying_network_change(page)
    _settle(window, focus_requests)
    _expect_listed(page, tasks)
    expect(groups).to_have_count(0)
    page.click(BUTTON)
    expect(_value(page, "group")).to_have_text("None")
    page.click(ROW("group"))
    assert _ticked(page, "group") == ["None"]
    _option(page, "group", "Chat").click()
    _expect_listed(page, CHATS)
    expect(_value(page, "group")).to_have_text("Chat")

    focus_requests = _count(window.received, "focusInput")
    reload_retrying_network_change(page)
    _settle(window, focus_requests)
    _expect_listed(page, CHATS)
    page.click(BUTTON)
    expect(_value(page, "group")).to_have_text("Chat")


_FILL_JS = "el => getComputedStyle(el).backgroundColor"
_HOLDS_FOCUS_JS = "el => el.contains(document.activeElement)"


def test_the_keyboard_walks_the_menu_and_its_submenus(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """The menu is driven from the keyboard as the ARIA menu pattern describes.

    Checks: Tab from the search box reaches the button and Enter opens the
    menu on its first row; Down walks the eight rows (the hidden Machine row
    is skipped) and wraps, Up wraps back, End and Home jump to the ends; the
    row under the keyboard is the filled one; Right opens the row's submenu
    on its first option, where Down, End and Home walk and wrap; Space and
    Enter tick and untick an option and close nothing; Left closes the
    submenu back onto its row; Enter on a row opens its submenu too; Escape
    closes the submenu onto its row and the next Escape the menu onto the
    button; Tab in the open menu closes it and moves on.
    """
    _seed_week(harness)
    page = windows.open().page
    _expect_listed(page, CHATS)
    menu = page.locator(MENU)
    _open_from_keyboard(page)

    walked = []
    for _ in SETTINGS:
        page.keyboard.press("ArrowDown")
        walked.append(_focus(page))
    assert walked == [f"row:{key}" for key in SETTINGS[1:]] + ["row:status"]
    page.keyboard.press("ArrowUp")
    _expect_focus(page, "row:group")
    page.keyboard.press("Home")
    _expect_focus(page, "row:status")
    page.keyboard.press("End")
    _expect_focus(page, "row:group")
    focused = page.locator(ROW("group")).evaluate(_FILL_JS)
    assert focused != page.locator(ROW("status")).evaluate(_FILL_JS), focused
    page.keyboard.press("Home")

    page.keyboard.press("ArrowRight")
    expect(page.locator(SUBMENU("status"))).to_be_visible()
    _expect_focus(page, "option:Running")
    page.keyboard.press("ArrowDown")
    _expect_focus(page, "option:Errored")
    page.keyboard.press("End")
    _expect_focus(page, "option:Succeeded")
    page.keyboard.press("ArrowDown")
    _expect_focus(page, "option:Running")
    page.keyboard.press("ArrowUp")
    _expect_focus(page, "option:Succeeded")
    page.keyboard.press("Home")
    _expect_focus(page, "option:Running")
    page.keyboard.press("Space")
    assert _ticked(page, "status") == ["Errored", "Succeeded"]
    page.keyboard.press("Enter")
    assert _ticked(page, "status") == ["Running", "Errored", "Succeeded"]
    assert _open_submenus(page) == ["status"]
    _expect_focus(page, "option:Running")

    page.keyboard.press("ArrowLeft")
    expect(page.locator(SUBMENU("status"))).to_be_hidden()
    expect(menu).to_be_visible()
    _expect_focus(page, "row:status")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    expect(page.locator(SUBMENU("favorites"))).to_be_visible()
    _expect_focus(page, "option:All")
    page.keyboard.press("Escape")
    expect(page.locator(SUBMENU("favorites"))).to_be_hidden()
    expect(menu).to_be_visible()
    _expect_focus(page, "row:favorites")
    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()
    _expect_focus(page, BUTTON)

    page.keyboard.press("Enter")
    expect(menu).to_be_visible()
    _expect_focus(page, "row:status")
    page.keyboard.press("Tab")
    expect(menu).to_be_hidden()
    assert not menu.evaluate(_HOLDS_FOCUS_JS)
    assert _focus(page) != "body"


def test_arrow_down_on_the_button_opens_the_menu(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """With the keyboard on the button, ArrowDown opens the menu on its first row, as Enter."""
    page = windows.open().page
    page.click("#history-search")
    page.keyboard.press("Tab")
    _expect_focus(page, BUTTON)
    page.keyboard.press("ArrowDown")
    expect(page.locator(MENU)).to_be_visible(timeout=2000)
    _expect_focus(page, "row:status")


def test_a_submenu_opened_from_the_keyboard_starts_on_its_first_option(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """Right on a row puts the keyboard on the submenu's first option, whatever is ticked."""
    page = windows.open().page
    _open_from_keyboard(page)
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowRight")
    _expect_focus(page, "option:All")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    assert _ticked(page, "favorites") == ["Only favorites"]
    page.keyboard.press("ArrowLeft")
    _expect_focus(page, "row:favorites")
    page.keyboard.press("ArrowRight")
    expect(page.locator(SUBMENU("favorites"))).to_be_visible()
    _expect_focus(page, "option:All")


_PICK_AND_PRESS_LEFT_JS = """() => {
  const option = document.activeElement;
  option.click();
  option.dispatchEvent(
    new KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true, cancelable: true}));
}"""


def test_a_submenu_closed_from_the_keyboard_stays_closed_under_a_resting_mouse(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """Left closes a submenu for good, also right after a pick and with the mouse on its row.

    The mouse rests on the value end of the Favorites row, which opened the
    submenu, and the keyboard is on "Only favorites".  The option is picked
    and Left is pressed in one task of the page, so the two are less than the
    hover wait apart however slow the machine is.  Checks: the submenu stays
    closed for the next two seconds and the keyboard is on the row.
    """
    page = windows.open().page
    page.click(BUTTON)
    row = _box(page, ROW("favorites"))
    page.mouse.move(row["right"] - 45, (row["top"] + row["bottom"]) / 2)
    expect(page.locator(SUBMENU("favorites"))).to_be_visible()
    _expect_focus(page, "option:All")
    page.keyboard.press("ArrowDown")
    _expect_focus(page, "option:Only favorites")
    page.evaluate(_PICK_AND_PRESS_LEFT_JS)
    assert _ticked(page, "favorites") == ["Only favorites"]
    _expect_submenus_stay(page, [])
    assert _focus(page) == "row:favorites"


def test_a_pick_from_the_keyboard_does_not_open_the_row_the_mouse_rests_on(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """A pick made with the keys keeps its submenu, wherever the mouse was left.

    The mouse rests on the value of the Status row, which opened that
    submenu.  The keys close it, walk to Favorites, open its submenu and pick
    "Only favorites".  Checks: for the next two seconds Favorites is the one
    open submenu, and the keyboard is still on the picked option.
    """
    page = windows.open().page
    page.click(BUTTON)
    value = _box(page, ROW("status") + " .fg-row-value")
    page.mouse.move((value["left"] + value["right"]) / 2, (value["top"] + value["bottom"]) / 2)
    expect(page.locator(SUBMENU("status"))).to_be_visible()
    _expect_focus(page, "option:Running")
    page.keyboard.press("ArrowLeft")
    _expect_focus(page, "row:status")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowRight")
    _expect_focus(page, "option:All")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    assert _ticked(page, "favorites") == ["Only favorites"]
    _expect_submenus_stay(page, ["favorites"])
    assert _focus(page) == "option:Only favorites"


def test_clear_dates_keeps_the_keyboard_in_the_date_submenu(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """Enter on Clear dates clears the dates and leaves the keyboard inside the Date submenu."""
    page = windows.open().page
    page.click(BUTTON)
    page.click(ROW("date"))
    _expect_focus(page, "#hf-from")
    page.keyboard.type((_today() - timedelta(days=10)).strftime("%m%d%Y"))
    expect(page.locator("#hf-date-clear")).to_be_visible()
    for _ in range(12):
        if _focus(page) == "#hf-date-clear":
            break
        page.keyboard.press("Tab")
    assert _focus(page) == "#hf-date-clear"
    page.keyboard.press("Enter")
    expect(page.locator("#hf-from")).to_have_value("")
    assert _open_submenus(page) == ["date"]
    assert page.locator(SUBMENU("date")).evaluate(_HOLDS_FOCUS_JS)


_LOOK_JS = """() => {
  const style = (el, pseudo) => getComputedStyle(el, pseudo);
  const box = el => {
    const r = el.getBoundingClientRect();
    return {left: r.left, top: r.top, right: r.right, bottom: r.bottom};
  };
  const card = el => ({
    radius: parseFloat(style(el).borderTopLeftRadius),
    shadow: style(el).boxShadow,
    background: style(el).backgroundColor,
    border: parseFloat(style(el).borderTopWidth),
  });
  const tick = option => ({
    mask: style(option, '::after').maskImage,
    fill: style(option, '::after').backgroundColor,
    ink: style(option).color,
    size: parseFloat(style(option, '::after').width),
  });
  const menu = document.getElementById('history-filter-menu');
  const sub = document.getElementById('fg-favorites');
  const row = menu.querySelector('.fg-row[data-fg="favorites"]');
  const name = row.querySelector('.fg-row-name');
  const value = row.querySelector('.fg-row-value');
  const chevron = style(row, '::after');
  const line = menu.querySelector('.fg-sep');
  return {
    cards: [card(menu), card(sub)],
    row: box(row), name: box(name), value: box(value),
    nameInk: style(name).color, valueInk: style(value).color,
    chevron: {
      size: parseFloat(chevron.width), rotate: chevron.rotate,
      top: parseFloat(chevron.borderTopWidth), right: parseFloat(chevron.borderRightWidth),
      bottom: parseFloat(chevron.borderBottomWidth), left: parseFloat(chevron.borderLeftWidth),
    },
    openRow: style(row).backgroundColor,
    otherRow: style(menu.querySelector('.fg-row[data-fg="status"]')).backgroundColor,
    chosen: tick(sub.querySelector('.fg-option[aria-checked="true"]')),
    other: tick(sub.querySelector('.fg-option[aria-checked="false"]')),
    line: {box: box(line), width: parseFloat(style(line).borderTopWidth)},
    above: box(menu.querySelector('.fg-row[data-fg="workdir"]')),
    below: box(menu.querySelector('.fg-row[data-fg="group"]')),
    dot: {
      content: style(document.getElementById('history-filter-btn'), '::after').content,
      fill: style(document.getElementById('history-filter-btn'), '::after').backgroundColor,
    },
  };
}"""

CLEAR = "rgba(0, 0, 0, 0)"


def _channels(colour: str) -> list[float]:
    """The red, green, blue and alpha of a computed ``rgb()`` / ``rgba()`` colour."""
    numbers = [float(number) for number in re.findall(r"[\d.]+", colour)]
    return numbers if len(numbers) == 4 else [*numbers, 1.0]


def _luminance(colour: str) -> float:
    """The WCAG relative luminance of an opaque computed colour."""
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        for c in (channel / 255 for channel in _channels(colour)[:3])
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(ink: str, paper: str) -> float:
    """The WCAG contrast ratio of the text colour *ink* on the background *paper*."""
    lighter, darker = sorted((_luminance(ink), _luminance(paper)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_the_menu_is_a_card_of_named_values_with_a_tick_on_the_choice(
    harness: _ServerHarness, windows: _Windows, dark: bool,
) -> None:
    """In the light and the dark theme the menu looks like a Filter and group menu.

    Favorites is set to "Only favorites" and its submenu is open.  Checks:
    the menu and the submenu are opaque cards with a border, corners rounded
    by 8px or more and a shadow; the row holds its name at the left and its
    value at the right in a second, still readable colour, ends in a chevron
    (a 6px box with a top and a right border, turned 45 degrees) and is
    filled while its submenu is open; the chosen option carries a tick in
    the text colour and the other option none; a hairline separates Work dir
    from Group by; the button wears its dot.
    """
    _seed_week(harness)
    page = windows.open().page
    _expect_listed(page, CHATS)
    if dark:
        page.click("#theme-btn")
    expect(page.locator("body")).to_have_class(
        re.compile(r"^(?!.*\blight-theme\b)" if dark else r"\blight-theme\b"),
    )
    page.click(BUTTON)
    page.click(ROW("favorites"))
    _option(page, "favorites", "Only favorites").click()
    _expect_listed(page, ["shop"])
    look = page.evaluate(_LOOK_JS)

    for card in look["cards"]:
        assert card["radius"] >= 8, card
        assert card["shadow"] != "none", card
        assert card["border"] == 1, card
        assert _channels(card["background"])[3] == 1, card
    paper = look["cards"][0]["background"]
    assert look["cards"][1]["background"] == paper

    row, name, value = look["row"], look["name"], look["value"]
    assert row["left"] < name["left"] < name["right"] <= value["left"], look
    assert value["left"] < value["right"] < row["right"], look
    assert look["valueInk"] != look["nameInk"], look
    assert _contrast(look["nameInk"], paper) >= 7, look
    assert _contrast(look["valueInk"], paper) >= 4.5, look
    assert look["chevron"] == {
        "size": 6, "rotate": "45deg",
        "top": pytest.approx(1.5, abs=0.5), "right": pytest.approx(1.5, abs=0.5),
        "bottom": 0, "left": 0,
    }
    assert look["openRow"] != CLEAR and look["otherRow"] == CLEAR, look

    chosen, other = look["chosen"], look["other"]
    assert chosen["mask"].startswith("url("), chosen
    assert chosen["fill"] == chosen["ink"] and chosen["size"] == 12, chosen
    assert other["fill"] == CLEAR, other

    line = look["line"]
    assert line["width"] == 1, line
    assert look["above"]["bottom"] <= line["box"]["top"], look
    assert line["box"]["bottom"] <= look["below"]["top"], look
    assert look["dot"]["content"] == '""' and look["dot"]["fill"] != CLEAR, look


def test_on_a_phone_the_menu_and_every_submenu_stay_inside_the_screen(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """On a 390 x 844 touch screen the menu and each submenu open whole, and close alone.

    A submenu has no room beside the menu there: it opens under its row, over
    the rows below, so the rows are tapped from the last to the first.
    Checks: a tap on the button opens the menu inside the screen; a tap on
    each row opens that row's submenu and no other, inside the screen; the
    long Tag list scrolls to its last option, whose tap ticks it and filters
    the list through the daemon; Escape closes the submenu, the next Escape
    the menu, and the Chats drawer under them stays open; a tap elsewhere in
    the drawer closes the menu and keeps the drawer too.
    """
    _seed_week(harness)
    page = windows.open(phone=True).page
    _show_chats(page)
    _expect_listed(page, CHATS)
    menu = page.locator(MENU)
    drawer = page.locator("#sidebar")
    page.tap(BUTTON)
    expect(menu).to_be_visible()
    assert _off_screen(page, MENU) == {}

    for key in reversed(SETTINGS):
        page.tap(ROW(key))
        expect(page.locator(SUBMENU(key))).to_be_visible()
        assert _open_submenus(page) == [key]
        assert _off_screen(page, SUBMENU(key)) == {}, key

    page.tap(ROW("tag"))
    last = _option(page, "tag", "failed")
    last.tap()
    expect(last).to_have_attribute("aria-checked", "true")
    assert _off_screen(page, SUBMENU("tag")) == {}
    _expect_listed(page, ["deploy"])

    page.keyboard.press("Escape")
    expect(page.locator(SUBMENU("tag"))).to_be_hidden()
    expect(menu).to_be_visible()
    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()
    expect(drawer).to_have_class(re.compile(r"\bopen\b"))

    page.tap(BUTTON)
    expect(menu).to_be_visible()
    page.touchscreen.tap(60, 600)
    expect(menu).to_be_hidden()
    expect(drawer).to_have_class(re.compile(r"\bopen\b"))


_HEIGHTS_JS = """menu => [...menu.querySelectorAll(
  ':scope > .fg-row, :scope > .fg-option, .fg-clear')]
  .filter(button => button.getClientRects().length > 0)
  .map(button => button.getBoundingClientRect().height)"""


def test_on_a_phone_rows_and_options_are_as_tall_as_the_more_menus_rows(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """On a phone every row and option is at least as tall as a row of the More menu.

    The More menu's rows get a touch-target height on a phone
    (``remote-codex.css``).  Checks: every row of the menu, every option of
    every submenu and Clear dates are at least that tall.
    """
    _seed_week(harness)
    page = windows.open(phone=True).page
    _show_chats(page)
    _expect_listed(page, CHATS)
    target = page.locator(".more-menu-item").first.evaluate(
        "item => parseFloat(getComputedStyle(item).minHeight)",
    )
    assert target >= 30, target
    page.tap(BUTTON)
    heights = {"rows": page.locator(MENU).evaluate(_HEIGHTS_JS)}
    for key in reversed(SETTINGS):
        page.tap(ROW(key))
        expect(page.locator(SUBMENU(key))).to_be_visible()
        heights[key] = page.locator(SUBMENU(key)).evaluate(_HEIGHTS_JS)
    short = {key: min(found) for key, found in heights.items() if min(found) < target}
    assert short == {}, (target, short)


def _add_work_dir(harness: _ServerHarness, *parts: str) -> Path:
    """Make a folder under the daemon's temp dir and adopt it as a recent work dir."""
    folder = Path(harness.tmpdir).joinpath(*parts)
    folder.mkdir(parents=True)
    user_config.record_recent_work_dir(str(folder))
    return folder


def _tick_box(page: Page, key: str) -> dict[str, float]:
    """The box of the tick of the chosen option of the setting *key*, in window coordinates.

    The tick is the option's ``::after``, which only the DevTools protocol
    can measure.
    """
    chosen = SUBMENU(key) + ' > .fg-option[aria-checked="true"]'
    cdp = page.context.new_cdp_session(page)
    try:
        found = cdp.send(
            "Runtime.evaluate", {"expression": f"document.querySelector({json.dumps(chosen)})"},
        )
        node = cdp.send("DOM.describeNode", {"objectId": found["result"]["objectId"]})["node"]
        after = [part for part in node["pseudoElements"] if part["pseudoType"] == "after"]
        model = cdp.send("DOM.getBoxModel", {"backendNodeId": after[0]["backendNodeId"]})["model"]
    finally:
        cdp.detach()
    xs, ys = model["border"][0::2], model["border"][1::2]
    return {"left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys)}


_NAME_SHOWN_JS = """(option, name) => {
  const card = option.closest('.fg-submenu').getBoundingClientRect();
  const inside = r => r.left >= card.left && r.right <= card.right &&
    r.top >= card.top && r.bottom <= card.bottom;
  const texts = document.createTreeWalker(option, NodeFilter.SHOW_TEXT);
  for (let node = texts.nextNode(); node; node = texts.nextNode()) {
    for (let at = node.data.indexOf(name); at >= 0; at = node.data.indexOf(name, at + 1)) {
      const range = document.createRange();
      range.setStart(node, at);
      range.setEnd(node, at + name.length);
      const rects = [...range.getClientRects()];
      if (rects.length > 0 && rects.every(inside)) return true;
    }
  }
  return false;
}"""


def test_on_a_phone_a_long_work_dir_option_shows_its_folder_and_its_tick(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """On a phone two folders with a long common path can be told apart and the choice seen.

    Two recent folders share a path longer than the screen is wide and
    differ in their last segment, the folder's own name.  Checks, with one
    of them chosen: each option shows its folder's name inside the submenu,
    and the tick of the chosen option is inside the submenu too.
    """
    _seed_week(harness)
    shared = ("projects", "customer-integrations", "production-services", "billing-platform")
    folders = [_add_work_dir(harness, *shared, name) for name in ("alpha", "beta")]
    page = windows.open(phone=True).page
    _show_chats(page)
    _expect_listed(page, CHATS)
    page.tap(BUTTON)
    page.tap(ROW("workdir"))
    chosen = _option(page, "workdir", str(folders[1]))
    chosen.tap()
    expect(chosen).to_have_attribute("aria-checked", "true")
    card = _box(page, SUBMENU("workdir"))
    named = {
        folder.name: _option(page, "workdir", str(folder)).evaluate(_NAME_SHOWN_JS, folder.name)
        for folder in folders
    }
    tick = _tick_box(page, "workdir")
    assert named == {"alpha": True, "beta": True}, named
    assert card["left"] <= tick["left"] and tick["right"] <= card["right"], (card, tick)


def test_on_a_phone_the_menu_stays_inside_the_screen_when_a_value_grows(
    harness: _ServerHarness, windows: _Windows,
) -> None:
    """On a phone the open menu is still inside the screen after a pick widens a row.

    A recent folder has a 40-character name, which the Work dir row shows
    once the folder is chosen.  Checks: after the pick, with the submenu
    closed again, no side of the menu is outside the screen.
    """
    _seed_week(harness)
    chosen = _add_work_dir(harness, "customer-integration-production-services")
    page = windows.open(phone=True).page
    _show_chats(page)
    _expect_listed(page, CHATS)
    page.tap(BUTTON)
    assert _off_screen(page, MENU) == {}
    page.tap(ROW("workdir"))
    _option(page, "workdir", str(chosen)).tap()
    expect(_value(page, "workdir")).not_to_have_text("All")
    page.keyboard.press("Escape")
    expect(page.locator(SUBMENU("workdir"))).to_be_hidden()
    assert _off_screen(page, MENU) == {}

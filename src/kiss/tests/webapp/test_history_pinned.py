# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pins in the Chats sidebar, against a real daemon in a real Chromium.

A task is pinned from the "…" menu of its row, a chat from the menu of
its header and a project from the menu of its row (``Pin task`` /
``Pin chat`` / ``Pin project``).  What is pinned is listed itself under
a "Pinned" separator ahead of the days -- the task's row on its own,
the chat's panel, the project's panel -- so the menu of every item
listed there reads ``Unpin ...``.  The daemon keeps the pins in its
history database (``setPinned`` / ``setChatPinned`` /
``setProjectPinned``), so a daemon that was restarted, a browser that
has nothing cached and a reloaded page list them there too; unpinned
from its menu, an item is back where it was.

Real daemon (``RemoteAccessServer``) with its own history database,
real Chromium (Playwright), at a desktop's size and on a phone.  What
a press does before the daemon answers (the move at once, a refetch
that predates the write, the filters) is tested in jsdom:
``webapp/test/historyPinned.test.js`` and, for the chats of a project,
``webapp/test/historyProjects.test.js``.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, Locator, Page, expect, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from kiss.agents.sorcar import persistence
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.persistence_helpers import close_db
from kiss.tests.server.test_content_tab_file_links import _ServerHarness
from kiss.tests.webapp.test_content_tab_file_links import _wait_ready

pytestmark = pytest.mark.timeout(240)

_TIMEOUT = 30000
_DAY = 86400.0
_FOLD = ":scope > .sidebar-item-collapse"
_MORE = ":scope > .sidebar-item-more"
_PINNED_SEP = '#history-list .history-day-sep[data-day="pinned"]'

# What the Chats list shows from top to bottom: its separators and,
# for every item of the list, what it is, the tasks it lists and, in
# brackets, what the pin line of its own menu reads -- a task's row
# (``task:``), a chat's panel (``chat:``) or a project's (``project:``,
# followed by what its panel lists when it is unfolded).  Anything
# else (the list's "Loading..." line) reads ``other:``.
_LIST_JS = """() => {
  const shown = el => el.style.display !== 'none';
  const tasks = el => Array.from(el.querySelectorAll('.sidebar-item-text'),
    text => text.textContent).join(',');
  const pin = (el, button) => ' (' + el.querySelector(button).getAttribute('aria-label') + ')';
  const item = el => {
    if (el.classList.contains('history-day-sep')) return 'sep:' + el.textContent;
    if (el.classList.contains('sidebar-item')) {
      return 'task:' + tasks(el) + pin(el, '.sidebar-item-pin');
    }
    if (el.classList.contains('history-chat-group')) {
      return 'chat:' + tasks(el) + pin(el, ':scope > .history-chat-header .history-chat-pin');
    }
    if (!el.classList.contains('history-project')) return 'other:' + el.textContent;
    const name = el.querySelector('.history-project-title').textContent
      + pin(el, ':scope > .history-project-header .history-project-pin');
    if (el.classList.contains('collapsed')) return 'project:' + name;
    const body = el.querySelector(':scope > .history-project-body');
    const note = body.querySelector(':scope > .history-project-note');
    const chats = Array.from(body.querySelectorAll(':scope > .history-chat-group')).filter(shown);
    return 'project:' + name + '{' + (getComputedStyle(note).display === 'none'
      ? chats.map(item).join(' ') : note.textContent) + '}';
  };
  return Array.from(document.getElementById('history-list').children).filter(shown).map(item);
}"""

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


def _seed_project(name: str, created: float) -> str:
    """Persist a project created at *created* (a project is listed by
    its newest task, or by its own age when that is later); returns
    its id."""
    project_id = persistence._save_project("", name, "", None)
    with persistence._rw_lock.write_lock():
        persistence._get_db().execute(
            "UPDATE projects SET created_at = ? WHERE id = ?", (created, project_id)
        )
    return project_id


async def _restarted(harness: _ServerHarness) -> dict:
    """Stop *harness*'s daemon and start another on the same port and files."""
    await harness.server.stop_async()
    # The process's database connections go with the daemon: the next
    # one reads the pins from the file.
    close_db()
    tmp = Path(harness.tmpdir)
    harness.server = RemoteAccessServer(
        host="127.0.0.1",
        port=harness.port,
        certfile=str(tmp / "cert.pem"),
        keyfile=str(tmp / "key.pem"),
        url_file=tmp / "remote-url.json",
        local_endpoint_file=tmp / "sorcar-local.json",
        work_dir=str(harness.work_dir),
    )
    await harness.server.start_async()
    return {}


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


def _project(page: Page, project_id: str) -> Locator:
    """The header of the panel of project *project_id*."""
    return page.locator(
        f'#history-list .history-project[data-project-id="{project_id}"] > .history-project-header'
    )


def _press_pin(row: Locator, reads: str) -> None:
    """Open the "…" menu of *row* -- a task's panel, a chat's header or
    a project's -- and press its pin line, which is on screen and
    reads *reads* (a line shows its button's ``aria-label``)."""
    row.locator(_MORE).click()
    pin = row.locator(":scope > .sidebar-item-actions .sidebar-item-pin")
    expect(pin).to_be_visible()
    expect(pin).to_have_attribute("aria-label", reads)
    pin.click()


def test_pinned_tasks_and_chats_are_listed_and_unpinned_across_a_restart(
    browser, harness
) -> None:
    """A chat pinned from its header's menu and a task pinned from its
    row's menu are each listed under "Pinned", newest first, ahead of
    the days: the chat's panel, and the task's row on its own, out of
    its chat's panel (a chat left without a task is not listed, nor is
    a day left without a chat).  The menu of each reads "Unpin ...",
    and the task's pin does not pin its chat, nor the chat's pin its
    tasks.  After a restart of the daemon a browser with nothing
    cached and the reloaded page show the same list and the same
    menus.  Unpinned from its menu, each item is back under its day,
    for another browser too.  The keyboard stays with what a press
    moved: on the "…" button of the row, or on the header of the
    row's chat when the row went back into a folded one."""
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
            "chat:alpha two,alpha one (Pin chat)",
            "sep:Yesterday",
            "chat:beta one (Pin chat)",
            f"sep:{old_day}",
            "chat:gamma one (Pin chat)",
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

        _press_pin(_header(page, chat_c), "Pin chat")
        _expect_list(
            page,
            [
                "sep:Pinned",
                "chat:gamma one (Unpin chat)",
                "sep:Today",
                "chat:alpha two,alpha one (Pin chat)",
                "sep:Yesterday",
                "chat:beta one (Pin chat)",
            ],
        )
        # A task of a chat with another task, then the one task of a chat.
        _header(page, chat_a).locator(_FOLD).click()
        _press_pin(_task(page, "alpha one"), "Pin task")
        _header(page, chat_b).locator(_FOLD).click()
        _press_pin(_task(page, "beta one"), "Pin task")
        # The keyboard is on the "…" button of the row that moved: out
        # of its chat and, unpinned, back in it (the chat, left without
        # a task, was not listed meanwhile).
        beta_more = _task(page, "beta one").locator(_MORE)
        expect(beta_more).to_be_focused()
        _press_pin(_task(page, "beta one"), "Unpin task")
        expect(_header(page, chat_b)).to_be_visible()
        expect(beta_more).to_be_focused()
        _press_pin(_task(page, "beta one"), "Pin task")
        pinned = [
            "sep:Pinned",
            "task:alpha one (Unpin task)",
            "task:beta one (Unpin task)",
            "chat:gamma one (Unpin chat)",
            "sep:Today",
            "chat:alpha two (Pin chat)",
        ]
        _expect_list(page, pinned)
        expect(page.locator(_PINNED_SEP)).to_be_visible()
        expect(_task(page, "gamma one").locator(".sidebar-item-pin")).to_have_attribute(
            "aria-label", "Pin task"
        )

        other_context, other = _open(browser, harness, **desktop)
        try:
            _expect_list(other, pinned)
        finally:
            other_context.close()

        harness.run(_restarted(harness))
        other_context, other = _open(browser, harness, **desktop)
        try:
            _expect_list(other, pinned)
        finally:
            other_context.close()
        page.reload()
        _wait_ready(page)
        _expect_list(page, pinned)

        # Every chat is folded again: what is pinned is in sight, and
        # its menu unpins it.
        _press_pin(_task(page, "alpha one"), "Unpin task")
        _expect_list(
            page,
            [
                "sep:Pinned",
                "task:beta one (Unpin task)",
                "chat:gamma one (Unpin chat)",
                "sep:Today",
                "chat:alpha two,alpha one (Pin chat)",
            ],
        )
        # The row is out of sight in its folded chat, whose header has
        # the keyboard: a chat that was listed, and one that was not.
        expect(_header(page, chat_a)).to_be_focused()
        _press_pin(_header(page, chat_c), "Unpin chat")
        _press_pin(_task(page, "beta one"), "Unpin task")
        _expect_list(page, by_day)
        expect(_header(page, chat_b)).to_be_focused()
        assert page.locator(_PINNED_SEP).count() == 0

        other_context, other = _open(browser, harness, **desktop)
        try:
            _expect_list(other, by_day)
            assert set(other.evaluate(_PINS_JS)) == {"Pin chat:false", "Pin task:false"}
        finally:
            other_context.close()
    finally:
        context.close()


def test_a_project_and_what_is_pinned_in_one_are_listed_and_unpinned_across_a_restart(
    browser, harness
) -> None:
    """A project pinned from the menu of its row is listed under
    "Pinned" and its menu reads "Unpin project".  A pinned chat of a
    project leaves the project's panel for "Pinned", and so does the
    row of a pinned task of a project's chat; an unfolded project left
    with nothing to list says where its chats are.  After a restart of
    the daemon a browser with nothing cached and the reloaded page
    list all three under "Pinned", every panel folded.  Unpinned from
    its menu, the project is back under "Projects", the chat under its
    project and the task in its chat; the header of the folded project
    that took either back has the keyboard."""
    today = _noon_today()
    paper = _seed_project("Paper", today - 30 * _DAY)
    site = _seed_project("Site", today - 20 * _DAY)
    chat_d = _seed("delta one", "", today - 2 * _DAY - 600)
    _seed("delta two", chat_d, today - 2 * _DAY)
    chat_e = _seed("epsilon one", "", today - 600)
    _seed("epsilon two", chat_e, today)
    _seed("zeta one", "", today - _DAY)
    persistence._set_chat_project(chat_d, paper)
    persistence._set_chat_project(chat_e, site)
    desktop = {"viewport": {"width": 1280, "height": 800}}
    context, page = _open(browser, harness, **desktop)
    try:
        # A project is listed by its newest task; both are folded.
        days = ["sep:Yesterday", "chat:zeta one (Pin chat)"]
        unpinned = ["sep:Projects", "project:Site (Pin project)", "project:Paper (Pin project)"]
        _expect_list(page, unpinned + days)

        _press_pin(_project(page, paper), "Pin project")
        _expect_list(
            page,
            [
                "sep:Pinned",
                "project:Paper (Unpin project)",
                "sep:Projects",
                "project:Site (Pin project)",
            ]
            + days,
        )
        # The one chat of Site, then a task of Paper's chat.
        _project(page, site).locator(_FOLD).click()
        _press_pin(_header(page, chat_e), "Pin chat")
        _project(page, paper).locator(_FOLD).click()
        _header(page, chat_d).locator(_FOLD).click()
        _press_pin(_task(page, "delta one"), "Pin task")
        _expect_list(
            page,
            [
                "sep:Pinned",
                "chat:epsilon two,epsilon one (Unpin chat)",
                "project:Paper (Unpin project){chat:delta two (Pin chat)}",
                "task:delta one (Unpin task)",
                "sep:Projects",
                "project:Site (Pin project){Listed under Pinned}",
            ]
            + days,
        )

        pinned = [
            "sep:Pinned",
            "chat:epsilon two,epsilon one (Unpin chat)",
            "project:Paper (Unpin project)",
            "task:delta one (Unpin task)",
            "sep:Projects",
            "project:Site (Pin project)",
        ] + days
        harness.run(_restarted(harness))
        other_context, other = _open(browser, harness, **desktop)
        try:
            _expect_list(other, pinned)
        finally:
            other_context.close()
        page.reload()
        _wait_ready(page)
        _expect_list(page, pinned)

        _press_pin(_project(page, paper), "Unpin project")
        _press_pin(_header(page, chat_e), "Unpin chat")
        expect(_project(page, site)).to_be_focused()
        _press_pin(_task(page, "delta one"), "Unpin task")
        expect(_project(page, paper)).to_be_focused()
        _expect_list(page, unpinned + days)
        assert page.locator(_PINNED_SEP).count() == 0
        # Unfolded, each project lists its chat with every task of it.
        _project(page, site).locator(_FOLD).click()
        _project(page, paper).locator(_FOLD).click()
        _expect_list(
            page,
            [
                "sep:Projects",
                "project:Site (Pin project){chat:epsilon two,epsilon one (Pin chat)}",
                "project:Paper (Pin project){chat:delta two,delta one (Pin chat)}",
            ]
            + days,
        )

        other_context, other = _open(browser, harness, **desktop)
        try:
            _expect_list(other, unpinned + days)
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
        _expect_list(
            page,
            [
                "sep:Today",
                "chat:alpha one (Pin chat)",
                "sep:Yesterday",
                "chat:beta one (Pin chat)",
            ],
        )
        _press_pin(_header(page, chat_b), "Pin chat")
        _expect_list(
            page,
            [
                "sep:Pinned",
                "chat:beta one (Unpin chat)",
                "sep:Today",
                "chat:alpha one (Pin chat)",
            ],
        )
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

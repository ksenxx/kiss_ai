# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
# test_explorer_scm_commands and re-exported here for pytest)
"""E2E tests: the chat header follows the machine a tab runs its tasks on.

The real remote webapp (an in-process ``RemoteAccessServer``) with a
real second daemon (:mod:`kiss.tests.remote_daemon`) registered as
machine ``box``, driven by headless Chromium:

* picking ``box`` in the machine drop-down lists box's working
  directory and the directories selected there so far, not this
  daemon's; picking this machine again lists this daemon's;
* opening one of box's chats from the CHATS panel switches the header
  to ``box`` and box's directories;
* picking one of box's directories makes it box's working directory
  (persisted there) and leaves this daemon's alone; "Choose ..." opens
  the folder picker on box's folders and a pick there is adopted by box;
* a machine registered behind the daemon's back (as ``/machine add``
  does) brings the Machines subpanel on screen once the daemon reports
  it;
* the phone layout keeps both drop-downs at the top of the chat.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from kiss.agents.sorcar import machines as mach
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.server.test_explorer_scm_commands import (
    ExplorerHarness,
    harness,  # noqa: F401  (module fixture)
)


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


class Box:
    """The remote daemon ``box``: its home, work dirs and seeded chats."""

    def __init__(self, harness: ExplorerHarness) -> None:
        root = Path(harness.tmpdir)
        self.work = root / "box-work"
        self.other = root / "box-other"
        (self.work / "sub-a").mkdir(parents=True)
        (self.work / "sub-b").mkdir()
        self.other.mkdir()
        self.home = root / "box-home"
        now = time.time()
        self.daemon: RemoteDaemon = start_remote_daemon(
            self.home, password="pw-box",
            seed=[
                {"chat_id": "box-chat-1", "task": "Box task one", "result": "done"},
                {"chat_id": "box-chat-2", "task": "Box task two", "result": "done"},
            ],
            config={
                "work_dir": str(self.work),
                "recent_work_dirs": [
                    {"path": str(self.work), "ts": now},
                    {"path": str(self.other), "ts": now - 60},
                ],
            },
        )
        mach.save_machines([])
        mach.add_machine("box", self.daemon.url, "pw-box")

    def work_dir(self) -> str:
        """The working directory box's daemon persisted."""
        config = json.loads((self.home / "config.json").read_text(encoding="utf-8"))
        return str(config.get("work_dir", ""))

    def stop(self) -> None:
        self.daemon.stop()
        mach.save_machines([])


@pytest.fixture(scope="module")
def box(harness: ExplorerHarness):
    b = Box(harness)
    yield b
    b.stop()


WORKDIR_OPTIONS = (
    "() => [...document.querySelectorAll('#chat-workdir option')].map(o => o.value)"
)


def _open(browser: Browser, harness: ExplorerHarness, width: int = 1280) -> Page:
    context = browser.new_context(
        ignore_https_errors=True, viewport={"width": width, "height": 900},
        service_workers="block", is_mobile=width < 700, has_touch=width < 700,
    )
    context.add_init_script("localStorage.setItem('kissTipsOptOut', '1');")
    page = context.new_page()
    page.goto(harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_function(
        "() => [...document.querySelectorAll('#chat-machine option')]"
        ".some(o => o.value === 'box')",
        timeout=20000,
    )
    return page


def _wait_workdirs(page: Page, expected: list[str], selected: str) -> None:
    page.wait_for_function(
        "([dirs, sel]) => JSON.stringify("
        "[...document.querySelectorAll('#chat-workdir option')].map(o => o.value)"
        ") === JSON.stringify(dirs) && document.getElementById('chat-workdir').value === sel",
        arg=[expected, selected],
        timeout=15000,
    )


def test_machine_pick_lists_the_machines_directories(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness)
    try:
        # This daemon's directories first (its working directory, then
        # "Choose ..."); the list arrives after the page, in its own message.
        page.wait_for_function(
            "(dir) => document.querySelector('#chat-workdir option')?.value === dir",
            arg=str(harness.work_dir),
            timeout=15000,
        )
        local = page.evaluate(WORKDIR_OPTIONS)
        assert local[0] == str(harness.work_dir) and local[-1] == ""
        page.select_option("#chat-machine", "box")
        _wait_workdirs(page, [str(box.work), str(box.other), ""], str(box.work))
        assert "box" in str(page.get_attribute("#chat-workdir", "title"))
        # Back to this machine: its own list again.
        page.select_option("#chat-machine", "")
        _wait_workdirs(page, local, str(harness.work_dir))
    finally:
        page.context.close()


def test_remote_chat_from_chats_switches_machine_and_dirs(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness)
    try:
        group = page.locator(
            "#history-list .history-chat-group",
            has=page.locator(".sidebar-item[data-machine='box']"),
        ).first
        group.locator(".history-chat-header").click(timeout=20000)
        page.wait_for_function(
            "() => document.getElementById('chat-machine').value === 'box'", timeout=20000,
        )
        _wait_workdirs(page, [str(box.work), str(box.other), ""], str(box.work))
    finally:
        page.context.close()


def test_pick_on_the_machine_adopts_there_and_leaves_this_daemon_alone(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness)
    try:
        page.select_option("#chat-machine", "box")
        _wait_workdirs(page, [str(box.work), str(box.other), ""], str(box.work))
        page.select_option("#chat-workdir", str(box.other))
        _wait_workdirs(page, [str(box.other), str(box.work), ""], str(box.other))
        assert box.work_dir() == str(box.other)
        assert harness.server._agent_server.work_dir == str(harness.work_dir)
        # "Choose ..." browses box's folders; the pick is adopted by box.
        page.evaluate(
            "() => { const s = document.getElementById('chat-workdir');"
            " s.selectedIndex = [...s.options].findIndex(o => o.id === 'chat-workdir-choose');"
            " s.dispatchEvent(new Event('change', {bubbles: true})); }"
        )
        page.wait_for_selector("#folder-picker:not([hidden])", timeout=10000)
        assert "on box" in str(page.text_content("#folder-picker-title"))
        page.fill("#folder-picker .folder-picker-input", str(box.work))
        page.press("#folder-picker .folder-picker-input", "Enter")
        page.wait_for_function(
            "(sub) => [...document.querySelectorAll('#folder-picker .folder-picker-item')]"
            ".some(i => i.dataset.path === sub)",
            arg=str(box.work / "sub-a"),
            timeout=15000,
        )
        sub = str(box.work / "sub-a")
        # A backslash is an escape in a CSS string: double it so a Windows
        # path matches the attribute the same way the forward-slash one does.
        css_path = sub.replace("\\", "\\\\")
        page.click(f"#folder-picker .folder-picker-item[data-path='{css_path}']")
        page.click("#folder-picker .folder-picker-select")
        _wait_workdirs(page, [sub, str(box.other), str(box.work), ""], sub)
        page.wait_for_selector("#folder-picker", state="hidden", timeout=10000)
        assert box.work_dir() == sub
        assert harness.server._agent_server.work_dir == str(harness.work_dir)
        # This daemon's own list is untouched.
        page.select_option("#chat-machine", "")
        local = page.evaluate(WORKDIR_OPTIONS)
        assert local[0] == str(harness.work_dir) and sub not in local
    finally:
        page.context.close()


def test_new_machine_reveals_the_machines_section(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness)
    try:
        # The Machines subpanel lists the registry of the machine the
        # drop-down shows, and an earlier test may leave one of box's
        # chats open: show this daemon's.
        page.select_option("#chat-machine", "")
        collapsed = (
            "() => document.getElementById('meta-machines').classList.contains('collapsed')"
        )
        page.wait_for_selector(
            "#meta-machines .machine-row[data-machine='box']", state="attached", timeout=15000,
        )
        # Folded (a fresh page folds it; otherwise the chevron does).
        if not page.evaluate(collapsed):
            page.click("#meta-machines .meta-section-toggle")
        assert page.evaluate(collapsed)
        # Registered behind the daemon's back, as `/machine add` does.
        mach.add_machine("box2", box.daemon.url, "pw-box")
        try:
            page.click("#meta-machines-refresh")
            # Visible: the section unfolded to show the new row.
            page.wait_for_selector(
                "#meta-machines .machine-row[data-machine='box2']", timeout=15000,
            )
            assert not page.evaluate(collapsed)
        finally:
            mach.remove_machine("box2")
    finally:
        page.context.close()


def test_phone_layout_keeps_machine_and_workdir_at_the_top(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness, width=390)
    try:
        page.select_option("#chat-machine", "box")
        # Box's current directory (whatever an earlier test picked there).
        page.wait_for_function(
            "(dir) => document.getElementById('chat-workdir').value === dir",
            arg=box.work_dir(),
            timeout=15000,
        )
        rects = page.evaluate(
            "() => ['chat-header', 'chat-machine', 'chat-workdir'].map(id => {"
            " const r = document.getElementById(id).getBoundingClientRect();"
            " return [r.x, r.y, r.width, r.height]; })"
        )
        header, machine, workdir = rects
        assert header[1] == 0 and header[2] == 390 and header[3] > 0
        for x, y, w, h in (machine, workdir):
            assert w > 0 and h > 0 and y >= 0 and y + h <= header[3]
            assert x >= 0 and x + w <= 390
        # The machine select is as wide as its widest option (the local
        # machine's name among them): the directory shares its row when
        # that leaves it at least half of the row, else it moves whole
        # onto a second line (main.css, ``@media (width < 900px)``).
        same_row = machine[0] + machine[2] <= workdir[0]
        second_line = workdir[1] >= machine[1] + machine[3]
        assert same_row or second_line, rects
    finally:
        page.context.close()

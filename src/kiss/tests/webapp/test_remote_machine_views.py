# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
# test_explorer_scm_commands and named as a parameter below)
"""E2E tests: the Explorer and Source Control follow the machine a tab runs on.

The real remote webapp (an in-process ``RemoteAccessServer`` whose
working directory is a git repository) with a real second daemon
(:mod:`kiss.tests.remote_daemon`) registered as machine ``box``, whose
working directory is another repository, driven by headless Chromium:

* picking ``box`` in the machine drop-down rebuilds the Explorer on
  box's working directory (its files, not this daemon's; the folder
  added to this daemon's Explorer is left out) and the Source Control
  view on box's repository (its branch and commit);
* a file clicked in box's Explorer opens with box's content, and a
  save writes it on box;
* picking this machine again brings this daemon's folders back.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from kiss.agents.sorcar import machines as mach
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.server.test_explorer_scm_commands import (
    ExplorerHarness,
    _git,
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
    """The remote daemon ``box`` whose working directory is a repository."""

    def __init__(self, harness: ExplorerHarness) -> None:
        root = Path(harness.tmpdir)
        self.work = root / "box-work"
        self.work.mkdir()
        _git(self.work, "init", "-q", "-b", "box-main")
        (self.work / "box-notes.txt").write_text("notes kept on the box\n", newline="\n")
        (self.work / "box-dir").mkdir()
        (self.work / "box-dir" / "inner.txt").write_text("inner\n", newline="\n")
        _git(self.work, "add", "-A")
        _git(self.work, "commit", "-q", "-m", "box: the only commit")
        self.home = root / "box-home"
        self.daemon: RemoteDaemon = start_remote_daemon(
            self.home, password="pw-box", config={"work_dir": str(self.work)},
        )
        mach.save_machines([])
        mach.add_machine("box", self.daemon.url, "pw-box")

    def stop(self) -> None:
        self.daemon.stop()
        mach.save_machines([])


@pytest.fixture(scope="module")
def box(harness: ExplorerHarness):
    b = Box(harness)
    yield b
    b.stop()


ROOT_PATHS = (
    "() => [...document.querySelectorAll('#explorer-tree > .explorer-row.is-root')]"
    ".map(r => r.dataset.explorerPath)"
)
FILE_NAMES = (
    "() => [...document.querySelectorAll('#explorer-tree .explorer-row.is-file')]"
    ".map(r => r.textContent.trim())"
)


def _open(browser: Browser, harness: ExplorerHarness) -> Page:
    context = browser.new_context(
        ignore_https_errors=True, viewport={"width": 1280, "height": 900},
        service_workers="block",
    )
    context.add_init_script(
        "localStorage.setItem('kissTipsOptOut', '1');"
        # A folder added to this daemon's Explorer earlier.
        "localStorage.setItem('kiss-explorer-roots', JSON.stringify(['"
        + str(harness.plain_dir).replace("\\", "\\\\") + "']));",
    )
    page = context.new_page()
    page.goto(harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_function(
        "() => [...document.querySelectorAll('#chat-machine option')]"
        ".some(o => o.value === 'box')",
        timeout=20000,
    )
    # The task-info panel (Explorer and Source Control) may have slid
    # behind its drawer on the desktop layout: bring it back.
    if page.evaluate("document.body.classList.contains('meta-hidden')"):
        page.click("#meta-drawer")
        page.wait_for_function(
            "() => !document.body.classList.contains('meta-hidden')", timeout=5000,
        )
    _show(page, "meta-explorer")
    page.wait_for_selector(".explorer-row.is-file", state="attached", timeout=15000)
    return page


def _show(page: Page, section: str) -> None:
    """Expand *section* (the panel shows one section at a time)."""
    if page.locator(f"#{section}.collapsed").count():
        page.click(f"#{section} .meta-section-toggle")
    page.wait_for_selector(f"#{section}:not(.collapsed)", state="attached", timeout=5000)


def _wait_roots(page: Page, roots: list[str]) -> None:
    page.wait_for_function(
        "(roots) => JSON.stringify((" + ROOT_PATHS + ")()) === JSON.stringify(roots)",
        arg=roots, timeout=15000,
    )


def _wait_branch(page: Page, branch: str) -> None:
    page.wait_for_function(
        "(b) => document.getElementById('scm-branch').textContent === b",
        arg=branch, timeout=15000,
    )


def test_machine_pick_shows_the_machines_explorer_and_source_control(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness)
    try:
        assert page.evaluate(ROOT_PATHS) == [str(harness.work_dir), str(harness.plain_dir)]
        assert "README.md" in page.evaluate(FILE_NAMES)
        page.select_option("#chat-machine", "box")
        _wait_roots(page, [str(box.work)])
        page.wait_for_function(
            "() => (" + FILE_NAMES + ")().includes('box-notes.txt')", timeout=15000,
        )
        names = page.evaluate(FILE_NAMES)
        assert "README.md" not in names, names
        assert page.is_hidden("#explorer-add-folder")
        _show(page, "meta-scm")
        _wait_branch(page, "box-main")
        page.wait_for_function(
            "() => [...document.querySelectorAll('#scm-graph .scm-commit')]"
            ".some(c => c.textContent.includes('box: the only commit'))",
            timeout=15000,
        )
        # Home again: this daemon's repository and folders.
        page.select_option("#chat-machine", "")
        _wait_branch(page, "main")
        _show(page, "meta-explorer")
        _wait_roots(page, [str(harness.work_dir), str(harness.plain_dir)])
        page.wait_for_function(
            "() => (" + FILE_NAMES + ")().includes('README.md')", timeout=15000,
        )
        assert page.is_visible("#explorer-add-folder")
    finally:
        page.context.close()


def test_a_file_of_the_machine_opens_and_saves_there(
    browser: Browser, harness: ExplorerHarness, box: Box,
) -> None:
    page = _open(browser, harness)
    try:
        page.select_option("#chat-machine", "box")
        page.wait_for_function(
            "() => (" + FILE_NAMES + ")().includes('box-notes.txt')", timeout=15000,
        )
        page.locator(".explorer-row.is-file", has_text="box-notes.txt").click()
        page.wait_for_function(
            "() => window.monaco && monaco.editor.getModels()"
            ".some(m => m.getValue().includes('notes kept on the box'))",
            timeout=20000,
        )
        page.evaluate(
            "() => monaco.editor.getModels().find(m => m.getValue().includes('notes kept'))"
            ".setValue('notes edited from afar\\n')"
        )
        page.wait_for_selector(".content-save-btn:not([disabled])", timeout=10000)
        page.click(".content-save-btn")
        page.wait_for_function(
            "() => [...document.querySelectorAll('.content-save-status')]"
            ".some(e => e.textContent === 'Saved')",
            timeout=15000,
        )
        assert (box.work / "box-notes.txt").read_text() == "notes edited from afar\n"
        assert not (harness.work_dir / "box-notes.txt").exists()
    finally:
        page.context.close()

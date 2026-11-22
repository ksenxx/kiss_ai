# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the Create pull request button of the task-info strip, pressed in a real browser.

Headless Chromium loads the page from the production
``RemoteAccessServer`` + daemon of ``ExplorerHarness``, whose work dir
is a real git repository.  A press travels through the page's transport
to the daemon, which pushes to a bare repository on this disk, so
nothing reaches a Git host.  ``gh`` is logged out for every test (an
empty ``GH_CONFIG_DIR`` and no token in the environment), whatever
account the machine running the tests has.

The daemon's outcomes one by one are in
``tests/server/test_create_pr_action.py`` and the page's handling of
each daemon event in ``webapp/test/createPrButton.test.js``; this module
checks that the page and the daemon meet.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Locator, Page, expect, sync_playwright

from kiss.agents.sorcar.git_worktree import repo_lock
from kiss.tests.server.test_create_pr_action import GH_LOGIN_ENV
from kiss.tests.server.test_explorer_scm_commands import ExplorerHarness, _git
from kiss.tests.webapp.test_remote_terminal_tab import _open_page
from kiss.tests.webapp.test_sidebar_menus import _show_panel

_OUTCOME_TOAST = '.kiss-notification[data-notification-id^="create-pr:"]'
_BUSY = re.compile(r"\bis-busy\b")


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
        yield b
        b.close()


@pytest.fixture
def daemon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[ExplorerHarness]:
    """A real daemon whose work dir is a real repository, with ``gh`` logged out."""
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "no-gh-login"))
    for name in GH_LOGIN_ENV:
        monkeypatch.delenv(name, raising=False)
    harness = ExplorerHarness()
    yield harness
    harness.stop()


def _strip_button(page: Page) -> Locator:
    """The Create pull request button, with the task-info panel on screen."""
    _show_panel(page)
    return page.locator("#create-pr-btn")


@pytest.mark.timeout(180)
def test_a_press_pushes_the_branch_and_reports_it_in_a_toast(
    browser: Browser, daemon: ExplorerHarness,
) -> None:
    """One press pushes the checked-out branch to its remote and says so in a toast.

    While the daemon works (here it waits for the repository's lock,
    which another git step holds) the button is disabled and named after
    what is going on; once the outcome is in, the button is ready again.
    The remote is not a Git host the daemon knows, so the toast offers no
    link.
    """
    work = daemon.work_dir
    bare = Path(daemon.tmpdir) / "origin.git"
    _git(work, "init", "-q", "--bare", "-b", "main", str(bare))
    _git(work, "remote", "add", "origin", str(bare))
    _git(work, "push", "-q", "-u", "origin", "main")
    _git(work, "checkout", "-q", "-b", "strip-button")
    (work / "strip.txt").write_text("strip\n")
    _git(work, "add", "strip.txt")
    _git(work, "commit", "-q", "-m", "Add strip.txt", "--", "strip.txt")
    head = _git(work, "rev-parse", "HEAD").strip()

    context, page = _open_page(browser, daemon)
    try:
        button = _strip_button(page)
        with repo_lock(work):
            button.click()
            expect(button).to_be_disabled()
            expect(button).to_have_class(_BUSY)
            expect(button).to_have_attribute("aria-label", "A pull request is being created")
        toast = page.locator(_OUTCOME_TOAST)
        expect(toast.locator(".kiss-notification-message")).to_have_text(
            "Pushed strip-button to origin. Open the pull request on your Git host."
            " (uncommitted changes are not included)",
            timeout=60000,
        )
        expect(toast.locator(".kiss-notification-action")).to_have_count(0)
        expect(button).to_be_enabled()
        expect(button).not_to_have_class(_BUSY)
        expect(button).to_have_attribute("aria-label", "Create pull request")
        assert _git(bare, "rev-parse", "strip-button").strip() == head
    finally:
        context.close()


@pytest.mark.timeout(180)
def test_a_repository_without_a_remote_gets_an_error_toast(
    browser: Browser, daemon: ExplorerHarness,
) -> None:
    """A press in a repository that has no remote reports the reason as an error.

    Nothing is pushed anywhere, and the button is ready for another
    press.
    """
    context, page = _open_page(browser, daemon)
    try:
        button = _strip_button(page)
        button.click()
        toast = page.locator(_OUTCOME_TOAST)
        expect(toast.locator(".kiss-notification-message")).to_have_text(
            "This repository has no remote to push to.", timeout=60000,
        )
        expect(toast).to_have_class(re.compile(r"\bkiss-notification-error\b"))
        expect(button).to_be_enabled()
        expect(button).to_have_attribute("aria-label", "Create pull request")
        assert _git(daemon.work_dir, "remote") == ""
    finally:
        context.close()

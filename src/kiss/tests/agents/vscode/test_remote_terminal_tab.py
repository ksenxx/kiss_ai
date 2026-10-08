# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_explorer_scm_commands and is intentionally
#   shadowed by test parameters of the same name)
"""E2E: the remote webapp's terminal tab, in a real browser.

The "..." menu of the composer has a "Terminal" item (remote webapp
only).  It opens a content tab holding an xterm.js terminal
(``media/terminalTab.js``) whose shell runs on the daemon's machine
(``kiss/server/terminal_tab.py``): typed commands run in the chat's
work dir, the pty follows the terminal's size, a second terminal gets
its own tab and shell, closing the tab hangs the shell up, an exited
shell can be restarted with Enter, a dropped WebSocket re-attaches the
same shell, and a theme switch recolours the terminal.

Driven against the production ``RemoteAccessServer`` + daemon of
``ExplorerHarness`` with headless Chromium; xterm.js is downloaded from
jsDelivr as in production.  The shell is the dotfile-free bash of
``tests/server/test_terminal_tab_service.py`` (``hermetic_shell``), so
the ``pwd`` and ``$``-prompt assertions do not depend on the
developer's own shell and rc files.
"""

from __future__ import annotations

import re
import sys
import time

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_activity_bar import (
    _dismiss_update_toast,
    _set_work_dir,
)
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.test_explorer_scm_commands import (
    ExplorerHarness,
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_terminal_tab_service import hermetic_shell

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="terminal tabs need a pty",
)


@pytest.fixture(scope="module", autouse=True)
def _bash_without_dotfiles(tmp_path_factory: pytest.TempPathFactory):
    """Every shell the in-process daemon spawns is a dotfile-free bash."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SHELL", str(hermetic_shell(tmp_path_factory.mktemp("shell"))))
        yield

_XTERM = ".terminal-tab-view .xterm"
_SCREEN_TEXT = (
    "Array.from(document.querySelectorAll('.terminal-tab-view'))"
    ".map(v => v.style.display === 'none' ? '' : v.innerText).join('')"
)


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
        yield b
        b.close()


@pytest.fixture(autouse=True)
def no_leftover_shells(harness: ExplorerHarness):
    """A page a failed test left behind keeps its shells for the grace
    period; hang them up so every test starts from zero shells."""
    harness.server._vscode_server.terminals.shutdown()
    yield


def _open_page(browser, harness: ExplorerHarness):
    context = browser.new_context(
        ignore_https_errors=True, viewport={"width": 1300, "height": 800},
    )
    page = context.new_page()
    for attempt in range(3):
        goto_retrying_network_change(page, harness.base_url + "/")
        try:
            page.wait_for_selector("#task-input", state="visible", timeout=30000)
            break
        except PlaywrightTimeoutError:
            if attempt == 2:
                raise
    page.wait_for_function(
        "document.getElementById('meta-workdir').textContent.length > 1",
        timeout=30000,
    )
    _dismiss_update_toast(page, harness)
    return context, page


def _open_terminal(page) -> None:
    """Open a terminal through the "..." menu and wait for its prompt."""
    before = page.locator(_XTERM).count()
    page.click("#more-btn")
    page.wait_for_selector("#terminal-btn", state="visible")
    page.click("#terminal-btn")
    page.wait_for_function(
        f"document.querySelectorAll('{_XTERM}').length === {before + 1}",
        timeout=60000,
    )
    # The shell's first prompt proves the pty round trip works.
    page.wait_for_function(f"({_SCREEN_TEXT}).includes('$')", timeout=30000)


def _screen(page) -> str:
    return str(page.evaluate(_SCREEN_TEXT))


def _wait_for_output(page, needle: str, timeout: int = 20000) -> str:
    page.wait_for_function(
        "needle => (" + _SCREEN_TEXT + ").includes(needle)",
        arg=needle, timeout=timeout,
    )
    return _screen(page)


def _wait_for_sizes(page, count: int) -> None:
    """Wait until the screen shows *count* ``rows cols`` lines (``stty size`` answers)."""
    page.wait_for_function(
        "n => ((" + _SCREEN_TEXT + ").match(/^\\d+ \\d+$/gm) || []).length >= n",
        arg=count, timeout=20000,
    )


def _sessions(harness: ExplorerHarness) -> int:
    return harness.server._vscode_server.terminals.session_count()


def _wait_sessions(harness: ExplorerHarness, want: int, timeout: float = 15) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _sessions(harness) == want:
            return
        time.sleep(0.05)
    raise AssertionError(f"expected {want} shells, found {_sessions(harness)}")


def _close_active_tab(page) -> None:
    """Close the content pane's shown tab (the desktop page is the split
    layout: terminal tabs live on ``#content-tab-list``)."""
    page.click("#content-tab-list .active .chat-tab-close")


def test_menu_item_opens_a_shell_in_the_work_dir(browser, harness):
    """The "..." menu's Terminal item opens a tab titled "Terminal"
    whose shell runs in the chat's work dir and whose pty is sized by
    xterm's fit."""
    context, page = _open_page(browser, harness)
    try:
        _open_terminal(page)
        tab = page.locator("#content-tab-list .active")
        assert tab.inner_text().startswith(">_")
        assert "Terminal" in tab.inner_text()
        page.keyboard.type("echo marker-$((40+2)); pwd; stty size\n")
        _wait_for_output(page, "marker-42")
        # The marker lands before pwd and stty answer: wait for the
        # ``rows cols`` line too before reading the screen.
        _wait_for_sizes(page, 1)
        text = _screen(page)
        assert str(harness.work_dir) in text
        size = re.search(r"(?m)^(\d+) (\d+)$", text)
        assert size is not None
        rows, cols = (int(x) for x in size.groups())
        assert rows > 10 and cols > 40
        # Split layout: the chat and composer stay visible beside the
        # terminal pane; the stacked page's content-tab-open mode is
        # never entered.
        assert page.locator("#output").is_visible()
        assert page.locator("#task-input").is_visible()
        assert page.locator(".terminal-tab-view").first.is_visible()
        assert not page.evaluate("document.body.classList.contains('content-tab-open')")
        assert page.evaluate("window._testApi.getActiveTabId()") == page.evaluate(
            "window._testApi.openTabs().find(t => !t.isContentTab).id"
        )
        assert _sessions(harness) == 1
        _close_active_tab(page)
        _wait_sessions(harness, 0)
        assert page.locator(_XTERM).count() == 0
    finally:
        context.close()


def test_shell_starts_in_the_workspace_the_page_browses(browser, harness):
    """After the "Working directory" panel points the page at another
    folder, a new terminal starts there (and the daemon's own folder is
    restored afterwards for the module's other tests)."""
    context, page = _open_page(browser, harness)
    try:
        _set_work_dir(page, harness, str(harness.plain_dir))
        _open_terminal(page)
        page.keyboard.type("pwd\n")
        text = _wait_for_output(page, str(harness.plain_dir) + "\n")
        assert str(harness.work_dir) + "\n" not in text
        _close_active_tab(page)
        _wait_sessions(harness, 0)
        _set_work_dir(page, harness, str(harness.work_dir))
    finally:
        context.close()


def test_resizing_the_window_resizes_the_pty(browser, harness):
    context, page = _open_page(browser, harness)
    try:
        _open_terminal(page)
        page.keyboard.type("stty size\n")
        _wait_for_sizes(page, 1)
        rows, cols = (
            int(x) for x in re.findall(r"(?m)^(\d+) (\d+)$", _screen(page))[-1]
        )
        page.set_viewport_size({"width": 900, "height": 500})
        page.wait_for_timeout(600)
        page.keyboard.type("stty size\n")
        _wait_for_sizes(page, 2)
        rows2, cols2 = (
            int(x) for x in re.findall(r"(?m)^(\d+) (\d+)$", _screen(page))[-1]
        )
        assert rows2 < rows and cols2 < cols
        _close_active_tab(page)
        _wait_sessions(harness, 0)
    finally:
        context.close()


def test_second_terminal_gets_its_own_tab_and_shell(browser, harness):
    context, page = _open_page(browser, harness)
    try:
        _open_terminal(page)
        page.keyboard.type("echo first-shell-$$\n")
        _wait_for_output(page, "first-shell-")
        _open_terminal(page)
        _wait_sessions(harness, 2)
        titles = page.locator("#content-tab-list .chat-tab").all_inner_texts()
        assert any(t.strip().endswith("Terminal 2") or "Terminal 2" in t for t in titles)
        page.keyboard.type("echo second-shell-$$\n")
        text = _wait_for_output(page, "second-shell-")
        assert "first-shell-" not in text
        # Back to the first tab: its screen is intact and still typeable.
        # Content tabs: "Terminal", "Terminal 2" (the chat is not on
        # that row).
        page.locator("#content-tab-list .chat-tab").nth(0).click()
        page.wait_for_function(f"({_SCREEN_TEXT}).includes('first-shell-')")
        page.keyboard.type("echo again-here\n")
        _wait_for_output(page, "again-here")
        # Closing one tab leaves the other shell running.
        _close_active_tab(page)
        _wait_sessions(harness, 1)
        assert page.locator(_XTERM).count() == 1
        _close_active_tab(page)
        _wait_sessions(harness, 0)
    finally:
        context.close()


def test_exited_shell_is_reported_and_enter_starts_a_new_one(browser, harness):
    context, page = _open_page(browser, harness)
    try:
        _open_terminal(page)
        page.keyboard.type("exit 7\n")
        text = _wait_for_output(page, "exited with code 7")
        assert "press Enter" in text
        _wait_sessions(harness, 0)
        # Keystrokes other than Enter go nowhere on an ended shell.
        page.keyboard.type("x")
        page.wait_for_timeout(200)
        assert _sessions(harness) == 0
        page.keyboard.press("Enter")
        _wait_sessions(harness, 1)
        page.keyboard.type("echo reborn-$((1+1))\n")
        text = _wait_for_output(page, "reborn-2")
        assert "while this page was disconnected" not in text
        _close_active_tab(page)
        _wait_sessions(harness, 0)
    finally:
        context.close()


def test_dropped_websocket_reattaches_the_same_shell(browser, harness):
    """The daemon closes the page's socket; the page reconnects, sends
    ``ready`` and claims its shell back: a variable set before the drop
    is still there, and no "new shell" note is printed."""
    context, page = _open_page(browser, harness)
    try:
        _open_terminal(page)
        page.keyboard.type("KEEP=alive-$$; echo set-$KEEP\n")
        _wait_for_output(page, "set-alive-")
        server = harness.server

        async def _drop() -> None:
            for ws in list(server._printer._remote_clients):
                await ws.close()

        harness.run(_drop())
        # The reconnect's ``ready`` re-opens the tab's shell on a new
        # connection; the daemon then has exactly one remote client again.
        deadline = time.time() + 20
        while time.time() < deadline and not (
            server._printer._remote_clients and
            all(w.state.name == "OPEN" for w in server._printer._remote_clients)
        ):
            time.sleep(0.05)
        page.wait_for_timeout(1000)
        page.keyboard.type("echo still-$KEEP\n")
        text = _wait_for_output(page, "still-alive-")
        assert "while this page was disconnected" not in text
        assert _sessions(harness) == 1
        _close_active_tab(page)
        _wait_sessions(harness, 0)
    finally:
        context.close()


def test_theme_switch_recolours_the_terminal(browser, harness):
    context, page = _open_page(browser, harness)
    try:
        _open_terminal(page)
        colour = "getComputedStyle(document.querySelector('.terminal-tab-view .xterm-rows')).color"
        before = page.evaluate(colour)
        page.click("#more-btn")
        page.click("#theme-btn")
        page.wait_for_function(f"{colour} !== {before!r}", timeout=10000)
        after = page.evaluate(colour)
        assert after != before
        # Switch back so the module's other tests see the default theme.
        page.click("#more-btn")
        page.click("#theme-btn")
        page.wait_for_function(f"{colour} === {before!r}", timeout=10000)
        _close_active_tab(page)
        _wait_sessions(harness, 0)
    finally:
        context.close()


def test_terminal_item_is_remote_only(browser, harness):
    """Without body.remote-chat (the VS Code webview) the menu has no
    Terminal item."""
    context, page = _open_page(browser, harness)
    try:
        display = page.evaluate(
            """() => {
              document.body.classList.remove('remote-chat');
              return getComputedStyle(
                document.getElementById('terminal-btn')).display;
            }"""
        )
        assert display == "none"
    finally:
        context.close()


def test_closing_the_page_hangs_the_shell_up_after_the_grace_period(
    browser, harness, monkeypatch,
):
    from kiss.server import terminal_tab

    monkeypatch.setattr(terminal_tab, "GRACE_SECONDS", 0.5)
    context, page = _open_page(browser, harness)
    _open_terminal(page)
    assert _sessions(harness) == 1
    context.close()
    _wait_sessions(harness, 0, timeout=20)

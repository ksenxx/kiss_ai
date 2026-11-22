# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_content_tab_file_links and is intentionally
#   shadowed by test parameters of the same name)
"""Browser E2E: the sign-in window of two browser windows on one daemon.

The daemon runs one sign-in at a time (:mod:`kiss.server.signin`): a
click in a second window replaces the first window's, which is told so
and forgets the token of its click.  A window whose connection is down
just then is not told: it still shows its paste row and holds its
token, and once it is connected again the daemon knows its paste by
that token alone.  What it pastes is for a sign-in that is gone, and
must not reach the one that runs.

REAL Chromium (Playwright) windows, each in a browser context of its
own, on a REAL :class:`RemoteAccessServer` over ``wss://`` whose
``$KISS_HOME/signin`` is the stub hook of
:mod:`kiss.tests.server.test_signin_window`: a bash script that says
"not logged in", and on ``login`` prints the page to sign in on, reads
one pasted line and records it.  The windows click their own buttons
(each opens a real tab), make up their own tokens and paste through the
paste row; a window's network goes down, the daemon closes its socket,
and the page connects again by itself once the network is back.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import sync_playwright

from kiss.core.locked import locked_call
from kiss.server import signin
from kiss.server.signin import SIGNIN_HOOK
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_signin_window import HOOK, NOT_LOGGED_IN, forget_status


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def hook_home(tmp_path: Path) -> Iterator[Path]:
    """A ``$KISS_HOME`` whose sign-in hook says "not logged in", read anew at every ``ready``."""
    home = tmp_path / "kiss-home"
    home.mkdir()
    hook = home / SIGNIN_HOOK
    hook.write_text(HOOK)
    hook.chmod(0o755)
    (home / "status.json").write_text(json.dumps(NOT_LOGGED_IN) + "\n")
    saved_home, saved_ttl = os.environ.get("KISS_HOME"), signin.STATUS_CACHE_TTL
    os.environ["KISS_HOME"] = str(home)
    signin.STATUS_CACHE_TTL = 0.0
    forget_status()
    try:
        yield home
    finally:
        signin.STATUS_CACHE_TTL = saved_ttl
        forget_status()
        if saved_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = saved_home


def _sockets(harness) -> set[Any]:
    """The daemon's ends of the connections its windows have now."""
    printer = harness.server._printer
    sockets: set[Any] = locked_call(printer._ws_lock, printer._remote_clients.copy)
    return sockets


def _open_window(browser, harness):
    """Open the web app in a browser context of its own; returns once its sign-in window is up.

    Returns ``(context, page, sockets)``: *sockets* are the daemon's
    ends of the connections the page has made.
    """
    before = _sockets(harness)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    goto_retrying_network_change(page, harness.base_url + "/")
    page.wait_for_selector("#signin-modal-buttons button", state="visible", timeout=30000)
    return context, page, _sockets(harness) - before


def _click_provider(page, provider: str) -> None:
    """Click the sign-in button of *provider*; returns once the paste row of its sign-in shows."""
    page.click(f'#signin-modal-buttons button[data-provider="{provider}"]')
    page.wait_for_selector("#signin-modal-paste", state="visible", timeout=30000)


def _paste(page, text: str) -> None:
    """Type *text* into the window's paste row and send it."""
    page.fill("#signin-modal-paste-input", text)
    page.click("#signin-modal-paste-btn")


def _lose_connection(context, harness, sockets: set[Any]) -> None:
    """Take the window's network down and have the daemon close *sockets*, its connections.

    The network first, so the page cannot connect again before the test
    says so (the page hears nothing the daemon sends meanwhile).
    """
    context.set_offline(True)

    async def _drop() -> None:
        for socket in sockets:
            await socket.close()

    asyncio.run_coroutine_threadsafe(_drop(), harness.loop).result(60)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and _sockets(harness) & sockets:
        time.sleep(0.05)
    assert not _sockets(harness) & sockets, "the daemon still has the window's connection"


def _connect_again(context, harness) -> None:
    """Bring the window's network back; returns once the daemon has its new connection."""
    known = _sockets(harness)
    context.set_offline(False)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        made = _sockets(harness) - known
        if made and all(socket.state.name == "OPEN" for socket in made):
            return
        time.sleep(0.05)
    raise AssertionError("the window did not connect again")


@pytest.mark.skipif(sys.platform == "win32", reason="the stub hook is a bash script")
class TestSigninWindowsOfOneDaemon:
    """Browser E2E: a window's paste reaches only the sign-in of its own click."""

    def test_reconnected_windows_paste_stays_out_of_the_next_windows_signin(
        self, browser, harness, hook_home: Path,
    ) -> None:
        """A window starts a sign-in and loses its connection.  A
        second window starts a sign-in of its own, which replaces the
        first; the first window cannot be told.  Connected again, it
        pastes into the paste row it still shows: the daemon refuses
        it, the hook of the second window's sign-in has read nothing,
        and the second window finishes its sign-in with its own paste,
        which closes the sign-in window in both."""
        first_context, first, first_sockets = _open_window(browser, harness)
        second_context, second, _ = _open_window(browser, harness)
        try:
            _click_provider(first, "google")
            _lose_connection(first_context, harness, first_sockets)
            _click_provider(second, "apple")
            _connect_again(first_context, harness)

            pasted = hook_home / "pasted"
            refused = first.locator("#signin-modal-note.signin-modal-error")
            _paste(first, "the-first-windows-code")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not (refused.count() or pasted.exists()):
                first.wait_for_timeout(50)
            assert not pasted.exists(), "the running sign-in read: " + pasted.read_text()
            assert "No sign-in is in progress" in refused.inner_text()

            _paste(second, "the-second-windows-code")
            for page in (second, first):
                page.wait_for_selector("#signin-modal", state="hidden", timeout=30000)
            assert pasted.read_text() == "the-second-windows-code\n"
        finally:
            first_context.close()
            second_context.close()

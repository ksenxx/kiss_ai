# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `daemon` / `page_server` fixtures are imported from
# the service test module and re-exported for pytest)
"""A browser that does not exit on ``Browser.close`` must not wedge the service.

Closing the last Browser tab tears the browser down under the lifecycle
lock (``_launch_lock``): ``_on_page_closed`` -> ``_teardown_locked`` ->
``context.close()``.  For a persistent context Playwright's driver answers
that by sending ``Browser.close`` and then waiting for the browser
*process* to exit, with no deadline of its own.  That await used to be
unbounded, so a Chromium that never exits (here: ``SIGSTOP`` just after it
closed the page) kept the lock forever: every later open queued behind it
and failed after ``_OPEN_TIMEOUT``, and the feature was dead until the
daemon restarted.

The test does exactly that to a real Chromium and checks that the lock is
released within the teardown bounds (the browser close gives up, the
driver stop kills the frozen browser) and that a new tab opens in a fresh
browser afterwards.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any

import pytest

from kiss.server.browser_tab import _TEARDOWN_TIMEOUT
from kiss.tests.server.test_browser_tab_service import (
    _PLAYWRIGHT_CACHE,
    _events,
    _wait,
    daemon,  # noqa: F401 - pytest fixture
    page_server,  # noqa: F401 - pytest fixture
)


async def _browser_pid(service: Any) -> int:
    """The pid of the Chromium browser process behind the service's context."""
    session = await service._context.browser.new_browser_cdp_session()
    try:
        info = await session.send("SystemInfo.getProcessInfo")
    finally:
        await session.detach()
    return next(int(proc["id"]) for proc in info["processInfo"] if proc["type"] == "browser")


def _process_state(pid: int) -> str:
    """The scheduler state of *pid* from ``/proc`` (``T`` stopped, ``Z`` zombie, ``-`` gone)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return "-"


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_browser_that_never_exits_releases_the_lifecycle_lock(
    daemon: Any, page_server: str
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c1"})
    tab_id = _wait(lambda: _events(printer, "openBrowserTab"), "tab opened", 90)[-1]["tab_id"]
    pid = asyncio.run_coroutine_threadsafe(_browser_pid(service), service._loop).result(30)

    # Close the only tab.  The page closes at once; its close event starts
    # the teardown, which sends Browser.close.  The browser is frozen as
    # soon as the tab is withdrawn: Chromium takes tens of milliseconds to
    # exit, so it is still alive, and its exit never comes.
    service.close(tab_id)
    deadline = time.monotonic() + 30
    while not _events(printer, "closeBrowserTab", tab_id=tab_id):
        assert time.monotonic() < deadline, "tab not withdrawn"
        time.sleep(0.0005)
    os.kill(pid, signal.SIGSTOP)
    try:
        time.sleep(0.05)
        assert _process_state(pid) == "T", "the browser exited before it could be frozen"
        assert service._context is None  # the teardown is under way

        # The lock is released without the browser's cooperation: the
        # browser close gives up after its bound, and stopping the driver
        # makes it kill the browser whose close is still pending.
        _wait(
            lambda: not service._launch_lock.locked(),
            "lifecycle lock released",
            2 * _TEARDOWN_TIMEOUT + 5,
        )
        assert service._playwright is None and not service._pages
        _wait(lambda: _process_state(pid) in ("Z", "-"), "frozen browser killed", 5)
    finally:
        # Only reached alive when the fix regressed: do not leave it behind.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # A new open launches a fresh browser instead of queueing behind the lock.
    printer.emitted.clear()
    server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c2"})
    answered = _wait(
        lambda: _events(printer, "openBrowserTab") or _events(printer, "browserError", connId="c2"),
        "reopen answered",
        90,
    )
    assert answered[-1]["type"] == "openBrowserTab", answered[-1]
    assert answered[-1]["tab_id"] != tab_id
    new_pid = asyncio.run_coroutine_threadsafe(_browser_pid(service), service._loop).result(30)
    assert new_pid != pid
    assert not service._launch_lock.locked()

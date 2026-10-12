# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `daemon` / `page_server` fixtures are imported from
# the service test module and re-exported for pytest)
"""A browser that freezes during a tab close and then dies must not wedge the service.

``BrowserTabService._close_page`` holds the browser lifecycle lock
(``_launch_lock``) while it closes a page: first through ``page.close()``
(bounded by ``_CLOSE_TIMEOUT``), then, when Chromium swallowed that close,
through the DevTools ``Page.close`` command.  That fallback used to be
unbounded.  A browser that stopped answering (here: ``SIGSTOP``) and then
exited (``SIGKILL``) left Playwright's request pending forever, so the lock
was never released: the browser's own close events (``_on_page_closed``,
``_teardown``) and every later open queued behind it, the dead context
stayed installed, and shutdown destroyed the pending tasks.

The test does exactly that to a real Chromium and checks that the lock is
released, the dead browser is torn down and a new tab opens afterwards.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any

import pytest

from kiss.server.browser_tab import _CLOSE_TIMEOUT
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


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_frozen_then_dead_browser_releases_the_lifecycle_lock(
    daemon: Any, page_server: str
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c1"})
    tab_id = _wait(lambda: _events(printer, "openBrowserTab"), "tab opened", 90)[-1]["tab_id"]
    pid = asyncio.run_coroutine_threadsafe(_browser_pid(service), service._loop).result(30)
    old_context = service._context

    os.kill(pid, signal.SIGSTOP)
    try:
        # The frozen browser answers neither close: the DevTools fallback is
        # in flight, under the lifecycle lock, when the browser dies.
        service.close(tab_id)
        time.sleep(_CLOSE_TIMEOUT + 1.0)
        assert service._launch_lock.locked()
        os.kill(pid, signal.SIGKILL)
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass

    # The browser's death reaches the service: the close gives up, the lock
    # is released and the dead context is torn down.
    _wait(
        lambda: not service._launch_lock.locked() and service._context is None,
        "lock released and dead browser torn down",
        3 * _CLOSE_TIMEOUT + 15,
    )
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "tab withdrawn", 10)
    assert not service._pages
    assert service._playwright is None

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
    assert service._context is not None and service._context is not old_context
    assert not service._launch_lock.locked()

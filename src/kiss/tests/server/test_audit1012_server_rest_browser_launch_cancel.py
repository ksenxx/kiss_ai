# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `daemon` / `page_server` fixtures are imported from
# the service test module and re-exported for pytest)
"""A browser open cancelled during the user-agent mask leaves no browser behind.

``BrowserTabService._launch_new`` records the Playwright driver in
``self._playwright`` before the launch, and ``_launch`` records the
context only once ``_launch_new`` returns.  Between the two the
user-agent mask (``_mask_headless_user_agent``) awaits the browser
several times.  An abandoned open (``asyncio.wait_for(..., _OPEN_TIMEOUT)``)
whose cancellation landed on one of those awaits used to escape the
cleanup ``try`` around the launch: the driver stayed in
``self._playwright`` with its browser holding the profile directory and
no ``_context`` to close it.  The next open then replaced
``self._playwright`` with a second driver (the first one leaked for the
daemon's lifetime) and its launch failed on the profile the leaked
browser still held.

The test drives the service's private loop directly: it starts the real
launch, cancels it the moment the coroutine chain is suspended inside
``_mask_headless_user_agent`` (observed through ``cr_await``, no test
double) and checks that the driver is gone and a following open works.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from typing import Any

import pytest

from kiss.tests.server.test_audit1005_browser_tab_lifecycle import _driver_processes
from kiss.tests.server.test_browser_tab_service import (
    _PLAYWRIGHT_CACHE,
    _events,
    _wait,
    daemon,  # noqa: F401 - pytest fixture
    page_server,  # noqa: F401 - pytest fixture
)


def _suspended_in(task: asyncio.Task[Any], name: str) -> bool:
    """Return True when *task*'s coroutine chain is awaiting inside the coroutine *name*."""
    coro: Any = task.get_coro()
    while coro is not None:
        code = getattr(coro, "cr_code", None)
        if code is not None and code.co_name == name:
            return True
        coro = getattr(coro, "cr_await", None)
    return False


def _launch_cancelled_in_mask(service: Any) -> bool:
    """Start one real launch and cancel it inside the user-agent mask.

    Returns:
        True when the cancellation landed inside ``_mask_headless_user_agent``;
        False when the launch *succeeded* before the mask was observed (the
        caller tears the browser down and tries again).

    Raises:
        Exception: Whatever a launch that failed on its own raised; a
            broken browser install must fail the test, not skip it.
    """
    loop = service._ensure_loop()
    future = asyncio.run_coroutine_threadsafe(service._launch(), loop)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if future.done():
            future.result()
            return False
        task = next(
            (
                t
                for t in asyncio.all_tasks(loop)
                if getattr(t.get_coro(), "cr_code", None) is service._launch.__code__
            ),
            None,
        )
        if task is not None and _suspended_in(task, "_mask_headless_user_agent"):
            loop.call_soon_threadsafe(task.cancel)
            with pytest.raises(concurrent.futures.CancelledError):
                future.result(30)
            return True
        time.sleep(0.0005)
    raise AssertionError("the launch neither finished nor reached the user-agent mask")


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_an_open_cancelled_in_the_user_agent_mask_stops_the_driver_and_frees_the_profile(
    daemon: Any, page_server: str
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    drivers_before = _driver_processes()

    caught = False
    for _attempt in range(5):
        if _launch_cancelled_in_mask(service):
            caught = True
            break
        # The mask window was missed: drop this browser and try again.
        loop = service._loop
        assert loop is not None
        asyncio.run_coroutine_threadsafe(service._teardown(announce=False), loop).result(30)
    if not caught:
        pytest.skip("the user-agent mask was never observed suspended (machine too fast)")

    # The abandoned launch took its driver (and the browser it launched) with it.
    assert service._context is None
    assert service._playwright is None
    _wait(lambda: _driver_processes() == drivers_before, "driver exit")

    # The profile is free again: the next open launches and streams a tab.
    server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c1"})
    opened = _wait(lambda: _events(printer, "openBrowserTab"), "tab opened", 90)
    assert opened and not _events(printer, "browserError", connId="c1")
    assert service._playwright is not None and service._context is not None

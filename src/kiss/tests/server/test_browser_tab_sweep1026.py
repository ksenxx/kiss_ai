# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `daemon` / `page_server` fixtures are imported from
# the service test module and re-exported for pytest)
"""A driver that does not stop after a failed launch must not wedge the service.

``BrowserTabService._launch_new`` runs under the lifecycle lock
(``_launch_lock``).  When the launch fails, or an abandoned open's
cancellation lands in it, it stops the Playwright driver so that no
driver (and no browser holding the profile) outlives the launch.  That
``playwright.stop()`` closes the driver's stdin and then waits for the
driver *process* to exit; the driver first asks its browser to close and
waits for it, and only force-exits (killing the browser) 30 s later.  The
stop used to be unbounded, unlike the same call in ``_teardown_locked``:
a browser that does not exit on ``Browser.close`` (wedged, stopped,
waiting on a native modal) kept the lock for those 30 s, or for good with
a driver that does not even manage that; every open meanwhile queued
behind the lock.

The test freezes the real browser (``SIGSTOP``) while a real launch is
suspended inside the user-agent mask, cancels the launch the way an
abandoned open does, and checks that the lock is released within the
teardown bound (not the driver's 30 s) and that a new tab opens once the
browser is let go.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import signal
import subprocess
import time
from typing import Any

import pytest

from kiss.server.browser_tab import _TEARDOWN_TIMEOUT
from kiss.tests.server.test_audit1012_server_rest_browser_launch_cancel import _suspended_in
from kiss.tests.server.test_browser_tab_service import (
    _PLAYWRIGHT_CACHE,
    _events,
    _wait,
    daemon,  # noqa: F401 - pytest fixture
    page_server,  # noqa: F401 - pytest fixture
)


def _children(parent: int, needle: str) -> set[int]:
    """The pids of *parent*'s direct children whose command line contains *needle*.

    ``-ww`` asks for the whole command line: GNU readline, loaded by the
    pytest session, exports ``COLUMNS=80`` into the environment when there
    is no terminal, and procps ``ps`` cuts every line at that width, which
    drops ``run-driver`` and ``--user-data-dir=`` from the long Playwright
    command lines.
    """
    listing = subprocess.run(
        ["ps", "-A", "-ww", "-o", "pid=,ppid=,args="], capture_output=True, text=True, check=False
    ).stdout
    pids = set()
    for line in listing.splitlines():
        pid, ppid, args = line.split(None, 2)
        if ppid == str(parent) and needle in args:
            pids.add(int(pid))
    return pids


def _driver_pids() -> set[int]:
    """The pids of this process's Playwright driver children."""
    return _children(os.getpid(), "run-driver")


def _launch_task(service: Any, loop: asyncio.AbstractEventLoop) -> asyncio.Task[Any] | None:
    """The task running ``service._new_tab()`` on *loop*, if it is still alive."""
    return next(
        (
            t
            for t in asyncio.all_tasks(loop)
            if getattr(t.get_coro(), "cr_code", None) is service._new_tab.__code__
        ),
        None,
    )


def _cancel_and_freeze(
    task: asyncio.Task[Any], browser: int, outcome: concurrent.futures.Future[bool]
) -> None:
    """On the loop thread: cancel *task* and freeze *browser* if the task is still in the mask.

    Both happen in one loop callback, so the cancellation is delivered at
    the mask's await even if the browser's answer to it is already in the
    pipe; a task that has left the mask meanwhile is left alone.
    """
    caught = _suspended_in(task, "_mask_headless_user_agent")
    if caught:
        task.cancel()
        os.kill(browser, signal.SIGSTOP)
    outcome.set_result(caught)


def _freeze_browser_in_mask(
    service: Any, drivers_before: set[int]
) -> tuple[concurrent.futures.Future[Any], int] | None:
    """Start one real tab open, freeze its browser while the launch is in the mask, cancel it.

    Returns:
        The open's future and the frozen browser's pid; ``None`` when the
        launch *succeeded* before the mask was observed (the caller tears
        the browser down and tries again).
    """
    loop = service._ensure_loop()
    future = asyncio.run_coroutine_threadsafe(service._new_tab(), loop)
    # The driver and its browser process exist well before the mask (the
    # browser's start-up sits in between); finding them here keeps the
    # slow ``ps`` off the mask window.
    (driver,) = _wait(lambda: _driver_pids() - drivers_before, "driver started", 60)
    (browser,) = _wait(lambda: _children(driver, "--user-data-dir="), "browser started", 60)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if future.done():
            future.result()
            return None
        task = _launch_task(service, loop)
        if task is not None and _suspended_in(task, "_mask_headless_user_agent"):
            outcome: concurrent.futures.Future[bool] = concurrent.futures.Future()
            loop.call_soon_threadsafe(_cancel_and_freeze, task, browser, outcome)
            if outcome.result(10):
                return future, browser
        time.sleep(0.0005)
    raise AssertionError("the launch neither finished nor reached the user-agent mask")


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_a_driver_that_does_not_stop_after_a_failed_launch_releases_the_lifecycle_lock(
    daemon: Any, page_server: str
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    drivers_before = _driver_pids()

    frozen = None
    for _attempt in range(5):
        frozen = _freeze_browser_in_mask(service, drivers_before)
        if frozen is not None:
            break
        # The mask window was missed: drop this browser and try again.
        loop = service._loop
        assert loop is not None
        asyncio.run_coroutine_threadsafe(service._teardown(announce=False), loop).result(30)
    if frozen is None:
        pytest.skip("the user-agent mask was never observed suspended (machine too fast)")
    future, browser = frozen

    try:
        # The cancelled launch gives up on its driver within the bound and
        # releases the lock, instead of holding it until the driver has
        # given up on the frozen browser (30 s).
        try:
            future.result(_TEARDOWN_TIMEOUT + 5)
        except concurrent.futures.CancelledError:
            pass
        except concurrent.futures.TimeoutError:
            pytest.fail("the cancelled launch waited on the frozen browser past the teardown bound")
        assert not service._launch_lock.locked()
        assert service._playwright is None and service._context is None
    finally:
        # Let the browser answer the driver's pending close: both exit.
        os.kill(browser, signal.SIGCONT)
    _wait(lambda: _driver_pids() == drivers_before, "driver exit")

    # A new open launches a fresh browser instead of queueing behind the lock.
    printer.emitted.clear()
    server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c1"})
    answered = _wait(
        lambda: _events(printer, "openBrowserTab") or _events(printer, "browserError", connId="c1"),
        "reopen answered",
        90,
    )
    assert answered[-1]["type"] == "openBrowserTab", answered[-1]
    assert service._playwright is not None and service._context is not None
    assert not service._launch_lock.locked()

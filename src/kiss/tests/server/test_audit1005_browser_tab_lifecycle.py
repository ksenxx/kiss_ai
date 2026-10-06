# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `daemon` / `page_server` fixtures are imported from
# test_browser_tab_service and re-bound as test parameters)
"""E2E: browser-tab lifecycle races (audit 2026-10-05, scope H).

Two races in :class:`kiss.server.browser_tab.BrowserTabService`:

* A browser that fails to launch used to leave the Playwright driver
  (a ``node`` child process) running: ``_launch_new`` started the
  driver before the launch and never stopped it on failure, and the
  next open started another one.  The service must stop the driver
  with the failed launch, so retries do not pile drivers up.
* Closing the last tab while another tab is being created tore the
  browser down under the ``new_page()`` call, which then failed and
  the user saw a ``browserError`` instead of the tab.  The teardown
  must wait while a page is being created.

Both run a real browser (or a real "browser" that exits at once)
through the daemon's command catalog, as ``test_browser_tab_service``
does.
"""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kiss.tests.server.test_browser_tab_service import (
    _PLAYWRIGHT_CACHE,
    _events,
    _wait,
    daemon,  # noqa: F401 - pytest fixture
    page_server,  # noqa: F401 - pytest fixture
)


def _driver_processes() -> int | None:
    """Count this process's Playwright driver children; ``None`` without ``ps``."""
    if os.name != "posix":
        return None
    listing = subprocess.run(
        ["ps", "-A", "-o", "ppid=,args="], capture_output=True, text=True, check=False
    ).stdout
    me = str(os.getpid())
    return sum(
        1 for line in listing.splitlines() if line.split(None, 1)[0] == me and "run-driver" in line
    )


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_a_failed_launch_stops_the_playwright_driver(
    daemon: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    fake = tmp_path / "not-a-browser"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("KISS_BROWSER", str(fake))
    monkeypatch.setenv("KISS_HEADLESS", "1")
    drivers_before = _driver_processes()

    for conn_id in ("c1", "c2"):
        server._handle_command({"type": "browserOpen", "url": "", "connId": conn_id})
        _wait(lambda: _events(printer, "browserError", connId=conn_id), "launch error", 60)
        # The driver started for the failed launch is stopped with it ...
        assert service._playwright is None
        # ... so no driver process is left behind for the next attempt.
        _wait(lambda: _driver_processes() == drivers_before, "driver exit")
    assert not _events(printer, "openBrowserTab")


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_a_shutdown_during_the_launch_keeps_the_launch_error(
    daemon: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The teardown takes the driver first; the launch's own error is still reported."""
    server, printer = daemon
    service = server.browser_tabs
    slow = tmp_path / "slow-browser"
    slow.write_text("#!/bin/sh\nsleep 5\nexit 1\n")
    slow.chmod(slow.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("KISS_BROWSER", str(slow))
    monkeypatch.setenv("KISS_HEADLESS", "1")
    server._handle_command({"type": "browserOpen", "url": "", "connId": "c1"})
    _wait(lambda: service._playwright is not None, "driver started")
    assert service._loop is not None
    asyncio.run_coroutine_threadsafe(service._teardown(announce=True), service._loop).result(30)
    err = _wait(lambda: _events(printer, "browserError", connId="c1"), "launch error")[0]
    assert "closed" in err["text"] and "NoneType" not in err["text"]
    assert service._playwright is None
    _wait(lambda: _driver_processes() in (None, 0), "driver exit")


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_closing_the_last_tab_while_another_opens_keeps_the_new_tab(
    daemon: Any, page_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    monkeypatch.setenv("KISS_HEADLESS", "1")
    server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c1"})
    current = _wait(lambda: _events(printer, "openBrowserTab"), "first tab", 90)[0]["tab_id"]
    seen = {current}

    # Open the next tab and close the current one back to back: the
    # close lands while the new page is still being created.
    for _ in range(6):
        server._handle_command({"type": "browserOpen", "url": page_server, "connId": "c1"})
        server._handle_command({"type": "browserClose", "tab_id": current, "connId": "c1"})
        _wait(lambda: _events(printer, "closeBrowserTab", tab_id=current), "old tab closed")
        # The event log is cumulative: the new tab is the one not announced before.
        opened = _wait(
            lambda: [e for e in _events(printer, "openBrowserTab") if e["tab_id"] not in seen],
            "new tab",
        )
        current = opened[-1]["tab_id"]
        seen.add(current)
        _wait(lambda: _events(printer, "browserState", tab_id=current, title="Tall"), "state")
        assert not _events(printer, "browserError")
        assert set(service._pages) == {current}

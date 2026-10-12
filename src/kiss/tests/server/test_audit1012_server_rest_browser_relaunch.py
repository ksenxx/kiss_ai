# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `daemon` / `page_server` fixtures are imported from
# the service test module and re-exported for pytest)
"""Closing a browser tab right after it opened, with or without a reopen.

Two things used to go wrong in ``BrowserTabService``:

* Chromium now and then swallows the close of a page whose navigation is
  committing (a tab closed right after it was announced): Playwright's
  ``page.close()`` never returned, ``_on_page_closed`` never ran, the
  page stayed in ``_pages`` unseen by every surface (the client drops a
  tab it closed at once) and kept the browser alive for good.  The
  service now bounds the close and closes such a page through the
  DevTools ``Page.close`` command.

* A reopen racing the teardown of the last tab launched a second
  browser on the profile the first one still held, and the first
  context's ``close`` event then tore down what the second launch had
  stored.  Launch, page creation, page close and teardown now share one
  lock on the private loop, and a stale context-close event is ignored.

Both are timing-dependent in the browser (the swallowed close hits
roughly one close in ten here), so each test repeats its round.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from kiss.tests.server.test_browser_tab_service import (
    _PLAYWRIGHT_CACHE,
    _events,
    _wait,
    daemon,  # noqa: F401 - pytest fixture
    page_server,  # noqa: F401 - pytest fixture
)


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_closing_a_tab_right_after_it_opened_always_closes_it(
    daemon: Any, page_server: str
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    for round_no in range(10):
        conn = f"c{round_no}"
        printer.emitted.clear()
        server._handle_command({"type": "browserOpen", "url": page_server, "connId": conn})
        tab_id = _wait(lambda: _events(printer, "openBrowserTab"), "tab opened", 90)[-1]["tab_id"]
        # The navigation is still committing: the close may be swallowed.
        server._handle_command({"type": "browserClose", "tab_id": tab_id, "connId": conn})
        _wait(
            lambda: _events(printer, "closeBrowserTab", tab_id=tab_id),
            f"round {round_no}: tab {tab_id} closed",
            15,
        )
        # Its page is gone, so the last tab's close tears the browser down.
        _wait(lambda: service._playwright is None, f"round {round_no}: browser torn down", 15)
        assert not service._pages


@pytest.mark.skipif(not _PLAYWRIGHT_CACHE.is_dir(), reason="Playwright browsers not installed")
def test_reopening_right_after_the_last_tab_closed_opens_a_tab_that_stays(
    daemon: Any, page_server: str
) -> None:
    server, printer = daemon
    service = server.browser_tabs
    for round_no in range(4):
        conn = f"c{round_no}"
        server._handle_command({"type": "browserOpen", "url": page_server, "connId": conn})
        opened = _wait(lambda: _events(printer, "openBrowserTab"), "tab opened", 90)
        tab_id = opened[-1]["tab_id"]
        printer.emitted.clear()
        # Close the last tab and reopen at once, before the teardown is done.
        server._handle_command({"type": "browserClose", "tab_id": tab_id, "connId": conn})
        server._handle_command({"type": "browserOpen", "url": page_server, "connId": conn})
        reopened = _wait(
            lambda: (
                _events(printer, "openBrowserTab") or _events(printer, "browserError", connId=conn)
            ),
            "second open answered",
            90,
        )
        assert reopened[-1]["type"] == "openBrowserTab", reopened[-1]
        new_tab = reopened[-1]["tab_id"]
        assert new_tab != tab_id
        # The new tab is not withdrawn by the old browser's close event.
        time.sleep(2.0)
        assert not _events(printer, "closeBrowserTab", tab_id=new_tab)
        assert not _events(printer, "browserError", connId=conn)
        printer.emitted.clear()
        server._handle_command({"type": "browserClose", "tab_id": new_tab, "connId": conn})
        # Closing it tears the browser down: no page the surfaces do not
        # show is left behind keeping it alive.
        _wait(lambda: service._playwright is None, f"round {round_no}: browser torn down", 30)
        assert not service._pages

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a page that closes while being attached never becomes a ghost browser tab.

``BrowserTabService._register`` attaches a CDP session to a new page,
asks ``Target.getTargetInfo`` for its target id and only then wires the
page's ``close`` listener.  Playwright's pipe transport dispatches every
message it has buffered without yielding, so the page's ``close`` can be
delivered, with no listener yet, between the ``getTargetInfo`` reply and
the attach coroutine resuming.  Before the fix the service then completed
and announced a record for a page that was already gone: ``_pages`` kept
it forever, every connecting client was told about it, and nothing could
close it (``page.close()`` on a closed page returns silently), so the
"last tab closes the browser" rule never fired either.

The test drives a REAL headless Chromium through the real service.  It
forces the race instead of hoping for it: a second listener on the
context's ``page`` event attaches its own CDP session (queued right
behind the service's) and then blocks the service loop for 300 ms.  By
then the popup, told by its opener to close 100 ms in, is gone, and the
``getTargetInfo`` reply and the ``close`` event sit in the pipe together.

That ordering is near-certain with more than one CPU.  On a single CPU
the loop can be descheduled between sending ``getTargetInfo`` and reading
the hog's attach reply, so both replies land in one batch, the hog's
stall starts too late and the popup outlives its attach: it is announced
and then closed normally.  Such a round proves nothing, so the test
repeats rounds until one hits the window, bounded by ``_MAX_ROUNDS``.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page

from kiss.agents.sorcar import web_stealth
from kiss.server.browser_tab import BrowserTabService
from kiss.tests.server._memory_printer import MemoryPrinter

_PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"
_MAX_ROUNDS = 10
_CLOSE_AFTER_MS = 100
_HOG_SECONDS = 0.3


class _Printer(MemoryPrinter):
    """A real printer that also keeps every event the service broadcasts, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(dict(event))
        super().broadcast(event)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        with self.lock:
            return [e for e in self.events if e["type"] == kind]


def _wait_for(predicate: Any, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for the browser")


async def _hog_after_attach(page: Page) -> None:
    """Attach to *page* like the service does, then stall the loop."""
    await page.context.new_cdp_session(page)
    time.sleep(_HOG_SECONDS)


def _on_page(page: Page) -> None:
    asyncio.ensure_future(_hog_after_attach(page))


async def _open_and_close_popup(opener: Page) -> None:
    """Open a popup from *opener* that closes itself ``_CLOSE_AFTER_MS`` later."""
    context = opener.context
    context.on("page", _on_page)
    try:
        await opener.evaluate(
            "d => { const w = window.open('about:blank'); setTimeout(() => w.close(), d); }",
            _CLOSE_AFTER_MS,
        )
        await asyncio.sleep(1.0)
    finally:
        context.remove_listener("page", _on_page)


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> Any:
    if not _PLAYWRIGHT_CACHE.is_dir():
        pytest.skip("Playwright browsers are not installed")
    monkeypatch.setenv("KISS_HEADLESS", "1")
    printer = _Printer()
    svc = BrowserTabService(printer, Path(tempfile.mkdtemp()))
    try:
        yield svc, printer
    finally:
        svc.shutdown()
        web_stealth.stop_virtual_display()


def _open_opener(svc: BrowserTabService) -> tuple[Any, asyncio.AbstractEventLoop]:
    """Open the opener tab and return its record once its navigation has committed.

    ``svc.open()`` returns at once and announces the tab *before* the
    initial ``goto``; evaluating in the page at that point can be cut
    short by the pending navigation ("Execution context was destroyed"),
    which actually happens on a single CPU.  Awaiting ``_open`` itself
    keeps the real announcements and waits for the committed navigation.
    """
    loop = svc._ensure_loop()
    asyncio.run_coroutine_threadsafe(svc._open("about:blank", "c1"), loop).result(60)
    with svc._lock:
        (opener_rec,) = svc._pages.values()
    return opener_rec, loop


def test_popup_closed_during_attach_is_not_announced_and_not_kept(service: Any) -> None:
    svc, printer = service
    opener_rec, loop = _open_opener(svc)
    hit = False
    for _ in range(_MAX_ROUNDS):
        announced_before = len(printer.of_type("openBrowserTab"))
        asyncio.run_coroutine_threadsafe(_open_and_close_popup(opener_rec.page), loop).result(30)
        # A popup gone before its attach completed is never announced.
        # Before the fix such a round announced a ghost instead, so the
        # loop then runs to its bound and the checks below catch it.
        if len(printer.of_type("openBrowserTab")) == announced_before:
            hit = True
            break
    # Nothing but the opener is kept, whether a popup closed during its
    # attach (never announced) or outlived it (announced, then closed
    # normally): no client is left with a tab it cannot close.
    with svc._lock:
        assert list(svc._pages) == [opener_rec.tab_id]
    assert [e["tab_id"] for e in svc.open_events()] == [opener_rec.tab_id]
    popups = [
        e["tab_id"] for e in printer.of_type("openBrowserTab") if e["tab_id"] != opener_rec.tab_id
    ]
    _wait_for(lambda: [e["tab_id"] for e in printer.of_type("closeBrowserTab")] == popups)
    assert hit, "no round closed the popup while it was being attached"


def test_closing_the_only_tab_still_shuts_the_browser_down(service: Any) -> None:
    """The control: a popup that lives long enough is announced and closed normally."""
    svc, printer = service
    opener_rec, loop = _open_opener(svc)
    asyncio.run_coroutine_threadsafe(
        opener_rec.page.evaluate("() => { window.open('about:blank'); }"), loop
    ).result(30)
    _wait_for(lambda: len(printer.of_type("openBrowserTab")) >= 3)
    with svc._lock:
        popup_id = next(t for t in svc._pages if t != opener_rec.tab_id)
    svc.close(popup_id)
    _wait_for(lambda: [e["tab_id"] for e in printer.of_type("closeBrowserTab")] == [popup_id])
    svc.close(opener_rec.tab_id)
    _wait_for(lambda: len(printer.of_type("closeBrowserTab")) == 2)
    _wait_for(lambda: svc._context is None)
    assert svc.open_events() == []

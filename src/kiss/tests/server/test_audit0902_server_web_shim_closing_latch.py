# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E (Playwright): the shim must reload after a wake-up reconnect too.

``_WS_SHIM_JS`` reloads the page on the first ``auth_ok`` that follows
the loss of an authenticated socket, so a page that survived a server
restart re-runs its load pipeline against the fresh backend instead of
showing stale tabs.  The latch for "we had a session and lost it"
(``_hadAuthThenClosed``) was set only in the old socket's ``onclose``.

Mobile Safari — the shim's own primary reconnect scenario — frequently
resumes JS with the dead socket still in ``CLOSING`` and delivers the
wake-up events (``focus`` / ``visibilitychange`` / ``pageshow``) BEFORE
the queued ``onclose``.  ``_reconnectNowIfNeeded`` then calls
``connect()``, which deliberately nulls the old socket's handlers to
make the swap atomic — and thereby discards the very ``onclose`` that
would have latched the flag.  The new socket's ``auth_ok`` therefore
did not reload the page: the user was left on a stale UI after every
app switch that coincided with a daemon restart.

The test drives the production shim string through the same
instrumented-``WebSocket`` page as ``test_remote_webapp_auto_reload``
and observes the reload from outside via Playwright frame navigations.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.server.test_remote_webapp_auto_reload import (
    _build_test_page,
)


@pytest.fixture(scope="module")
def _browser():
    """Module-scoped headless Chromium."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


def _load(browser):
    context = browser.new_context()
    page = context.new_page()
    navs = [0]

    def _on_nav(frame) -> None:
        if frame == page.main_frame:
            navs[0] += 1

    page.on("framenavigated", _on_nav)
    page.set_content(_build_test_page(), wait_until="load")
    navs[0] = 0
    return context, page, navs


def _authenticate_current_socket(page) -> None:
    page.evaluate("window.__fireOpen()")
    page.evaluate("window.__fireAuthOk()")


@pytest.mark.parametrize("wake_event", ["focus", "pageshow", "online"])
def test_wakeup_reconnect_while_old_socket_closing_reloads(
    _browser, wake_event: str,
) -> None:
    """Wake-up reconnect from CLOSING + fresh ``auth_ok`` must reload."""
    context, page, navs = _load(_browser)
    try:
        _authenticate_current_socket(page)
        page.wait_for_timeout(100)
        assert navs[0] == 0, "first auth_ok must not reload"

        # The OS killed the socket while the tab was backgrounded: it
        # is CLOSING, but its onclose has NOT been delivered yet.
        page.evaluate("window.__openSocket.readyState = 2")
        page.evaluate(
            f"window.dispatchEvent(new Event({wake_event!r}))"
        )
        assert page.evaluate("window.__sockets.length") == 2, (
            "wake-up listener must open a replacement socket"
        )
        # The replacement swap silently closed the old socket; its
        # onclose handler was nulled first, so the latch has to be
        # taken by connect() itself.
        _authenticate_current_socket(page)
        page.wait_for_timeout(500)
        assert navs[0] >= 1, (
            "BUG: a wake-up reconnect that replaces a still-CLOSING "
            "authenticated socket never reloads the page on the next "
            "auth_ok, leaving the webapp on stale pre-restart state"
        )
    finally:
        context.close()


def test_visibilitychange_reconnect_while_closing_reloads(_browser) -> None:
    """``visibilitychange`` (visible) is the fourth wake-up path."""
    context, page, navs = _load(_browser)
    try:
        _authenticate_current_socket(page)
        page.wait_for_timeout(100)
        page.evaluate("window.__openSocket.readyState = 2")
        # Headless pages report visibilityState 'visible', so the
        # listener's visible-branch runs.
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        assert page.evaluate("window.__sockets.length") == 2
        _authenticate_current_socket(page)
        page.wait_for_timeout(500)
        assert navs[0] >= 1
    finally:
        context.close()


def test_wakeup_with_unauthenticated_closing_socket_does_not_reload(
    _browser,
) -> None:
    """A never-authenticated CLOSING socket must not arm the reload.

    Guards the fix against over-latching: a fresh page whose FIRST
    socket died before ``auth_ok`` must still treat the replacement's
    first ``auth_ok`` as the initial handshake (no reload loop).
    """
    context, page, navs = _load(_browser)
    try:
        page.evaluate("window.__fireOpen()")  # opened, never authenticated
        page.evaluate("window.__openSocket.readyState = 2")
        page.evaluate("window.dispatchEvent(new Event('focus'))")
        assert page.evaluate("window.__sockets.length") == 2
        _authenticate_current_socket(page)
        page.wait_for_timeout(300)
        assert navs[0] == 0, (
            f"unauthenticated replacement must not reload; saw {navs[0]}"
        )
        events = page.evaluate("window.__daemonStatusEvents")
        assert any(e.get("connected") is True for e in events), events
    finally:
        context.close()

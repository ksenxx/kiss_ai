# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_content_tab_file_links and is intentionally
#   shadowed by test parameters of the same name)
"""End-to-end regression test: the loading overlay must always lift.

The remote webapp's inline socket shim (served by
:func:`kiss.server.web_server._build_html`) sits at the TOP of the
body script list and opens the WebSocket immediately, while
``media/main.js`` — whose window ``message`` listener performs the
``daemonStatus`` handling that hides the "KISS Sorcar Server is
starting ..." overlay — is a LATER, separately fetched script.  When
the ``auth_ok`` frame arrived while the HTML parser was still fetching
``main.js``, the shim's one-shot ``daemonStatus connected:true``
MessageEvent was dispatched with no listener registered and silently
lost: the overlay covered ``#app`` forever even though the socket was
authenticated and every later frame flowed normally.  Observed as an
intermittent (~1 in 5) 30 s timeout waiting for ``#task-input`` to
become visible in the content-tab browser tests.

The fix queues app-bound dispatches while ``document.readyState`` is
``'loading'`` and flushes them on ``DOMContentLoaded``.  This test
makes the race deterministic instead of 1-in-5: a Playwright route
stalls the ``main.js`` request long enough that the WebSocket
handshake and authentication ALWAYS finish first, then asserts the
overlay still lifts and the composer becomes visible.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` — no mocks.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


def test_overlay_lifts_when_auth_beats_main_js_load(browser, harness) -> None:
    """``auth_ok`` before ``main.js`` finishes loading must still reveal the app.

    The stalled ``main.js`` request guarantees the WebSocket
    authenticates while the parser is still waiting on the script, so
    the synthesised ``daemonStatus connected:true`` is always produced
    before the app's ``message`` listener exists — the exact window in
    which the pre-fix shim lost it forever.
    """
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()

    def _stall_main_js(route) -> None:
        # Localhost WS handshake + empty-password auth completes in
        # well under a second; 1.5 s makes the ordering deterministic.
        time.sleep(1.5)
        route.continue_()

    page.route("**/main.js*", _stall_main_js)
    try:
        page.goto(harness.base_url + "/")
        page.wait_for_selector("#task-input", state="visible", timeout=15000)
        overlay_display = page.evaluate(
            "() => document.getElementById('kiss-server-loading').style.display"
        )
        assert overlay_display == "none", (
            "loading overlay must be hidden once the authenticated "
            f"daemonStatus is delivered; got display={overlay_display!r}"
        )
        app_display = page.evaluate(
            "() => document.getElementById('app').style.display"
        )
        assert app_display == "", (
            f"#app must be revealed; got display={app_display!r}"
        )
    finally:
        context.close()


def test_overlay_lifts_on_normal_load(browser, harness) -> None:
    """Control: an unstalled page load reveals the composer as before."""
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    try:
        page.goto(harness.base_url + "/")
        page.wait_for_selector("#task-input", state="visible", timeout=15000)
    finally:
        context.close()

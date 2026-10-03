# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_explorer_scm_commands and is intentionally
#   shadowed by test parameters of the same name)
"""E2E: the remote webapp recovers from a page asset that failed to load.

Chromium aborts every in-flight request with ``ERR_NETWORK_CHANGED``
when the network path changes during a page load (a phone's
Wi-Fi/cellular hand-over; on a busy CI host, container network churn).
Before the fix a page whose ``api.js`` never arrived stayed at the
"Server is starting" overlay for good: ``main.js`` threw
``createSorcarApi is not defined`` while the WebSocket shim happily
authenticated; a page whose ``remote-codex.css`` never arrived came up
with the history panel as main.css's 90vw drawer, open on top of the
chat, for its whole lifetime.  The asset-load guard
(``_ASSET_LOAD_GUARD_JS`` in ``web_server.py``, the first script of
``<head>`` so it is installed before any stylesheet fetch can fail)
now reloads the page once when a same-origin ``<script src>`` or
``<link rel=stylesheet>`` fails to load while the document is still
parsing.

Driven against the production ``RemoteAccessServer`` + daemon of
``test_explorer_scm_commands.harness`` and a real headless Chromium.
The failing load is injected with ``page.route`` (a single aborted
request, or every request for the loop-guard test).

Not covered here: the shim's "no reload when ``sessionStorage`` throws"
branch (a browser with site storage blocked).  Playwright has no switch
for that, and replacing ``window.sessionStorage`` from an init script
would test a stand-in rather than the browser.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from kiss.tests.server.test_explorer_scm_commands import (
    ExplorerHarness,
    harness,  # noqa: F401  (module fixture)
)

RELOADED_AT_KEY = "sorcar-script-reloaded-at"


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


class _Loads:
    """Counts document navigations and the aborted asset's requests."""

    def __init__(self, asset: str, abort_first_n: int) -> None:
        self.asset = asset
        self.abort_first_n = abort_first_n
        self.reset()

    def reset(self) -> None:
        """Forget every request seen so far (used when a navigation is retried)."""
        self.documents = 0
        self.asset_requests = 0
        self.aborted = 0
        self.network_changed: list[str] = []

    def on_request(self, request) -> None:
        if request.is_navigation_request() and request.resource_type == "document":
            self.documents += 1

    def on_request_failed(self, request) -> None:
        """Record requests the host's network churn aborted (not the injected failure)."""
        if "ERR_NETWORK_CHANGED" in (request.failure or ""):
            self.network_changed.append(request.url)

    def route(self, route) -> None:
        self.asset_requests += 1
        if self.aborted < self.abort_first_n:
            self.aborted += 1
            route.abort("connectionaborted")
        else:
            route.continue_()


def _open(browser: Browser, harness: ExplorerHarness, loads: _Loads) -> tuple:
    # Service workers are blocked so every ``/media/`` request reaches
    # the route handler (a worker installed by the first load would
    # otherwise answer the reload from its cache and bypass ``route``).
    context = browser.new_context(
        ignore_https_errors=True,
        viewport={"width": 1400, "height": 900},
        service_workers="block",
    )
    page = context.new_page()
    page.on("request", loads.on_request)
    page.on("requestfailed", loads.on_request_failed)
    page.route(f"**/media/{loads.asset}*", loads.route)
    # The self-reload interrupts the first navigation's ``load`` event,
    # so wait for the commit only and let the assertions drive the rest.
    # The real ``ERR_NETWORK_CHANGED`` (host interface churn on a busy
    # CI box) can also hit the *document* request, before the injected
    # script failure gets a chance; that is not what is under test, so
    # retry the navigation and forget the requests the aborted attempt
    # recorded, keeping the exact counts asserted below meaningful.
    for attempt in range(3):
        try:
            page.goto(harness.base_url + "/", wait_until="commit")
            break
        except PlaywrightError as exc:
            if "net::ERR_NETWORK_CHANGED" not in str(exc) or attempt == 2:
                raise
            loads.reset()
            time.sleep(1.0)
    return context, page


def _reloaded_at(page: Page) -> float:
    return float(page.evaluate(f"Number(sessionStorage.getItem('{RELOADED_AT_KEY}')) || 0"))


_CHAT_CLEAR_OF_SIDEBAR_JS = """
() => {
  const app = document.getElementById('app').getBoundingClientRect();
  const sidebar = document.getElementById('sidebar').getBoundingClientRect();
  return app.left >= sidebar.right && sidebar.width > 0;
}
"""


@pytest.mark.parametrize(
    "asset", ["api.js", "highlight.min.js", "remote-codex.css", "main.css"],
)
def test_page_reloads_itself_once_when_an_asset_fails_to_load(browser, harness, asset):
    _run_unless_network_churn(
        browser, harness, _Loads(asset, abort_first_n=1), _check_recovered_after_one_reload,
    )


def _run_unless_network_churn(
    browser: Browser, harness: ExplorerHarness, loads: _Loads, check,
) -> None:
    """Run ``check(page, loads)`` on a fresh page; retry when the host's network churn interfered.

    The guard reloads once per 30 s, so when the host's network churn
    (Docker containers of concurrent tests on a CI box) aborts one of
    the *reloaded* page's own assets, or the reload's document request
    itself, with the real ``ERR_NETWORK_CHANGED``, the page stays
    half-booted or unstyled by design (or is Chromium's error page).
    That second, uninjected failure is not what is under test: when one
    was seen the scenario is retried, up to three times, in a fresh
    context (fresh ``sessionStorage``).  Any other failure propagates.
    """
    for attempt in range(3):
        loads.reset()
        context, page = _open(browser, harness, loads)
        try:
            check(page, loads)
            return
        except (AssertionError, PlaywrightError):
            if not loads.network_changed or attempt == 2:
                raise
        finally:
            context.close()


def _check_recovered_after_one_reload(page: Page, loads: _Loads) -> None:
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_selector("body.remote-desktop", state="attached")
    assert loads.aborted == 1
    assert loads.asset_requests == 2, loads.asset_requests
    assert loads.documents == 2, loads.documents
    assert _reloaded_at(page) > 0
    # The recovered page is fully live: the config reply landed.
    page.wait_for_function(
        "document.getElementById('meta-workdir').textContent.length > 1",
        timeout=30000,
    )
    # ... and styled: the docked history panel sits beside the chat
    # (without remote-codex.css it is a 90vw drawer over the chat,
    # and every click on the chat lands on the history list).
    page.wait_for_function(_CHAT_CLEAR_OF_SIDEBAR_JS, timeout=10000)


def test_a_script_that_keeps_failing_reloads_once_then_stays(browser, harness):
    # A really broken asset (every fetch fails) must not put the page in
    # a reload loop: the timestamp guard allows one reload per 30 s.
    _run_unless_network_churn(
        browser, harness, _Loads("api.js", abort_first_n=10**6), _check_reloaded_once_then_stayed,
    )


def _check_reloaded_once_then_stayed(page: Page, loads: _Loads) -> None:
    # ``wait_for_timeout`` (not ``time.sleep``): the sync API runs
    # route handlers only while a Playwright call is in progress.
    deadline = time.monotonic() + 15
    while loads.documents < 2 and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert loads.documents == 2, loads.documents
    page.wait_for_load_state("load")
    stamp = _reloaded_at(page)
    assert stamp > 0
    page.wait_for_timeout(3000)
    assert loads.documents == 2, loads.documents
    assert loads.aborted == 2, loads.aborted
    assert _reloaded_at(page) == stamp
    # The failure stays visible instead of a blank flicker loop.
    assert page.evaluate(
        "getComputedStyle(document.getElementById('app')).display"
    ) == "none"
    assert page.evaluate(
        "getComputedStyle(document.getElementById('kiss-server-loading')).display"
    ) != "none"


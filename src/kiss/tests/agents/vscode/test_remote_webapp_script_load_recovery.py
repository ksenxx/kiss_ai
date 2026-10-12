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
now reloads the page when a same-origin ``<script src>`` or
``<link rel=stylesheet>`` fails to load while the document is still
parsing: up to three times per 30 s (the network change can outlast
the first reload; full-suite run of 2026-10-11, where the reload's
``remote-codex.css`` was aborted too), then the failure stays visible.

Driven against the production ``RemoteAccessServer`` + daemon of
``test_explorer_scm_commands.harness`` and a real headless Chromium.
The failing load is injected with ``page.route`` (the first one or two
requests aborted, or every request for the loop-guard test).

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

RELOAD_STAMPS_KEY = "sorcar-asset-reloads"
# Reloads the guard allows per 30 s window (``_MAX_RELOADS`` in the guard).
RELOAD_BUDGET = 3


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
        self.commits = 0
        self.asset_requests = 0
        self.aborted = 0
        self.host_failures: list[str] = []
        self.asset_log: list[str] = []

    def on_request(self, request) -> None:
        if request.is_navigation_request() and request.resource_type == "document":
            self.documents += 1

    def on_frame_navigated(self, frame) -> None:
        """Count the main frame's committed navigations: ``documents`` grows
        at the request, while the previous document is still the one
        ``page.evaluate`` runs in."""
        if frame.parent_frame is None:
            self.commits += 1

    def on_request_failed(self, request) -> None:
        """Record requests the host, not the injected failure, made fail.

        ``ERR_NETWORK_CHANGED`` is the host's interface churn.
        ``ERR_TOO_MANY_RETRIES`` hits the reload's document request or
        one of the reloaded page's assets about once in five module runs
        on an idle box: Chromium's socket pool restarts a request whose
        connection reported the self-signed certificate
        (``ignore_https_errors``) and gives up after a few restarts
        (crbug "Socket pools should be aware of HTTPS certificate
        errors"); the burst of fresh connections after a reload makes it
        likely.  The guard's next reload recovers the asset case.
        """
        failure = request.failure or ""
        if "ERR_NETWORK_CHANGED" in failure or "ERR_TOO_MANY_RETRIES" in failure:
            self.host_failures.append(f"{request.url}: {failure}")

    def route(self, route) -> None:
        self.asset_requests += 1
        self.asset_log.append(f"document {self.documents}: {route.request.url}")
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
    page.on("framenavigated", loads.on_frame_navigated)
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


def _reload_stamps(page: Page) -> list[float]:
    """The guard's reload timestamps (ms) stored in the tab's sessionStorage."""
    return list(
        page.evaluate(f"JSON.parse(sessionStorage.getItem('{RELOAD_STAMPS_KEY}')) || []")
    )


_CHAT_CLEAR_OF_SIDEBAR_JS = """
() => {
  const app = document.getElementById('app').getBoundingClientRect();
  const sidebar = document.getElementById('sidebar').getBoundingClientRect();
  return app.left >= sidebar.right && sidebar.width > 0;
}
"""


@pytest.mark.parametrize(
    ("asset", "failing_loads"),
    [
        ("api.js", 1),
        ("highlight.min.js", 1),
        ("remote-codex.css", 1),
        ("main.css", 1),
        # The network change outlasts the first reload: the reloaded
        # page's asset fails too, and the second reload recovers.
        ("api.js", 2),
        ("remote-codex.css", 2),
    ],
)
def test_page_reloads_itself_when_an_asset_fails_to_load(
    browser, harness, asset, failing_loads,
):
    _run_unless_network_churn(
        browser, harness, _Loads(asset, abort_first_n=failing_loads), _check_recovered,
    )


def _run_unless_network_churn(
    browser: Browser, harness: ExplorerHarness, loads: _Loads, check,
) -> None:
    """Run ``check(page, loads)`` on a fresh page; retry when the host's network churn interfered.

    The guard reloads at most three times per 30 s, so when the host
    fails one of a *reloaded* page's own assets, or a reload's document
    request itself (``_Loads.on_request_failed`` lists the two ways), the
    counts asserted below are off by that extra reload, the reload budget
    is spent, or the page is Chromium's error page.  That uninjected
    failure is not what is under test: when one was seen the scenario is
    retried, up to three times, in a fresh context (fresh
    ``sessionStorage``).  Any other failure propagates.
    """
    for attempt in range(3):
        loads.reset()
        context, page = _open(browser, harness, loads)
        try:
            check(page, loads)
            return
        except (AssertionError, PlaywrightError):
            if not loads.host_failures or attempt == 2:
                raise
        finally:
            context.close()


def _wait_for_commits(page: Page, loads: _Loads, count: int) -> None:
    """Wait until *count* documents have committed in the main frame.

    ``wait_for_timeout`` (not ``time.sleep``): the sync API runs route
    handlers and delivers events only while a Playwright call is in
    progress.
    """
    deadline = time.monotonic() + 30
    while loads.commits < count and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert loads.commits == count, loads.commits


def _check_recovered(page: Page, loads: _Loads) -> None:
    """The page booted after one reload per aborted load (and no more)."""
    _wait_for_commits(page, loads, loads.abort_first_n + 1)
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_selector("body.remote-desktop", state="attached")
    assert loads.aborted == loads.abort_first_n
    assert loads.asset_requests == loads.abort_first_n + 1, loads.asset_log
    assert loads.documents == loads.abort_first_n + 1, loads.documents
    assert len(_reload_stamps(page)) == loads.abort_first_n
    # The recovered page is fully live: the config reply landed.
    page.wait_for_function(
        "document.getElementById('meta-workdir').textContent.length > 1",
        timeout=30000,
    )
    # ... and styled: the docked history panel sits beside the chat
    # (without remote-codex.css it is a 90vw drawer over the chat,
    # and every click on the chat lands on the history list).
    page.wait_for_function(_CHAT_CLEAR_OF_SIDEBAR_JS, timeout=10000)


def test_a_script_that_keeps_failing_reloads_three_times_then_stays(browser, harness):
    # A really broken asset (every fetch fails) must not put the page in
    # a reload loop: the stamps guard allows three reloads per 30 s.
    _run_unless_network_churn(
        browser, harness, _Loads("api.js", abort_first_n=10**6), _check_reloaded_thrice_then_stayed,
    )


def _check_reloaded_thrice_then_stayed(page: Page, loads: _Loads) -> None:
    # Three reloads plus their page loads.
    loads_expected = RELOAD_BUDGET + 1
    _wait_for_commits(page, loads, loads_expected)
    assert loads.documents == loads_expected, loads.documents
    page.wait_for_load_state("load")
    stamps = _reload_stamps(page)
    assert len(stamps) == RELOAD_BUDGET, stamps
    page.wait_for_timeout(5000)
    assert loads.documents == loads_expected, loads.documents
    assert loads.aborted == loads_expected, loads.aborted
    assert _reload_stamps(page) == stamps
    # The failure stays visible instead of a blank flicker loop.
    assert page.evaluate(
        "getComputedStyle(document.getElementById('app')).display"
    ) == "none"
    assert page.evaluate(
        "getComputedStyle(document.getElementById('kiss-server-loading')).display"
    ) != "none"


"""``goto_retrying_network_change`` and the webapp's dead self-reload.

The remote webapp reloads itself once when one of its scripts fails to
load (``media/assetLoadGuard.js``). In the full-suite run of 2026-09-30
``test_remote_model_dropdown_mobile`` timed out 30 s waiting for
``#model-btn`` although that button is static markup in ``chat.html``:
a host network change had aborted a page script AND the reload's own
document request, so the page sat on ``chrome-error://chromewebdata/``
with an empty DOM while ``page.goto`` had returned normally.

The helper now recognises that state. The retry it performs is
reserved for ``net::ERR_NETWORK_CHANGED``, which only Chromium's network
stack raises for an interface change on the host (no Playwright or CDP
abort code produces it), so that branch is not driven here. It does
strike these tests, though: on this box other jobs' containers add and
remove bridge interfaces (adding and removing Docker networks in a loop
aborted 83 of 2,190 page loads that way on 2026-11-05), so a test that
counts a navigation's document requests routes only the requests it
kills and allows the helper a retry only after a network change struck
the attempt before it. What the tests cover: a healthy load returns at
once, and a self-reload that dies for any other reason is reported with
the failed navigation instead of being retried or left for the caller's
selector wait to time out.
"""

from __future__ import annotations

import functools
import itertools
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from kiss.tests.conftest import (
    goto_retrying_network_change,
    reload_retrying_network_change,
)
from kiss.tests.webapp.test_remote_model_dropdown_mobile import (
    _start_live_server,
)


@pytest.fixture
def base_url(tmp_path: Path) -> Iterator[str]:
    """The production RemoteAccessServer on an ephemeral port."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server, args=(tmp_path, ready, done, state), daemon=True
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        error = state.get("error")
        if isinstance(error, BaseException):
            raise AssertionError("RemoteAccessServer startup failed") from error
        yield f"https://127.0.0.1:{state['port']}/"
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"


#: Gives the app's service-worker registration a second to settle: on a
#: context that allowed workers, ``ready`` resolves and the worker claims
#: the page; on the blocked context it never does.
_SETTLE_SERVICE_WORKER = (
    "() => Promise.race([navigator.serviceWorker.ready, new Promise(r => setTimeout(r, 1000))])"
)


def _note_network_change(network_changes: list[str], request) -> None:
    """``requestfailed`` listener: record a request Chromium aborted for a host network change."""
    if "net::ERR_NETWORK_CHANGED" in str(request.failure):
        network_changes.append(request.url)


@pytest.fixture
def page() -> Iterator[object]:
    """A fresh page in a browser that accepts the server's self-signed cert.

    Service workers are blocked: the app registers ``/sw.js``
    (``transport.js``) as the page loads, and once the worker is active
    it serves a reload's own document request itself, out of the reach
    of ``page.route``, so the abort a test scripts for that request never
    happens.  Whether the worker is active by the time a test reloads is
    a matter of timing (the full-suite run of 2026-11-05 lost the abort
    that way); without a worker the request always reaches the route.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--ignore-certificate-errors"])
        try:
            yield browser.new_page(ignore_https_errors=True, service_workers="block")
        finally:
            browser.close()


def test_a_healthy_load_returns_with_the_app_markup(page, base_url: str) -> None:
    goto_retrying_network_change(page, base_url, wait_until="domcontentloaded")
    assert page.url == base_url
    assert page.query_selector("#model-btn") is not None


def test_a_self_reload_that_dies_is_reported_not_retried(page, base_url: str) -> None:
    scripts_aborted: list[str] = []
    network_changes: list[str] = []
    # The network changes seen as each attempt of the helper began: the
    # first load's document request and, after the self-reload died,
    # each retry's.
    attempts: list[int] = []
    killed: list[str] = []

    def kill_the_reload(route) -> None:
        # The first load goes through (so does its retry after a host
        # network change); every document request after the aborted
        # script, the self-reload first, dies.
        if not scripts_aborted:
            attempts.append(len(network_changes))
            route.continue_()
            return
        if killed:
            attempts.append(len(network_changes))
        killed.append(route.request.url)
        route.abort("connectionaborted")

    def abort_api_once(route) -> None:
        if scripts_aborted:
            route.continue_()
        else:
            scripts_aborted.append(route.request.url)
            route.abort("connectionaborted")

    page.route(base_url, kill_the_reload)
    page.route("**/media/api.js*", abort_api_once)
    page.on("requestfailed", functools.partial(_note_network_change, network_changes))

    with pytest.raises(AssertionError) as info:
        goto_retrying_network_change(page, base_url, wait_until="domcontentloaded")

    message = str(info.value)
    assert "chrome-error://" in message, message
    assert "net::ERR_CONNECTION_ABORTED" in message, message
    # The self-reload after the aborted script died, and the helper
    # retried only after a host network change struck the attempt
    # before (a change seen earlier buys no retry): never for the abort.
    assert killed, (attempts, network_changes)
    assert all(later > earlier for earlier, later in itertools.pairwise(attempts)), (
        attempts, killed, network_changes,
    )
    assert page.url.startswith("chrome-error://")
    assert page.query_selector("#model-btn") is None


def test_a_healthy_reload_returns_with_the_app_markup(page, base_url: str) -> None:
    goto_retrying_network_change(page, base_url, wait_until="domcontentloaded")
    page.evaluate("() => { window.__beforeReload = true; }")
    reload_retrying_network_change(page, wait_until="domcontentloaded")
    assert page.url == base_url
    assert page.query_selector("#model-btn") is not None
    assert page.evaluate("() => window.__beforeReload") is None


def test_a_reload_that_dies_for_another_reason_is_raised_not_retried(page, base_url: str) -> None:
    goto_retrying_network_change(page, base_url, wait_until="domcontentloaded")
    # The case the route must still catch: the worker's registration has
    # had time to activate and claim the page (see the ``page`` fixture).
    page.evaluate(_SETTLE_SERVICE_WORKER)
    assert page.evaluate("() => navigator.serviceWorker.controller") is None

    documents: list[str] = []

    def kill_the_reload(route) -> None:
        documents.append(route.request.url)
        route.abort("connectionaborted")

    # Routed only now: the first load is retried when a host network
    # change strikes it, and a route counting its requests took that
    # retry for the reload (full-suite run of 2026-11-05).
    page.route(base_url, kill_the_reload)
    with pytest.raises(PlaywrightError, match="net::ERR_CONNECTION_ABORTED"):
        reload_retrying_network_change(page, wait_until="domcontentloaded")

    # Exactly one reload attempt: no retry for a failure that is not a
    # network change.
    assert documents == [base_url], documents

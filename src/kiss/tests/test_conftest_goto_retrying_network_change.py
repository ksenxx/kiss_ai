"""``goto_retrying_network_change`` and the webapp's dead self-reload.

The remote webapp reloads itself once when one of its scripts fails to
load (``web_server._WS_SHIM_JS``). In the full-suite run of 2026-09-30
``test_remote_model_dropdown_mobile`` timed out 30 s waiting for
``#model-btn`` although that button is static markup in ``chat.html``:
a host network change had aborted a page script AND the reload's own
document request, so the page sat on ``chrome-error://chromewebdata/``
with an empty DOM while ``page.goto`` had returned normally.

The helper now recognises that state. The retry it performs is
reserved for ``net::ERR_NETWORK_CHANGED``, which only Chromium's network
stack raises for a real interface change on the host (no Playwright or
CDP abort code produces it, and Docker network/container churn did not
trigger it in headless Chromium when tried on 2026-09-30), so that
branch is documented here rather than driven by a test double. What the
tests cover: a healthy load returns at once, and a self-reload that dies
for any other reason is reported with the failed navigation instead of
being retried or left for the caller's selector wait to time out.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_remote_model_dropdown_mobile import (
    _start_live_server,
)
from kiss.tests.conftest import (
    goto_retrying_network_change,
    reload_retrying_network_change,
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


@pytest.fixture
def page() -> Iterator[object]:
    """A fresh page in a browser that accepts the server's self-signed cert."""
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--ignore-certificate-errors"])
        try:
            yield browser.new_page(ignore_https_errors=True)
        finally:
            browser.close()


def test_a_healthy_load_returns_with_the_app_markup(page, base_url: str) -> None:
    goto_retrying_network_change(page, base_url, wait_until="domcontentloaded")
    assert page.url == base_url
    assert page.query_selector("#model-btn") is not None


def test_a_self_reload_that_dies_is_reported_not_retried(page, base_url: str) -> None:
    documents: list[str] = []
    scripts_aborted: list[str] = []

    def kill_the_reload(route) -> None:
        documents.append(route.request.url)
        if len(documents) == 2:
            route.abort("connectionaborted")
        else:
            route.continue_()

    def abort_api_once(route) -> None:
        if scripts_aborted:
            route.continue_()
        else:
            scripts_aborted.append(route.request.url)
            route.abort("connectionaborted")

    page.route(base_url, kill_the_reload)
    page.route("**/media/api.js*", abort_api_once)

    with pytest.raises(AssertionError) as info:
        goto_retrying_network_change(page, base_url, wait_until="domcontentloaded")

    message = str(info.value)
    assert "chrome-error://" in message, message
    assert "net::ERR_CONNECTION_ABORTED" in message, message
    # One self-reload after the aborted script, and no retry of the
    # navigation for a failure that is not a network change.
    assert documents == [base_url, base_url], documents
    assert page.url.startswith("chrome-error://")
    assert page.query_selector("#model-btn") is None


def test_a_healthy_reload_returns_with_the_app_markup(page, base_url: str) -> None:
    goto_retrying_network_change(page, base_url, wait_until="domcontentloaded")
    page.evaluate("() => { window.__beforeReload = true; }")
    reload_retrying_network_change(page, wait_until="domcontentloaded")
    assert page.url == base_url
    assert page.query_selector("#model-btn") is not None
    assert page.evaluate("() => window.__beforeReload") is None


def test_a_reload_that_dies_for_another_reason_is_raised_not_retried(
    page, base_url: str
) -> None:
    documents: list[str] = []

    def kill_the_reload(route) -> None:
        documents.append(route.request.url)
        if len(documents) == 1:
            route.continue_()
        else:
            route.abort("connectionaborted")

    page.route(base_url, kill_the_reload)
    goto_retrying_network_change(page, base_url, wait_until="domcontentloaded")

    with pytest.raises(PlaywrightError, match="net::ERR_CONNECTION_ABORTED"):
        reload_retrying_network_change(page, wait_until="domcontentloaded")

    # The first load plus exactly one reload attempt: no retry for a
    # failure that is not a network change.
    assert documents == [base_url, base_url], documents

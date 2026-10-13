# Author: Koushik Sen (ksen@berkeley.edu)
"""Real browser regressions with a live, TLS-trusted daemon.

WebSocket authentication does not count as an HTML login submission. Previously
Chrome kept the hidden login password pending, then treated Terminal's xterm
fetch as a successful login and opened its native Save Password bubble.

Headless/incognito browsers and untrusted certificates can suppress saving, so
native-prompt tests use headed, persistent Chromium with an isolated NSS trust
store. chrome://password-manager-internals reports native UI decisions that page
screenshots cannot see. No network, browser APIs, or shell behavior are mocked.

The production HTML always has the auth input; the missing-input guard is only
for consumers loading the shim without that HTML. A synchronous exception from
constructing a valid PasswordCredential or calling an available credentials.store
requires a broken/replaced browser API, so it is not manufactured here. Real
promise rejection (cross-origin iframe) and missing API (Firefox) are covered.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, Page, expect, sync_playwright

from kiss.core.vscode_config import save_config
from kiss.server.web_server import RemoteAccessServer, _tls_dir

_PASSWORD = "terminal-regression-secret"
_CM_SAVE = "CM API save a credential:"
_NATIVE_PROMPT = "Message:  Show password prompt"


async def _serve(
    server: RemoteAccessServer, ready: threading.Event, stopped: threading.Event,
) -> None:
    await server.start_async()
    ready.set()
    try:
        while not stopped.is_set():
            await asyncio.sleep(0.02)
    finally:
        await server.stop_async()


def _run_server(
    server: RemoteAccessServer, ready: threading.Event, stopped: threading.Event,
) -> None:
    asyncio.run(_serve(server, ready, stopped))


@pytest.fixture
def password_server(tmp_path: Path) -> Iterator[RemoteAccessServer]:
    """Run the production server with a password and isolated test persistence."""
    save_config({"remote_password": _PASSWORD, "work_dir": str(tmp_path)})
    server = RemoteAccessServer(host="127.0.0.1", port=0, work_dir=str(tmp_path))
    ready, stopped = threading.Event(), threading.Event()
    thread = threading.Thread(target=_run_server, args=(server, ready, stopped))
    thread.start()
    try:
        assert ready.wait(30), "test daemon failed to start"
        yield server
    finally:
        stopped.set()
        thread.join(timeout=30)
        assert not thread.is_alive(), "test daemon failed to stop"


def _url(server: RemoteAccessServer) -> str:
    port = server._ws_server.sockets[0].getsockname()[1]
    return f"https://127.0.0.1:{port}"


@pytest.fixture
def password_browser(
    password_server: RemoteAccessServer, tmp_path: Path,
) -> Iterator[BrowserContext]:
    """Launch a persistent browser with native password saving enabled."""
    if not shutil.which("Xvfb") or not shutil.which("certutil"):
        pytest.skip("native password-manager test needs Xvfb and certutil")
    browser_home = tmp_path / "browser-home"
    nss = browser_home / ".pki" / "nssdb"
    nss.mkdir(parents=True)
    subprocess.run(
        ["certutil", "-N", "-d", f"sql:{nss}", "--empty-password"], check=True,
    )
    subprocess.run(
        ["certutil", "-A", "-d", f"sql:{nss}", "-n", "Test daemon CA",
         "-t", "C,,", "-i", str(_tls_dir() / "ca.pem")], check=True,
    )
    read_fd, write_fd = os.pipe()
    with subprocess.Popen(
        ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "1440x1000x24", "-ac"],
        pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) as xvfb:
        os.close(write_fd)
        try:
            with os.fdopen(read_fd) as display:
                display_name = ":" + display.readline().strip()
            with sync_playwright() as p:
                context = p.chromium.launch_persistent_context(
                    str(tmp_path / "profile"), headless=False,
                    ignore_default_args=["--enable-automation"],
                    env={**os.environ, "HOME": str(browser_home), "DISPLAY": display_name},
                )
                try:
                    yield context
                finally:
                    context.close()
        finally:
            xvfb.terminate()
            xvfb.wait(timeout=10)


def _open_login(context: BrowserContext, url: str) -> tuple[Page, Page]:
    internals = context.pages[0]
    internals.goto("chrome://password-manager-internals")
    page = context.new_page()
    page.goto(url)
    expect(page.locator("#auth-modal")).to_be_visible()
    expect(internals.locator("body")).to_contain_text(
        "IsPasswordManagementEnabledForCurrentPage: true",
    )
    return page, internals


def _unlock(page: Page) -> None:
    page.locator("#auth-modal-input").press_sequentially(_PASSWORD, delay=15)
    page.locator("#auth-modal-input").press("Enter")
    expect(page.locator("#auth-modal")).to_be_hidden()
    expect(page.locator("#auth-modal-input")).to_have_value("")
    page.keyboard.press("Escape")


async def _disconnect_clients(server: RemoteAccessServer) -> None:
    await asyncio.gather(*(
        client.close(code=1012) for client in list(server._ws_server.connections)
    ))


@pytest.mark.timeout(120)
def test_terminal_does_not_trigger_password_save(
    password_browser: BrowserContext, password_server: RemoteAccessServer, tmp_path: Path,
) -> None:
    """Cold Terminal, another Terminal, reconnect, and cached login never re-offer."""
    page, internals = _open_login(password_browser, _url(password_server))
    cdp = password_browser.new_cdp_session(page)
    cdp.send("Profiler.enable")
    cdp.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
    page.reload()  # Compile the inline shim with block coverage enabled.
    _unlock(page)
    expect(internals.locator("body")).to_contain_text(_CM_SAVE)
    page.wait_for_timeout(500)
    before = internals.locator("body").inner_text()
    (tmp_path / "password-manager-before.txt").write_text(before)
    for index in range(2):
        page.locator("#more-btn").click()
        page.locator("#terminal-btn").click()
        terminal = page.locator(".terminal-view").nth(index)
        expect(terminal.locator(".xterm-helper-textarea")).to_be_focused(timeout=30000)
        page.keyboard.type("printf 'terminal-%s\\n' verified")
        page.keyboard.press("Enter")
        expect(terminal).to_contain_text("terminal-verified")

    terminal = page.locator(".terminal-view").last
    navigations: list[object] = []
    page.on("framenavigated", partial(navigations.append))
    assert password_server._loop is not None
    with page.expect_websocket():
        asyncio.run_coroutine_threadsafe(
            _disconnect_clients(password_server), password_server._loop,
        ).result(timeout=30)
    expect(page.locator("#kiss-server-loading")).to_be_hidden()
    terminal.locator(".xterm-helper-textarea").focus()
    page.keyboard.type("printf 'reconnect-%s\\n' verified")
    page.keyboard.press("Enter")
    expect(terminal).to_contain_text("reconnect-verified")
    assert not navigations, "reconnect must not reload the application"
    coverage = cdp.send("Profiler.takePreciseCoverage")
    (tmp_path / "auth-js-coverage.json").write_text(json.dumps(coverage))
    cdp.detach()
    page.reload()
    expect(page.locator("#auth-modal")).to_be_hidden()
    page.wait_for_timeout(1000)
    after = internals.locator("body").inner_text()
    (tmp_path / "password-manager-after.txt").write_text(after)
    assert after.count(_CM_SAVE) == before.count(_CM_SAVE) == 1
    assert after.count(_NATIVE_PROMPT) == before.count(_NATIVE_PROMPT), (
        "Terminal or reconnect triggered Chrome's native Save Password prompt"
    )


@pytest.mark.timeout(120)
def test_other_tabs_cached_login_does_not_offer_an_unsubmitted_password(
    password_browser: BrowserContext, password_server: RemoteAccessServer,
) -> None:
    """An untouched dialog must not turn another tab's cached login into a save offer."""
    waiting, internals = _open_login(password_browser, _url(password_server))
    active = password_browser.new_page()
    active.goto(_url(password_server))
    _unlock(active)
    expect(internals.locator("body")).to_contain_text(_CM_SAVE)
    expect(waiting.locator("#auth-modal")).to_be_visible()
    assert password_server._loop is not None
    with waiting.expect_websocket():
        asyncio.run_coroutine_threadsafe(
            _disconnect_clients(password_server), password_server._loop,
        ).result(timeout=30)
    expect(waiting.locator("#auth-modal")).to_be_hidden()
    expect(waiting.locator("#kiss-server-loading")).to_be_hidden()
    waiting.wait_for_timeout(500)
    assert internals.locator("body").inner_text().count(_CM_SAVE) == 1


@pytest.mark.timeout(120)
def test_wrong_password_is_not_offered_but_successful_retry_is(
    password_browser: BrowserContext, password_server: RemoteAccessServer,
) -> None:
    """Keep rejected text editable, but offer only the accepted password."""
    page, internals = _open_login(password_browser, _url(password_server))
    page.locator("#auth-modal-input").fill("wrong-password")
    page.locator("#auth-modal-ok").click()
    expect(page.locator("#auth-modal-error")).not_to_be_empty()
    expect(page.locator("#auth-modal-input")).to_have_value("wrong-password")
    assert _CM_SAVE not in internals.locator("body").inner_text()
    page.locator("#auth-modal-input").fill(_PASSWORD)
    page.locator("#auth-modal-ok").click()
    expect(page.locator("#auth-modal")).to_be_hidden()
    expect(page.locator("#auth-modal-input")).to_have_value("")
    expect(internals.locator("body")).to_contain_text(_CM_SAVE)
    assert internals.locator("body").inner_text().count(_CM_SAVE) == 1


@pytest.mark.timeout(120)
@pytest.mark.parametrize("cancel", ["button", "escape"])
def test_cancel_clears_password_without_saving(
    password_browser: BrowserContext, password_server: RemoteAccessServer, cancel: str,
) -> None:
    """Cancelled text is erased; reopening starts clean and still authenticates."""
    page, internals = _open_login(password_browser, _url(password_server))
    page.locator("#auth-modal-input").fill("not-submitted")
    if cancel == "button":
        page.locator("#auth-modal-cancel").click()
    else:
        page.locator("#auth-modal-input").press("Escape")
    expect(page.locator("#auth-modal")).to_be_hidden()
    expect(page.locator("#auth-modal-input")).to_have_value("")
    assert _CM_SAVE not in internals.locator("body").inner_text()
    page.get_by_role("button", name="Enter password", exact=True).click()
    expect(page.locator("#auth-modal-input")).to_have_value("")
    _unlock(page)
    expect(internals.locator("body")).to_contain_text(_CM_SAVE)


@pytest.mark.timeout(120)
def test_firefox_authenticates_without_password_credential_api(
    password_server: RemoteAccessServer,
) -> None:
    """Unsupported browsers keep the existing password login and reconnect flow."""
    with sync_playwright() as p:
        if not Path(p.firefox.executable_path).is_file():
            pytest.skip("install Playwright Firefox for the unsupported-API test")
        browser = p.firefox.launch()
        try:
            page = browser.new_page(ignore_https_errors=True)
            page.goto(_url(password_server))
            assert page.evaluate("typeof PasswordCredential") == "undefined"
            _unlock(page)
            page.reload()
            expect(page.locator("#auth-modal")).to_be_hidden()
            expect(page.locator("#task-input")).to_be_visible()
        finally:
            browser.close()


@pytest.mark.timeout(120)
def test_browser_refusing_credential_storage_does_not_block_authentication(
    password_browser: BrowserContext, password_server: RemoteAccessServer,
) -> None:
    """A real cross-origin iframe rejects credentials.store without breaking login."""
    page, _ = _open_login(password_browser, _url(password_server))
    errors: list[object] = []
    page.on("pageerror", partial(errors.append))
    # Different hostname, same real daemon: browsers forbid storing credentials
    # in a cross-origin frame. No authentication or browser method is replaced.
    page.evaluate("""url => {
      const frame = document.createElement('iframe');
      frame.src = url;
      frame.style = 'position:fixed;inset:0;width:100%;height:100%;z-index:999999';
      document.body.appendChild(frame);
    }""", _url(password_server).replace("127.0.0.1", "localhost"))
    frame = page.frame_locator("iframe")
    refusal = frame.locator("#auth-modal-input").evaluate("""async () => {
      try {
        await navigator.credentials.store(new PasswordCredential({
          id: 'permission-probe', password: 'not-a-login'
        }));
        return 'allowed';
      } catch (error) { return error.name; }
    }""")
    assert refusal in ("NotAllowedError", "SecurityError")
    frame.locator("#auth-modal-input").fill(_PASSWORD)
    frame.locator("#auth-modal-ok").click()
    expect(frame.locator("#auth-modal")).to_be_hidden()
    expect(frame.locator("#auth-modal-input")).to_have_value("")
    expect(frame.locator("#task-input")).to_be_visible()
    assert not errors

# Author: Koushik Sen (ksen@berkeley.edu)
"""Real browser regressions with a live daemon on its plain-http loopback URL.

WebSocket authentication does not count as an HTML login submission. Previously
Chrome kept the hidden login password pending, then treated Terminal's xterm
fetch as a successful login and opened its native Save Password bubble.

Headless/incognito browsers can suppress saving, so native-prompt tests use
headed, persistent Chromium. The page is loaded from the ``http://127.0.0.1``
loopback URL the daemon reports (a secure context for browsers, so the
Credential Management API is available without any certificate trust).  A
browser directly on loopback is admitted without a password, so every
browser here sends the ``X-Forwarded-For`` header a cloudflared visitor
carries (``FORWARDED_VISITOR``), which makes the daemon ask for the password.
The header is stripped again from the xterm.js CDN fetches, whose CORS
preflight would otherwise refuse it.
chrome://password-manager-internals reports native UI decisions that page
screenshots cannot see. No network, browser APIs, or shell behavior are mocked.

The production HTML always has the auth input; the missing-input guard is only
for consumers loading the shim without that HTML. A synchronous exception from
constructing a valid PasswordCredential or calling an available credentials.store
requires a broken/replaced browser API, so it is not manufactured here. Real
promise rejection (cross-origin iframe) and missing API (Firefox) are covered.

The last test puts a TCP relay (:class:`_Link`) where a remote visitor's
tunnel stands: the browser's own WebSocket and the daemon exchange their own
bytes through it, late or not at all, as over a slow or a dead link.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import socket
import subprocess
import threading
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest
from playwright.sync_api import (
    BrowserContext,
    Locator,
    Page,
    Request,
    Route,
    WebSocket,
    expect,
    sync_playwright,
)

from kiss.core.user_config import save_config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import FORWARDED_VISITOR
from kiss.tests.webapp.test_sidebar_menus import _show_panel

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
    save_config({
        "remote_password": _PASSWORD, "work_dir": str(tmp_path),
    })
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


def _without_forwarded_header(route: Route, request: Request) -> None:
    """Continue a CDN fetch without the forwarded-IP header meant for the daemon."""
    route.continue_(headers={
        k: v for k, v in request.headers.items() if k.lower() != "x-forwarded-for"
    })


def _as_forwarded_visitor(context: BrowserContext) -> None:
    """Make every page of *context* reach the daemon as a cloudflared visitor would."""
    context.set_extra_http_headers(FORWARDED_VISITOR)
    context.route("https://cdn.jsdelivr.net/**", _without_forwarded_header)


def _url(server: RemoteAccessServer) -> str:
    """The ``http://127.0.0.1:PORT`` URL the daemon itself reports."""
    url = server._loopback_url
    assert url.startswith("http://127.0.0.1:"), url
    return url


@pytest.fixture
def password_browser(
    password_server: RemoteAccessServer, tmp_path: Path,
) -> Iterator[BrowserContext]:
    """Launch a persistent browser with native password saving enabled."""
    if not shutil.which("Xvfb"):
        pytest.skip("native password-manager test needs Xvfb")
    browser_home = tmp_path / "browser-home"
    browser_home.mkdir(parents=True)
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
                _as_forwarded_visitor(context)
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


def _open_terminal(page: Page, index: int) -> Locator:
    """Open a Terminal from the task-info strip; return its view, the *index*-th on the page.

    A tab that opens slides the task-info panel away on the desktop:
    the panel is brought back before the press.

    xterm.js is downloaded from jsDelivr when the terminal opens
    (``terminalTab.js``, 15 s budget).  Under full-suite load that
    download sporadically dies (Chromium drops every request in flight
    when a host interface appears or goes away; the fetch rejects with
    "Failed to fetch") and the tab shows ``.terminal-load-error``
    instead of a terminal.  Such a tab is closed and the terminal opened
    again, three attempts in all, as ``test_remote_terminal_tab`` does.
    """
    for attempt in range(3):
        _show_panel(page)
        page.locator("#terminal-btn").click()
        terminal = page.locator(".terminal-view").nth(index)
        expect(
            terminal.locator(".xterm-helper-textarea, .terminal-load-error"),
        ).to_have_count(1, timeout=30000)
        load_error = terminal.locator(".terminal-load-error")
        if load_error.count() == 0:
            return terminal
        assert attempt < 2, f"xterm.js failed to download three times: {load_error.inner_text()}"
        page.locator("#content-tab-list .active .chat-tab-close").click()
        expect(page.locator(".terminal-load-error")).to_have_count(0)
    raise AssertionError("unreachable")


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
        terminal = _open_terminal(page, index)
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
            context = browser.new_context()
            _as_forwarded_visitor(context)
            page = context.new_page()
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


class _Link:
    """A remote visitor's link to the daemon: a TCP relay on loopback.

    Each side's own bytes pass through it.  While ``answers`` is clear it
    keeps back what the daemon sends on a WebSocket after its answer to the
    upgrade (a slow link); while ``down`` is set every new connection ends
    as it is made (a dead link, see :meth:`cut`).
    """

    def __init__(self, daemon_port: int) -> None:
        self._daemon_port = daemon_port
        self.answers = threading.Event()
        self.answers.set()
        self.down = threading.Event()
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                browser, _ = self._listener.accept()
            except OSError:
                return  # closed
            with self._lock:
                if self.down.is_set():
                    browser.close()
                    continue
                daemon = socket.create_connection(("127.0.0.1", self._daemon_port))
                self._sockets += [browser, daemon]
            for source, sink in ((browser, daemon), (daemon, browser)):
                threading.Thread(
                    target=self._carry, args=(source, sink, source is daemon), daemon=True,
                ).start()

    def _carry(self, source: socket.socket, sink: socket.socket, from_daemon: bool) -> None:
        """Pass what *source* sends on to *sink* until either ends, then end both."""
        head = b""
        try:
            while data := source.recv(65536):
                if b"\r\n\r\n" not in head:
                    head += data  # HTTP, to the end of the connection's first header
                elif from_daemon and head.startswith(b"HTTP/1.1 101"):
                    self.answers.wait()  # a WebSocket frame of the daemon
                sink.sendall(data)
        except OSError:
            pass
        _hang_up(source, sink)
        source.close()

    def cut(self) -> None:
        """Take the link down: its connections end, and each new one as it is made."""
        with self._lock:
            self.down.set()
            _hang_up(*self._sockets)
            self._sockets.clear()

    def close(self) -> None:
        """End the relay: its connections, what it kept back, and its listener."""
        self.cut()
        self.answers.set()
        _hang_up(self._listener)
        self._listener.close()


def _hang_up(*sockets: socket.socket) -> None:
    """Shut *sockets* down both ways, which wakes the threads blocked on them."""
    for sock in sockets:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)


# One WebSocket's frames as Chromium reports them: (direction, type, password).
_SocketLog = list[tuple[str, str, str]]


def _log_socket(logs: list[_SocketLog], ws: WebSocket) -> None:
    """Start the log of *ws*, a socket the page just made, at the end of *logs*."""
    log: _SocketLog = []
    logs.append(log)
    ws.on("framesent", partial(_log_frame, log, "sent"))
    ws.on("framereceived", partial(_log_frame, log, "received"))


def _log_frame(log: _SocketLog, direction: str, payload: str | bytes) -> None:
    frame = json.loads(payload)
    log.append((direction, frame.get("type", ""), frame.get("password", "")))


def _wait_for_frame(page: Page, logs: list[_SocketLog], line: tuple[str, str, str]) -> _SocketLog:
    """Let *page*'s events arrive until its newest socket has logged *line*; return that log."""
    for _ in range(300):
        if line in logs[-1]:
            return logs[-1]
        page.wait_for_timeout(100)
    raise AssertionError(f"no {line} on the newest socket: {logs}")


@pytest.mark.timeout(120)
@pytest.mark.parametrize("first", [_PASSWORD, "mistyped-secret"], ids=["accepted", "refused"])
def test_password_typed_while_an_earlier_one_is_unanswered_waits_for_the_answer(
    password_server: RemoteAccessServer, first: str,
) -> None:
    """A second Unlock never overtakes the password a reconnect has just sent.

    The link dies under the open dialog, so the first Unlock's password goes
    out when the link is back, on a new socket, with the dialog armed again;
    a second Unlock follows while the daemon's answer is still on the link.
    The daemon judges one ``auth`` at a time and reads a frame after an
    accepted one as a command, whose wrong password ends the session: the
    newer password is sent only once the first is refused (as the socket's
    second ``auth``), and the accepted one is stored and goes with the frames
    that follow.  The newer one used to go out at once, and to be stored when
    the first was the one accepted.
    """
    newer = "typed-meanwhile" if first == _PASSWORD else _PASSWORD
    with contextlib.closing(_Link(password_server.port)) as link, sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            _as_forwarded_visitor(context)
            page = context.new_page()
            logs: list[_SocketLog] = []
            page.on("websocket", partial(_log_socket, logs))
            page.goto(link.url)
            dialog, field = page.locator("#auth-modal"), page.locator("#auth-modal-input")
            expect(dialog).to_be_visible()
            with page.expect_websocket():  # the page noticed: it tries again
                link.cut()
            field.fill(first)
            page.locator("#auth-modal-ok").click()
            expect(page.locator("#auth-modal-error")).to_have_text(
                "Cannot reach the server; retrying ...",
            )
            link.answers.clear()
            link.down.clear()
            log = _wait_for_frame(page, logs, ("sent", "auth", first))
            field.fill(newer)
            page.locator("#auth-modal-ok").click()
            expect(dialog).to_be_visible()
            assert log == [("sent", "auth", first)], "only its answer may release the newer one"
            link.answers.set()
            expect(dialog).to_be_hidden()
            expect(field).to_have_value("")
            assert _wait_for_frame(page, logs, ("received", "pong", "")) is log, (
                "the daemon ended the session"
            )
            handshake = [line for line in log if line[1].startswith("auth")]
            refused = [("sent", "auth", first), ("received", "auth_required", "")]
            accepted = [("sent", "auth", _PASSWORD), ("received", "auth_ok", "")]
            assert handshake == (accepted if first == _PASSWORD else refused + accepted)
            later = {password for way, kind, password in log if way == "sent" and kind != "auth"}
            assert later == {_PASSWORD}
            assert page.evaluate("localStorage.getItem('sorcar-remote-pwd')") == _PASSWORD
        finally:
            browser.close()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""UI anti-pattern fixes in the remote webapp server (audit F1-F12).

Every test drives real code: the ``_WS_SHIM_JS`` string runs verbatim
in headless Chromium (Playwright) against an instrumented
``WebSocket`` stub, exactly like
``test_remote_webapp_reconnect_no_reload.py``; the server-side checks
open real WSS / HTTPS connections to a live :class:`RemoteAccessServer`
like ``test_remote_webapp_password_prompt.py``.

Behaviours locked in:

* F1  a wrong password is reported inside the password dialog
  (``role="alert"``), the typed text is kept and focused, and the
  server's close that follows does not bring back the spinner.
* F2  Cancel is no dead end: the overlay explains that a password is
  needed and offers "Enter password" to re-open the dialog.
* F3  the prompt is idempotent: a repeated ``auth_required`` (the
  server dropping an idle socket while the user types) re-uses the
  open dialog, keeps the typed value and sends ONE ``auth`` frame.
* F8  dialog markup: ``role="dialog"``, ``aria-modal``,
  ``aria-labelledby``, a real ``<label>``, "Unlock" as the verb.
* F4  the connect overlay escalates after 8 s with the elapsed time,
  a plain hint and a working "Retry now" button.
* F9  the lockout text counts down and does not blame the reader.
* F5  a prompt over the 1 MB cap is refused with the limit spelled
  out (the draft stays); attachments over the cap produce a notice
  naming what was dropped.
* F11 HTTP 403/404/502 bodies are small plain-language HTML pages.
* F12 the shared chat page is named after the chat title with the
  chat id as suffix; untitled chats keep the old name.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import re
import socket
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import pytest
from playwright.sync_api import sync_playwright
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.config import kiss_home
from kiss.core.models.model_info import get_available_models
from kiss.core.utils import finish
from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.sorcar import TIPS_OPT_OUT_MARKER
from kiss.server.web_server import (
    _MAX_ATTACHMENTS,
    _MAX_PROMPT_BYTES,
    _WS_SHIM_JS,
    RemoteAccessServer,
    _build_html,
    _share_page_filename,
)
from kiss.tests.conftest import goto_retrying_network_change

pytestmark = pytest.mark.usefixtures("stubbed_agent_model")

PAGE_URL = "https://shim-ui.test/"
_PASSWORD = "correct-horse-battery-staple"


# --------------------------------------------------------------------------
# Shim (browser) tests
# --------------------------------------------------------------------------


def _auth_modal_markup() -> str:
    """Return the auth-modal block exactly as ``_build_html`` emits it."""
    page = _build_html()
    start = page.index('<div id="auth-modal"')
    end = page.index('id="auth-modal-ok"')
    for _ in range(4):  # button, actions, content, modal
        end = page.index("</div>", end) + len("</div>")
    return page[start:end]


def _build_test_page() -> str:
    """A page with the real overlay/modal markup, a WS stub and the shim.

    The stub records every constructed socket on ``window.__sockets``
    and exposes ``__fireOpen`` / ``__fireMessage`` / ``__fireClose``;
    ``Date.now`` can be advanced through ``window.__clockOffset`` so
    the 8 s overlay escalation is testable without waiting.  The
    ``daemonStatus`` contract of ``main.js`` (overlay vs ``#app``
    visibility) is wired the same way ``remotePasswordBypass.test.js``
    does it.
    """
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>shim ui test</title></head>
<body>
  <div id="kiss-server-loading" role="status" aria-live="polite">
    <div class="kiss-server-loading-inner">
      <div id="kiss-server-loading-msg">KISS Sorcar Server is starting ...</div>
    </div>
  </div>
  <div id="app" style="display:none;">
{_auth_modal_markup()}
  </div>
  <script>
    window.__sockets = [];
    window.__clockOffset = 0;
    var _realNow = Date.now;
    Date.now = function() {{ return _realNow() + window.__clockOffset; }};
    window.addEventListener('message', function(e) {{
      var d = e && e.data;
      if (!d || d.type !== 'daemonStatus') return;
      document.getElementById('kiss-server-loading').style.display =
        d.connected ? 'none' : '';
      document.getElementById('app').style.display = d.connected ? '' : 'none';
    }});
    function FakeWebSocket(url) {{
      this.url = url; this.readyState = 0; this.sent = [];
      window.__sockets.push(this);
    }}
    FakeWebSocket.CONNECTING = 0; FakeWebSocket.OPEN = 1;
    FakeWebSocket.CLOSING = 2; FakeWebSocket.CLOSED = 3;
    FakeWebSocket.prototype.send = function(d) {{ this.sent.push(d); }};
    FakeWebSocket.prototype.close = function() {{ this.readyState = 3; }};
    window.WebSocket = FakeWebSocket;
    window.__last = function() {{
      return window.__sockets[window.__sockets.length - 1];
    }};
    window.__fireOpen = function() {{
      var ws = window.__last(); ws.readyState = 1;
      if (ws.onopen) ws.onopen({{}});
    }};
    window.__fireMessage = function(msg) {{
      var ws = window.__last();
      if (ws.onmessage) ws.onmessage({{data: JSON.stringify(msg)}});
    }};
    window.__fireClose = function() {{
      var ws = window.__last(); ws.readyState = 3;
      if (ws.onclose) ws.onclose({{}});
    }};
    window.__sentTypes = function() {{
      return window.__last().sent.map(function(d) {{ return JSON.parse(d).type; }});
    }};
    // Run the shim's reconnect backoff immediately (250 ms .. 5 s);
    // shorter delays (the focus call) and longer ones are left alone.
    // With ``__holdReconnect`` set the backoff is parked for 5 s
    // instead, so a test can act while the socket is really CLOSED
    // and a reconnect timer is pending.
    window.__holdReconnect = false;
    var _origSetTimeout = window.setTimeout;
    window.setTimeout = function(fn, ms) {{
      if (typeof fn === 'function' && ms >= 100 && ms <= 5000) {{
        if (window.__holdReconnect) return _origSetTimeout(fn, 5000);
        try {{ fn(); }} catch (e) {{}}
        return 0;
      }}
      return _origSetTimeout(fn, ms);
    }};
  </script>
  <script>
{_WS_SHIM_JS}
  </script>
</body></html>
"""


@pytest.fixture
def _browser():
    """Headless Chromium for one shim test.

    Function-scoped on purpose: the sync Playwright driver keeps an
    asyncio loop running for as long as it is open, and the
    ``IsolatedAsyncioTestCase`` classes below cannot start their own
    loop while it is.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


def _load_page(browser):
    """Open the shim page at a routed origin; return (context, page)."""
    context = browser.new_context()
    page = context.new_page()
    html = _build_test_page()
    page.route(
        PAGE_URL + "**",
        lambda route: route.fulfill(body=html, content_type="text/html"),
    )
    goto_retrying_network_change(page, PAGE_URL, wait_until="load")
    return context, page


def _visible(page, selector: str) -> bool:
    return bool(page.evaluate(
        f"document.querySelector({selector!r}).style.display !== 'none'"
    ))


def _text(page, selector: str) -> str:
    return str(page.evaluate(f"document.querySelector({selector!r}).textContent"))


def _open_and_require_password(page) -> None:
    """Fresh page: socket opens, empty probe answered with auth_required."""
    page.evaluate("window.__fireOpen()")
    page.evaluate("window.__fireMessage({type: 'auth_required'})")
    page.wait_for_timeout(50)
    assert _visible(page, "#auth-modal"), "auth_required opens the dialog"


def test_wrong_password_is_reported_inside_the_dialog(_browser):
    """F1: inline alert, typed text kept and focused, no spinner on close."""
    context, page = _load_page(_browser)
    try:
        _open_and_require_password(page)
        page.fill("#auth-modal-input", "wrong-guess")
        page.press("#auth-modal-input", "Enter")
        assert page.evaluate("window.__sentTypes()") == ["auth", "auth"]
        assert _visible(page, "#auth-modal"), (
            "the dialog stays open while the server checks the password"
        )
        page.evaluate(
            "window.__fireMessage({type: 'error', code: 'auth_failed',"
            " text: 'That password is not correct. Try again.'})"
        )
        page.wait_for_timeout(50)
        assert _text(page, "#auth-modal-error") == (
            "That password is not correct. Try again."
        )
        assert page.evaluate(
            "document.getElementById('auth-modal-error').getAttribute('role')"
        ) == "alert"
        assert _visible(page, "#auth-modal"), "dialog still open after the error"
        assert page.input_value("#auth-modal-input") == "wrong-guess", (
            "the typed text is kept so the user can correct it"
        )
        assert page.evaluate("document.activeElement.id") == "auth-modal-input"
        assert page.evaluate(
            "window.localStorage.getItem('sorcar-remote-pwd')"
        ) is None, "the rejected password is not kept for the reconnect"

        # The server closes the socket after the error: the dialog and
        # the app behind it stay, the spinner does not come back, and
        # the shim reconnects underneath.
        page.evaluate("window.__fireClose()")
        page.wait_for_timeout(50)
        assert _visible(page, "#auth-modal")
        assert not _visible(page, "#kiss-server-loading"), (
            "an auth failure must not show the 'Server is starting' overlay"
        )
        assert page.evaluate("window.__sockets.length") == 2, "reconnected"
        page.evaluate("window.__fireOpen()")
        assert page.evaluate("window.__sentTypes()") == ["auth"], (
            "the reconnect probes with the (empty) stored password"
        )
        assert json.loads(page.evaluate("window.__last().sent[0]"))["password"] == ""
        page.evaluate("window.__fireMessage({type: 'auth_required'})")
        page.wait_for_timeout(50)
        assert page.input_value("#auth-modal-input") == "wrong-guess", (
            "a repeated auth_required never clears what the user typed"
        )

        page.fill("#auth-modal-input", _PASSWORD)
        page.click("#auth-modal-ok")
        sent = [json.loads(d) for d in page.evaluate("window.__last().sent")]
        assert [m["type"] for m in sent] == ["auth", "auth"]
        assert sent[-1]["password"] == _PASSWORD
        page.evaluate("window.__fireMessage({type: 'auth_ok'})")
        page.wait_for_timeout(50)
        assert not _visible(page, "#auth-modal"), "auth_ok closes the dialog"
        assert _text(page, "#auth-modal-error") == ""
        assert _visible(page, "#app")
    finally:
        context.close()


def test_cancel_explains_and_offers_enter_password(_browser):
    """F2: Cancel re-gates the app behind a message with a way back in."""
    context, page = _load_page(_browser)
    try:
        _open_and_require_password(page)
        page.click("#auth-modal-cancel")
        page.wait_for_timeout(50)
        assert not _visible(page, "#auth-modal")
        assert _visible(page, "#kiss-server-loading"), "app is re-gated"
        assert not _visible(page, "#app")
        assert _text(page, "#kiss-server-loading-msg") == (
            "A password is needed to use this server."
        )
        assert _text(page, "#kiss-server-loading-action") == "Enter password"
        assert page.evaluate(
            "document.getElementById('kiss-server-loading-action').tagName"
        ) == "BUTTON"

        page.click("#kiss-server-loading-action")
        page.wait_for_timeout(50)
        assert _visible(page, "#auth-modal"), "the dialog is back"
        assert not _visible(page, "#kiss-server-loading")
        assert page.evaluate(
            "document.getElementById('kiss-server-loading-action')"
        ) is None, "the button is gone with the message it belonged to"
        page.fill("#auth-modal-input", _PASSWORD)
        page.click("#auth-modal-ok")
        sent = [json.loads(d) for d in page.evaluate("window.__last().sent")]
        assert sent[-1] == {"type": "auth", "password": _PASSWORD}
    finally:
        context.close()


def test_escape_behaves_like_cancel(_browser):
    """F2: Escape takes the same non-dead-end path as the Cancel button."""
    context, page = _load_page(_browser)
    try:
        _open_and_require_password(page)
        page.press("#auth-modal-input", "Escape")
        page.wait_for_timeout(50)
        assert not _visible(page, "#auth-modal")
        assert _text(page, "#kiss-server-loading-action") == "Enter password"
    finally:
        context.close()


def test_repeated_auth_required_sends_one_auth_frame(_browser):
    """F3: the dialog is idempotent — one prompt, one listener set."""
    context, page = _load_page(_browser)
    try:
        _open_and_require_password(page)
        page.fill("#auth-modal-input", "half-ty")
        # The server drops the idle socket (60 s recv timeout) while
        # the user is typing; the shim reconnects and gets a second
        # auth_required for the same open dialog.
        page.evaluate("window.__fireClose()")
        page.wait_for_timeout(50)
        assert _visible(page, "#auth-modal"), "close keeps the dialog open"
        assert not _visible(page, "#kiss-server-loading"), (
            "no overlay flicker while the user is typing the password"
        )
        assert page.evaluate("window.__sockets.length") == 2
        page.evaluate("window.__fireOpen()")
        page.evaluate("window.__fireMessage({type: 'auth_required'})")
        page.evaluate("window.__fireMessage({type: 'auth_required'})")
        page.wait_for_timeout(50)
        assert page.input_value("#auth-modal-input") == "half-ty"
        page.fill("#auth-modal-input", _PASSWORD)
        page.press("#auth-modal-input", "Enter")
        types = page.evaluate("window.__sentTypes()")
        assert types == ["auth", "auth"], (
            f"probe + exactly one user auth frame expected, got {types}"
        )
        # A second Enter after submit is inert (listeners detached)
        # until the server answers.
        page.press("#auth-modal-input", "Enter")
        assert page.evaluate("window.__sentTypes()") == ["auth", "auth"]
    finally:
        context.close()


def test_submit_on_a_dead_socket_reconnects_at_once(_browser):
    """F3: Unlock with the socket closed stores the password and reconnects."""
    context, page = _load_page(_browser)
    try:
        _open_and_require_password(page)
        # Park the backoff so the socket is CLOSED with a reconnect
        # pending when the user presses Unlock.
        page.evaluate("window.__holdReconnect = true")
        page.evaluate("window.__fireClose()")
        page.wait_for_timeout(50)
        assert page.evaluate("window.__sockets.length") == 1
        page.fill("#auth-modal-input", _PASSWORD)
        page.click("#auth-modal-ok")
        page.wait_for_timeout(50)
        assert page.evaluate(
            "window.localStorage.getItem('sorcar-remote-pwd')"
        ) == _PASSWORD
        assert page.evaluate("window.__sockets.length") == 2, (
            "submitting on a dead socket must reconnect immediately"
        )
        page.evaluate("window.__fireOpen()")
        sent = [json.loads(d) for d in page.evaluate("window.__last().sent")]
        assert sent == [{"type": "auth", "password": _PASSWORD}]
    finally:
        context.close()


def test_overlay_escalates_with_elapsed_time_and_retry(_browser):
    """F4: after 8 s the overlay says how long, what to check, and offers Retry."""
    context, page = _load_page(_browser)
    try:
        page.evaluate("window.__fireClose()")  # server not reachable
        page.wait_for_timeout(50)
        assert _text(page, "#kiss-server-loading-msg") == (
            "KISS Sorcar Server is starting ..."
        )
        assert page.evaluate(
            "document.getElementById('kiss-server-loading-action')"
        ) is None, "no Retry button before the escalation"
        page.evaluate("window.__clockOffset = 12000")
        page.wait_for_timeout(1300)  # the shim's 1 s overlay tick
        assert re.fullmatch(
            r"Still trying to reach the KISS Sorcar server \(1[23] s\)\. "
            r"Check that it is running on this machine\.",
            _text(page, "#kiss-server-loading-msg"),
        )
        assert _text(page, "#kiss-server-loading-action") == "Retry now"
        # A further failed attempt keeps the escalated text (no flip
        # back to the bare label) and the seconds keep counting.
        page.evaluate("window.__fireClose()")
        page.wait_for_timeout(50)
        assert _text(page, "#kiss-server-loading-msg").startswith(
            "Still trying to reach the KISS Sorcar server ("
        )
        sockets_before = page.evaluate("window.__sockets.length")
        page.click("#kiss-server-loading-action")
        page.wait_for_timeout(50)
        assert page.evaluate("window.__sockets.length") == sockets_before + 1, (
            "Retry now opens a new socket immediately"
        )
        # Once the server answers, the escalation and its button are gone.
        page.evaluate("window.__fireOpen()")
        page.evaluate("window.__fireMessage({type: 'auth_ok'})")
        page.wait_for_timeout(50)
        assert not _visible(page, "#kiss-server-loading")
        assert page.evaluate(
            "document.getElementById('kiss-server-loading-action')"
        ) is None
    finally:
        context.close()


def test_lockout_message_counts_down_without_blame(_browser):
    """F9: the lockout wording is neutral and the seconds tick down."""
    context, page = _load_page(_browser)
    try:
        page.evaluate("window.__fireOpen()")
        page.evaluate("window.__fireMessage({type: 'auth_locked', retry_after: 3})")
        page.wait_for_timeout(50)
        assert _text(page, "#kiss-server-loading-msg") == (
            "Too many wrong passwords from this network. You can try again in 3 s."
        )
        page.evaluate("window.__fireClose()")
        page.wait_for_timeout(1300)
        assert _text(page, "#kiss-server-loading-msg") == (
            "Too many wrong passwords from this network. You can try again in 2 s."
        )
        assert _visible(page, "#kiss-server-loading")
    finally:
        context.close()


def test_pre_auth_error_replaces_the_spinner_text(_browser):
    """F1: a non-password pre-auth error is shown, not hidden behind a spinner."""
    context, page = _load_page(_browser)
    try:
        page.evaluate("window.__fireOpen()")
        page.evaluate(
            "window.__fireMessage({type: 'error', code: 'localhost_only',"
            " text: 'Remote access is turned off.'})"
        )
        page.evaluate("window.__fireClose()")
        page.wait_for_timeout(50)
        assert _text(page, "#kiss-server-loading-msg") == "Remote access is turned off."
        assert _visible(page, "#kiss-server-loading")
    finally:
        context.close()


def test_auth_modal_markup_is_a_labelled_dialog():
    """F8: dialog semantics, a real label, 'Unlock' as the verb."""
    page = _build_html()
    modal = _auth_modal_markup()
    assert 'role="dialog"' in modal
    assert 'aria-modal="true"' in modal
    assert 'aria-labelledby="auth-modal-title"' in modal
    assert 'id="auth-modal-title"' in modal
    assert '<label for="auth-modal-input"' in modal
    assert 'autocomplete="current-password"' in modal
    assert 'id="auth-modal-error" role="alert"' in modal
    assert ">Unlock</button>" in modal
    assert ">OK</button>" not in modal
    assert page.count('id="auth-modal"') == 1


# --------------------------------------------------------------------------
# Server tests (real WSS / HTTPS)
# --------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """Permissive SSL context for the dev self-signed cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _lan_ip() -> str | None:
    """Return a non-loopback IP of this machine, or None if it has none."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))  # TEST-NET-1; nothing is sent.
            ip = str(sock.getsockname()[0])
    except OSError:
        return None
    if ipaddress.ip_address(ip).is_loopback:
        return None
    return ip


class _ScriptedAgent(WorktreeSorcarAgent):
    """Real agent subclass whose ``run`` finishes at once with a fixed result.

    Stands in for the model call only: it allocates the ``task_history``
    row and publishes ``_last_task_id`` the way ``ChatSorcarAgent.run``
    does under ``_skip_persistence=True``, and records the keyword
    arguments the runner passed so a test can inspect them.
    """

    run_kwargs: dict[str, Any] = {}

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate a task row, remember *kwargs*, return a finished result."""
        self.run_kwargs = kwargs
        task_id, self._chat_id = _add_task(
            kwargs.get("prompt_template", ""), chat_id=self._chat_id or "",
        )
        with self._task_id_lock:
            self._last_task_id = task_id
        return finish(True, summary_in_html="<p>done</p>")


def _pop_states(tab_id: str) -> None:
    """Unregister every agent state bound to *tab_id*."""
    with agent_state.STATE_LOCK:
        stale = [st.task_id for st in agent_state.snapshot() if st.tab_id == tab_id]
    for task_id in stale:
        agent_state.unregister(task_id)


class _LiveServer(IsolatedAsyncioTestCase):
    """A real ``RemoteAccessServer`` with a configurable remote password."""

    remote_password = _PASSWORD
    host = "127.0.0.1"

    async def asyncSetUp(self) -> None:
        self._port = _pick_free_port()
        self._work_dir = tempfile.mkdtemp()
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": self.remote_password})
        self._server = RemoteAccessServer(
            host=self.host,
            port=self._port,
            work_dir=self._work_dir,
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        try:
            await self._server.stop_async()
        finally:
            if self._orig_config is not None:
                CONFIG_PATH.write_text(self._orig_config)
            elif CONFIG_PATH.exists():
                CONFIG_PATH.unlink()

    async def _ws(self, host: str = "127.0.0.1") -> Any:
        return await connect(f"wss://{host}:{self._port}/ws", ssl=_no_verify_ssl())

    async def _recv(self, ws: Any) -> dict[str, Any]:
        msg: dict[str, Any] = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        return msg

    async def _authenticated_ws(self) -> Any:
        ws = await self._ws()
        await ws.send(json.dumps({"type": "auth", "password": _PASSWORD}))
        while (await self._recv(ws)).get("type") != "auth_ok":
            pass
        return ws

    async def _http_get(
        self, path: str, host: str = "127.0.0.1",
    ) -> tuple[int, dict[str, str], bytes]:
        """GET *path* over raw HTTPS; return (status, headers, body)."""
        reader, writer = await asyncio.open_connection(
            host, self._port, ssl=_no_verify_ssl(), limit=16 * 1024 * 1024,
        )
        try:
            writer.write(
                f"GET {path} HTTP/1.1\r\nHost: {host}:{self._port}\r\n"
                "Connection: close\r\n\r\n".encode("ascii")
            )
            await writer.drain()
            raw = await asyncio.wait_for(reader.read(), timeout=10)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        head, _, body = raw.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
        return int(lines[0].split(" ")[1]), headers, body


class TestAuthFailureFrame(_LiveServer):
    """F1: the server's wrong-password frame is typed and in plain words."""

    async def test_second_wrong_password_gets_auth_failed_code(self) -> None:
        async with await self._ws() as ws:
            await ws.send(json.dumps({"type": "auth", "password": "wrong"}))
            self.assertEqual((await self._recv(ws))["type"], "auth_required")
            await ws.send(json.dumps({"type": "auth", "password": "still wrong"}))
            err = await self._recv(ws)
        self.assertEqual(err["type"], "error")
        self.assertEqual(err["code"], "auth_failed")
        self.assertEqual(err["text"], "That password is not correct. Try again.")
        self.assertNotIn("Authentication failed", err["text"])


class TestSubmitLimits(_LiveServer):
    """F5: no silent truncation of the prompt or the attachments."""

    async def test_oversize_prompt_is_refused_with_the_limit(self) -> None:
        ws = await self._authenticated_ws()
        try:
            big = "x" * (_MAX_PROMPT_BYTES + 1)
            await ws.send(json.dumps({
                "type": "submit", "tabId": "tab-big", "prompt": big,
                "workDir": self._work_dir, "useWorktree": False,
            }))
            seen: list[dict[str, Any]] = []
            while True:
                msg = await self._recv(ws)
                if msg.get("tabId") != "tab-big":
                    continue
                seen.append(msg)
                if msg.get("type") == "error":
                    break
            types = [m["type"] for m in seen]
            self.assertEqual(types[-2:], ["status", "error"], seen)
            self.assertFalse(seen[-2]["running"], "the composer is re-enabled")
            # R4-2: typed so the webview can hand the draft back to the
            # composer instead of losing what the user typed.
            self.assertEqual(seen[-1]["code"], "prompt_refused")
            self.assertIn("too long to send", seen[-1]["text"])
            self.assertIn("limit is 1 MB", seen[-1]["text"])
            self.assertIn("1.0 MB", seen[-1]["text"])
            # Nothing was started for the tab: no ``clear`` follows.
            with self.assertRaises(asyncio.TimeoutError):
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
                    self.assertNotEqual(
                        (msg.get("type"), msg.get("tabId")), ("clear", "tab-big"),
                        "a refused prompt must not start a task",
                    )
        finally:
            await ws.close()

    async def test_dropped_attachments_notice_survives_the_new_task_clear(
        self,
    ) -> None:
        """R4-3: the notice lands AFTER the run's ``clear``, so it stays.

        The whole server path is real (submit → ``_cmd_run`` → worker
        thread); only the model call is replaced by a scripted agent
        pre-registered for the tab, which ``_cmd_run`` carries into the
        new run.  Every frame of the tab is recorded in wire order.
        """
        models = get_available_models()
        if not models:
            self.skipTest("no models configured in this environment")
        agent = _ScriptedAgent("Sorcar VS Code")
        agent_state.register(
            AgentState("pre-tab-att", agent=agent, tab_id="tab-att", server_owned=True),
        )
        ws = await self._authenticated_ws()
        try:
            count = _MAX_ATTACHMENTS + 3
            await ws.send(json.dumps({
                "type": "submit", "tabId": "tab-att", "prompt": "hi",
                "workDir": self._work_dir, "useWorktree": False,
                "model": models[0],
                "attachments": [{"path": f"f{i}.txt"} for i in range(count)],
            }))
            seen: list[dict[str, Any]] = []
            while True:
                msg = await self._recv(ws)
                if msg.get("tabId") != "tab-att":
                    continue
                seen.append(msg)
                if msg.get("type") == "status" and msg.get("running") is False:
                    break
            types = [m["type"] for m in seen]
            self.assertIn("clear", types, seen)
            self.assertIn("notice", types, seen)
            notice_at = types.index("notice")
            self.assertLess(
                types.index("clear"), notice_at,
                "the notice must not precede the clear that would erase it",
            )
            self.assertNotIn(
                "clear", types[notice_at:],
                "no later reset erases the notice from the transcript",
            )
            self.assertEqual(
                seen[notice_at]["text"],
                f"Only the first {_MAX_ATTACHMENTS} attachments were sent (the "
                f"limit per prompt); the last 3 of your {count} were left out.",
            )
            self.assertEqual(
                len(agent.run_kwargs["attachments"]), _MAX_ATTACHMENTS,
                "the agent received exactly the kept attachments",
            )
        finally:
            _pop_states("tab-att")
            await ws.close()

    async def test_attachments_within_the_cap_produce_no_notice(self) -> None:
        captured: list[dict[str, Any]] = []

        async def _capture(cmd: dict[str, Any]) -> None:
            captured.append(cmd)

        self._server._run_cmd = _capture  # type: ignore[method-assign]
        ws = await self._authenticated_ws()
        try:
            await ws.send(json.dumps({
                "type": "submit", "tabId": "tab-ok", "prompt": "hi",
                "attachments": [{"path": "a.txt"}],
            }))
            with self.assertRaises(asyncio.TimeoutError):
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.7))
                    self.assertNotEqual(msg.get("type"), "notice", msg)
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["attachments"], [{"path": "a.txt"}])
        finally:
            await ws.close()


class TestErrorPages(_LiveServer):
    """F11: HTTP error bodies are readable HTML, not bare log lines."""

    async def test_unknown_path_is_a_plain_language_page(self) -> None:
        status, headers, body = await self._http_get("/definitely/not/here")
        self.assertEqual(status, 404)
        self.assertEqual(headers["content-type"], "text/html; charset=utf-8")
        text = body.decode("utf-8")
        self.assertIn("<!DOCTYPE html>", text)
        self.assertIn("There is nothing at this address", text)
        self.assertIn('<a href="/">', text)
        self.assertIn("<details>", text)
        self.assertIn("/definitely/not/here", text)
        self.assertNotIn("Not Found", text)

    async def test_media_traversal_gets_the_same_page_escaped(self) -> None:
        status, _, body = await self._http_get("/media/..%2F<b>x</b>")
        self.assertEqual(status, 404)
        text = body.decode("utf-8")
        self.assertIn("There is nothing at this address", text)
        self.assertNotIn("<b>x</b>", text, "the echoed path is HTML-escaped")
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", text)


class TestRemoteAccessOffPages(_LiveServer):
    """F11/F1: a LAN visitor with remote access off gets a helpful answer."""

    remote_password = ""
    host = "0.0.0.0"

    def _require_lan_ip(self) -> str:
        ip = _lan_ip()
        if ip is None:
            self.skipTest("machine has no non-loopback interface")
        return ip

    async def test_403_page_points_at_settings_not_config_json(self) -> None:
        ip = self._require_lan_ip()
        status, headers, body = await self._http_get("/", host=ip)
        self.assertEqual(status, 403)
        self.assertEqual(headers["content-type"], "text/html; charset=utf-8")
        text = body.decode("utf-8")
        self.assertIn("This device is not allowed yet", text)
        self.assertIn("open Settings and set a Remote password", text)
        # The config path is kept for administrators, folded away.
        details = text[text.index("<details>"):text.index("</details>")]
        self.assertIn("~/.kiss/config.json", details)
        self.assertNotIn("config.json", text[:text.index("<details>")])

    async def test_ws_from_lan_is_refused_with_the_same_page(self) -> None:
        ip = self._require_lan_ip()
        with self.assertRaises(InvalidStatus) as cm:
            await self._ws(host=ip)
        self.assertEqual(cm.exception.response.status_code, 403)


class TestTipsOptOut(_LiveServer):
    """The remote page's "Don't show tips again" choice is persisted.

    ``tipsOptOut`` writes / removes ``$KISS_HOME/TIPS_DISABLED``, the
    marker the VS Code extension's ``tipsDisabled()`` reads, so the
    choice holds on every surface.
    """

    # ``$KISS_HOME`` is the session-wide temporary directory the test
    # conftest installs (config.json lives there too, so it cannot be
    # swapped per test without breaking the server's password check).
    # The conftest also writes this very marker so the Tips window
    # stays out of the browser suites: it is put back after each test.
    _marker = kiss_home() / TIPS_OPT_OUT_MARKER

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._session_opt_out = self._marker.is_file()
        self._marker.unlink(missing_ok=True)

    async def asyncTearDown(self) -> None:
        self._marker.unlink(missing_ok=True)
        if self._session_opt_out:
            self._marker.write_text("test session opt-out\n")
        await super().asyncTearDown()

    async def _send_and_settle(self, ws: Any, cmd: dict[str, Any]) -> None:
        await ws.send(json.dumps(cmd))
        # No reply is defined for the command; an unknown command
        # would come back as an ``error`` frame, so give it a moment.
        with contextlib.suppress(asyncio.TimeoutError):
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                self.assertNotEqual(msg.get("type"), "error", msg)

    async def test_marker_written_then_removed(self) -> None:
        marker = self._marker
        ws = await self._authenticated_ws()
        try:
            await self._send_and_settle(ws, {"type": "tipsOptOut", "optOut": True})
            self.assertTrue(marker.is_file())
            await self._send_and_settle(ws, {"type": "tipsOptOut"})
            self.assertTrue(marker.is_file(), "absent optOut means opt out")
            await self._send_and_settle(ws, {"type": "tipsOptOut", "optOut": False})
            self.assertFalse(marker.exists())
            await self._send_and_settle(ws, {"type": "tipsOptOut", "optOut": False})
            self.assertFalse(marker.exists(), "removing twice is fine")
        finally:
            await ws.close()

    async def test_marker_in_the_way_is_tolerated(self) -> None:
        """A directory where the marker file should be: logged, not raised."""
        self._marker.mkdir()
        try:
            ws = await self._authenticated_ws()
            try:
                await self._send_and_settle(
                    ws, {"type": "tipsOptOut", "optOut": True},
                )
                await self._send_and_settle(
                    ws, {"type": "tipsOptOut", "optOut": False},
                )
            finally:
                await ws.close()
            self.assertTrue(self._marker.is_dir(), "the directory is left alone")
        finally:
            self._marker.rmdir()


class TestSharePageFilename(_LiveServer):
    """F12: a shared chat is saved under its title, id as suffix."""

    BODY = '<div class="task-panel">Fix the bug</div>'

    async def _share(self, chat_id: str, title: str) -> dict[str, Any]:
        ws = await self._authenticated_ws()
        try:
            await ws.send(json.dumps({
                "type": "shareChat", "chatId": chat_id, "title": title,
                "html": self.BODY, "tabId": "tab-share",
            }))
            while True:
                event = await self._recv(ws)
                if event.get("type") == "share_done":
                    return event
        finally:
            await ws.close()

    async def test_title_slug_then_chat_id(self) -> None:
        event = await self._share("0123abcd" * 4, "My Chat: Fix the bug!")
        self.assertTrue(event["ok"], event)
        expected = (
            Path(self._work_dir) / "reports"
            / f"chat-my-chat-fix-the-bug-{'0123abcd' * 4}.html"
        )
        self.assertEqual(event["path"], str(expected))
        self.assertIn(self.BODY, expected.read_text(encoding="utf-8"))
        # Sharing the same chat again overwrites the same file.
        again = await self._share("0123abcd" * 4, "My Chat: Fix the bug!")
        self.assertEqual(again["path"], str(expected))

    async def test_untitled_chat_keeps_the_old_name(self) -> None:
        event = await self._share("plain-id", "")
        self.assertTrue(event["ok"], event)
        self.assertEqual(
            event["path"], str(Path(self._work_dir) / "reports" / "chat-plain-id.html"),
        )

    def test_filename_rules(self) -> None:
        self.assertEqual(_share_page_filename("", "abc"), "chat-abc.html")
        self.assertEqual(_share_page_filename("!!!", "abc"), "chat-abc.html")
        self.assertEqual(
            _share_page_filename("Hello, World  ", "a/b"), "chat-hello-world-a-b.html",
        )
        long_title = "word " * 40
        name = _share_page_filename(long_title, "id")
        self.assertTrue(name.startswith("chat-word-word"))
        self.assertTrue(name.endswith("-id.html"))
        self.assertLessEqual(len(name), len("chat-") + 60 + 1 + 2 + len(".html"))
        self.assertNotIn("--id", name, "no trailing hyphen from the slug cut")
        self.assertEqual(_share_page_filename("t", "///"), "chat-t-chat.html")

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: clicking a file link in a remote-webapp chat tab.

When the user clicks a file link (``span[data-path]``) in a chat
webview served by the remote webapp, the frontend sends ``openFile``
over the WebSocket; :meth:`RemoteAccessServer._handle_open_file` reads
the file and replies with a ``fileContent`` event; ``media/main.js``
then opens the content in a SEPARATE content tab — code in a read-only
Monaco editor (with a ``pre``/highlight fallback when the CDN is
unreachable) and ``.html`` rendered as a webpage inside a sandboxed
iframe.

Content tabs must never interfere with the chat tabs of agents:

* opening/closing a content tab never sends ``closeTab``/``newTab``
  (or any other message) about a chat tab to the backend;
* the chat tab's input text, output DOM, and tab-bar entry survive
  switching to and from content tabs;
* closing a content tab leaves every chat tab intact.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` — no mocks.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import sync_playwright
from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
import kiss.agents.vscode.vscode_config as vc
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)

_PY_SOURCE = 'def greet(name):\n    return "hello " + name\n'
_HTML_SOURCE = (
    "<!DOCTYPE html><html><body>"
    "<h1 id='marker'>KISS-HTML-MARKER</h1>"
    "</body></html>"
)


def _find_free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class _ServerHarness:
    """A real RemoteAccessServer running on a background event loop."""

    def __init__(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-content-tab-")
        tmp = Path(self.tmpdir)
        # Redirect persistence + config into the sandbox so the test
        # cannot touch (or be influenced by) the developer's ~/.kiss.
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = tmp / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = tmp / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        # Work dir with the files the chat links will point at.
        self.work_dir = tmp / "repo"
        self.work_dir.mkdir()
        (self.work_dir / "sample.py").write_text(_PY_SOURCE)
        (self.work_dir / "page.html").write_text(_HTML_SOURCE)
        (self.work_dir / "binary.bin").write_bytes(b"\x00\x01\x02\x03")

        certfile = tmp / "cert.pem"
        keyfile = tmp / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _find_free_port()
        self.base_url = f"https://127.0.0.1:{self.port}"
        self.ws_url = f"wss://127.0.0.1:{self.port}/ws"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp / "remote-url.json",
            uds_path=tmp / "sorcar.sock",
            work_dir=str(self.work_dir),
        )
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self._thread.start()
        asyncio.run_coroutine_threadsafe(
            self.server.start_async(), self.loop,
        ).result(60)

    def run(self, coro: Coroutine[Any, Any, dict]) -> dict:
        """Run *coro* on the server loop and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(60)

    def stop(self) -> None:
        """Stop the server, its loop, and restore redirected globals."""
        try:
            asyncio.run_coroutine_threadsafe(
                self.server.stop_async(), self.loop,
            ).result(60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=30)
            self.loop.close()
            if th._db_conn is not None:
                th._db_conn.close()
            th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
            vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
            shutil.rmtree(self.tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def harness():
    """One shared real server for every browser test in this module."""
    h = _ServerHarness()
    yield h
    h.stop()


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


def _open_page(browser, harness):
    """Open the webapp, wait for auth, and record every sent WS frame.

    Returns ``(context, page, sent_frames)`` where *sent_frames* is a
    live list of JSON-decoded frames the page sent over the WebSocket.
    """
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    sent_frames: list[dict] = []

    def _on_ws(ws) -> None:
        def _on_sent(payload) -> None:
            try:
                sent_frames.append(json.loads(payload))
            except Exception:
                pass

        ws.on("framesent", _on_sent)

    page.on("websocket", _on_ws)
    page.goto(harness.base_url + "/")
    # Empty remote_password → auth succeeds automatically; the shim
    # reveals #app on auth_ok.
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_selector(".chat-tab", timeout=30000)
    return context, page, sent_frames


def _inject_file_link(page, path: str, link_id: str) -> None:
    """Insert a file link span into the chat output, like linkified
    tool-call output would produce (``span.kiss-filelink[data-path]``).
    """
    page.evaluate(
        """([path, linkId]) => {
             const out = document.getElementById('output');
             const span = document.createElement('span');
             span.className = 'kiss-filelink';
             span.id = linkId;
             span.dataset.path = path;
             span.textContent = path;
             out.appendChild(span);
           }""",
        [path, link_id],
    )


class TestContentTabFileLinks:
    """Browser E2E: file links open formatted content tabs."""

    def test_code_link_opens_separate_tab_with_code(
        self, browser, harness,
    ) -> None:
        """Clicking a .py link opens a new content tab showing the code
        (Monaco, or the pre/code fallback when the CDN is unreachable)
        while the chat tab and its input text stay intact."""
        context, page, sent = _open_page(browser, harness)
        try:
            page.fill("#task-input", "my precious draft")
            real_tabs = page.locator(
                ".chat-tab:not(.chat-tab-add):not(.chat-tab-settings)",
            )
            n_tabs_before = real_tabs.count()
            _inject_file_link(
                page, str(harness.work_dir / "sample.py"), "lnk-code",
            )
            page.click("#lnk-code")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            # New, separate tab — chat tabs untouched.
            n_tabs_after = real_tabs.count()
            assert n_tabs_after == n_tabs_before + 1
            label = page.locator(".chat-tab.content-tab .chat-tab-label")
            assert label.inner_text() == "sample.py"
            # The content tab is active and shows the file content.
            assert "active" in (
                page.locator(".chat-tab.content-tab").get_attribute("class")
            )
            page.wait_for_selector(
                "#content-tab-area .content-tab-view", timeout=30000,
            )
            # Chat surface hidden while the content tab is active.
            assert page.locator("#output").is_hidden()
            assert page.locator("#input-area").is_hidden()
            # Monaco (preferred) or the offline fallback must render
            # the file text.
            page.wait_for_function(
                """() => {
                     const area = document.getElementById('content-tab-area');
                     if (!area) return false;
                     // Monaco renders spaces as U+00A0.
                     const text = area.innerText.replace(/\u00a0/g, ' ');
                     return text.includes('def greet');
                   }""",
                timeout=30000,
            )
            monaco_used = page.locator(
                "#content-tab-area .monaco-editor",
            ).count() > 0
            fallback_used = page.locator(
                "#content-tab-area .content-code-fallback",
            ).count() > 0
            assert monaco_used or fallback_used
            # Switch back to the chat tab: input text preserved, chat
            # surface re-revealed, content area hidden.
            page.click(".chat-tab:not(.content-tab) .chat-tab-label")
            page.wait_for_selector("#task-input", state="visible")
            assert page.input_value("#task-input") == "my precious draft"
            assert page.locator("#output").is_visible()
            assert page.locator("#content-tab-area").is_hidden()
            # And back to the content tab again — still rendered.
            page.click(".chat-tab.content-tab .chat-tab-label")
            page.wait_for_selector("#content-tab-area", state="visible")
        finally:
            context.close()

    def test_html_link_renders_webpage_in_sandboxed_iframe(
        self, browser, harness,
    ) -> None:
        """Clicking an .html link renders the page in a sandboxed
        iframe inside a separate content tab."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page, str(harness.work_dir / "page.html"), "lnk-html",
            )
            page.click("#lnk-html")
            page.wait_for_selector(
                "#content-tab-area .content-html-frame", timeout=30000,
            )
            iframe = page.locator("#content-tab-area .content-html-frame")
            assert iframe.get_attribute("sandbox") == "allow-scripts"
            frame = page.frame_locator("#content-tab-area .content-html-frame")
            assert (
                frame.locator("#marker").inner_text() == "KISS-HTML-MARKER"
            )
            label = page.locator(".chat-tab.content-tab .chat-tab-label")
            assert label.inner_text() == "page.html"
        finally:
            context.close()

    def test_closing_content_tab_never_touches_backend_or_chat_tabs(
        self, browser, harness,
    ) -> None:
        """Opening and closing a content tab must not send closeTab (or
        any tab-lifecycle message) to the backend and must leave the
        chat tab fully intact."""
        context, page, sent = _open_page(browser, harness)
        try:
            page.fill("#task-input", "still here")
            chat_tab_id = page.locator(
                ".chat-tab:not(.chat-tab-add):not(.chat-tab-settings)",
            ).first.get_attribute("data-tab-id")
            _inject_file_link(
                page, str(harness.work_dir / "sample.py"), "lnk-close",
            )
            page.click("#lnk-close")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            content_tab_id = page.locator(
                ".chat-tab.content-tab",
            ).get_attribute("data-tab-id")
            sent.clear()
            page.click(".chat-tab.content-tab .chat-tab-close")
            page.wait_for_selector(
                ".chat-tab.content-tab", state="detached", timeout=30000,
            )
            # Chat tab restored as the active tab with its state.
            page.wait_for_selector("#task-input", state="visible")
            assert page.input_value("#task-input") == "still here"
            remaining = page.locator(
                ".chat-tab:not(.chat-tab-add):not(.chat-tab-settings)",
            )
            assert remaining.count() >= 1
            assert remaining.first.get_attribute("data-tab-id") == chat_tab_id
            # No backend message referenced the content tab, and no
            # closeTab was sent for ANY tab.
            page.wait_for_timeout(500)
            for frame in sent:
                assert frame.get("type") != "closeTab"
                assert frame.get("tabId") != content_tab_id
        finally:
            context.close()

    def test_missing_file_shows_error_notification_no_tab(
        self, browser, harness,
    ) -> None:
        """A link to a nonexistent file shows an error toast and opens
        no content tab."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page, str(harness.work_dir / "nope.py"), "lnk-missing",
            )
            page.click("#lnk-missing")
            page.wait_for_selector(
                ".kiss-notification-error", timeout=30000,
            )
            toast = page.locator(".kiss-notification-error")
            assert "File not found" in toast.inner_text()
            assert page.locator(".chat-tab.content-tab").count() == 0
            assert page.locator("#output").is_visible()
        finally:
            context.close()

    def test_relative_path_resolves_against_work_dir(
        self, browser, harness,
    ) -> None:
        """A relative file link resolves against the tab's work dir."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(page, "sample.py", "lnk-rel")
            page.click("#lnk-rel")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            page.wait_for_function(
                """() => {
                     const area = document.getElementById('content-tab-area');
                     if (!area) return false;
                     // Monaco renders spaces as U+00A0.
                     const text = area.innerText.replace(/\u00a0/g, ' ');
                     return text.includes('def greet');
                   }""",
                timeout=30000,
            )
        finally:
            context.close()

    def test_line_suffix_link_opens_content_tab(
        self, browser, harness,
    ) -> None:
        """A ``path:line`` link (as linkifyFilePaths produces) opens the
        file — the ``:line`` suffix is parsed off, not sent as path."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page,
                str(harness.work_dir / "sample.py") + ":2",
                "lnk-line",
            )
            page.click("#lnk-line")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            label = page.locator(".chat-tab.content-tab .chat-tab-label")
            assert label.inner_text() == "sample.py"
        finally:
            context.close()

    def test_clicking_same_link_twice_reuses_tab(
        self, browser, harness,
    ) -> None:
        """Clicking the same file link twice opens exactly one tab."""
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(
                page, str(harness.work_dir / "sample.py"), "lnk-dup",
            )
            page.click("#lnk-dup")
            page.wait_for_selector(".chat-tab.content-tab", timeout=30000)
            # Go back to the chat tab so the second click is possible.
            page.click(".chat-tab:not(.content-tab) .chat-tab-label")
            page.wait_for_selector("#lnk-dup", state="visible")
            page.click("#lnk-dup")
            page.wait_for_selector(
                ".chat-tab.content-tab.active", timeout=30000,
            )
            assert page.locator(".chat-tab.content-tab").count() == 1
        finally:
            context.close()


async def _ws_request(harness: _ServerHarness, payload: dict) -> dict:
    """Authenticate over wss:// , send *payload*, return the
    ``fileContent`` reply."""
    async with connect(harness.ws_url, ssl=_no_verify_ssl()) as ws:
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == "auth_ok":
                break
        await ws.send(json.dumps(payload))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if msg.get("type") == "fileContent":
                reply: dict = msg
                return reply


class TestOpenFileBackend:
    """Protocol-level tests for the ``openFile`` → ``fileContent``
    request/reply over a real ``wss://`` connection (no browser).

    Each request coroutine runs on the harness's own background event
    loop so these tests coexist with the sync-Playwright tests above
    (which forbid a running event loop in the pytest thread).
    """

    def _request(self, harness: _ServerHarness, payload: dict) -> dict:
        return harness.run(_ws_request(harness, payload))

    def test_absolute_path_returns_content(self, harness) -> None:
        path = str(harness.work_dir / "sample.py")
        reply = self._request(
            harness, {"type": "openFile", "path": path, "tabId": "t-1"},
        )
        assert reply["name"] == "sample.py"
        assert reply["content"] == _PY_SOURCE
        assert reply["tabId"] == "t-1"
        assert "error" not in reply

    def test_relative_path_uses_work_dir_field(self, harness) -> None:
        reply = self._request(harness, {
            "type": "openFile",
            "path": "page.html",
            "workDir": str(harness.work_dir),
        })
        assert reply["name"] == "page.html"
        assert reply["content"] == _HTML_SOURCE

    def test_missing_file_returns_error(self, harness) -> None:
        reply = self._request(
            harness, {"type": "openFile", "path": "/no/such/file.py"},
        )
        assert "File not found" in reply["error"]
        assert "content" not in reply

    def test_binary_file_returns_error(self, harness) -> None:
        path = str(harness.work_dir / "binary.bin")
        reply = self._request(harness, {"type": "openFile", "path": path})
        assert "binary" in reply["error"].lower()
        assert "content" not in reply

    def test_oversized_file_returns_error(self, harness) -> None:
        big = harness.work_dir / "big.txt"
        big.write_text("x" * 2_500_000)
        try:
            reply = self._request(
                harness, {"type": "openFile", "path": str(big)},
            )
            assert "too large" in reply["error"]
            assert "content" not in reply
        finally:
            big.unlink()

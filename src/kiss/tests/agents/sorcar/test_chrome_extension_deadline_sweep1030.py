# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-30 (S11): the extension's command deadline is the caller's ``timeout``.

The shipped extension (``chrome_extension/``, installed by
``ChromeBridge.start``) runs in a real headless Chromium against a real
bridge and a real loopback HTTP page; no part of either side is replaced.

The extension used to fail every command after a fixed 30 s, counted from
the moment the command's queue slot started:

* a ``timeout`` above 30 s could not lengthen it (an ``evaluate`` that
  needs 31 s failed with "command exceeded its 30s deadline");
* at or below 30 s the bridge's own clock (started before the send) always
  ran out first, and ``call()``'s timeout path drops the whole connection,
  failing every other command in flight and forcing a reconnect.

``call()`` now sends ``timeout_ms`` (its remaining wait less a small
margin) and the extension answers by that deadline, counted from the
command's arrival, queued or running, so the connection survives an
overrun.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.request
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from patchright.sync_api import Page, sync_playwright

from kiss.agents.sorcar.chrome_bridge import ChromeBridge, ChromeBridgeError, extension_dir

_PAGE = b"<!doctype html><title>deadline page</title><p>deadline page</p>"


class _PageHandler(BaseHTTPRequestHandler):
    """Serves the one test page."""

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(_PAGE)))
        self.end_headers()
        self.wfile.write(_PAGE)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output quiet."""


def _serve_page() -> ThreadingHTTPServer:
    """Serve the test page on a loopback port, one thread per connection.

    Chromium opens speculative connections it may never send a request
    on; a single-threaded server blocks reading such a connection's
    request line and never answers the page load.
    """
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _PageHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _stop_serving(httpd: ThreadingHTTPServer) -> None:
    httpd.shutdown()
    httpd.server_close()


def test_an_idle_connection_does_not_hold_up_the_page() -> None:
    """A connection that never sends a request (a preconnect) leaves the page served."""
    httpd = _serve_page()
    port = httpd.server_address[1]
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as page:
                assert page.read() == _PAGE
    finally:
        _stop_serving(httpd)


@dataclass
class _Chrome:
    bridge: ChromeBridge
    tab_id: int
    page: Page  # the same tab, driven over CDP by the test (outside the extension)


@pytest.fixture(scope="module")
def chrome(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Chrome]:
    """A bridge, and Chromium running the installed extension connected to it."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KISS_HOME", str(tmp_path_factory.mktemp("kiss_home")))
        httpd = _serve_page()
        url = f"http://127.0.0.1:{httpd.server_address[1]}/"
        bridge = ChromeBridge(write_config=True)
        bridge.start(takeover=True)
        ext = str(extension_dir())
        profile: Path = tmp_path_factory.mktemp("chrome_profile")
        try:
            with sync_playwright() as p:
                context = p.chromium.launch_persistent_context(
                    str(profile),
                    channel="chromium",
                    headless=True,
                    args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}"],
                )
                try:
                    page = context.new_page()
                    page.goto(url)
                    assert bridge.wait_for_extension(30), "the extension never connected"
                    tabs = bridge.call("list_tabs", timeout=10)
                    tab_id = next(tab["id"] for tab in tabs if tab["url"] == url)
                    yield _Chrome(bridge, tab_id, page)
                finally:
                    context.close()
        finally:
            bridge.stop()
            _stop_serving(httpd)


def _evaluate(chrome: _Chrome, script: str, timeout: float) -> object:
    return chrome.bridge.call("evaluate", {"tabId": chrome.tab_id, "script": script}, timeout)


def test_an_overrun_is_answered_and_the_connection_stays(chrome: _Chrome) -> None:
    """A 5 s script with ``timeout=2`` fails with the extension's reply; the socket lives on."""
    with pytest.raises(ChromeBridgeError, match=r"command exceeded its 1\.\d+s deadline"):
        _evaluate(chrome, "new Promise(r => setTimeout(() => r(1), 5000))", timeout=2)
    assert chrome.bridge.extension_connected
    assert _evaluate(chrome, "6 * 7", timeout=10) == 42


def test_a_command_queued_past_its_deadline_is_answered_and_never_runs(chrome: _Chrome) -> None:
    """Queued behind a 3 s script on its tab, a 1.5 s command is refused in time, unrun."""
    chrome.page.evaluate("delete window.__slow_started; delete window.__queued_ran")
    box: dict[str, object] = {}

    def slow() -> None:
        script = "window.__slow_started = true; new Promise(r => setTimeout(() => r(3), 3000))"
        box["slow"] = _evaluate(chrome, script, 10)

    worker = threading.Thread(target=slow)
    worker.start()
    try:
        chrome.page.wait_for_function("window.__slow_started === true", timeout=5000)
        with pytest.raises(ChromeBridgeError, match=r"command exceeded its 1\.\d+s deadline"):
            _evaluate(chrome, "window.__queued_ran = true", timeout=1.5)
        assert chrome.bridge.extension_connected
    finally:
        worker.join(15)
    assert box["slow"] == 3
    assert _evaluate(chrome, "window.__queued_ran === undefined", timeout=10) is True


def test_a_command_without_timeout_ms_keeps_the_30s_deadline(chrome: _Chrome) -> None:
    """An older KISS sends no ``timeout_ms``: a 1 s script still gets the fixed 30 s.

    ``call()`` always sends the field, so the frame is written on the
    bridge's live connection by hand, its reply collected the way
    ``call()`` collects one.
    """
    conn = chrome.bridge._conn
    assert conn is not None
    reply: Future = Future()
    with conn.pending_lock:
        conn.pending[10**9] = reply
    script = "new Promise(r => setTimeout(() => r('old'), 1000))"
    with conn.lock:
        conn.ws.send(json.dumps({
            "id": 10**9, "cmd": "evaluate", "args": {"tabId": chrome.tab_id, "script": script},
        }))
    assert reply.result(10) == "old"


@pytest.mark.slow
def test_a_timeout_above_30s_lengthens_the_deadline(chrome: _Chrome) -> None:
    """A 31 s script succeeds with ``timeout=40`` (the old fixed deadline was 30 s)."""
    script = "new Promise(r => setTimeout(() => r('done'), 31000))"
    assert _evaluate(chrome, script, timeout=40) == "done"

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

The last three tests are of the audit of 2026-11-10 (B/S2-F2, F3, F5): the
snapshot handles and the connection's life, in the same setting.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from patchright.sync_api import BrowserContext, Page, Playwright, sync_playwright
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from kiss.agents.sorcar.chrome_bridge import (
    CONFIG_FILE_NAME,
    ChromeBridge,
    ChromeBridgeError,
    extension_dir,
    install_extension,
)
from kiss.tests.conftest import goto_retrying_network_change

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
def launcher() -> Iterator[Playwright]:
    """What launches this module's Chromiums (a thread runs one ``sync_playwright()`` at a time)."""
    with sync_playwright() as p:
        yield p


@contextmanager
def _chromium(launcher: Playwright, profile: Path) -> Iterator[BrowserContext]:
    """Chromium running the extension installed in ``$KISS_HOME``, on the profile dir *profile*."""
    ext = str(extension_dir())
    context = launcher.chromium.launch_persistent_context(
        str(profile),
        channel="chromium",
        headless=True,
        args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}"],
    )
    try:
        yield context
    finally:
        context.close()


@pytest.fixture(scope="module")
def chrome(tmp_path_factory: pytest.TempPathFactory, launcher: Playwright) -> Iterator[_Chrome]:
    """A bridge, and Chromium running the installed extension connected to it."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KISS_HOME", str(tmp_path_factory.mktemp("kiss_home")))
        httpd = _serve_page()
        url = f"http://127.0.0.1:{httpd.server_address[1]}/"
        bridge = ChromeBridge(write_config=True)
        bridge.start(takeover=True)
        try:
            with _chromium(launcher, tmp_path_factory.mktemp("chrome_profile")) as context:
                page = context.new_page()
                goto_retrying_network_change(page, url)
                assert bridge.wait_for_extension(30), "the extension never connected"
                tabs = bridge.call("list_tabs", timeout=10)
                tab_id = next(tab["id"] for tab in tabs if tab["url"] == url)
                yield _Chrome(bridge, tab_id, page)
        finally:
            bridge.stop()
            _stop_serving(httpd)


def _evaluate(chrome: _Chrome, script: str, timeout: float) -> object:
    return chrome.bridge.call("evaluate", {"tabId": chrome.tab_id, "script": script}, timeout)


def test_an_overrun_is_answered_and_the_connection_stays(chrome: _Chrome) -> None:
    """A 15 s script with ``timeout=10`` fails with the extension's reply; the socket lives on.

    ``timeout=10`` leaves the reply the full 1 s margin (``timeout=2`` left
    0.2 s, which a loaded machine's Chromium missed).
    """
    with pytest.raises(ChromeBridgeError, match=r"command exceeded its (8\.\d+|9)s deadline"):
        _evaluate(chrome, "new Promise(r => setTimeout(() => r(1), 15000))", timeout=10)
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


#: Makes a button that marks the page when clicked, and a text field, the page's content.
_CONTROLS = (
    "document.body.innerHTML = '<button onclick=\"document.title = `hit`\">%s</button>"
    "<input placeholder=Field>'"
)
#: A little over the extension's PING_INTERVAL_MS: a live connection ran a keepalive meanwhile.
_KEEPALIVE_SECS = 27.0


def _uid(snapshot: str, label: str) -> int:
    """The handle *snapshot* gives the element labelled *label*."""
    line = next(ln for ln in snapshot.splitlines() if f'"{label}"' in ln)
    return int(line[1:line.index("]")])


def _assert_stale(bridge: ChromeBridge, tab_id: int, *uids: int) -> None:
    """``click`` and ``type_text`` with each of the handles *uids* fail as stale."""
    for uid in uids:
        for cmd in ("click", "type_text"):
            with pytest.raises(ChromeBridgeError, match="no longer exists"):
                bridge.call(cmd, {"tabId": tab_id, "uid": uid, "text": "x"}, 30)


def test_a_handle_of_an_earlier_document_is_stale_in_the_next(chrome: _Chrome) -> None:
    """B/S2-F5: once the tab shows another document, the old one's handles name nothing.

    Agent A snapshots "Save draft"; agent B navigates the tab and snapshots
    "Delete account".  The uids counted from 1 in every document, so A's
    saved uid clicked B's button and typed into B's field.  A fresh uid now
    comes from the block of the snapshot command's id, which the bridge
    never repeats.
    """
    tab = {"tabId": chrome.tab_id}
    _evaluate(chrome, _CONTROLS % "Save draft", 30)
    first = chrome.bridge.call("snapshot", tab, 30)
    stale = (_uid(first, "Save draft"), _uid(first, "Field"))
    goto_retrying_network_change(chrome.page, chrome.page.url + "next")
    _evaluate(chrome, _CONTROLS % "Delete account", 30)
    _assert_stale(chrome.bridge, chrome.tab_id, *stale)  # no snapshot has seen this document
    later = chrome.bridge.call("snapshot", tab, 30)
    _assert_stale(chrome.bridge, chrome.tab_id, *stale)  # it has handles of its own now
    state = "[document.title, document.querySelector('input').value]"
    assert _evaluate(chrome, state, 30) == ["deadline page", ""]
    own = {**tab, "uid": _uid(later, "Delete account")}
    assert chrome.bridge.call("click", own, 30).startswith("clicked")
    assert _evaluate(chrome, "document.title", 30) == "hit"


def _say_nothing(accepted: threading.Event, left: threading.Event, ws: ServerConnection) -> None:
    """An unrelated listener: takes the connection, sends no challenge, keeps it open."""
    accepted.set()
    try:
        ws.recv()
    except ConnectionClosed:
        left.set()


def test_a_handshake_that_never_completes_is_given_up_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launcher: Playwright
) -> None:
    """B/S2-F3: an open socket that never authenticates does not hold the connection for good.

    A stale ``bridge-config.json`` names a port where an unrelated listener
    accepts the connection and sends no challenge.  ``connect()`` leaves an
    open socket alone and only a close scheduled a retry, so the extension
    stayed there whatever KISS published later.  A socket now has a
    handshake deadline, after which it is closed and the usual retry
    re-reads the config.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss_home"))
    accepted, left = threading.Event(), threading.Event()
    listener = serve(partial(_say_nothing, accepted, left), "127.0.0.1", 0)
    threading.Thread(target=listener.serve_forever, daemon=True).start()
    stale = {"port": listener.socket.getsockname()[1], "token": "stale"}
    (install_extension() / CONFIG_FILE_NAME).write_text(json.dumps(stale), encoding="utf-8")
    bridge = ChromeBridge(write_config=True)
    try:
        with _chromium(launcher, tmp_path / "profile"):
            assert accepted.wait(60), "the extension never reached the stale listener"
            bridge.start(takeover=True)  # KISS publishes; the listener still holds the socket
            assert bridge.wait_for_extension(120), "the extension stayed with the listener"
            assert left.wait(30), "the stalled socket was left open"
    finally:
        bridge.stop()
        listener.shutdown()


@pytest.mark.slow
def test_a_live_connection_follows_the_bridge_the_config_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launcher: Playwright
) -> None:
    """B/S2-F2: a daemon that takes the config over gets the browser from a live CLI bridge.

    The config was read on connection attempts only, and an authenticated
    socket makes none: the extension stayed with the CLI run's bridge for
    as long as that process ran.  The keepalive now re-reads the file and
    gives the socket up when its token is another bridge's.  The handles
    of the old connection die with it: the next bridge numbers its
    commands from 1 again.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss_home"))
    httpd = _serve_page()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    cli, daemon = ChromeBridge(write_config=True), ChromeBridge(write_config=True)
    # Chrome is up first, with the config of a bridge that is gone: the deadline of that
    # refused attempt comes due on the CLI run's live connection, which is not its to close.
    with socket.socket() as gone:
        gone.bind(("127.0.0.1", 0))
        stale = {"port": gone.getsockname()[1], "token": "gone"}
    (install_extension() / CONFIG_FILE_NAME).write_text(json.dumps(stale), encoding="utf-8")
    try:
        with _chromium(launcher, tmp_path / "profile") as context:
            cli.start(takeover=True)
            assert cli.wait_for_extension(60), "the extension never connected"
            goto_retrying_network_change(context.new_page(), url)
            tab = next(t["id"] for t in cli.call("list_tabs", timeout=30) if t["url"] == url)
            cli.call("evaluate", {"tabId": tab, "script": _CONTROLS % "Save draft"}, 30)
            uid = _uid(cli.call("snapshot", {"tabId": tab}, 30), "Save draft")
            # The file's publisher withdrew it: the bridge this socket has is the only one.
            # A keepalive later the same connection (its handle holds) still serves it.
            (extension_dir() / CONFIG_FILE_NAME).unlink()
            time.sleep(_KEEPALIVE_SECS)
            assert cli.call("click", {"tabId": tab, "uid": uid}, 30).startswith("clicked")
            daemon.start(takeover=True)  # the CLI run stays up and keeps its socket open
            assert daemon.wait_for_extension(120), "the daemon never got the browser"
            closed_by = time.monotonic() + 30
            while cli.extension_connected and time.monotonic() < closed_by:
                time.sleep(0.1)
            assert not cli.extension_connected, "the CLI run's socket was left open"
            _assert_stale(daemon, tab, uid)
            # The daemon's own snapshot starts the document's handles over.
            own = _uid(daemon.call("snapshot", {"tabId": tab}, 30), "Save draft")
            assert daemon.call("click", {"tabId": tab, "uid": own}, 30).startswith("clicked")
    finally:
        daemon.stop()
        cli.stop()
        _stop_serving(httpd)

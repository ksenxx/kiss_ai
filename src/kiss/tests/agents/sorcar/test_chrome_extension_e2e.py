# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The bridge extension end to end, every line of it run.

The extension ``install_extension()`` writes runs in a real headless
Chromium, connects to real bridges and drives real loopback pages; no
part of it is replaced.  The last test fails while any code of
``background.js`` has not run: V8 itself counts what ran, in the service
worker and in the pages it injects code into
(``chrome_extension_harness.py``).

Whatever a bridge can drive is driven through a real ``ChromeBridge``.
A :class:`_Peer`, a socket that speaks the bridge's protocol frame by
frame, stands in its place only for what no bridge does: a listener that
is none (silent, sending garbage, a wrong proof, or a command ahead of
its proof), a refusal of the installed extension's own worker, the
frame of an older KISS, a count of the sockets the
extension opened, and a deadline shorter than a bridge's own wait (a
bridge sends nine tenths of its caller's timeout and drops a connection
whose reply is later than the timeout, so the reply to a deadline of
milliseconds never comes back through one).

The waits are the extension's own constants: its keepalive (25 s), its
handshake deadline (10 s), its load wait (15 s) and its retry (3 s).
The one test that waits for the reconnect alarm (30 s) is marked slow:
it runs no code the others leave out (run it with ``-m slow``).
"""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import os
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from patchright.sync_api import sync_playwright
from PIL import Image
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

from kiss.agents.sorcar.chrome_bridge import (
    CONFIG_FILE_NAME,
    MAX_MESSAGE_BYTES,
    ChromeBridge,
    ChromeBridgeError,
    extension_dir,
    extension_version,
    install_extension,
)
from kiss.core.brand import DEFAULT_BRAND, render_brand
from kiss.tests.agents.sorcar.chrome_extension_harness import (
    BACKGROUND_JS,
    ExtensionChromium,
    start_chromium,
    stop_chromium,
)
from kiss.tests.branded_package import SEAMLESS_LOOP, make_branded_package

# What a command answers once its deadline has passed, wherever it was then.
_EXPIRED = re.compile(r"command (exceeded|waited out) its ")
_PIXEL = "data:image/gif;base64,R0lGODlhAQABAAAAACw="

# Every kind of element a snapshot lists or leaves out.  The page logs the
# events that reach its document: "<type>:<id of the target>".
_CONTROLS = f"""<!doctype html><title>Controls</title>
<style>.card {{ cursor: pointer; }} .box {{ display: inline-block; width: 10px; height: 10px; }}
</style>
<button id="save">Save</button>
<button style="display:none">hidden by display</button>
<button style="visibility:hidden">hidden by visibility</button>
<a id="docs" href="/blank">Docs</a>
<a id="away" href="/blank?away" target="_blank">Elsewhere</a>
<a id="stay" href="#top" target="_self">Stay</a>
<input id="q" placeholder="Search">
<input id="filled" value="typed already">
<input name="by-name">
<input id="bare">
<input id="count" type="number" value="5">
<input id="shade" type="color" value="#ff0000" readonly>
<input id="level" type="range" value="20" readonly>
<input id="secret" type="password" value="hunter2">
<input id="kept" value="read only" readonly>
<input type="hidden" value="secret">
<input id="on" type="checkbox" checked>
<input type="checkbox" aria-label="Agree">
<input type="radio" name="plan">
<select id="pick"><option>one</option><option selected>two</option></select>
<select id="empty"></select>
<textarea id="notes" placeholder="Notes"></textarea>
<div id="tenant" role="button" aria-selected="true">Tenant</div>
<div role="presentation" tabindex="0">decoration</div>
<div tabindex="-1">focus target only</div>
<div id="focusable" tabindex="0">Focusable</div>
<button id="bold" tabindex="-1">Bold</button>
<button disabled>Disabled</button>
<fieldset disabled><input id="off" value="switched off"><button id="idle">Idle</button></fieldset>
<div role="checkbox" aria-checked="true">Dark mode</div>
<div id="rich" contenteditable="true">rich text</div>
<div id="editor" contenteditable>bare editor</div>
<div id="plain" contenteditable="plaintext-only">plain editor</div>
<div contenteditable="false">read only</div>
<div contenteditable="true"><input id="inner" value="inside"></div>
<button aria-label="Close"></button>
<img id="logo" class="box" alt="Logo" onclick="note('logo')" src="{_PIXEL}">
<span class="box" title="Tip" tabindex="0"></span>
<div class="card"><span>Nested label</span></div>
<div class="card">Pointer card</div>
<div class="card" aria-label="Close dialog"><svg width="10" height="10"><path d="M1 1l8 8"/></svg>
</div>
<div class="card"><i class="box"></i></div>
<div class="card"><i class="box"></i><span>Labelled</span></div>
<i class="box card"></i>
<ul class="card"><li>Row<button>Edit</button></li></ul>
<svg id="chart" role="button" aria-label="Chart" width="20" height="20">
<rect width="20" height="20"/></svg>
<svg width="20" height="20"><a id="drawn" href="/blank?drawn" target="report" aria-label="Drawn">
<path d="M0 0h20v20H0z"/></a></svg>
<progress id="progress" value="0.5" max="1" onclick="note('progress')"></progress>
<button>{"L" * 100}</button>
<div id="host"></div>
<button id="far" style="display:block;margin-top:3000px">Far away</button>
<script>
window.log = [];
function note(what) {{ log.push(what); }}
const heard = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click', 'input', 'change'];
for (const type of heard) {{
  document.addEventListener(type, (e) => note(type + ':' + e.target.id), true);
}}
const root = document.getElementById('host').attachShadow({{ mode: 'open' }});
root.innerHTML = '<input placeholder="Shadow field"><button>Shadow button</button><div></div>';
root.lastChild.attachShadow({{ mode: 'open' }}).innerHTML = '<button>Deep button</button>';
for (const name of ['gadget', 'select']) {{  // elements of no namespace the browser knows
  const odd = document.createElementNS('urn:odd', name);
  odd.id = name;
  odd.setAttribute('role', 'button');
  odd.setAttribute('aria-label', 'Odd ' + name);
  odd.textContent = 'x';
  document.getElementById('host').before(odd);
}}
</script>"""

# What a snapshot of _CONTROLS lists, its handles left out.
_LISTED = [
    '<button> "Save"',
    '<a> "Docs" -> {site}/blank',
    '<a> "Elsewhere" -> {site}/blank?away',
    '<a> "Stay" -> {site}/controls#top',
    '<input type=text> "Search"',
    '<input type=text> "typed already"',
    '<input type=text> "by-name"',
    "<input type=text>",
    '<input type=number> "5"',
    '<input type=color> "#ff0000"',
    '<input type=range> "20"',
    "<input type=password>",  # what a password field holds is not listed
    '<input type=text> "read only"',
    "<input type=checkbox> [checked]",
    '<input type=checkbox> "Agree"',
    '<input type=radio> "plan"',
    '<select> "two"',
    "<select>",
    '<textarea> "Notes"',
    '<div role=button> "Tenant" [selected]',
    '<div> "Focusable"',
    '<button> "Bold"',
    '<button> "Disabled" [disabled]',
    '<input type=text> "switched off" [disabled]',  # disabled by their fieldset
    '<button> "Idle" [disabled]',
    '<div role=checkbox> "Dark mode" [checked]',
    '<div> "rich text"',
    '<div> "bare editor"',
    '<div> "plain editor"',
    '<input type=text> "inside"',
    '<button> "Close"',
    '<img> "Logo"',
    '<span> "Tip"',
    '<span> "Nested label"',
    '<div> "Pointer card"',
    '<div> "Close dialog"',
    "<div>",
    "<i>",
    '<span> "Labelled"',
    "<i>",
    '<button> "Edit"',
    '<svg role=button> "Chart"',
    '<a> "Drawn" -> /blank?drawn',  # a link of a drawing, by the address it is written with
    "<progress>",
    f'<button> "{"L" * 77}..."',
    '<gadget role=button> "Odd gadget"',
    '<select role=button> "Odd select"',
    '<input type=text> "Shadow field"',
    '<button> "Shadow button"',
    '<button> "Deep button"',
    '<button> "Far away"',
]

_BUSY = "const end = Date.now() + 1500; while (Date.now() < end);"  # 1.5 s of a renderer's time
_PAGES = {
    "/blank": "<!doctype html><title>Blank</title><p>blank</p>",
    "/red": '<!doctype html><title>Red</title><body style="background:#f00">',
    "/blue": '<!doctype html><title>Blue</title><body style="background:#00f">',
    "/controls": _CONTROLS,
    "/many": "<!doctype html><title>Many</title><script>"
    "for (let i = 0; i < 205; i++) document.write('<button>B' + i + '</button>');</script>",
    # Never done loading: its image is still on its way after a minute.
    "/pending": '<!doctype html><title>Pending</title><img src="/blank?ms=60000">',
    # Not even read to its end: the script it stops at is as far away.
    "/unread": '<!doctype html><title>Unread</title><script src="/blank?ms=60000"></script>',
    # Pages that replaced eval: one refuses with a string, one after a busy while.
    "/noeval": "<!doctype html><title>No eval</title><script>"
    "window.eval = function () { throw 'eval is off here'; };</script>",
    "/sloweval": "<!doctype html><title>Slow eval</title><script>"
    f"window.eval = function () {{ {_BUSY} throw new EvalError('eval is off'); }};</script>",
    # Busy once its policy has refused a script (served as /csp/stall).
    "/stall": "<!doctype html><title>Stall</title><script>"
    f"document.addEventListener('securitypolicyviolation', () => {{ {_BUSY} }}, {{ once: true }});"
    "</script>",
}


class _Site(BaseHTTPRequestHandler):
    """Serves ``_PAGES``; ``/csp/<page>`` forbids ``eval``; ``?ms=N`` answers N ms late."""

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urlsplit(self.path)
        time.sleep(int(parse_qs(url.query).get("ms", ["0"])[0]) / 1000)
        csp = url.path.startswith("/csp/")
        body = _PAGES.get(url.path[4:] if csp else url.path, "").encode()
        try:
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "text/html")
            if csp:
                self.send_header("Content-Security-Policy", "script-src 'unsafe-inline'")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass  # the tab was closed before its late answer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output quiet."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hello(token: str, nonce: str) -> dict[str, str]:
    """The extension's answer to the challenge *nonce*: its proof, and the install it is of."""
    return {"type": "hello", "proof": _sha(f"{token}:{nonce}"), "version": extension_version()}


def _publish(port: int, token: str, padding: int = 0) -> None:
    """Write bridge-config.json as a bridge does: the whole file or the old one.

    *padding* bytes of white space after it make the file slow to read.
    """
    config = extension_dir() / CONFIG_FILE_NAME
    staged = config.with_name("staged-config")
    staged.write_text(json.dumps({"port": port, "token": token}) + " " * padding)
    staged.replace(config)


def _send_together(ws: ServerConnection, *texts: str) -> None:
    """Write *texts* as frames in one TCP segment: the extension has them all at once.

    Each is shorter than 126 bytes, the most a frame's two-byte header announces.
    """
    assert all(len(text.encode()) < 126 for text in texts)
    ws.socket.sendall(b"".join(bytes([0x81, len(text)]) + text.encode() for text in texts))


class _Peer:
    """A process on the bridge's socket that speaks its protocol frame by frame.

    Stands where a bridge stands: it listens on a loopback port and
    names it, with its token, in ``bridge-config.json``.  A test takes
    each connection of the extension with :meth:`accept` and writes and
    reads its frames itself.
    """

    def __init__(self, padding: int = 0) -> None:
        """Listen and publish the config, followed by *padding* bytes of white space."""
        self.token = secrets.token_hex(16)
        self._accepted: queue.Queue[ServerConnection] = queue.Queue()
        self._closed = threading.Event()
        self._server = serve(self._hold, "127.0.0.1", 0)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        _publish(self._server.socket.getsockname()[1], self.token, padding)

    def _hold(self, ws: ServerConnection) -> None:
        self._accepted.put(ws)
        self._closed.wait()  # the connection lives until the peer closes

    def accept(self, timeout: float = 30.0) -> ServerConnection:
        """The extension's next connection, not yet greeted."""
        return self._accepted.get(timeout=timeout)

    def login(self) -> _Link:
        """The extension's next connection, authenticated as the bridge does it."""
        ws = self.accept()
        nonce = secrets.token_hex(16)
        ws.send(json.dumps({"type": "challenge", "nonce": nonce}))
        assert json.loads(ws.recv(timeout=10)) == _hello(self.token, nonce)
        ws.send(json.dumps({"type": "hello_ok", "proof": _sha(f"{nonce}:{self.token}")}))
        return _Link(ws)

    def close(self) -> None:
        """Close every connection and stop listening."""
        self._closed.set()
        self._server.shutdown()


class _Link:
    """A connection of the extension a :class:`_Peer` authenticated: commands out, replies in."""

    def __init__(self, ws: ServerConnection) -> None:
        self.ws = ws
        self._ids = itertools.count(1)
        self._replies: dict[int, dict[str, Any]] = {}

    def send(self, cmd: str, args: dict[str, Any], timeout_ms: float = 15000) -> int:
        """Send a command frame with the deadline *timeout_ms*; return its id."""
        msg_id = next(self._ids)
        frame = {"id": msg_id, "cmd": cmd, "args": args, "timeout_ms": timeout_ms}
        self.ws.send(json.dumps(frame))
        return msg_id

    def reply(self, msg_id: int) -> dict[str, Any]:
        """The reply frame to command *msg_id*, whatever order the replies come in."""
        while msg_id not in self._replies:
            frame = json.loads(self.ws.recv(timeout=30))
            if "id" in frame:  # not a ping
                self._replies[frame["id"]] = frame
        return self._replies.pop(msg_id)

    def ask(self, cmd: str, args: dict[str, Any], timeout_ms: float = 15000) -> dict[str, Any]:
        """Send a command frame and return its reply frame."""
        return self.reply(self.send(cmd, args, timeout_ms))

    def result(self, cmd: str, args: dict[str, Any]) -> Any:
        """Run a command that must succeed and return its result."""
        reply = self.ask(cmd, args)
        assert reply["ok"], reply
        return reply["result"]


def _in_thread(fn: Callable[..., Any], *args: Any) -> Future[Any]:
    """Run ``fn(*args)`` in a thread; the future holds its result or its exception."""
    outcome: Future[Any] = Future()

    def run() -> None:
        try:
            outcome.set_result(fn(*args))
        except BaseException as e:  # noqa: BLE001 - handed to the test
            outcome.set_exception(e)

    threading.Thread(target=run, daemon=True).start()
    return outcome


def _until(condition: Callable[[], Any], timeout: float = 10.0) -> None:
    """Wait until ``condition()`` is true."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the condition never came true"
        time.sleep(0.02)


def _tab_shows(bridge: ChromeBridge, wanted: dict[str, Any]) -> bool:
    """Whether a tab of the browser has every one of the *wanted* properties."""
    return any(wanted.items() <= tab.items() for tab in bridge.call("list_tabs", timeout=10))


@dataclass
class _Chrome:
    """The browser with the extension, the test site, and the bridge the extension is on."""

    browser: ExtensionChromium
    site: str
    bridge: ChromeBridge | None = None

    def live_bridge(self) -> ChromeBridge:
        """A real bridge the extension is connected to (the last test's, while it lasts)."""
        if self.bridge is not None and self.bridge.extension_connected:
            return self.bridge
        return self.new_bridge()

    def new_bridge(self) -> ChromeBridge:
        """Start a bridge and wake the extension, which connects to it anew."""
        self.release()
        self.bridge = ChromeBridge(write_config=True)
        self.bridge.start(takeover=True)
        self.browser.click_toolbar_icon()
        assert self.bridge.wait_for_extension(30), "the extension never connected"
        return self.bridge

    def release(self) -> None:
        """Stop the bridge: the extension is free for another one, or for a peer."""
        if self.bridge is not None:
            self.bridge.stop()
            self.bridge = None

    @contextmanager
    def peer(self, padding: int = 0, clicks: int = 1) -> Iterator[_Peer]:
        """A :class:`_Peer` in the bridge's place, the extension woken *clicks* times at once."""
        self.release()
        peer = _Peer(padding)
        self.browser.click_toolbar_icon(clicks)
        try:
            yield peer
        finally:
            peer.close()

    def fresh_worker(self) -> None:
        """Start a new service worker whose coverage is measured (after one Chrome ended)."""
        self.release()
        self.browser.new_worker()

    def open(self, path: str) -> tuple[int, str]:
        """Open ``site + path`` in a new tab through the bridge; return its tab id and URL."""
        url = self.site + path
        return self.live_bridge().call("open_tab", {"url": url}, 20)["id"], url

    def evaluate(self, tab: int, script: str, timeout: float = 10.0) -> Any:
        """Run *script* in *tab* through the bridge."""
        return self.live_bridge().call("evaluate", {"tabId": tab, "script": script}, timeout)

    def snapshot(self, tab: int) -> list[str]:
        """The lines of a snapshot of *tab*: its head, then one line per element."""
        return str(self.live_bridge().call("snapshot", {"tabId": tab}, 10)).split("\n")

    def click(self, tab: int, uid: int) -> str:
        """Click the element *uid* of *tab*; return the extension's description of it."""
        return str(self.live_bridge().call("click", {"tabId": tab, "uid": uid}, 10))

    def type_text(self, tab: int, uid: int, text: str) -> str:
        """Type *text* into the element *uid* of *tab*."""
        args = {"tabId": tab, "uid": uid, "text": text}
        return str(self.live_bridge().call("type_text", args, 10))


def _uid(lines: list[str], element: str) -> int:
    """The handle of the one snapshot line that reads ``[N] <element>``."""
    (line,) = [line for line in lines if line.split("] ", 1)[-1] == element]
    return int(line[1 : line.index("]")])


@pytest.fixture(scope="module")
def chromium() -> str:
    """The Chromium this module's browsers run: the one installed for Playwright."""
    with sync_playwright() as p:
        return str(p.chromium.executable_path)


@pytest.fixture(scope="module")
def chrome(tmp_path_factory: pytest.TempPathFactory, chromium: str) -> Iterator[_Chrome]:
    """Chromium running the extension installed in a temporary ``$KISS_HOME``."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KISS_HOME", str(tmp_path_factory.mktemp("kiss_home")))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            browser = ExtensionChromium(
                chromium, tmp_path_factory.mktemp("chrome_profile"), install_extension()
            )
            chrome = _Chrome(browser, f"http://127.0.0.1:{httpd.server_address[1]}")
            try:
                yield chrome
            finally:
                chrome.release()
                browser.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


@pytest.fixture(autouse=True)
def _pages_closed(request: pytest.FixtureRequest) -> Iterator[None]:
    """After each test with the browser: record what its pages ran, then close them all."""
    yield
    if "chrome" in request.fixturenames:
        request.getfixturevalue("chrome").browser.close_pages()


# ------------------------------------------------------------ the connection

_ALARMS = "chrome.alarms.getAll().then((alarms) => alarms.length)"


def _close_code(ws: ServerConnection) -> int:
    """Read *ws* until the extension closes it; return its close code."""
    with pytest.raises(ConnectionClosed) as closed:
        while True:
            ws.recv(timeout=20)
    assert closed.value.rcvd is not None
    return closed.value.rcvd.code


def test_no_command_runs_before_both_sides_proved_the_token(chrome: _Chrome) -> None:
    """The extension proves the token without sending it; commands wait for the server's proof."""
    early = chrome.site + "/blank?early"
    with chrome.peer() as peer:
        ws = peer.accept()
        ws.send(json.dumps({"id": 1000, "cmd": "open_tab", "args": {"url": early}}))
        nonce = secrets.token_hex(16)
        ws.send(json.dumps({"type": "challenge", "nonce": nonce}))
        assert json.loads(ws.recv(timeout=20)) == _hello(peer.token, nonce)
        # The proof with a command right behind it, as the bridge may send
        # them: the command waits for the check of the proof, then runs.
        proof = json.dumps({"type": "hello_ok", "proof": _sha(f"{nonce}:{peer.token}")})
        _send_together(ws, proof, json.dumps({"id": 7, "cmd": "list_tabs"}))
        reply = json.loads(ws.recv(timeout=20))
        assert (reply["id"], reply["ok"]) == (7, True)
        assert early not in [tab["url"] for tab in reply["result"]]


def test_a_listener_without_the_token_runs_no_command(chrome: _Chrome) -> None:
    """A wrong server proof closes the socket with 4003; the command behind it never runs."""
    intruded = chrome.site + "/blank?intruded"
    command = json.dumps({"id": 1, "cmd": "open_tab", "args": {"url": intruded}})
    began = time.monotonic()
    with chrome.peer() as rogue:
        ws = rogue.accept()
        ws.send(json.dumps({"type": "challenge", "nonce": secrets.token_hex(16)}))
        hello = json.loads(ws.recv(timeout=20))
        # The one proof a listener without the token has: the extension's own, sent back.
        reflected = json.dumps({"type": "hello_ok", "proof": hello["proof"]})
        _send_together(ws, reflected, command)
        assert _close_code(ws) == 4003
    bridge = chrome.new_bridge()
    assert intruded not in [tab["url"] for tab in bridge.call("list_tabs", timeout=10)]
    # The refused socket's handshake deadline (10 s) passes on a socket that
    # is gone: the connection that replaced it is not the one given up.
    time.sleep(max(began + 11 - time.monotonic(), 0))
    assert bridge.extension_connected
    assert intruded not in [tab["url"] for tab in bridge.call("list_tabs", timeout=10)]


def test_frames_that_are_no_commands_are_passed_over(chrome: _Chrome) -> None:
    """Garbage, a pong, an id that is no number, a frame without a command: no reply, no harm."""
    with chrome.peer() as peer:
        link = peer.login()
        for frame in (
            "no json",
            "null",
            '{"type": "pong"}',
            '{"id": "7", "cmd": "list_tabs"}',
            '{"id": 7}',
        ):
            link.ws.send(frame)
        # An older KISS sends neither args nor timeout_ms.
        link.ws.send(json.dumps({"id": 8, "cmd": "list_tabs"}))
        first = json.loads(link.ws.recv(timeout=10))  # nothing answered the five frames before it
        assert (first["id"], first["ok"]) == (8, True)


def test_wake_ups_that_come_together_make_one_connection(chrome: _Chrome) -> None:
    """Two toolbar clicks at once, the second while the first still reads the config: one socket."""
    # Counted by a peer: a bridge does not say how many sockets it accepted.
    with chrome.peer(clicks=2) as peer:
        peer.login()
        with pytest.raises(queue.Empty):
            peer.accept(timeout=2)
    # Clicks on a live connection open no socket either: the bridge would take a second
    # one for another profile's, replace the connection and fail the command in flight.
    tab, _ = chrome.open("/blank?clicks")
    slow = "new Promise((ok) => setTimeout(() => ok('done'), 1500))"
    in_flight = _in_thread(chrome.evaluate, tab, slow)
    time.sleep(0.3)
    chrome.browser.click_toolbar_icon(2)
    assert in_flight.result(10) == "done"


def test_a_port_that_never_answers_is_given_up_after_ten_seconds(chrome: _Chrome) -> None:
    """What is no bridge holds the port of the file: one socket at a time, each given up."""
    chrome.release()
    with closing(socket.create_server(("127.0.0.1", 0))) as listener:
        listener.settimeout(15)
        _publish(listener.getsockname()[1], "stale")
        chrome.browser.click_toolbar_icon()
        first, _ = listener.accept()
        with closing(first):
            began = time.monotonic()
            first.settimeout(15)
            assert b"Upgrade: websocket" in first.recv(65536)
            # A wake-up while that socket is still connecting opens no second one.
            chrome.browser.click_toolbar_icon()
            listener.settimeout(2)
            with pytest.raises(TimeoutError):
                listener.accept()
            assert first.recv(65536) == b""  # given up
            assert 7 < time.monotonic() - began < 14
        listener.settimeout(15)
        listener.accept()[0].close()  # and tried again


def test_the_extension_finds_a_bridge_started_later(chrome: _Chrome) -> None:
    """No bridge, then a dead port in the file: it keeps trying and finds the next bridge."""
    chrome.new_bridge()
    # The alarm that starts a new worker after Chrome ended this one is
    # armed on a live connection, and still armed with the bridge gone.
    assert chrome.browser.worker_eval(_ALARMS) == 1
    chrome.release()  # the bridge withdraws its file
    time.sleep(3.5)  # one retry finds no file
    assert chrome.browser.worker_eval(_ALARMS) == 1
    with closing(socket.create_server(("127.0.0.1", 0))) as gone:
        port = gone.getsockname()[1]
    _publish(port, "left behind")  # a bridge that died left its file: the port refuses
    time.sleep(3.5)
    chrome.bridge = ChromeBridge(write_config=True)
    chrome.bridge.start(takeover=True)
    assert chrome.bridge.wait_for_extension(15), "the retry did not find the new bridge"
    assert isinstance(chrome.bridge.call("list_tabs", timeout=10), list)


def test_a_second_profile_keeps_the_first_away_until_its_icon_is_clicked(chrome: _Chrome) -> None:
    """Replaced by another profile's extension (close 4002), this one stays down, alarm disarmed."""
    bridge = chrome.new_bridge()
    config = json.loads((extension_dir() / CONFIG_FILE_NAME).read_text())
    with connect(f"ws://127.0.0.1:{config['port']}") as other:  # the other profile's extension
        nonce = json.loads(other.recv(timeout=10))["nonce"]
        other.send(json.dumps(_hello(config["token"], nonce)))
        assert json.loads(other.recv(timeout=10))["type"] == "hello_ok"
        _until(lambda: chrome.browser.worker_eval(_ALARMS) == 0)  # no alarm to wake a later worker
        # A tick that comes with the alarm gone (one on its way when it was
        # disarmed) wakes nothing: it is over once its one-shot alarm is gone.
        chrome.browser.worker_eval("chrome.alarms.create('tick', { when: Date.now() })")
        _until(lambda: chrome.browser.worker_eval(_ALARMS) == 0)
        # A reconnect would replace this connection in turn; none comes,
        # from the tick or from the 3 s retry.
        with pytest.raises(TimeoutError):
            other.recv(timeout=5)
    _until(lambda: not bridge.extension_connected)
    chrome.browser.click_toolbar_icon()
    assert bridge.wait_for_extension(15), "the toolbar icon did not wake the extension"
    assert isinstance(bridge.call("list_tabs", timeout=10), list)


def test_the_commands_of_a_connection_that_closed_do_not_run(chrome: _Chrome) -> None:
    """Still queued when its connection closes, a command is dropped: its page never sees it."""
    tab, url = chrome.open("/blank?closed")
    slow = "new Promise((ok) => setTimeout(() => ok('slow'), 1500))"
    running = _in_thread(chrome.evaluate, tab, slow)
    time.sleep(0.3)  # the first is running: the second waits for the tab
    queued = _in_thread(chrome.evaluate, tab, "document.title = 'ran after the close'")
    time.sleep(0.3)
    chrome.release()  # the bridge stops, and its connection closes
    for call in (running, queued):
        with pytest.raises(ChromeBridgeError):
            call.result(10)
    time.sleep(2.5)  # the first is over by now, and the second's turn has come
    assert chrome.browser.page_eval(url, "document.title") == "Blank"
    assert chrome.evaluate(tab, "document.title") == "Blank"  # the next bridge's commands run


def test_the_keepalive_follows_the_config_file(chrome: _Chrome) -> None:
    """Every 25 s the file is read again: gone, the connection stays; with a new token

    (the daemon published its bridge over a CLI run's), the extension moves there.
    """
    first = chrome.new_bridge()
    (extension_dir() / CONFIG_FILE_NAME).unlink()
    time.sleep(26)  # a keepalive without the file
    assert first.extension_connected
    assert isinstance(first.call("list_tabs", timeout=10), list)
    second = ChromeBridge(write_config=True)
    second.start(takeover=True)
    chrome.bridge = second
    try:
        assert second.wait_for_extension(40), "the keepalive did not notice the new bridge"
        assert not first.extension_connected
        assert isinstance(second.call("list_tabs", timeout=10), list)
    finally:
        first.stop()


@pytest.mark.slow
def test_a_worker_chrome_ended_comes_back_on_its_own(chrome: _Chrome) -> None:
    """Chrome ends the worker under a live connection: its alarm (30 s) starts the next one."""
    bridge = chrome.new_bridge()
    try:
        assert chrome.browser.worker_eval(_ALARMS) == 1
        chrome.browser.kill_worker()
        _until(lambda: not bridge.extension_connected)
        assert bridge.wait_for_extension(45), "nothing woke the extension"
        assert isinstance(bridge.call("list_tabs", timeout=10), list)
    finally:
        chrome.fresh_worker()


def test_a_debugger_left_by_an_ended_worker_does_not_block_its_tab(chrome: _Chrome) -> None:
    """The attachment of a worker Chrome ended mid-script is dropped by the next worker."""
    tab, url = chrome.open("/csp/blank?orphan")
    bridge = chrome.live_bridge()
    script = "document.title = 'attached'; new Promise(() => {})"
    # Why the extension is refused a debugger on the tab ("" when it gets one):
    # it has one there already.
    refusal = (
        f"chrome.debugger.attach({{ tabId: {tab} }}, '1.3')"
        f".then(() => chrome.debugger.detach({{ tabId: {tab} }}))"
        ".then(() => '', (e) => e.message)"
    )
    try:
        stuck = _in_thread(chrome.evaluate, tab, script, 60)
        _until(lambda: chrome.browser.page_eval(url, "document.title") == "attached")
        assert "Another debugger is already attached" in chrome.browser.worker_eval(refusal)
        chrome.browser.kill_worker()
        with pytest.raises(ChromeBridgeError, match="extension disconnected"):
            stuck.result(10)
        chrome.browser.click_toolbar_icon()  # the next worker
        assert bridge.wait_for_extension(30)
        # Dropped before the first command, on a tab no command has named since.
        assert chrome.browser.worker_eval(refusal) == ""
        assert chrome.evaluate(tab, "1 + 1") == 2
    finally:
        chrome.fresh_worker()


# ------------------------------------------------------------------ the tabs

_RED, _BLUE = (255, 0, 0), (0, 0, 255)
# Deadlines in milliseconds, from one that is over at once to one a quick command meets.
_SHORT = [0.2, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 40, 80]


def _centre(data_url: str) -> object:
    """The colour in the middle of a screenshot (a PNG data URL)."""
    assert data_url.startswith("data:image/png;base64,")
    image = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")
    return image.getpixel((image.width // 2, image.height // 2))


def _sweep(
    chrome: _Chrome, link: _Link, cmd: str, tabs: list[int], budgets: list[float]
) -> list[dict[str, Any]]:
    """Send *cmd* for *tabs* in turn with each deadline of *budgets*; return the replies.

    Repeated until every step of the command has been the one a deadline
    passed in: V8's record of what the worker ran tells.
    """
    replies = []
    until = time.monotonic() + 60
    while True:
        for index, ms in enumerate(budgets):
            replies.append(link.ask(cmd, {"tabId": tabs[index % len(tabs)]}, ms))
            assert replies[-1]["ok"] or _EXPIRED.match(replies[-1]["error"]), replies[-1]
        missing = chrome.browser.uncovered(cmd)
        if not missing:
            return replies
        assert time.monotonic() < until, missing


def test_tabs_are_listed_opened_activated_navigated_and_closed(chrome: _Chrome) -> None:
    """The tab commands, each answering with the tab as Chrome reports it."""
    bridge = chrome.live_bridge()
    first, _ = chrome.open("/blank?first")
    second = bridge.call("open_tab", {"url": chrome.site + "/blank?second", "active": False}, 20)
    assert second == {
        "id": second["id"],
        "windowId": second["windowId"],
        "title": "Blank",
        "url": chrome.site + "/blank?second",
        "active": False,
        "status": "complete",
    }
    listed = {tab["id"]: tab for tab in bridge.call("list_tabs", timeout=10)}
    assert listed[first]["active"] and listed[second["id"]] == second
    assert bridge.call("activate_tab", {"tabId": second["id"]}, 10) == {**second, "active": True}
    moved = bridge.call("navigate", {"tabId": first, "url": chrome.site + "/red"}, 20)
    assert (moved["id"], moved["title"], moved["url"]) == (first, "Red", chrome.site + "/red")
    # A URL without its scheme is a file of the extension to Chrome, a page that does not
    # exist: refused, and no tab is opened or moved.
    for cmd, target in (("open_tab", {}), ("navigate", {"tabId": first})):
        with pytest.raises(ChromeBridgeError, match="'example.com' is not a complete URL"):
            bridge.call(cmd, {**target, "url": "example.com"}, 10)
    assert {tab["id"]: tab["url"] for tab in bridge.call("list_tabs", timeout=10)} == {
        **{tab_id: tab["url"] for tab_id, tab in listed.items()},
        first: chrome.site + "/red",
    }
    assert bridge.call("close_tab", {"tabId": first}, 10) == {"closed": first}
    assert first not in [tab["id"] for tab in bridge.call("list_tabs", timeout=10)]
    blank = bridge.call("open_tab", {}, 10)  # no URL: an empty tab, not waited for
    assert (blank["url"], blank["active"]) == ("about:blank", True)


def test_a_page_load_is_waited_for_fifteen_seconds_at_most(chrome: _Chrome) -> None:
    """A page still loading after 15 s is returned as it is; a tab closed meanwhile fails."""
    bridge = chrome.live_bridge()
    began = time.monotonic()
    pending = bridge.call("open_tab", {"url": chrome.site + "/pending"}, 25)
    assert 14 < time.monotonic() - began < 20
    assert (pending["title"], pending["status"]) == ("Pending", "loading")
    tab, url = chrome.open("/blank?closing")
    target = {"tabId": tab, "url": chrome.site + "/blank?ms=3000"}
    slow = _in_thread(bridge.call, "navigate", target, 20)
    time.sleep(0.5)  # the navigation waits for the site's late answer
    chrome.browser.close_page(chrome.browser.pages()[url])
    with pytest.raises(ChromeBridgeError, match=f"No tab with id: {tab}"):
        slow.result(20)


def test_a_command_for_a_tab_that_is_gone_fails_by_name(chrome: _Chrome) -> None:
    """Every command that takes a tab says which tab is missing; none hangs."""
    bridge = chrome.live_bridge()
    tab, _ = chrome.open("/blank")
    bridge.call("close_tab", {"tabId": tab}, 10)
    commands: dict[str, dict[str, Any]] = {
        "navigate": {"url": chrome.site + "/blank"},
        "activate_tab": {},
        "close_tab": {},
        "screenshot": {},
        "evaluate": {"script": "1"},
        "snapshot": {},
        "click": {"uid": 1},
        "type_text": {"uid": 1, "text": "x"},
    }
    for cmd, args in commands.items():
        with pytest.raises(ChromeBridgeError, match=f"No tab with id: {tab}"):
            bridge.call(cmd, {"tabId": tab, **args}, 10)
    with pytest.raises(ChromeBridgeError, match="Missing required property 'tabId'"):
        bridge.call("evaluate", {"script": "1"}, 10)
    # What is no command is refused by name too, the members of every object included.
    for name in ("no_such_command", "__proto__", "constructor"):
        with pytest.raises(ChromeBridgeError, match=f"unknown command '{name}'"):
            bridge.call(name, {}, 10)
    assert isinstance(bridge.call("list_tabs", timeout=10), list)


def test_a_tab_gets_a_window_of_its_own_when_chrome_has_none(chrome: _Chrome) -> None:
    """Chrome running with no window, as on a Mac once the last one is closed, opens a tab."""
    bridge = chrome.live_bridge()
    try:
        for tab in bridge.call("list_tabs", timeout=10):
            bridge.call("close_tab", {"tabId": tab["id"]}, 10)
        _until(lambda: bridge.call("list_tabs", timeout=10) == [])
        red = bridge.call("open_tab", {"url": chrome.site + "/red"}, 20)
        assert (red["title"], red["status"], red["active"]) == ("Red", "complete", True)
        assert _centre(bridge.call("screenshot", {"tabId": red["id"]}, 20)) == _RED
        blue = bridge.call("open_tab", {"url": chrome.site + "/blue"}, 20)
        assert blue["windowId"] == red["windowId"]  # the next tab joins that window
        # What Chrome refuses to open in a window it has, it refuses in its own words.
        with pytest.raises(ChromeBridgeError, match="JavaScript URLs are not allowed"):
            bridge.call("open_tab", {"url": "javascript:void 0"}, 10)
    finally:
        chrome.browser.open_manager()  # closed with the rest: new_worker() works through it


def _unload(chrome: _Chrome, tab: int, url: str, crashed: bool) -> None:
    """Leave *tab*, which shows *url*, with no page loaded in it.

    Discarded, as Chrome's Memory Saver leaves a tab nobody looked at for a while, or with
    its page *crashed*.
    """
    if crashed:
        chrome.browser.crash_page(chrome.browser.pages()[url])
    else:
        chrome.browser.discard_tab(tab)
    _until(partial(_tab_shows, chrome.live_bridge(), {"id": tab, "status": "unloaded"}))


def test_a_tab_with_no_page_loaded_is_loaded_before_it_is_read_or_shown(chrome: _Chrome) -> None:
    """A tab Chrome unloaded to save memory, or whose page crashed, holds no document.

    A script in it would read an empty page and a capture show a blank one: every command
    that reads or shows the page loads it again first, where the tab is.
    """
    bridge = chrome.live_bridge()
    tab, url = chrome.open("/controls?unloaded")
    save = _uid(chrome.snapshot(tab), '<button> "Save"')
    chrome.open("/blank?front")  # the tab is in the background from here on
    _unload(chrome, tab, url, crashed=False)
    state = "document.title + ' ' + document.visibilityState"
    assert chrome.evaluate(tab, state) == "Controls hidden"  # loaded, and not brought forward
    _unload(chrome, tab, url, crashed=True)
    lines = chrome.snapshot(tab)
    assert lines[0].startswith("Controls — ") and '<button> "Save"' in lines[1]
    # A handle of the page that was is one of a page that changed.
    stale = rf"element \[{save}\] no longer exists — the page changed; take a fresh snapshot"
    _unload(chrome, tab, url, crashed=False)
    with pytest.raises(ChromeBridgeError, match=stale):
        chrome.click(tab, save)
    _unload(chrome, tab, url, crashed=True)
    with pytest.raises(ChromeBridgeError, match=stale):
        chrome.type_text(tab, save, "x")
    # A page slow to come back is waited for: a capture taken as soon as the tab is in front
    # would show an empty one.
    red, red_url = chrome.open("/red?ms=1500")
    chrome.open("/blank?front-again")
    _unload(chrome, red, red_url, crashed=False)
    assert _centre(bridge.call("screenshot", {"tabId": red}, 20)) == _RED
    # Another command is already having the page loaded, and the tab says "loading", not
    # "unloaded": the capture waits for that same load.
    chrome.open("/blank?front-once-more")
    _unload(chrome, red, red_url, crashed=False)
    reading = _in_thread(chrome.evaluate, red, "document.title", 20)
    _until(partial(_tab_shows, bridge, {"id": red, "status": "loading"}))
    assert _centre(bridge.call("screenshot", {"tabId": red}, 20)) == _RED
    assert reading.result(10) == "Red"
    chrome.open("/blank?front-at-last")
    _unload(chrome, red, red_url, crashed=True)
    shown = bridge.call("activate_tab", {"tabId": red}, 20)
    assert (shown["title"], shown["status"], shown["active"]) == ("Red", "complete", True)


def test_screenshots_in_a_row_each_show_the_tab_asked_for(chrome: _Chrome) -> None:
    """Back to back and at once: each capture is of its own tab, within Chrome's capture quota."""
    bridge = chrome.live_bridge()
    red, _ = chrome.open("/red")
    blue, _ = chrome.open("/blue")  # the active tab now
    for tab, colour in ((red, _RED), (blue, _BLUE), (red, _RED), (blue, _BLUE)):
        assert _centre(bridge.call("screenshot", {"tabId": tab}, 20)) == colour
    shots = [_in_thread(bridge.call, "screenshot", {"tabId": tab}, 20) for tab in (red, blue, red)]
    assert [_centre(shot.result(25)) for shot in shots] == [_RED, _BLUE, _RED]


def test_a_focus_command_past_its_deadline_is_refused_at_every_step(chrome: _Chrome) -> None:
    """Deadlines from a fraction of a millisecond up: each step of a screenshot and of an

    activation is the one a deadline passes in, the reply is the refusal, the tabs work on.
    Sent by a peer: a bridge waits a tenth longer than the deadline it sends, drops a
    connection whose reply is not in by then, and so never shows a reply this late.
    """
    with chrome.peer() as peer:
        link = peer.login()
        pages = [chrome.site + "/red", chrome.site + "/blue"]
        tabs = [link.result("open_tab", {"url": url})["id"] for url in pages]
        assert _centre(link.result("screenshot", {"tabId": tabs[0]})) == _RED
        # None of these deadlines outlasts the wait for the tab to paint.
        refused = _sweep(chrome, link, "screenshot", tabs, [*_SHORT, 150, 200])
        assert not any(reply["ok"] for reply in refused)
        _sweep(chrome, link, "activate_tab", tabs, _SHORT)
        # A deadline that is over on arrival: refused before anything starts, in every queue.
        late = {"tabId": tabs[0], "script": "document.title = 'ran late'"}
        for cmd in ("list_tabs", "activate_tab", "evaluate"):
            reply = link.ask(cmd, late, 0)
            assert reply["error"] == "command waited out its 0s deadline in the queue"
        assert link.result("evaluate", {"tabId": tabs[0], "script": "document.title"}) == "Red"
        assert _centre(link.result("screenshot", {"tabId": tabs[1]})) == _BLUE
        assert link.result("activate_tab", {"tabId": tabs[0]})["active"] is True


def test_a_command_waiting_behind_another_fails_at_its_own_deadline(chrome: _Chrome) -> None:
    """Queued behind a slow script on its tab: answered at its deadline, and never run."""
    tab, _ = chrome.open("/blank?queue")
    slow = "new Promise((ok) => setTimeout(() => ok('slow'), 2500))"
    ahead = _in_thread(chrome.evaluate, tab, slow)
    time.sleep(0.3)  # the slow script is running
    began = time.monotonic()
    # Of the caller's second the bridge gives the extension nine tenths.
    refusal = _fails(chrome, tab, "document.title = 'ran late'", 1.0)
    assert "command exceeded its 0.9s deadline" in refusal
    assert time.monotonic() - began < 1.6  # before the script ahead of it was done
    assert ahead.result(10) == "slow"
    # Its turn has come by now, and it did not run.
    assert chrome.evaluate(tab, "document.title") == "Blank"


# ------------------------------------------------------------ scripts in pages


def _fails(chrome: _Chrome, tab: int, script: str, timeout: float = 10.0) -> str:
    """The error *script* fails with in *tab*."""
    with pytest.raises(ChromeBridgeError) as failed:
        chrome.evaluate(tab, script, timeout)
    return str(failed.value)


def test_a_script_runs_in_the_page_and_its_value_comes_back(chrome: _Chrome) -> None:
    """In the page's own world: values as JSON carries them, promises awaited, throws as errors."""
    tab, url = chrome.open("/blank")
    values: dict[str, Any] = {
        "6 * 7": 42,
        "undefined": None,
        "[undefined, 'x', {n: 1}]": [None, "x", {"n": 1}],
        "new Promise((ok) => setTimeout(() => ok('later'), 50))": "later",
        "window.kissMark = 'page world'": "page world",
    }
    for script, value in values.items():
        assert chrome.evaluate(tab, script) == value, script
    assert chrome.browser.page_eval(url, "window.kissMark") == "page world"
    errors = {
        "throw new Error('boom')": r"Error: boom\n.*",
        "Promise.reject(new Error('rejected'))": r"Error: rejected\n.*",
        "this is not javascript": r"SyntaxError: .*",
        "throw {message: 'no stack'}": "no stack",
        "throw 'just a string'": "just a string",
        "throw null": "null",
        "throw ''": "Error",
    }
    for script, message in errors.items():
        assert re.fullmatch(message, _fails(chrome, tab, script), re.DOTALL), script
    # A value of more bytes than the bridge takes in one frame would end the connection,
    # with every other command on it: it fails by name and the connection stays. One of
    # as many characters and a third of the bytes comes back whole.
    bridge = chrome.live_bridge()
    long = 22 * 1024 * 1024
    assert MAX_MESSAGE_BYTES // 3 < long < MAX_MESSAGE_BYTES < 3 * long
    assert len(chrome.evaluate(tab, f"'x'.repeat({long})", 60)) == long
    refusal = "the result is larger than the 64 MB a reply may be: return less of it"
    assert _fails(chrome, tab, f"'\u20ac'.repeat({long})", 60) == refusal
    assert chrome.live_bridge() is bridge and chrome.evaluate(tab, "6 * 7") == 42


def test_a_page_that_forbids_eval_is_reached_through_the_debugger(chrome: _Chrome) -> None:
    """Refused by the page's policy before it ran, a script runs once, through chrome.debugger."""
    tab, _ = chrome.open("/csp/blank")
    values: dict[str, Any] = {
        "6 * 7": 42,
        "void 0": None,
        "new Promise((ok) => setTimeout(() => ok('later'), 50))": "later",
        "document.body.dataset.runs = (document.body.dataset.runs || '') + 'x'": "x",
        # A 'use strict' directive still opens the script the page is given ...
        "'use strict'; (function () { return this === undefined; })()": True,
        "'other'; 'use strict'; (function () { return this === undefined; })()": True,
        "'use strict'": "use strict",
        # ... and a string that only looks like one makes nothing strict.
        "'use strict'.length; kissMember = 73; kissMember": 73,
        "'use strict'\n+ 1; kissPlus = 73; kissPlus": 73,
    }
    for script, value in values.items():
        assert chrome.evaluate(tab, script) == value, script
    errors = {
        "throw new Error('boom')": "debugger: Error: boom\n",
        "throw 'str'": "debugger: str)",
        "throw null": "debugger: Uncaught)",
        '// c\n"use strict"\nkissUnbound = 73; kissUnbound': "ReferenceError: kissUnbound",
        "window": "debugger: ",  # no value JSON can carry
    }
    for script, message in errors.items():
        error = _fails(chrome, tab, script)
        assert error.startswith("evaluate failed (page world blocked by CSP: "), script
        assert message in error, script
    # A page that replaced eval itself (its eval throws a string): the same way round.
    other, _ = chrome.open("/noeval")
    assert chrome.evaluate(other, "6 * 7") == 42
    # Snapshots, clicks and typing take that way too.
    controls, _ = chrome.open("/csp/controls")
    lines = chrome.snapshot(controls)
    assert chrome.type_text(controls, _uid(lines, '<input type=text> "Search"'), "hi")
    assert chrome.click(controls, _uid(lines, '<button> "Save"')).endswith('<button> "Save"')
    state = "[document.getElementById('q').value, log.at(-1)]"
    assert chrome.evaluate(controls, state) == ["hi", "click:save"]


def test_a_script_past_its_deadline_never_runs_late(chrome: _Chrome) -> None:
    """A renderer busy past the deadline: the command fails on time and the page is left alone."""
    late = "document.title = 'ran late'; 1"
    # Busy when the script arrives; busy before it says that it forbids eval;
    # busy once it has said so, while the debugger's evaluation waits.
    stages = {
        "/blank?busy": f"setTimeout(() => {{ {_BUSY} }}, 0); 'armed'",
        "/sloweval": "",
        "/csp/stall": "",
    }
    for path, arm in stages.items():
        tab, url = chrome.open(path)
        title = chrome.browser.page_eval(url, "document.title")
        if arm:
            assert chrome.evaluate(tab, arm) == "armed"
        began = time.monotonic()
        assert _EXPIRED.search(_fails(chrome, tab, late, 1.0)), path
        assert time.monotonic() - began < 1.2, path  # at the deadline, not when the page was free
        time.sleep(1.5)
        assert chrome.browser.page_eval(url, "document.title") == title, path
        assert chrome.evaluate(tab, "6 * 7") == 42, path  # the tab takes the next command


def test_a_deadline_that_passes_while_the_debugger_attaches_harms_no_command(
    chrome: _Chrome,
) -> None:
    """Deadlines of a few milliseconds on a page behind a policy, until one has passed in

    the attach itself: every reply is the value or the refusal, and the command queued
    behind each has a debugger of its own.  Sent by a peer, as every deadline shorter
    than a bridge's own wait is (see the sweep of the focus commands).
    """
    bump = "document.body.dataset.n = String(Number(document.body.dataset.n || 0) + 1); 1"
    with chrome.peer() as peer:
        link = peer.login()
        tab = link.result("open_tab", {"url": chrome.site + "/csp/blank?sweep"})["id"]
        ran = 0
        until = time.monotonic() + 60
        while any("deadlineError" in line for line in chrome.browser.uncovered("evalViaDebugger")):
            assert time.monotonic() < until, "no deadline passed during an attach"
            for ms in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80):
                first = link.send("evaluate", {"tabId": tab, "script": bump}, ms)
                behind = link.send("evaluate", {"tabId": tab, "script": "6 * 7"}, 5000)
                reply = link.reply(first)
                assert reply["ok"] or _EXPIRED.search(reply["error"]), reply
                ran += reply["ok"]
                assert link.reply(behind).get("result") == 42
        count = link.result("evaluate", {"tabId": tab, "script": "document.body.dataset.n || '0'"})
        assert int(count) >= ran  # a script whose reply came back ran


def _started(chrome: _Chrome, url: str) -> bool:
    """Whether the page showing *url* runs the script that titles it "started"."""
    return bool(chrome.browser.page_eval(url, "document.title") == "started")


def test_a_script_whose_page_goes_away_fails_without_waiting_for_its_deadline(
    chrome: _Chrome,
) -> None:
    """Left, reloaded or closed under a running script: the command fails at once, and by name."""
    script = "document.title = 'started'; new Promise(() => {})"
    gone = r"the page went away \(it navigated, reloaded or closed\) before the script finished"
    # Chrome keeps a page that was left for another one, frozen: the script in it never ends,
    # and only the page says that it is going. A page reloaded or closed is destroyed, which
    # Chrome may tell before the page's word is in.
    leaves = {
        "left": (f"location.href = '{chrome.site}/blue'; 1", gone, "Blue"),
        "reloaded": ("location.reload(); 1", f"{gone}|the page returned nothing", "Blank"),
    }
    tabs = chrome.live_bridge()
    for name, (leave, error, title) in leaves.items():
        tab, url = chrome.open(f"/blank?{name}")
        stuck = _in_thread(chrome.evaluate, tab, script, 30)
        _until(partial(_started, chrome, url))
        chrome.browser.page_eval(url, leave)
        with pytest.raises(ChromeBridgeError, match=error):
            stuck.result(10)
        _until(partial(_tab_shows, tabs, {"id": tab, "title": title, "status": "complete"}))
        assert chrome.evaluate(tab, "document.title") == title  # the tab takes commands
    tab, url = chrome.open("/blank?closed")
    stuck = _in_thread(chrome.evaluate, tab, script, 30)
    _until(partial(_started, chrome, url))
    chrome.browser.close_page(chrome.browser.pages()[url])
    with pytest.raises(ChromeBridgeError, match=f"{gone}|was removed|No tab with id"):
        stuck.result(10)
    # A page can also announce a leaving that does not happen, with a pagehide event of its
    # own: the wait ends the same way. Chrome forgets what ran in a document that was left,
    # so this is the run of that answer which the coverage gate can count.
    tab, url = chrome.open("/blank?stays")
    stuck = _in_thread(chrome.evaluate, tab, script, 30)
    _until(partial(_started, chrome, url))
    chrome.browser.page_eval(url, "dispatchEvent(new PageTransitionEvent('pagehide')); 1")
    with pytest.raises(ChromeBridgeError, match=gone):
        stuck.result(10)
    assert chrome.evaluate(tab, "document.title") == "started"
    # The wait ends with the command as well: scripts that outlive their deadline in a page
    # that stays leave it nothing to carry, no listener for its leaving either.
    for _ in range(3):
        with pytest.raises(ChromeBridgeError, match="command exceeded its 0.9s deadline"):
            chrome.evaluate(tab, script, 1)
    _until(lambda: chrome.browser.page_listeners(url, "pagehide") == 0)
    # A script that still waits for its page to be read never started in it: when the page
    # goes, Chrome has nothing of it to return.
    tab, url = chrome.open("/blank?unread")
    unread = chrome.site + "/unread"
    chrome.browser.page_eval(url, f"location.href = '{unread}'; 1")
    _until(lambda: unread in chrome.browser.pages())
    _until(lambda: chrome.browser.page_eval(unread, "document.title") == "Unread")
    waiting = _in_thread(chrome.evaluate, tab, "document.title = 'ran'", 30)
    # The worker has the command once the tab has a queue. One more answer of Chrome to the
    # worker, and the command's own question about the tab is answered too: its script is
    # handed to the page, where it waits for the end of the reading.
    _until(partial(chrome.browser.worker_eval, f"tabChains.has({tab})"))
    chrome.browser.worker_eval(f"chrome.tabs.get({tab}).then(() => 1)")
    chrome.browser.page_eval(unread, f"location.href = '{chrome.site}/blue?after'; 1")
    with pytest.raises(ChromeBridgeError, match="the page returned nothing"):
        waiting.result(10)
    _until(lambda: chrome.site + "/blue?after" in chrome.browser.pages())
    assert chrome.evaluate(tab, "document.title") == "Blue"


# ------------------------------------------------- snapshots, clicks and typing


def _handles(lines: list[str]) -> list[int]:
    """The handles of a snapshot's element lines."""
    return [int(line[1 : line.index("]")]) for line in lines[1:]]


def test_a_snapshot_lists_what_a_user_can_act_on(chrome: _Chrome) -> None:
    """Controls, editors and clickable boxes, each once and under a handle that stays its own."""
    tab, url = chrome.open("/controls")
    lines = chrome.snapshot(tab)
    assert lines[0] == f"Controls — {url} — {len(_LISTED)} interactive elements"
    listed = [line.split("] ", 1)[1] for line in lines[1:]]
    assert listed == [element.format(site=chrome.site) for element in _LISTED]
    assert chrome.snapshot(tab) == lines  # the same handles, snapshot after snapshot
    # A new element gets a number no element had; one that left the page loses its handle.
    docs = _uid(lines, f'<a> "Docs" -> {chrome.site}/blank')
    change = (
        "document.getElementById('save').insertAdjacentHTML('afterend', '<button>New</button>');"
        "document.getElementById('docs').style.display = 'none'; 1"
    )
    chrome.evaluate(tab, change)
    grown = chrome.snapshot(tab)
    new = _uid(grown, '<button> "New"')
    assert new > max(_handles(lines))
    assert set(grown[1:]) - set(lines[1:]) == {f'[{new}] <button> "New"'}
    with pytest.raises(ChromeBridgeError, match=rf"element \[{docs}\] no longer exists"):
        chrome.click(tab, docs)


def test_a_snapshot_shows_the_first_two_hundred_elements(chrome: _Chrome) -> None:
    """A page with more says how many it has; the handles are the listed elements'."""
    tab, url = chrome.open("/many")
    lines = chrome.snapshot(tab)
    assert lines[0] == f"Many — {url} — 205 interactive elements (showing first 200)"
    assert len(lines) == 201 and lines[-1].endswith('<button> "B199"')
    assert len(set(_handles(lines))) == 200


# What a click on a link into another tab adds to its reply, before the link's address.
_NEW_TAB = (
    " — the link targets {}, and Chrome blocks a new tab for a scripted click: "
    "if none opened, open_tab "
)


def test_a_click_reaches_the_element_as_a_users_click_would(chrome: _Chrome) -> None:
    """The pointer and mouse events, then the click, on the element the handle names."""
    tab, _ = chrome.open("/controls")
    lines = chrome.snapshot(tab)
    events = ["pointerdown", "mousedown", "pointerup", "mouseup", "click"]
    # What the reply calls the element: its text, its label, its value, or its tag.
    clicks = {
        '<button> "Save"': ("save", '<button> "Save"'),
        '<button> "Close"': ("", '<button> "Close"'),
        '<input type=text> "typed already"': ("filled", '<input> "typed already"'),
        "<input type=text>": ("bare", '<input> "INPUT"'),
        # What a password field holds is not repeated.
        "<input type=password>": ("secret", '<input> "INPUT"'),
        # A link that stays in its tab, and two into another tab (a new one, a named one):
        # Chrome's pop-up blocker opens none for a scripted click, and the reply gives the
        # address open_tab takes, also that of a drawing's link, which holds it as written.
        f'<a> "Stay" -> {chrome.site}/controls#top': ("stay", '<a> "Stay"'),
        f'<a> "Elsewhere" -> {chrome.site}/blank?away': (
            "away",
            f'<a> "Elsewhere"{_NEW_TAB.format("_blank")}{chrome.site}/blank?away',
        ),
        '<a> "Drawn" -> /blank?drawn': (
            "drawn",
            f'<a> "Drawn"{_NEW_TAB.format("report")}{chrome.site}/blank?drawn',
        ),
        # No click() of its own (SVG, an element of an unknown kind): the click is dispatched.
        '<svg role=button> "Chart"': ("chart", '<svg> "Chart"'),
        '<gadget role=button> "Odd gadget"': ("gadget", '<gadget> "Odd gadget"'),
        # Its value is a number, and it is described all the same.
        "<progress>": ("progress", '<progress> "0.5"'),
        # Inside shadow roots: the page hears every event, at the host.
        '<button> "Shadow button"': ("host", '<button> "Shadow button"'),
        '<button> "Deep button"': ("host", '<button> "Deep button"'),
    }
    bridge = chrome.live_bridge()
    open_tabs = [t["id"] for t in bridge.call("list_tabs", timeout=10)]
    for element, (target, described) in clicks.items():
        uid = _uid(lines, element)
        assert chrome.click(tab, uid) == f"clicked [{uid}] {described}", element
        heard = chrome.evaluate(tab, "log.splice(0)")
        assert [e for e in heard if ":" in e] == [f"{event}:{target}" for event in events], element
        assert ("progress" in heard) == (element == "<progress>")  # its onclick ran
    # Neither link into another tab opened one: what its reply says is what happened.
    assert [t["id"] for t in bridge.call("list_tabs", timeout=10)] == open_tabs
    # A disabled control takes no click, a user's or this one: refused, the page hears nothing.
    for element in ('<button> "Disabled" [disabled]', '<button> "Idle" [disabled]'):
        uid = _uid(lines, element)
        refusal = rf"element \[{uid}\] is disabled: nothing was clicked"
        with pytest.raises(ChromeBridgeError, match=refusal):
            chrome.click(tab, uid)
    assert chrome.evaluate(tab, "log") == []
    # An element out of view is scrolled to; one in view leaves the page where it is.
    assert chrome.evaluate(tab, "scrollY") == 0
    chrome.click(tab, _uid(lines, '<button> "Far away"'))
    scrolled = chrome.evaluate(tab, "scrollY")
    assert scrolled > 1000
    chrome.click(tab, _uid(lines, '<button> "Far away"'))
    assert chrome.evaluate(tab, "scrollY") == scrolled
    chrome.click(tab, _uid(lines, '<button> "Save"'))
    assert chrome.evaluate(tab, "scrollY") < scrolled


def test_typing_replaces_what_a_field_holds_and_tells_the_page(chrome: _Chrome) -> None:
    """Inputs, text areas and editors, in a tab that is not the one in front."""
    tab, _ = chrome.open("/controls")
    chrome.open("/blank")  # the tab typed into is in the background from here on
    lines = chrome.snapshot(tab)
    shadow = "document.getElementById('host').shadowRoot.querySelector('input').value"
    # element: its id, the text typed, where the text is read, the events the page hears
    fields = {
        '<input type=text> "Search"': ("q", "hello", "value", ["input:q", "change:q"]),
        '<textarea> "Notes"': ("notes", "line1\nline2", "value", ["input:notes", "change:notes"]),
        '<div> "rich text"': ("rich", "new rich", "textContent", ["input:rich"]),
        '<div> "bare editor"': ("editor", "typed", "textContent", ["input:editor"]),
        '<div> "plain editor"': ("plain", "typed", "textContent", ["input:plain"]),
        '<input type=number> "5"': ("count", "42", "value", ["input:count", "change:count"]),
        # A field inside an editor is typed into as a field: replaced, not added to.
        '<input type=text> "inside"': ("inner", "hello", "value", ["input:inner", "change:inner"]),
    }
    for element, (name, text, where, heard) in fields.items():
        uid = _uid(lines, element)
        tag = element[1:].split(">")[0].split(" ")[0]
        state = f"[document.getElementById('{name}').{where}, log.splice(0)]"
        for typed in (text, ""):  # an empty text clears the field, and the page hears of it
            reply = chrome.type_text(tab, uid, typed)
            assert reply == f"typed {len(typed)} chars into [{uid}] <{tag}>", element
            assert chrome.evaluate(tab, state) == [typed, heard], (element, typed)
    # In a shadow root: the page hears the input at the host.
    uid = _uid(lines, '<input type=text> "Shadow field"')
    chrome.type_text(tab, uid, "deep")
    assert chrome.evaluate(tab, f"[{shadow}, log.splice(0)]") == ["deep", ["input:host"]]
    # A one-line field keeps a text without its line break, as when a user pastes one.
    uid = _uid(lines, '<input type=text> "Search"')
    assert chrome.type_text(tab, uid, "a\nb") == f"typed 3 chars into [{uid}] <input>"
    assert chrome.evaluate(tab, "document.getElementById('q').value") == "ab"


def test_typing_into_what_takes_no_text_fails_and_says_so(chrome: _Chrome) -> None:
    """A checkbox, a button, a list, a number field given letters: refused, and left as it was."""
    tab, _ = chrome.open("/controls")
    lines = chrome.snapshot(tab)
    refused = {
        "<input type=checkbox> [checked]": "<input type=checkbox>",
        '<button> "Save"': "<button>",
        '<select> "two"': "<select>",
        '<gadget role=button> "Odd gadget"': "<gadget>",
    }
    for element, kind in refused.items():
        uid = _uid(lines, element)
        with pytest.raises(ChromeBridgeError) as failed:
            chrome.type_text(tab, uid, "abc")
        assert f"element [{uid}] is a {kind}, not a text field" in str(failed.value), element
    state = "[document.getElementById('on').value, document.getElementById('on').checked, log]"
    assert chrome.evaluate(tab, state) == ["on", True, []]
    # A number field takes no letters: it keeps its number, and the page hears nothing.
    uid = _uid(lines, '<input type=number> "5"')
    with pytest.raises(ChromeBridgeError) as failed:
        chrome.type_text(tab, uid, "abc")
    dropped = f"element [{uid}] <input type=number> did not take the text and was left unchanged"
    assert dropped in str(failed.value)
    assert chrome.evaluate(tab, "[document.getElementById('count').value, log]") == ["5", []]
    # A colour well and a slider take a value of their own kind (in any case, spaces
    # around it) and nothing else: what they would turn into black or the middle is refused.
    picked = {
        '<input type=color> "#ff0000"': ("shade", " #00FF00 ", "#00ff00"),
        '<input type=range> "20"': ("level", "70", "70"),
    }
    for element, (name, good, held) in picked.items():
        well = _uid(lines, element)
        state = f"[document.getElementById('{name}').value, log.splice(0)]"
        chrome.type_text(tab, well, good)
        assert chrome.evaluate(tab, state) == [held, [f"input:{name}", f"change:{name}"]]
        for bad in ("abc", ""):
            with pytest.raises(ChromeBridgeError, match="did not take the text"):
                chrome.type_text(tab, well, bad)
        assert chrome.evaluate(tab, state) == [held, []], element
    # A user types into no field that is disabled or read-only: each keeps what it holds.
    locked = {
        '<input type=text> "switched off" [disabled]': ("off", "disabled", "switched off"),
        '<input type=text> "read only"': ("kept", "read-only", "read only"),
    }
    for element, (name, why, held) in locked.items():
        field = _uid(lines, element)
        refusal = rf"element \[{field}\] <input type=text> is {why} and was left unchanged"
        with pytest.raises(ChromeBridgeError, match=refusal):
            chrome.type_text(tab, field, "abc")
        state = f"[document.getElementById('{name}').value, log]"
        assert chrome.evaluate(tab, state) == [held, []], element
    with pytest.raises(ChromeBridgeError, match="type_text needs a 'text' string"):
        chrome.live_bridge().call("type_text", {"tabId": tab, "uid": uid}, 10)


def test_a_handle_names_its_element_or_nothing(chrome: _Chrome) -> None:
    """Copied, removed, of an earlier connection, or no number: never another element's."""
    tab, _ = chrome.open("/controls")
    lines = chrome.snapshot(tab)
    save = _uid(lines, '<button> "Save"')
    # A copy carries the handle's attribute along, in front of the original: not the handle.
    copy = "const s = document.getElementById('save'), c = s.cloneNode(true);"
    chrome.evaluate(tab, copy + "c.id = 'copy'; s.before(c); log.splice(0); 1")
    chrome.click(tab, save)
    assert chrome.evaluate(tab, "log.splice(0).at(-1)") == "click:save"
    chrome.evaluate(tab, "document.getElementById('save').remove(); 1")
    gone = rf"element \[{save}\] no longer exists — the page changed; take a fresh snapshot"
    with pytest.raises(ChromeBridgeError, match=gone):
        chrome.click(tab, save)
    with pytest.raises(ChromeBridgeError, match=gone):
        chrome.type_text(tab, save, "x")
    assert chrome.evaluate(tab, "log") == []
    for uid in ("7", 1.5, None):
        with pytest.raises(ChromeBridgeError, match="needs an integer 'uid' from a snapshot"):
            chrome.live_bridge().call("click", {"tabId": tab, "uid": uid}, 10)
    lines = chrome.snapshot(tab)
    assert _uid(lines, '<button> "Save"') > save  # the copy, under a handle of its own
    # Another connection numbers its commands from 1 again: this one's handles are void.
    disabled = _uid(lines, '<button> "Disabled" [disabled]')
    chrome.new_bridge()
    with pytest.raises(ChromeBridgeError, match=rf"element \[{disabled}\] no longer exists"):
        chrome.click(tab, disabled)
    untouched, _ = chrome.open("/blank")  # no snapshot was taken of this page
    with pytest.raises(ChromeBridgeError, match=r"element \[1\] no longer exists"):
        chrome.click(untouched, 1)


# ------------------------------------------------------- the product's name


# A bridge of a copy of the package, in a home of its own: it installs
# the extension, says where and what the Settings label of its own page
# reads, waits for a line on stdin (Chromium is up), then drives it.
_INSTALL = """
import html, json, re, sys
from kiss.agents.sorcar.chrome_bridge import ChromeBridge, install_extension
from kiss.server import web_server
bridge = ChromeBridge(write_config=True)
bridge.start()
label = re.search(r'id="cfg-chrome-bridge">(.*?)</label>', web_server._build_html(), re.S)
text = html.unescape(re.sub("<[^>]+>", "", label[1]))  # as a page with no daemon yet shows it
print(json.dumps([str(install_extension()), " ".join(text.split())]), flush=True)
sys.stdin.readline()
connected = bridge.wait_for_extension(30)
tab = bridge.call("open_tab", {"url": sys.argv[1]}, 20)
answer = bridge.call("evaluate", {"tabId": tab["id"], "script": "6 * 7"}, 10)
print(json.dumps([connected, tab["title"], answer]), flush=True)
bridge.stop()
"""


@pytest.mark.parametrize(
    ("brand", "product", "folder"),
    [(DEFAULT_BRAND, "KISS Sorcar", ".kiss"), (SEAMLESS_LOOP, "Seamless Loop", ".s10s")],
    ids=["stock", "branded"],
)
def test_an_install_is_the_same_extension_under_the_products_own_name(
    brand: dict[str, str],
    product: str,
    folder: str,
    chrome: _Chrome,
    chromium: str,
    tmp_path: Path,
) -> None:
    """Unbranded or branded: the folder and the name the Settings label gives, the same worker.

    The product runs from a copy of the package carrying *brand*, in a
    home of its own: the checkout's brand and ``$KISS_HOME`` play no part.
    """
    package = make_branded_package(tmp_path / "package", brand)
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "KISS_HOME"}
    # Path.home() follows USERPROFILE on Windows and HOME on POSIX.
    env |= {"HOME": str(home), "USERPROFILE": str(home), "PYTHONPATH": str(package)}
    bridge = subprocess.Popen(
        [sys.executable, "-c", _INSTALL, chrome.site + "/red"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert bridge.stdout is not None and bridge.stdin is not None
        installed, label = json.loads(bridge.stdout.readline())
        assert label == (
            "Drive my Chrome: tick this box and close Settings. Then, in Chrome on the computer "
            f"that runs {product}, open chrome://extensions, turn on Developer mode, click "
            f'"Load unpacked" and select the folder ~/{folder}/chrome_extension. '
            f'It is listed there as "{product} Bridge". '
            "Leave Developer mode on: without it Chrome switches the extension off."
        )
        # The label's folder and name are the install's own.
        assert Path(installed) == home / folder / "chrome_extension"
        manifest = json.loads((Path(installed) / "manifest.json").read_text())
        worker = Path(installed) / manifest["background"]["service_worker"]
        assert worker.read_bytes() == BACKGROUND_JS.read_bytes()
        browser = ExtensionChromium(chromium, tmp_path / "profile", Path(installed))
        try:
            name = browser.worker_eval("chrome.runtime.getManifest().name")
            assert name == f"{product} Bridge"
            # What the toolbar icon says under the pointer.
            title = browser.worker_eval("chrome.action.getTitle({})")
            assert title == f"{product} Bridge — click to reconnect"
            bridge.stdin.write("\n")
            bridge.stdin.flush()
            assert json.loads(bridge.stdout.readline()) == [True, "Red", 42]
        finally:
            browser.close()
    finally:
        bridge.kill()
        bridge.wait()


# The bridge of a copy of the package, in the home of this module's browser.  For every
# line on stdin, the seconds to wait for the extension, it says what a command through it
# gave: the kind of its result, or the text of its error.
_SERVE = """
import json
import sys

from kiss.agents.sorcar.chrome_bridge import ChromeBridge, ChromeBridgeError, extension_version

bridge = ChromeBridge(write_config=True)
bridge.start()
print(extension_version(), flush=True)
for seconds in sys.stdin:
    bridge.wait_for_extension(float(seconds))
    try:
        answer = type(bridge.call("list_tabs", timeout=10)).__name__
    except ChromeBridgeError as e:
        answer = str(e)
    print(json.dumps(answer), flush=True)
bridge.stop()
"""

# The install Chrome loaded, and whether its worker is the newer product's.
_LOADED = "[chrome.runtime.getManifest().version, globalThis.newer]"


class _NewerProduct:
    """The bridge of a newer product, started where this product's extension is installed."""

    def __init__(self, root: Path) -> None:
        """Copy the package into *root*, its worker one line longer, and start its bridge."""
        package = make_branded_package(root / "newer", DEFAULT_BRAND)
        worker = package / "kiss" / "agents" / "sorcar" / "chrome_extension" / "background.js"
        worker.write_text(worker.read_text() + "globalThis.newer = true;\n")
        self._bridge = subprocess.Popen(
            [sys.executable, "-c", _SERVE],
            env=os.environ | {"PYTHONPATH": str(package)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._bridge.stdout is not None
        self.version = self._bridge.stdout.readline().strip()  # of the extension it installed

    def command(self, wait: float) -> str:
        """Wait *wait* seconds at most for the extension, then run a command through the bridge.

        Returns:
            ``"list"`` when the command listed the tabs, else the error's text.
        """
        assert self._bridge.stdin is not None and self._bridge.stdout is not None
        self._bridge.stdin.write(f"{wait}\n")
        self._bridge.stdin.flush()
        return str(json.loads(self._bridge.stdout.readline()))

    def stop(self) -> None:
        """Stop the bridge, which takes its config file with it."""
        assert self._bridge.stdin is not None
        self._bridge.stdin.close()
        try:
            self._bridge.wait(30)
        finally:
            self._bridge.kill()
            self._bridge.wait()


def test_a_newer_install_reaches_a_running_chrome_by_itself(
    chrome: _Chrome, tmp_path: Path
) -> None:
    """Chrome runs the worker it loaded until the extension is reloaded: the worker sees to that.

    A newer product installs its extension where Chrome loaded this one's from, and its
    bridge refuses the worker of this one.  So refused, the worker finds another version on
    disk than the one Chrome loaded and reloads the extension: the bridge gets the worker it
    installed.

    Back under this product the extension is switched off while the files are installed.
    Switched on, it starts the worker of the manifest Chrome holds, the newer product's:
    that worker's file is still there, and the same happens again.
    """
    assert chrome.browser.worker_eval(_LOADED) == [extension_version(), None]
    # The product's install instructions require this: Chrome disables an
    # unpacked extension when it reloads without Developer mode.
    chrome.browser.set_developer_mode(True)
    chrome.release()
    chrome.browser.stop_worker_at("chrome.runtime.reload()")
    newer = _NewerProduct(tmp_path)
    try:
        assert newer.version != extension_version()
        chrome.browser.click_toolbar_icon()
        # The reload ends the worker and what Chrome knows of its run: that is recorded
        # with the worker stopped before the line, which then runs as it is.
        chrome.browser.let_worker_go()
        assert newer.command(60) == "list"  # the newer bridge drives the extension it installed
        assert chrome.browser.worker_eval(_LOADED) == [newer.version, True]
    finally:
        newer.stop()
        with chrome.browser.switched_off():
            chrome.bridge = ChromeBridge(write_config=True)
            chrome.bridge.start(takeover=True)  # installs this product's files again
    assert chrome.bridge.wait_for_extension(30), "the extension never came back"
    assert chrome.browser.worker_eval(_LOADED) == [extension_version(), None]
    assert isinstance(chrome.bridge.call("list_tabs", timeout=10), list)


def test_an_upgrade_with_developer_mode_off_is_back_once_the_mode_is_on(
    chromium: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Developer mode off, Chrome switches an unpacked extension that reloads off.

    An extension Chrome has not reloaded since it was loaded runs on when the mode is
    switched off (one reloaded before is switched off there and then, so the browser here
    is a new one).  No worker can tell the mode, and the reload button of chrome://extensions
    ends the same way, so the worker reloads as for every newer install.  The bridge that
    refused it says what to switch on, and with the mode on the extension is back by itself,
    on the newer install.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    browser = ExtensionChromium(chromium, tmp_path / "profile", install_extension())
    chrome = _Chrome(browser, site="")
    try:
        chrome.live_bridge()
        browser.set_developer_mode(False)
        assert isinstance(chrome.live_bridge().call("list_tabs", timeout=10), list)  # runs on
        chrome.release()  # from here on the worker looks for a bridge every three seconds
        newer = _NewerProduct(tmp_path)
        try:
            _until(lambda: browser.extension_state() == "DISABLED", timeout=20)
            assert newer.command(5) == (
                "Chrome loaded its copy of the KISS Sorcar Bridge extension before the one in "
                f"{extension_dir()} was installed, and it is not running the new one. Open "
                "chrome://extensions, switch on Developer mode if it is off (Chrome runs the "
                "extension only with it), and click the reload button (the round arrow) on "
                "the extension's card. With Developer mode on, later updates load by themselves."
            )
            browser.set_developer_mode(True)
            assert newer.command(60) == "list"
            assert browser.worker_eval(_LOADED) == [newer.version, True]
        finally:
            newer.stop()
    finally:
        chrome.release()
        browser.close()


def test_a_refusal_with_no_other_install_on_disk_is_asked_again(chrome: _Chrome) -> None:
    """Refused while the version it runs is the one on disk, the worker stays and asks again.

    A bridge refuses a worker after it has put its own extension in place, and the worker
    reloads onto that.  Where the refusal leaves the loaded version on disk there is nothing
    to load, and a reload would only be refused again; a manifest that cannot be read, as
    while it is being replaced, says nothing either way.  Only a peer refuses the worker of
    the installed extension: a bridge takes it.
    """
    manifest = extension_dir() / "manifest.json"
    installed = manifest.read_bytes()
    staged = manifest.with_name("staged-manifest")
    chrome.browser.worker_eval("globalThis.sameWorker = true")
    try:
        with chrome.peer() as peer:
            ws = peer.accept()
            for readable in (True, False):
                nonce = secrets.token_hex(16)
                ws.send(json.dumps({"type": "challenge", "nonce": nonce}))
                assert json.loads(ws.recv(timeout=10)) == _hello(peer.token, nonce)
                if not readable:
                    staged.write_bytes(installed[: len(installed) // 2])
                    staged.replace(manifest)
                ws.send(json.dumps({"type": "hello_error", "error": "outdated extension"}))
                ws.close(4004, "outdated extension")
                ws = peer.accept(timeout=10)  # asked again, three seconds later
    finally:
        staged.write_bytes(installed)
        staged.replace(manifest)
    assert chrome.browser.extension_state() == "ENABLED"
    assert chrome.browser.worker_eval("globalThis.sameWorker") is True  # it never reloaded


# This product's install and the one of a product from before the worker's file was named
# after its build (both files under fixed names, each replaced when it differs) take turns
# in one folder, for as long as the process lives.
_UPGRADING = """
import itertools
import sys
import time
from pathlib import Path

from kiss.agents.sorcar.chrome_bridge import install_extension
from kiss.core.utils import atomic_write_text

before = {Path(path).name: Path(path).read_text() for path in sys.argv[1:]}


def install_as_before(target):
    for name, text in before.items():
        if not (target / name).exists() or (target / name).read_text() != text:
            atomic_write_text(target / name, text, create_mode=0o644)


target = install_extension()
print(target, flush=True)
for turn in itertools.count():
    if turn % 2:
        install_extension()
    else:
        install_as_before(target)
    time.sleep(0.0002)
"""


def _runs_while_upgraded(chromium: str, before: tuple[Path, Path], root: Path) -> bool:
    """Whether the extension of a browser started in *root* runs, installs under way all the while.

    The browser loads the folder the two installs write into, the files *before* being those
    of the older one.  A worker that runs dials the bridge its config names, here a bare
    listener.
    """
    installing = subprocess.Popen(
        [sys.executable, "-c", _UPGRADING, *map(str, before)],
        env=os.environ | {"KISS_HOME": str(root / "home")},
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert installing.stdout is not None
        extension = Path(installing.stdout.readline().strip())
        with closing(socket.create_server(("127.0.0.1", 0))) as bridge:
            bridge.settimeout(30)
            config = {"port": bridge.getsockname()[1], "token": secrets.token_hex(16)}
            (extension / CONFIG_FILE_NAME).write_text(json.dumps(config))
            browser = start_chromium(chromium, root / "profile", extension)
            try:
                bridge.accept()[0].close()
                return True
            except TimeoutError:
                return False
            finally:
                stop_chromium(browser)
    finally:
        installing.kill()
        installing.wait()


def test_an_upgrade_installed_while_chrome_starts_leaves_the_extension_running(
    chromium: str, tmp_path: Path
) -> None:
    """An install under way while Chrome loads the extension must leave it a whole worker.

    Chrome reads a worker's size and, about a millisecond later, its bytes: a file replaced in
    between by a longer one ends at the old size, parses no more, and leaves the extension
    without a worker until it is reloaded.  The install of a product from before this one
    (its worker, shorter, under the fixed name ``background.js``) and this product's replace
    each other here while browsers start, far more often than the one time an upgrade does.
    """
    lines = BACKGROUND_JS.read_text().split("\n")
    comment = next(n for n, line in enumerate(lines) if line.startswith("// ") and len(line) > 60)
    before = (tmp_path / "manifest.json", tmp_path / "background.js")
    manifest = BACKGROUND_JS.with_name("manifest.json").read_text()
    assert '"service_worker": "background.js"' in manifest
    before[0].write_text(render_brand(manifest, DEFAULT_BRAND))
    before[1].write_text("\n".join(lines[:comment] + lines[comment + 1 :]))  # shorter, same code
    roots = [tmp_path / f"browser{n}" for n in range(24)]
    with ThreadPoolExecutor(8) as starts:
        running = list(starts.map(partial(_runs_while_upgraded, chromium, before), roots))
    assert all(running), f"{running.count(False)} of 24 browsers got no running extension"


# ------------------------------------------------------------- the coverage


def test_the_count_names_the_code_that_did_not_run(
    chrome: _Chrome, chromium: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count the gate below reads tells the code that ran from the code that did not.

    A browser of its own has run one command, ``list_tabs``: none of that command's code is
    named, and ``close_tab``, which nothing sent, is.  What the worker puts into pages is
    counted where it runs: ``runCode`` is named until a page has run a script.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    browser = ExtensionChromium(chromium, tmp_path / "profile", install_extension())
    own = _Chrome(browser, chrome.site)
    try:
        assert isinstance(own.live_bridge().call("list_tabs", timeout=10), list)
        assert browser.uncovered("list_tabs") == []
        close_tab = browser.uncovered("close_tab")
        assert len(close_tab) == 1 and "chrome.tabs.remove(args.tabId)" in close_tab[0], close_tab
        assert close_tab[0] in browser.uncovered()  # the list of the whole file, the gate's
        in_pages = "async function runCode("
        assert any(in_pages in code for code in browser.uncovered())
        tab, _ = own.open("/red")
        assert own.evaluate(tab, "document.title") == "Red"
        assert not any(in_pages in code for code in browser.uncovered())
    finally:
        own.release()
        browser.close()


def _slow(test: object) -> bool:
    return any(mark.name == "slow" for mark in getattr(test, "pytestmark", []))


def test_every_line_of_the_extension_ran(chrome: _Chrome, request: pytest.FixtureRequest) -> None:
    """V8's own count, in the worker and in the pages: no code of ``background.js`` is left.

    The tests above are what runs it: when some were not selected, there
    is nothing to conclude, and this one is skipped.
    """
    selected = {
        getattr(item, "originalname", item.name)  # a test's name, less its parameters
        for item in request.session.items
        if item.path == request.path
    }
    needed = {
        name
        for name, test in vars(sys.modules[__name__]).items()
        if name.startswith("test_") and not _slow(test)
    }
    if needed - selected:
        pytest.skip(f"needs the whole module; not selected: {sorted(needed - selected)}")
    never_ran = chrome.browser.uncovered()
    assert not never_ran, "code of background.js that no test ran:\n" + "\n".join(never_ran)

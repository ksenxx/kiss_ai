# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for the Chrome extension bridge (chrome_bridge.py + use_my_chrome).

A real loopback WebSocket server is started per test and driven by a real
WebSocket client playing the extension's side of the protocol over the
socket (the same shape as the HTTP server in test_web_devtools_tools.py):
the JavaScript half of the extension cannot run inside pytest, so the
protocol peer is scripted — no mocks, patches, or test doubles of the code
under test.  The extension itself is verified manually in real Chrome.

Branches not covered here, and why (unreachable without test doubles):

* ``_handle``'s broad ``except Exception`` on a malformed frame: a client
  sending garbage is dropped the same way as a clean disconnect, which the
  disconnect test covers; only the log line differs.
* ``call``'s send-failure branch (socket dies between the connected check
  and send): a nanosecond race not arrangeable deterministically.
* Connection replacement's ``old.ws.close`` exception branch: the replaced
  socket dying inside the close call.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import json
import linecache
import os
import queue
import re
import socket
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from websockets.client import ClientProtocol
from websockets.exceptions import ConnectionClosed
from websockets.frames import Frame, Opcode
from websockets.sync.client import connect
from websockets.uri import parse_uri

from kiss.agents.sorcar import chrome_bridge
from kiss.agents.sorcar.chrome_bridge import (
    CONFIG_FILE_NAME,
    ChromeBridge,
    ChromeBridgeError,
    extension_dir,
    extension_version,
    start_if_enabled,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool, _format_bridge_result
from kiss.core.brand import PRODUCT_NAME
from kiss.core.user_config import load_config, save_config
from kiss.tests.conftest import IS_WINDOWS
from kiss.tests.lock_injection import inject

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _proof(key: str, message: str) -> str:
    return hashlib.sha256(f"{key}:{message}".encode()).hexdigest()


def _hello(token: str, nonce: str) -> dict[str, str]:
    """The hello of the extension this product installs, in answer to the challenge *nonce*."""
    return {"type": "hello", "proof": _proof(token, nonce), "version": extension_version()}


class ExtensionPeer:
    """The extension's side of the bridge protocol, over a real socket.

    Performs the mutual challenge-response handshake (and verifies the
    server's proof, as background.js does).  With ``responder`` set, the
    reader thread answers every command immediately, so the synchronous
    WebUseTool API can be called directly.  The responder returns the
    command's result or raises ChromeBridgeError to send an error reply.

    The constructor returns once the bridge has published the connection.
    The bridge sends ``hello_ok`` first (the extension needs the server's
    proof before any command) and reads this peer's frames only after the
    publish, so the pong to one ping is the sign; returning on ``hello_ok``
    let a ``bridge.call`` right after it find no connection.  A test that
    holds the publish back itself passes ``wait_published=False``.
    """

    def __init__(
        self,
        port: int,
        token: str,
        responder: Callable[[dict[str, Any]], Any] | None = None,
        wait_published: bool = True,
    ):
        self.ws = connect(f"ws://127.0.0.1:{port}")
        challenge = json.loads(self.ws.recv(timeout=5))
        assert challenge["type"] == "challenge"
        nonce = challenge["nonce"]
        self.ws.send(json.dumps(_hello(token, nonce)))
        hello = json.loads(self.ws.recv(timeout=5))
        assert hello["type"] == "hello_ok"
        assert hello["proof"] == _proof(nonce, token)  # server proves itself too
        self.incoming: queue.Queue[Any] = queue.Queue()
        self._responder = responder
        self._published = threading.Event()
        if wait_published:
            self.ws.send(json.dumps({"type": "ping"}))
        else:  # no ping of the constructor's: every pong is the test's own
            self._published.set()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        assert self._published.wait(5), "the bridge never served this connection"

    def _pump(self) -> None:
        try:
            while True:
                msg = json.loads(self.ws.recv(timeout=30))
                if msg == {"type": "pong"} and not self._published.is_set():
                    self._published.set()  # the constructor's ping, not a test's
                elif self._responder is not None and isinstance(msg.get("id"), int):
                    try:
                        self.reply(msg["id"], result=self._responder(msg))
                    except ChromeBridgeError as e:
                        self.reply(msg["id"], error=str(e))
                else:
                    self.incoming.put(msg)
        except Exception:  # noqa: BLE001 - closed: signal with None
            self.incoming.put(None)

    def next_command(self, timeout: float = 10.0) -> dict[str, Any]:
        msg = self.incoming.get(timeout=timeout)
        assert isinstance(msg, dict), "extension socket closed unexpectedly"
        return msg

    def reply(self, msg_id: int, result: Any = None, error: str | None = None) -> None:
        if error is not None:
            self.ws.send(json.dumps({"id": msg_id, "ok": False, "error": error}))
        else:
            self.ws.send(json.dumps({"id": msg_id, "ok": True, "result": result}))

    def close(self) -> None:
        self.ws.close()


@pytest.fixture(autouse=True)
def _fresh_bridge_state():
    """No singleton and no config file before or after any test.

    The tool tests start the process-wide bridge lazily (as a CLI run
    does) and would otherwise hand it, config file included, to whatever
    test runs next; the config file alone makes a later ``start()`` defer
    to the "live" bridge it names.
    """
    ChromeBridge.shutdown()
    (extension_dir() / CONFIG_FILE_NAME).unlink(missing_ok=True)
    yield
    ChromeBridge.shutdown()
    (extension_dir() / CONFIG_FILE_NAME).unlink(missing_ok=True)


@pytest.fixture
def bridge():
    b = ChromeBridge(port=0, write_config=False)
    b.start()
    yield b
    b.stop()


def _call_in_thread(bridge: ChromeBridge, cmd: str, args: dict | None = None, timeout: float = 5.0):
    """Run bridge.call on a worker thread; return (thread, outcome box)."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = bridge.call(cmd, args, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - asserted by the test
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def test_hello_ok_marks_connected(bridge: ChromeBridge) -> None:
    peer = ExtensionPeer(bridge.port, bridge.token)
    try:
        assert bridge.extension_connected
    finally:
        peer.close()


def test_bad_token_rejected(bridge: ChromeBridge) -> None:
    ws = connect(f"ws://127.0.0.1:{bridge.port}")
    challenge = json.loads(ws.recv(timeout=5))
    ws.send(json.dumps({"type": "hello", "proof": _proof("wrong", challenge["nonce"])}))
    assert json.loads(ws.recv(timeout=5))["type"] == "hello_error"
    with pytest.raises(ConnectionClosed) as exc_info:
        ws.recv(timeout=5)
    assert exc_info.value.rcvd is not None and exc_info.value.rcvd.code == 4001
    assert not bridge.extension_connected


@pytest.mark.parametrize("installs", [True, False], ids=["installing", "not-installing"])
def test_an_extension_chrome_loaded_before_this_install_is_refused_and_named(
    installs: bool,
) -> None:
    """Chrome keeps running the worker it loaded: one of another install gets no command.

    The worker of an install from before this check names no version, one
    of another build names its own.  A bridge that installs the extension
    puts its files back, whatever replaced them, so that the reload (the
    refused worker's own, or the one the user is asked for) loads this
    install.  Chrome switches an unpacked extension that reloads off
    while Developer mode is off, so both complaints name the mode.
    """
    ext = chrome_bridge.install_extension()
    another = b'{"version": "1.0.0"}'
    b = ChromeBridge(write_config=installs)
    b.start()
    try:
        (ext / "manifest.json").write_bytes(another)
        for names in ({}, {"version": "1.0.0"}):
            ws = connect(f"ws://127.0.0.1:{b.port}")
            nonce = json.loads(ws.recv(timeout=5))["nonce"]
            ws.send(json.dumps({"type": "hello", "proof": _proof(b.token, nonce)} | names))
            refusal = {"type": "hello_error", "error": "outdated extension"}
            assert json.loads(ws.recv(timeout=5)) == refusal
            with pytest.raises(ConnectionClosed) as closed:
                ws.recv(timeout=5)
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 4004
            assert not b.extension_connected
        mine = chrome_bridge.bundled_extension()["manifest.json"]
        assert (ext / "manifest.json").read_bytes() == (mine if installs else another)
        with pytest.raises(ChromeBridgeError, match="click the reload button") as refused:
            b.call("list_tabs", timeout=1.0)
        assert str(ext) in str(refused.value)
        assert "switch on Developer mode if it is off" in str(refused.value)
        # The extension of this install is taken, and with it the complaint is over.
        ExtensionPeer(b.port, b.token).close()
        assert _eventually(lambda: not b.extension_connected)
        with pytest.raises(ChromeBridgeError, match="is not connected") as absent:
            b.call("list_tabs", timeout=1.0)
        assert "keep Developer mode on" in str(absent.value)
    finally:
        b.stop()
        chrome_bridge.install_extension()


def test_first_message_not_hello_rejected(bridge: ChromeBridge) -> None:
    ws = connect(f"ws://127.0.0.1:{bridge.port}")
    assert json.loads(ws.recv(timeout=5))["type"] == "challenge"
    ws.send(json.dumps({"type": "ping"}))
    with pytest.raises(ConnectionClosed):
        while True:
            ws.recv(timeout=5)


def test_rogue_server_learns_nothing_reusable(bridge: ChromeBridge) -> None:
    """A proof for one nonce must not authenticate against a fresh nonce."""
    ws = connect(f"ws://127.0.0.1:{bridge.port}")
    nonce1 = json.loads(ws.recv(timeout=5))["nonce"]
    stolen = _proof(bridge.token, nonce1)  # what a rogue listener could capture
    ws.close()
    ws2 = connect(f"ws://127.0.0.1:{bridge.port}")
    nonce2 = json.loads(ws2.recv(timeout=5))["nonce"]
    assert nonce2 != nonce1
    ws2.send(json.dumps({"type": "hello", "proof": stolen}))  # replayed: wrong nonce
    assert json.loads(ws2.recv(timeout=5))["type"] == "hello_error"
    ws2.close()


def test_command_round_trip(bridge: ChromeBridge) -> None:
    peer = ExtensionPeer(bridge.port, bridge.token)
    try:
        t, box = _call_in_thread(bridge, "list_tabs")
        cmd = peer.next_command()
        assert cmd["cmd"] == "list_tabs"
        assert isinstance(cmd["id"], int)
        tabs = [{"id": 7, "title": "Inbox", "url": "https://mail.google.com", "active": True}]
        peer.reply(cmd["id"], result=tabs)
        t.join(timeout=10)
        assert box["result"] == tabs
    finally:
        peer.close()


def test_error_reply_raises(bridge: ChromeBridge) -> None:
    peer = ExtensionPeer(bridge.port, bridge.token)
    try:
        t, box = _call_in_thread(bridge, "evaluate", {"tabId": 1, "script": "x"})
        cmd = peer.next_command()
        assert cmd["args"] == {"tabId": 1, "script": "x"}
        peer.reply(cmd["id"], error="boom")
        t.join(timeout=10)
        assert isinstance(box["error"], ChromeBridgeError)
        assert "boom" in str(box["error"])
    finally:
        peer.close()


def test_command_timeout(bridge: ChromeBridge) -> None:
    peer = ExtensionPeer(bridge.port, bridge.token)
    t, box = _call_in_thread(bridge, "evaluate", {"tabId": 1, "script": "x"}, timeout=1.0)
    peer.next_command()  # receive but never reply
    t.join(timeout=10)
    assert isinstance(box["error"], ChromeBridgeError)
    assert "did not answer" in str(box["error"])
    # A silent peer is dropped so it cannot wedge later commands behind a
    # stuck socket; the extension (or a new one) reconnects and works.
    assert not bridge.extension_connected
    assert peer.incoming.get(timeout=10) is None
    peer2 = ExtensionPeer(bridge.port, bridge.token)
    try:
        t2, box2 = _call_in_thread(bridge, "list_tabs")
        cmd = peer2.next_command()
        peer2.reply(cmd["id"], result=[])
        t2.join(timeout=10)
        assert box2["result"] == []
    finally:
        peer2.close()


def _parked_in_a_wait_of(thread: threading.Thread, function: str) -> bool:
    """Whether *thread* is blocked in a condition wait under a call of *function*."""
    assert thread.ident is not None
    frame = sys._current_frames().get(thread.ident)
    if frame is None:
        return False
    names = [entry.name for entry in traceback.extract_stack(frame)]
    return function in names and names[-1] == "wait"


def _catching(caught: list[BaseException], call: Callable[..., Any], *args: Any) -> None:
    """Thread body: make *call*, keeping what it raises (a Stop is the expected outcome)."""
    try:
        call(*args)
    except BaseException as exc:  # noqa: BLE001
        caught.append(exc)


def test_a_stop_while_waiting_for_the_reply_ends_the_wait_within_a_slice(
    bridge: ChromeBridge,
) -> None:
    """A Stop injected while ``call`` waits for the extension's reply ends the wait at once.

    ``Future.result(timeout)`` blocks in one C-level lock wait, where an
    injected ``KeyboardInterrupt`` is not delivered: a Stop waited out the
    whole command timeout (up to 120 s for ``use_my_chrome``).  The wait
    is sliced now, like every lock acquisition on a task thread.
    """
    peer = ExtensionPeer(bridge.port, bridge.token)  # receives, never answers
    caught: list[BaseException] = []
    thread = threading.Thread(
        target=_catching, args=(caught, bridge.call, "list_tabs"), daemon=True,
    )
    thread.start()
    try:
        peer.next_command()  # the frame is on the wire: the caller now waits for the reply
        assert _eventually(lambda: _parked_in_a_wait_of(thread, "result"))
        started = time.monotonic()
        assert inject(thread) == 1
        thread.join(5)
        assert not thread.is_alive(), "the Stop did not end the reply wait"
        assert time.monotonic() - started < 3
        assert [type(exc) for exc in caught] == [KeyboardInterrupt]
        conn = bridge._conn
        assert conn is not None and conn.pending == {}  # the stopped command was unregistered
        assert bridge.extension_connected  # the timeout path (a drop) was not taken
    finally:
        peer.close()
    # The wait for an extension to connect (the first call of a bridge it
    # has not found yet) is sliced the same way.
    assert _eventually(lambda: not bridge.extension_connected)
    caught.clear()
    waiter = threading.Thread(
        target=_catching, args=(caught, bridge.wait_for_extension, 30.0), daemon=True,
    )
    waiter.start()
    assert _eventually(lambda: _parked_in_a_wait_of(waiter, "wait_for_extension"))
    started = time.monotonic()
    assert inject(waiter) == 1
    waiter.join(5)
    assert not waiter.is_alive(), "the Stop did not end the wait for the extension"
    assert time.monotonic() - started < 3
    assert [type(exc) for exc in caught] == [KeyboardInterrupt]


def test_late_reply_for_unknown_id_ignored(bridge: ChromeBridge) -> None:
    peer = ExtensionPeer(bridge.port, bridge.token)
    try:
        peer.reply(999, result="late")  # never sent: must not confuse routing
        t, box = _call_in_thread(bridge, "list_tabs")
        cmd = peer.next_command()
        peer.reply(cmd["id"], result=[])
        t.join(timeout=10)
        assert box["result"] == []
    finally:
        peer.close()


def test_disconnect_mid_command_fails_call(bridge: ChromeBridge) -> None:
    peer = ExtensionPeer(bridge.port, bridge.token)
    t, box = _call_in_thread(bridge, "screenshot", {"tabId": 1}, timeout=30.0)
    peer.next_command()
    peer.close()
    t.join(timeout=10)
    assert isinstance(box["error"], ChromeBridgeError)
    assert "disconnected" in str(box["error"])
    assert not bridge.extension_connected


def test_second_connection_replaces_first(bridge: ChromeBridge) -> None:
    first = ExtensionPeer(bridge.port, bridge.token)
    second = ExtensionPeer(bridge.port, bridge.token)
    try:
        assert first.incoming.get(timeout=10) is None  # closed as replaced
        assert bridge.extension_connected
        t, box = _call_in_thread(bridge, "list_tabs")
        cmd = second.next_command()
        second.reply(cmd["id"], result=[])
        t.join(timeout=10)
        assert box["result"] == []
    finally:
        second.close()


def test_ping_pong(bridge: ChromeBridge) -> None:
    peer = ExtensionPeer(bridge.port, bridge.token)
    try:
        peer.ws.send(json.dumps({"type": "ping"}))
        assert peer.next_command() == {"type": "pong"}
    finally:
        peer.close()


class StalledPeer:
    """An authenticated extension socket that reads only when the test says so.

    Driven by hand over a raw socket with websockets' sans-I/O client
    protocol (the threaded client reads on its own and would answer, or
    fail on, whatever arrives): after the handshake nothing drains the
    socket, as behind an extension whose service worker stalled.
    """

    def __init__(self, bridge: ChromeBridge):
        self.sock = socket.socket()
        # An explicit size pins the buffer: the kernel grows an unpinned
        # one while the pipe fills, and a write that found it full fits
        # a moment later.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.sock.connect(("127.0.0.1", bridge.port))
        self.proto = ClientProtocol(parse_uri(f"ws://127.0.0.1:{bridge.port}"))
        self.events: deque[Any] = deque()
        self.proto.send_request(self.proto.connect())
        self._flush()
        assert self._next_event().status_code == 101
        assert self.proto.handshake_exc is None
        nonce = self.recv_json()["nonce"]
        self.send_json(_hello(bridge.token, nonce))
        assert self.recv_json()["type"] == "hello_ok"
        assert bridge.wait_for_extension(5)

    def _flush(self) -> None:
        for data in self.proto.data_to_send():
            self.sock.sendall(data)

    def _next_event(self) -> Any:
        # ``events_received`` hands over every event parsed so far and
        # forgets them: one read can carry the handshake response and the
        # challenge frame together, so the events are queued.
        while not self.events:
            self.proto.receive_data(self.sock.recv(65536))
            self.events.extend(self.proto.events_received())
        return self.events.popleft()

    def recv_json(self) -> dict[str, Any]:
        return dict(json.loads(self._next_event().data))

    def send_json(self, obj: dict[str, Any]) -> None:
        self.proto.send_text(json.dumps(obj).encode())
        self._flush()

    def fill_pipe(self, bridge: ChromeBridge) -> None:
        """Fill the kernel buffers from the bridge's socket towards this peer.

        The next write of a single byte on the bridge's side then blocks in
        ``sendall``: the state a stalled extension leaves the bridge in.
        The writes go through a non-blocking duplicate of the bridge's
        socket (``MSG_DONTWAIT`` does not exist on Windows, and macOS
        still blocks a large send that finds the buffer partly full).  The
        flag is shared with the bridge's socket, so the writes start only
        once the bridge's reader is parked in a blocking ``recv``
        (:func:`_wait_until_reader_parked`), and the socket is blocking
        again before this returns.  Filled in
        rounds: this peer's kernel still opens its window a little after
        a write found no room (its window updates ride delayed acks, at
        most 200 ms apart), so a round ends only when a one-byte probe
        still finds the pipe full after that.
        """
        conn = bridge._conn
        assert conn is not None
        conn.ws.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # pinned too
        _wait_until_reader_parked(conn.ws.recv_events_thread)
        with conn.ws.socket.dup() as sock:
            sock.setblocking(False)
            try:
                while True:
                    try:
                        sock.send(b"\0" * 65536)
                        continue
                    except BlockingIOError:
                        time.sleep(0.3)
                    try:
                        sock.send(b"\0")
                    except BlockingIOError:
                        return
            finally:
                sock.setblocking(True)

    def abort(self) -> None:
        """Close with data unread: the bridge's blocked write fails at once."""
        self.sock.close()


def _reader_at_recv(reader: threading.Thread) -> bool:
    """Whether websockets' *reader* thread waits for the socket in ``recv_events``.

    websockets 15 blocks in ``socket.recv`` there; 17 waits in a selector
    first, whose frames lie under that of ``recv_events``.
    """
    frame = sys._current_frames().get(reader.ident or 0)
    while frame is not None and frame.f_code.co_name != "recv_events":
        frame = frame.f_back
    if frame is None:
        return False
    line = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
    return "socket.recv(" in line or "poller.select()" in line


def _wait_until_reader_parked(reader: threading.Thread) -> None:
    """Wait until websockets' *reader* thread is parked waiting for the socket.

    On the ``recv`` line in two samples 50 ms apart, with nothing
    arriving: the thread is inside the call, which chose to block when
    it was entered, so a non-blocking flag set afterwards cannot reach it.
    (Parked in websockets 17's selector, it reads only once data is there.)
    """
    deadline = time.monotonic() + 10
    while True:
        if _reader_at_recv(reader):
            time.sleep(0.05)
            if _reader_at_recv(reader):
                return
        assert time.monotonic() < deadline, "the bridge's reader never blocked in recv"
        time.sleep(0.01)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="only Linux's loopback keeps a pipe full once fill_pipe's probe found it so: "
    "macOS keeps opening room for seconds under load, and Windows still completes a "
    "small blocking send, so the pong is written instead of stalling",
)
def test_a_pong_the_extension_stopped_reading_is_cut_and_the_connection_dropped(
    bridge: ChromeBridge,
) -> None:
    """A peer that still pings but no longer drains its socket must not wedge the bridge.

    The reply to its ping blocks in ``sendall`` with the send lock held, as
    behind a stalled extension.  Bounded like a command's write, the pong is
    cut by its watchdog, the connection is dropped and a fresh extension
    connection works.  Unbounded, every later command failed as "busy"
    while the bridge still reported the extension as connected.
    """
    stalled = StalledPeer(bridge)
    try:
        stalled.fill_pipe(bridge)
        stalled.send_json({"type": "ping"})
        assert _eventually(lambda: not bridge.extension_connected, timeout=10.0)
        peer = ExtensionPeer(bridge.port, bridge.token, responder=lambda cmd: "answered")
        try:
            assert bridge.call("list_tabs", timeout=5) == "answered"
        finally:
            peer.close()
    finally:
        stalled.abort()


@pytest.mark.skipif(sys.platform != "linux", reason="needs a pipe that stays full, as above")
def test_a_pong_written_as_its_deadline_passes_keeps_the_connection(bridge: ChromeBridge) -> None:
    """A watchdog that fired as the write returned cuts nothing once the write has returned.

    ``Timer.cancel`` does not stop a timer whose thread is already in its
    body: the watchdog of a pong that got through at the very end of its
    budget shut the socket down afterwards, under whatever the healthy
    connection carried next.  The watchdog's thread is parked as it
    enters its body (a profile hook: nothing is replaced), the peer
    drains the pipe so the write returns, and the watchdog is let go.
    """
    stalled = StalledPeer(bridge)
    fired, written = threading.Event(), threading.Event()
    watchdogs: list[threading.Thread] = []

    def park(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is chrome_bridge._cut_socket.__code__:
            watchdogs.append(threading.current_thread())
            fired.set()
            written.wait(30)

    conn = bridge._conn
    assert conn is not None
    try:
        stalled.fill_pipe(bridge)
        threading.setprofile(park)  # for the watchdog's thread, started by the pong's write
        try:
            stalled.send_json({"type": "ping"})
            assert fired.wait(chrome_bridge._PONG_TIMEOUT_SECS + 10), "the watchdog never fired"
        finally:
            threading.setprofile(None)
        stalled.sock.settimeout(0.2)
        deadline = time.monotonic() + 10
        while not conn.lock.acquire(blocking=False):  # held by the pong's write
            assert time.monotonic() < deadline, "the drained pong was never written"
            try:
                stalled.sock.recv(1 << 20)
            except TimeoutError:
                pass
        conn.lock.release()
        written.set()
        watchdogs[0].join(5)
        assert not watchdogs[0].is_alive()
        try:  # what was still in the pipe: the fill's last bytes, the pong
            while stalled.sock.recv(1 << 20):
                pass
        except TimeoutError:
            pass
        assert bridge.extension_connected
        pong = Frame(Opcode.TEXT, json.dumps({"type": "pong"}).encode()).serialize(mask=False)
        stalled.send_json({"type": "ping"})
        stalled.sock.settimeout(5)
        received = b""
        while len(received) < len(pong):
            part = stalled.sock.recv(len(pong) - len(received))
            assert part, "the watchdog cut the connection after its write had returned"
            received += part
        assert received == pong
    finally:
        written.set()
        stalled.abort()


def test_stop_does_not_wait_for_a_peer_that_stopped_reading(bridge: ChromeBridge) -> None:
    """``stop()`` shuts the live socket down before closing it, as the timeout path does.

    A bare close writes a close frame and waits for the peer's, up to
    websockets' 10 s close_timeout: a peer that stopped reading held the
    daemon's shutdown for that long.
    """
    stalled = StalledPeer(bridge)
    stopper = threading.Thread(target=bridge.stop)
    try:
        stalled.fill_pipe(bridge)
        stopper.start()
        stopper.join(5)
        assert not stopper.is_alive(), "stop() waited for the peer"
        assert not bridge.extension_connected
    finally:
        stalled.abort()
        stopper.join(30)


def test_a_handshake_in_flight_cannot_connect_a_stopped_bridge() -> None:
    """The extension's hello lands after ``stop()``: the stopped bridge must not keep it.

    Closing the listener does not end an accepted handshake; published,
    the connection would keep the extension away from any live bridge.
    """
    b = ChromeBridge(port=0, write_config=False)
    b.start()
    ws = connect(f"ws://127.0.0.1:{b.port}")
    try:
        nonce = json.loads(ws.recv(timeout=5))["nonce"]
        b.stop()
        # websockets 17 closes the connection as its server shuts down: there
        # the hello itself is refused.
        with pytest.raises(ConnectionClosed):
            ws.send(json.dumps(_hello(b.token, nonce)))
            ws.send(json.dumps({"type": "ping"}))
            while True:
                msg = json.loads(ws.recv(timeout=5))
                assert msg.get("type") != "pong", "a stopped bridge serves the extension"
        assert not b.extension_connected
    finally:
        ws.close()
        b.stop()


def test_call_without_extension(bridge: ChromeBridge) -> None:
    with pytest.raises(ChromeBridgeError, match="not connected"):
        bridge.call("list_tabs", timeout=1.0)


def test_bind_conflict(bridge: ChromeBridge) -> None:
    other = ChromeBridge(port=bridge.port, write_config=False)
    with pytest.raises(ChromeBridgeError, match="cannot listen"):
        other.start()


# -- extension install dir, config file, ownership between processes ------
# All under the test session's temporary KISS_HOME (conftest), never the
# user's ~/.kiss.


def _config() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((extension_dir() / CONFIG_FILE_NAME).read_text())
    return data


def test_start_installs_extension_and_writes_config_under_kiss_home() -> None:
    ext = extension_dir()
    assert ext == Path(os.environ["KISS_HOME"]) / "chrome_extension"
    bundled = Path(chrome_bridge.__file__).parent / "chrome_extension"
    b = ChromeBridge(write_config=True)
    b.start()
    try:
        assert b.port != 0  # ephemeral: whatever the kernel handed out
        # The manifest is installed with the brand filled in: Chrome lists
        # the extension under the product's name, never the placeholder
        # or the stock name.  Its version is the one the bridge takes.
        manifest = json.loads((ext / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["name"] == f"{PRODUCT_NAME} Bridge"
        assert manifest["version"] == extension_version()
        # The worker it names is the bundled one, in a file of this build's own.
        worker = ext / manifest["background"]["service_worker"]
        assert worker.name == f"background-{extension_version()}.js"
        assert worker.read_bytes() == (bundled / "background.js").read_bytes()
        assert "{{" not in (ext / "manifest.json").read_text(encoding="utf-8")
        assert "{{PRODUCT_NAME}}" in (bundled / "manifest.json").read_text(encoding="utf-8")
        cfg_path = ext / CONFIG_FILE_NAME
        assert _config() == {"port": b.port, "token": b.token}
        # Windows chmod only toggles the read-only bit.
        assert IS_WINDOWS or cfg_path.stat().st_mode & 0o777 == 0o600
        # The extension dir is where the user is told to "Load unpacked".
        with pytest.raises(ChromeBridgeError, match=re.escape(str(ext))):
            b.call("list_tabs", timeout=1.0)
    finally:
        b.stop()
    # Stopping withdraws the file: the extension reports a missing bridge
    # instead of retrying a dead port.
    assert not cfg_path.exists()


def test_install_neither_replaces_nor_removes_a_worker_file_chrome_may_read() -> None:
    """Chrome reads a worker's size, then its bytes: no install changes the file in between.

    Every build's worker has a file of its own, which its manifest names,
    so an install leaves the worker of every other build as it is (here
    the ``background.js`` of an install from before workers were named
    after their build, and one of another build), however long ago this
    one was installed: a Chrome that loaded another build's manifest
    starts that build's worker whenever the extension is switched on, and
    another process may be installing that build right now.
    """
    ext = extension_dir()
    ext.mkdir(parents=True, exist_ok=True)
    stale = b"// the worker of another install\n"
    others = [ext / "background.js", ext / "background-1.2.3.4.js"]
    for other in others:
        other.write_bytes(stale)
    try:
        chrome_bridge.install_extension()
        files = chrome_bridge.bundled_extension()
        worker = ext / json.loads(files["manifest.json"])["background"]["service_worker"]
        assert worker.name == f"background-{extension_version()}.js"
        assert {name: (ext / name).read_bytes() for name in files} == files
        bundled = (chrome_bridge._EXTENSION_SOURCE / "background.js").read_bytes()
        assert files[worker.name] == bundled
        assert worker.stat().st_mode & 0o444 == 0o444  # Chrome can read it
        written = [(ext / name).stat().st_mtime_ns for name in files]
        chrome_bridge.install_extension()  # nothing changed: nothing is written again
        assert [(ext / name).stat().st_mtime_ns for name in files] == written
        a_year_ago = time.time() - 365 * 24 * 3600
        os.utime(ext / "manifest.json", (a_year_ago, a_year_ago))
        chrome_bridge.install_extension()
        assert [other.read_bytes() for other in others] == [stale, stale]
    finally:
        for other in others:
            other.unlink(missing_ok=True)


@pytest.mark.parametrize("at", ["publication", "registration"])
def test_a_stop_during_the_lazy_start_leaves_no_orphan_bridge(at: str) -> None:
    """A Stop landing in ``instance()``'s start strands no server: the next start succeeds.

    The server thread and the config file are published before the
    singleton is installed; a Stop (an injected ``KeyboardInterrupt``)
    landing after the publication, or after ``start`` returned and before
    the singleton was set, left a live, ownerless server that every later
    ``instance()`` deferred to as "another KISS process".
    """
    caught: list[str] = []
    ports: list[int] = []

    def stop_at(frame: Any, event: str, arg: Any) -> None:
        if at == "publication" and event == "call" and frame.f_code.co_name == "info":
            if frame.f_back.f_code is ChromeBridge.start.__code__:  # start's final log line
                ports.append(frame.f_back.f_locals["self"].port)
                raise KeyboardInterrupt
        if at == "registration" and event == "c_call" and arg is atexit.register:
            ports.append(frame.f_locals["bridge"].port)
            raise KeyboardInterrupt

    def worker() -> None:
        sys.setprofile(stop_at)
        try:
            ChromeBridge.instance()
        except BaseException as exc:  # noqa: BLE001 - the Stop is the expected outcome
            caught.append(type(exc).__name__)
        finally:
            sys.setprofile(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(30)
    assert not thread.is_alive()
    assert caught == ["KeyboardInterrupt"]
    assert ChromeBridge._instance is None
    assert not (extension_dir() / CONFIG_FILE_NAME).exists()  # withdrawn with the server
    with pytest.raises(OSError):  # nothing serves the stranded port any more
        socket.create_connection(("127.0.0.1", ports[0]), timeout=2).close()
    again = ChromeBridge.instance()  # not "another KISS process already serves"
    assert again.port != 0 and _config() == {"port": again.port, "token": again.token}


def test_bundled_manifest_survives_quotes_and_backslashes_in_the_brand() -> None:
    # A product name with JSON-special characters still yields a valid
    # manifest (the values are escaped into the JSON strings).
    brand = {"product_name": 'ACME "Assistant" C:\\Tools é'}
    files = chrome_bridge.bundled_extension(brand)
    manifest = json.loads(files["manifest.json"])
    assert manifest["name"] == 'ACME "Assistant" C:\\Tools é Bridge'
    # default_title: the key Chrome reads the toolbar tooltip from.
    assert manifest["action"]["default_title"].startswith('ACME "Assistant" C:\\Tools é Bridge')
    assert 'ACME "Assistant" C:\\Tools é agent' in manifest["description"]
    # The version names the build, in numbers Chrome takes (four of at most
    # 65535): another brand's manifest is another build.
    version = [int(part) for part in manifest["version"].split(".")]
    assert len(version) == 4 and max(version) <= 65535
    assert manifest["version"] == extension_version(brand) != extension_version()
    # The worker is the same under every brand, in the file its manifest names.
    worker = manifest["background"]["service_worker"]
    assert worker == f"background-{manifest['version']}.js"
    stock = chrome_bridge.bundled_extension()
    assert files[worker] == stock[f"background-{extension_version()}.js"]


def test_second_process_defers_to_a_live_bridge_and_daemon_takes_over() -> None:
    daemon = ChromeBridge(write_config=True)
    daemon.start(takeover=True)
    try:
        # A CLI run beside the daemon: the extension stays with the daemon,
        # and the refused start binds nothing (websockets never closes a
        # Server's shutdown pipe: every refused use_my_chrome call leaked
        # its two descriptors).
        open_fds = 0 if IS_WINDOWS else len(os.listdir("/dev/fd"))
        for _ in range(3):
            cli = ChromeBridge(write_config=True)
            with pytest.raises(ChromeBridgeError, match=f"port {daemon.port}"):
                cli.start()
            assert cli.port == 0
        assert IS_WINDOWS or _eventually(lambda: len(os.listdir("/dev/fd")) <= open_fds)
        assert _config()["token"] == daemon.token
        # A restarted daemon (its predecessor still up during the overlap)
        # takes the file over...
        successor = ChromeBridge(write_config=True)
        successor.start(takeover=True)
        try:
            assert _config() == {"port": successor.port, "token": successor.token}
        finally:
            # ...and the predecessor's stop leaves the newer file alone.
            daemon.stop()
            assert _config()["token"] == successor.token
            successor.stop()
        assert not (extension_dir() / CONFIG_FILE_NAME).exists()
    finally:
        daemon.stop()


def test_two_lazy_starts_at_once_publish_exactly_one_bridge() -> None:
    """Two CLI runs start their bridge together: one publishes, the other defers to it.

    The "is another bridge live?" probe and the publication are one critical
    section; apart, both starts found no live bridge and both published, and
    the one whose file lost served a port the extension never reads.
    """
    for _ in range(5):
        bridges = [ChromeBridge(write_config=True) for _ in range(2)]
        errors: list[ChromeBridgeError | None] = [None, None]
        gate = threading.Barrier(2)

        def run(index: int) -> None:
            gate.wait()
            try:
                bridges[index].start()
            except ChromeBridgeError as e:
                errors[index] = e

        threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        winners = [b for b, e in zip(bridges, errors, strict=True) if e is None]
        try:
            assert len(winners) == 1, errors
            assert _config() == {"port": winners[0].port, "token": winners[0].token}
            assert f"port {winners[0].port}" in str(next(e for e in errors if e is not None))
        finally:
            for b in winners:
                b.stop()
        assert not (extension_dir() / CONFIG_FILE_NAME).exists()


def test_stale_config_of_a_dead_bridge_is_replaced() -> None:
    dead = ChromeBridge(write_config=True)
    dead.start()
    dead_port = dead.port
    assert dead._server is not None
    dead._server.shutdown()  # the process died: no stop(), the file stays
    dead._server = None
    try:
        assert _config()["port"] == dead_port
        b = ChromeBridge(write_config=True)
        b.start()  # no takeover needed: nothing answers on the stale port
        try:
            assert _config() == {"port": b.port, "token": b.token}
        finally:
            b.stop()
    finally:
        (extension_dir() / CONFIG_FILE_NAME).unlink(missing_ok=True)


def test_config_naming_a_non_bridge_listener_is_replaced() -> None:
    # The stale port was reused by an unrelated service (here: a bridge
    # test peer that never sends a challenge): not a bridge, so replaced.
    other = socket.socket()
    other.bind(("127.0.0.1", 0))
    other.listen()
    ext = extension_dir()
    ext.mkdir(parents=True, exist_ok=True)
    (ext / CONFIG_FILE_NAME).write_text(json.dumps({"port": other.getsockname()[1], "token": "x"}))
    try:
        b = ChromeBridge(write_config=True)
        b.start()
        try:
            assert _config()["token"] == b.token
        finally:
            b.stop()
    finally:
        other.close()
        (ext / CONFIG_FILE_NAME).unlink(missing_ok=True)


def test_garbage_config_is_replaced() -> None:
    ext = extension_dir()
    ext.mkdir(parents=True, exist_ok=True)
    (ext / CONFIG_FILE_NAME).write_text("[not an object")
    b = ChromeBridge(write_config=True)
    b.start()
    try:
        assert _config()["token"] == b.token
    finally:
        b.stop()


def test_wait_for_extension(bridge: ChromeBridge) -> None:
    assert bridge.wait_for_extension(0.1) is False
    peer_box: dict[str, ExtensionPeer] = {}
    timer = threading.Timer(
        0.3, lambda: peer_box.setdefault("peer", ExtensionPeer(bridge.port, bridge.token))
    )
    timer.start()
    try:
        assert bridge.wait_for_extension(5.0) is True
        assert bridge.extension_connected
    finally:
        timer.join()
        peer_box["peer"].close()
    assert _eventually(lambda: not bridge.extension_connected)
    assert bridge.wait_for_extension(0.1) is False  # cleared with the connection


def test_start_if_enabled_follows_the_setting() -> None:
    saved = load_config().get("chrome_bridge", False)
    try:
        save_config({"chrome_bridge": False})
        start_if_enabled()
        assert ChromeBridge._instance is None
        save_config({"chrome_bridge": True})
        start_if_enabled()
        started = ChromeBridge._instance
        assert started is not None and started.port != 0
        assert _config()["token"] == started.token
        start_if_enabled()  # idempotent: the running bridge is kept
        assert ChromeBridge._instance is started
        assert ChromeBridge.instance() is started
    finally:
        save_config({"chrome_bridge": saved})
        ChromeBridge.shutdown()
    assert ChromeBridge._instance is None
    assert not (extension_dir() / CONFIG_FILE_NAME).exists()
    ChromeBridge.shutdown()  # a no-op without a bridge


def _eventually(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


# -- use_my_chrome (WebUseTool) --------------------------------------------
# These use the process-wide singleton the tool itself uses, started lazily
# by the first call of each test and shut down by the autouse fixture.


def test_tool_validates_before_connecting() -> None:
    tool = WebUseTool()
    assert "needs tab_id" in tool.use_my_chrome("evaluate", script="1+1")
    assert "needs a url" in tool.use_my_chrome("open_tab")
    assert "needs a url" in tool.use_my_chrome("navigate", tab_id=3)
    assert "needs a script" in tool.use_my_chrome("evaluate", tab_id=3)
    assert "needs tab_id" in tool.use_my_chrome("snapshot")
    assert "needs tab_id" in tool.use_my_chrome("click", uid=3)
    assert "needs a uid" in tool.use_my_chrome("click", tab_id=3)
    assert "needs a uid" in tool.use_my_chrome("type_text", tab_id=3, text="hi")
    assert "unknown action" in tool.use_my_chrome("hack", tab_id=3)


def test_tool_reports_missing_extension() -> None:
    tool = WebUseTool()
    out = tool.use_my_chrome("list_tabs")
    assert out.startswith("Error using my Chrome:")
    assert "chrome://extensions" in out


def test_tool_list_tabs_formats_output() -> None:
    def respond(cmd: dict[str, Any]) -> Any:
        assert cmd["cmd"] == "list_tabs"
        return [
            {
                "id": 1, "title": "Inbox (3)",
                "url": "https://mail.google.com/mail/u/0/", "active": True,
            },
            {"id": 2, "title": "Docs", "url": "https://docs.hunters.ai", "active": False},
        ]

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        out = WebUseTool().use_my_chrome("list_tabs")
        assert "id=1 [active] Inbox (3) — https://mail.google.com/mail/u/0/" in out
        assert "id=2 Docs — https://docs.hunters.ai" in out
    finally:
        peer.close()


def test_tool_error_becomes_readable_string() -> None:
    def respond(cmd: dict[str, Any]) -> Any:
        raise ChromeBridgeError("No tab with id: 5")

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        out = WebUseTool().use_my_chrome("navigate", tab_id=5, url="https://example.com")
        assert out == "Error using my Chrome: No tab with id: 5"
    finally:
        peer.close()


def test_tool_open_tab_confirmation() -> None:
    def respond(cmd: dict[str, Any]) -> Any:
        assert cmd["cmd"] == "open_tab" and cmd["args"] == {"url": "https://example.com"}
        return {"id": 9, "title": "Example", "url": "https://example.com", "active": True}

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        out = WebUseTool().use_my_chrome("open_tab", url="https://example.com")
        assert out == "Done: tab id=9 — Example — https://example.com"
    finally:
        peer.close()


def test_tool_evaluate_formats_json() -> None:
    def respond(cmd: dict[str, Any]) -> Any:
        assert cmd["args"]["script"] == "document.title"
        return {"unread": 3}

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        out = WebUseTool().use_my_chrome("evaluate", tab_id=1, script="document.title")
        assert out == '{\n "unread": 3\n}'
    finally:
        peer.close()


def test_tool_snapshot_passes_text_through() -> None:
    def respond(cmd: dict[str, Any]) -> Any:
        assert cmd["cmd"] == "snapshot" and cmd["args"] == {"tabId": 7}
        return (
            "Hunters - SOC Platform — https://app.hunters.ai/ — 2 interactive elements\n"
            '[1] <button> "Reve.Art"\n[2] <a role=link> "Detectors"'
        )

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        out = WebUseTool().use_my_chrome("snapshot", tab_id=7)
        assert '[1] <button> "Reve.Art"' in out
    finally:
        peer.close()


def test_tool_click_sends_uid() -> None:
    def respond(cmd: dict[str, Any]) -> Any:
        assert cmd["cmd"] == "click" and cmd["args"] == {"tabId": 7, "uid": 12}
        return 'clicked [12] <button> "Save"'

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        out = WebUseTool().use_my_chrome("click", tab_id=7, uid=12)
        assert out == 'clicked [12] <button> "Save"'
    finally:
        peer.close()


def test_tool_type_text_sends_args() -> None:
    """The text reaches the extension as given; an empty one clears the field (A15-8)."""
    def respond(cmd: dict[str, Any]) -> Any:
        assert cmd["cmd"] == "type_text"
        assert cmd["args"] in (
            {"tabId": 7, "uid": 4, "text": "hello detectors"},
            {"tabId": 7, "uid": 4, "text": ""},
        )
        return f"typed {len(cmd['args']['text'])} chars into [4] <input>"

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        out = WebUseTool().use_my_chrome("type_text", tab_id=7, uid=4, text="hello detectors")
        assert out == "typed 15 chars into [4] <input>"
        out = WebUseTool().use_my_chrome("type_text", tab_id=7, uid=4, text="")
        assert out == "typed 0 chars into [4] <input>"
    finally:
        peer.close()


def test_tool_screenshot_saves_png_under_the_work_dir(tmp_path: Path) -> None:
    def respond(cmd: dict[str, Any]) -> Any:
        if cmd["cmd"] == "screenshot":
            return f"data:image/png;base64,{_TINY_PNG_B64}"
        if cmd["cmd"] == "close_tab":
            return {"closed": cmd["args"]["tabId"]}
        raise ChromeBridgeError(f"unexpected command {cmd['cmd']}")

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        # The task's work dir, not the process CWD (the daemon's), like
        # screenshot().
        work_dir = tmp_path / "task"
        work_dir.mkdir()
        tool = WebUseTool(work_dir=str(work_dir))
        out = tool.use_my_chrome("screenshot", tab_id=1)
        assert out.startswith("Screenshot saved to ")
        shots = list((work_dir / "tmp").glob("chrome-shot-*.png"))
        assert len(shots) == 1
        assert str(shots[0].resolve()) in out
        assert base64.b64encode(shots[0].read_bytes()).decode() == _TINY_PNG_B64
        assert tool.use_my_chrome("close_tab", tab_id=1) == "Tab closed."
        # A sub-agent's goes to its private folder, not to the work tree.
        private = tmp_path / "kiss-sub-x"
        private.mkdir()
        sub = WebUseTool(work_dir=str(work_dir), private_dir=str(private))
        assert str(private / "chrome-shot-") in sub.use_my_chrome("screenshot", tab_id=1)
        assert len(list(private.glob("chrome-shot-*.png"))) == 1
        assert len(list((work_dir / "tmp").iterdir())) == 1
        # A malformed payload is an error string, not a crash.
        assert "unexpected screenshot payload" in tool._save_bridge_shot("not-a-data-url")
    finally:
        peer.close()


# -- _format_bridge_result branches ----------------------------------------


def test_format_empty_and_truncated_tab_list() -> None:
    assert _format_bridge_result("list_tabs", []) == "The user's Chrome reports no open tabs."
    many = [
        {"id": i, "title": f"t{i}", "url": f"https://x/{i}", "active": False}
        for i in range(55)
    ]
    out = _format_bridge_result("list_tabs", many)
    assert "id=49" in out and "id=50" not in out
    assert "and 5 more tabs" in out


def test_format_scalars() -> None:
    assert _format_bridge_result("evaluate", None) == "undefined"
    assert _format_bridge_result("evaluate", "plain") == "plain"
    assert _format_bridge_result("evaluate", [1, 2]) == "[\n 1,\n 2\n]"

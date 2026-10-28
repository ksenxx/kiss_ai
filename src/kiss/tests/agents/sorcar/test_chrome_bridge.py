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

import base64
import hashlib
import json
import os
import queue
import re
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from kiss.agents.sorcar import chrome_bridge
from kiss.agents.sorcar.chrome_bridge import (
    CONFIG_FILE_NAME,
    ChromeBridge,
    ChromeBridgeError,
    extension_dir,
    start_if_enabled,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool, _format_bridge_result
from kiss.core.brand import PRODUCT_NAME
from kiss.core.user_config import load_config, save_config
from kiss.tests.conftest import IS_WINDOWS

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _proof(key: str, message: str) -> str:
    return hashlib.sha256(f"{key}:{message}".encode()).hexdigest()


class ExtensionPeer:
    """The extension's side of the bridge protocol, over a real socket.

    Performs the mutual challenge-response handshake (and verifies the
    server's proof, as background.js does).  With ``responder`` set, the
    reader thread answers every command immediately, so the synchronous
    WebUseTool API can be called directly.  The responder returns the
    command's result or raises ChromeBridgeError to send an error reply.
    """

    def __init__(
        self,
        port: int,
        token: str,
        responder: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.ws = connect(f"ws://127.0.0.1:{port}")
        challenge = json.loads(self.ws.recv(timeout=5))
        assert challenge["type"] == "challenge"
        nonce = challenge["nonce"]
        self.ws.send(json.dumps({"type": "hello", "proof": _proof(token, nonce)}))
        hello = json.loads(self.ws.recv(timeout=5))
        assert hello["type"] == "hello_ok"
        assert hello["proof"] == _proof(nonce, token)  # server proves itself too
        self.incoming: queue.Queue[Any] = queue.Queue()
        self._responder = responder
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            while True:
                msg = json.loads(self.ws.recv(timeout=30))
                if self._responder is not None and isinstance(msg.get("id"), int):
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
    (ext / "background.js").parent.mkdir(parents=True, exist_ok=True)
    (ext / "background.js").write_text("// stale copy from an older install")
    b = ChromeBridge(write_config=True)
    b.start()
    try:
        assert b.port != 0  # ephemeral: whatever the kernel handed out
        assert (ext / "background.js").read_bytes() == (bundled / "background.js").read_bytes()
        # The manifest is installed with the brand filled in: Chrome lists
        # the extension under the product's name, never the placeholder
        # or the stock name.
        manifest = json.loads((ext / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["name"] == f"{PRODUCT_NAME} Bridge"
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


def test_bundled_manifest_survives_quotes_and_backslashes_in_the_brand() -> None:
    # A product name with JSON-special characters still yields a valid
    # manifest (the values are escaped into the JSON strings).
    brand = {"product_name": 'ACME "Assistant" C:\\Tools é'}
    manifest = json.loads(chrome_bridge.bundled_extension_file("manifest.json", brand))
    assert manifest["name"] == 'ACME "Assistant" C:\\Tools é Bridge'
    assert manifest["action"]["title"].startswith('ACME "Assistant" C:\\Tools é Bridge')
    assert 'ACME "Assistant" C:\\Tools é agent' in manifest["description"]
    # The other files are copied byte for byte.
    bundled = Path(chrome_bridge.__file__).parent / "chrome_extension"
    assert (
        chrome_bridge.bundled_extension_file("background.js", brand)
        == (bundled / "background.js").read_bytes()
    )


def test_second_process_defers_to_a_live_bridge_and_daemon_takes_over() -> None:
    daemon = ChromeBridge(write_config=True)
    daemon.start(takeover=True)
    try:
        # A CLI run beside the daemon: the extension stays with the daemon.
        cli = ChromeBridge(write_config=True)
        with pytest.raises(ChromeBridgeError, match=f"port {daemon.port}"):
            cli.start()
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
    assert "needs text" in tool.use_my_chrome("type_text", tab_id=3, uid=3)
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
    def respond(cmd: dict[str, Any]) -> Any:
        assert cmd["cmd"] == "type_text"
        assert cmd["args"] == {"tabId": 7, "uid": 4, "text": "hello detectors"}
        return "typed 15 chars into [4] <input>"

    peer = ExtensionPeer(ChromeBridge.instance().port, ChromeBridge.instance().token, respond)
    try:
        out = WebUseTool().use_my_chrome("type_text", tab_id=7, uid=4, text="hello detectors")
        assert out == "typed 15 chars into [4] <input>"
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

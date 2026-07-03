# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing bugs in irc/feishu/zalo/line agents — no mocks.

Bugs covered:
(A) IRC use_tls persisted as str(bool) so "False" is truthy -> TLS on plain sockets.
(B) IRC PRIVMSG text parsed with lstrip(":") corrupting text starting with ":".
(C) IRC substring check "PRIVMSG" in line misclassifies non-PRIVMSG lines.
(D) IRC send_message silently no-ops when not connected (breaks retry contract).
(E/F) Feishu send_message swallows failures / must raise when unconfigured.
(G) Zalo send_message discards error JSON and never raises.
(H) LINE send_message silently returns when unconfigured / swallows errors.
(I) Zalo/LINE poll_messages ignore channel_id when draining the webhook queue.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

import kiss.agents.third_party_agents.zalo_agent as zalo_agent
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.feishu_agent import FeishuChannelBackend
from kiss.agents.third_party_agents.irc_agent import IRCChannelBackend
from kiss.agents.third_party_agents.irc_agent import _config as _irc_config
from kiss.agents.third_party_agents.line_agent import LineChannelBackend
from kiss.agents.third_party_agents.zalo_agent import ZaloChannelBackend


def _backup_config(path: Path) -> str | None:
    if path.exists():
        backup = path.read_text()
        path.unlink()
        return backup
    return None


def _restore_config(path: Path, backup: str | None) -> None:
    if backup is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(backup)
    elif path.exists():
        path.unlink()


class _FakeIRCServer:
    """A real plaintext TCP server speaking just enough IRC for the tests."""

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        self.received: list[str] = []
        self._conn: socket.socket | None = None
        self._connected = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._listener.accept()
        except OSError:
            return
        self._conn = conn
        conn.settimeout(0.5)
        buf = b""
        first = True
        while True:
            try:
                data = conn.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not data:
                break
            if first:
                first = False
                # A TLS ClientHello starts with 0x16; close so a wrongly-TLS
                # client fails its handshake quickly instead of hanging.
                if data[:1] == b"\x16":
                    conn.close()
                    return
                self._connected.set()
            buf += data
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                self.received.append(line.decode("utf-8", errors="replace"))

    def send_line(self, line: str) -> None:
        assert self._connected.wait(timeout=5.0), "client never connected"
        assert self._conn is not None
        self._conn.sendall(line.encode("utf-8") + b"\r\n")

    def wait_received(self, needle: str, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(needle in line for line in self.received):
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        try:
            self._listener.close()
        except OSError:
            pass
        self._thread.join(timeout=5.0)


class TestIRCBugs:
    """IRC bugs (A)-(D) against a real local TCP server."""

    def setup_method(self) -> None:
        self._backup = _backup_config(_irc_config.path)
        self.server = _FakeIRCServer()
        self.backend = IRCChannelBackend()

    def teardown_method(self) -> None:
        self.backend.disconnect()
        self.server.close()
        _restore_config(_irc_config.path, self._backup)

    def test_use_tls_false_connects_plaintext(self) -> None:
        """(A) connect_irc(use_tls=False) must not attempt TLS."""
        result = json.loads(
            self.backend.connect_irc(
                server="127.0.0.1", port=self.server.port, nick="kissbot", use_tls=False
            )
        )
        assert result["ok"] is True, result
        assert self.server.wait_received("NICK kissbot")
        cfg = json.loads(_irc_config.path.read_text())
        assert cfg["use_tls"].lower() in ("", "false")

    def test_legacy_str_false_config_connects_plaintext(self) -> None:
        """(A) legacy configs storing use_tls="False" must connect plaintext."""
        _irc_config.save(
            {
                "server": "127.0.0.1",
                "nick": "kissbot",
                "port": str(self.server.port),
                "password": "",
                "use_tls": "False",
            }
        )
        assert self.backend.connect() is True, self.backend._connection_info
        assert self.server.wait_received("NICK kissbot")

    def test_privmsg_text_and_command_parsing(self) -> None:
        """(B)+(C) text keeps a single leading ':'; non-PRIVMSG lines ignored."""
        result = json.loads(
            self.backend.connect_irc(
                server="127.0.0.1", port=self.server.port, nick="kissbot", use_tls=False
            )
        )
        assert result["ok"] is True, result
        self.server.send_line(":alice!u@h PRIVMSG #ch ::) hello")
        self.server.send_line(":srv NOTICE kissbot :please use PRIVMSG to chat")
        messages: list = []
        deadline = time.time() + 5.0
        while time.time() < deadline and not messages:
            messages, _ = self.backend.poll_messages("", "")
            time.sleep(0.1)
        time.sleep(0.5)
        more, _ = self.backend.poll_messages("", "")
        messages.extend(more)
        assert len(messages) == 1, messages
        assert messages[0]["user"] == "alice"
        assert messages[0]["target"] == "#ch"
        assert messages[0]["text"] == ":) hello"

    def test_send_message_unconnected_raises(self) -> None:
        """(D) send_message must raise when not connected so retries work."""
        backend = IRCChannelBackend()
        with pytest.raises(RuntimeError):
            backend.send_message("#ch", "hi")


class _JSONResponder(BaseHTTPRequestHandler):
    body: bytes = b"{}"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *args) -> None:  # type: ignore[override]
        pass


class TestZaloBugs:
    """Zalo bugs (G) send errors swallowed and (I) poll ignores channel_id."""

    def setup_method(self) -> None:
        self._saved_base = zalo_agent._API_BASE
        self._server: ThreadedHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.backend = ZaloChannelBackend()
        self.backend._access_token = "tok"

    def teardown_method(self) -> None:
        zalo_agent._API_BASE = self._saved_base
        self._server, self._thread = stop_http_server(self._server, self._thread)

    def _start_server(self, body: bytes) -> None:
        handler = type("Handler", (_JSONResponder,), {"body": body})
        self._server = ThreadedHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        port = self._server.server_address[1]
        zalo_agent._API_BASE = f"http://127.0.0.1:{port}/v2.0/oa"

    def test_send_message_raises_on_api_error(self) -> None:
        """(G) send_message must raise when the Zalo API reports an error."""
        self._start_server(b'{"error": 1, "message": "token expired"}')
        with pytest.raises(RuntimeError, match="token expired"):
            self.backend.send_message("U1", "hello")

    def test_send_message_success_does_not_raise(self) -> None:
        """(G) send_message succeeds silently when the API accepts the message."""
        self._start_server(b'{"error": 0, "data": {"message_id": "m1"}}')
        self.backend.send_message("U1", "hello")

    def test_poll_messages_filters_by_channel_id(self) -> None:
        """(I) poll_messages must only return messages from the given user."""
        self.backend._message_queue.put({"ts": "1", "user": "A", "text": "from A"})
        self.backend._message_queue.put({"ts": "2", "user": "B", "text": "from B"})
        messages, _ = self.backend.poll_messages("B", "")
        assert [m["user"] for m in messages] == ["B"]

    def test_poll_messages_empty_channel_returns_all(self) -> None:
        """(I) poll_messages with empty channel_id returns everything."""
        self.backend._message_queue.put({"ts": "1", "user": "A", "text": "from A"})
        self.backend._message_queue.put({"ts": "2", "user": "B", "text": "from B"})
        messages, _ = self.backend.poll_messages("", "")
        assert [m["user"] for m in messages] == ["A", "B"]


class TestLineBugs:
    """LINE bugs (H) silent send failures and (I) poll ignores channel_id."""

    def setup_method(self) -> None:
        self.backend = LineChannelBackend()

    def _put(self, user: str, group_id: str, room_id: str, text: str) -> None:
        self.backend._message_queue.put(
            {
                "ts": "1",
                "user": user,
                "text": text,
                "reply_token": "",
                "group_id": group_id,
                "room_id": room_id,
            }
        )

    def test_send_message_unconfigured_raises(self) -> None:
        """(H) send_message must raise when the LINE API is not configured."""
        with pytest.raises(RuntimeError):
            self.backend.send_message("U1", "hello")

    def test_poll_messages_filters_by_group(self) -> None:
        """(I) poll_messages("G1") returns only messages from group G1."""
        self._put("U1", "", "", "dm")
        self._put("U2", "G1", "", "group msg")
        self._put("U3", "", "R1", "room msg")
        messages, _ = self.backend.poll_messages("G1", "")
        assert [m["text"] for m in messages] == ["group msg"]

    def test_poll_messages_filters_by_user(self) -> None:
        """(I) poll_messages("U1") returns only the DM from user U1."""
        self._put("U1", "", "", "dm")
        self._put("U2", "G1", "", "group msg")
        messages, _ = self.backend.poll_messages("U1", "")
        assert [m["text"] for m in messages] == ["dm"]

    def test_poll_messages_empty_channel_returns_all(self) -> None:
        """(I) poll_messages("") returns all queued messages."""
        self._put("U1", "", "", "dm")
        self._put("U2", "G1", "", "group msg")
        self._put("U3", "", "R1", "room msg")
        messages, _ = self.backend.poll_messages("", "")
        assert [m["text"] for m in messages] == ["dm", "group msg", "room msg"]


class TestFeishuBugs:
    """Feishu bug (E): send_message must not swallow failures."""

    def test_send_message_unconfigured_raises(self) -> None:
        """(E) send_message must raise when no client is configured."""
        backend = FeishuChannelBackend()
        with pytest.raises(RuntimeError):
            backend.send_message("chat1", "hello")

    def test_send_message_errors_propagate(self) -> None:
        """(E) errors from the lark SDK layer must propagate, not be swallowed."""
        backend = FeishuChannelBackend()
        backend._client = object()
        with pytest.raises((ImportError, AttributeError)):
            backend.send_message("chat1", "hello")

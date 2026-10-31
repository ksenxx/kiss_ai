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
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from kiss.agents.third_party_agents.feishu.feishu_sea import FeishuChannelBackend
from kiss.agents.third_party_agents.irc.irc_sea import IRCChannelBackend
from kiss.agents.third_party_agents.irc.irc_sea import _config as _irc_config
from kiss.agents.third_party_agents.line.line_sea import LineChannelBackend
from kiss.agents.third_party_agents.zalo.zalo_sea import ZaloChannelBackend
from kiss.tests.agents.third_party_agents.irc_test_utils import IRCServer, read_lines
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    recording_server,
)


class TestIRCBugs:
    """IRC bugs (A)-(D) against a real loopback TCP listener.

    Nothing answers a TLS ClientHello, so a regression of (A) fails
    ``connect`` only after the backend's 30 s handshake timeout.
    """

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_kiss_home: Path):
        """Clean IRC config, a loopback IRC listener and a backend per test."""
        with closing(IRCServer()) as self.server:
            self.backend = IRCChannelBackend()
            try:
                yield
            finally:
                self.backend.disconnect()

    def test_use_tls_false_connects_plaintext(self) -> None:
        """(A) connect_irc(use_tls=False) must not attempt TLS."""
        result = json.loads(
            self.backend.connect_irc(
                server="127.0.0.1", port=self.server.port, nick="kissbot", use_tls=False
            )
        )
        assert result["ok"] is True, result
        with self.server.accept() as conn:
            assert "NICK kissbot" in read_lines(conn, "NICK")
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
        with self.server.accept() as conn:
            assert "NICK kissbot" in read_lines(conn, "NICK")

    def test_privmsg_text_and_command_parsing(self) -> None:
        """(B)+(C) text keeps a single leading ':'; non-PRIVMSG lines ignored."""
        result = json.loads(
            self.backend.connect_irc(
                server="127.0.0.1", port=self.server.port, nick="kissbot", use_tls=False
            )
        )
        assert result["ok"] is True, result
        with self.server.accept() as conn:
            conn.sendall(b":alice!u@h PRIVMSG #ch ::) hello\r\n")
            conn.sendall(b":srv NOTICE kissbot :please use PRIVMSG to chat\r\n")
            # Sentinel PRIVMSG on the same TCP connection: FIFO ordering guarantees
            # the NOTICE was processed before the sentinel is queued, so once the
            # sentinel arrives we know the NOTICE was (correctly) not enqueued.
            conn.sendall(b":bob!u@h PRIVMSG #ch :sentinel\r\n")
            messages: list = []
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and len(messages) < 2:
                more, _ = self.backend.poll_messages("", "")
                messages.extend(more)
                time.sleep(0.1)
        assert [m["text"] for m in messages] == [":) hello", "sentinel"], messages
        assert messages[0]["user"] == "alice"
        assert messages[0]["target"] == "#ch"
        assert messages[1]["user"] == "bob"


def test_irc_send_message_unconnected_raises() -> None:
    """(D) send_message must raise when not connected so retries work."""
    backend = IRCChannelBackend()
    with pytest.raises(RuntimeError):
        backend.send_message("#ch", "hi")


class TestIRCFreshDaemonTools:
    """A fresh daemon-side agent's IRC tools must really reach the server.

    The ``kiss.server.sorcar.run`` SEA contract builds a FRESH
    ``IRCAgent`` inside the daemon via this module's ``IrcSea().tools([])``.
    That backend starts disconnected, so its messaging tools must
    connect on demand from the persisted config instead of silently
    reporting success while sending nothing.
    """

    @pytest.fixture(autouse=True)
    def _isolated(self, isolated_kiss_home: Path):
        """Clean IRC config and a loopback IRC listener per test."""
        with closing(IRCServer()) as self.server:
            yield

    def test_tools_post_message_connects_on_demand(self) -> None:
        """``IrcSea().tools([])``' fresh backend lazily connects and really sends."""
        from kiss.agents.third_party_agents.irc import irc_sea

        _irc_config.save(
            {
                "server": "127.0.0.1",
                "nick": "kissbot",
                "port": str(self.server.port),
                "password": "",
                "use_tls": "false",
            }
        )
        tools = {t.__name__: t for t in irc_sea.IrcSea().tools([])}
        assert "post_message" in tools, "authenticated backend tools expected"
        backend: IRCChannelBackend = tools["post_message"].__self__
        try:
            result = json.loads(tools["post_message"]("#chan", "from daemon"))
            assert result["ok"] is True, result
            with self.server.accept() as conn:
                assert "PRIVMSG #chan :from daemon" in read_lines(conn, "PRIVMSG"), (
                    "the lazily-connected backend must really send the line"
                )
        finally:
            tools["clear_irc_auth"]()
            backend.disconnect()
        assert backend._reader_thread is None

    def test_fresh_backend_post_message_fails_loudly_without_config(self) -> None:
        """No config: the tool must report an error, never a false ok."""
        backend = IRCChannelBackend()
        backend._nick = "kissbot"
        result = json.loads(backend.post_message("#chan", "hi"))
        assert result["ok"] is False
        assert "Not connected to IRC" in result["error"]


class _JSONResponder(RecordingHandler):
    body: bytes = b"{}"

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._read_body()
        self._reply(200, type(self).body.decode())


def _zalo_backend(server: RecordingServer | None = None) -> ZaloChannelBackend:
    """Return an authenticated Zalo backend, pointed at *server* when given."""
    backend = ZaloChannelBackend(api_base=f"{server.base_url}/v2.0/oa" if server else "")
    backend._access_token = "tok"
    return backend


@contextmanager
def _zalo_api(body: bytes) -> Iterator[ZaloChannelBackend]:
    """Serve *body* to every POST and yield a backend whose API base is that server."""
    handler = type("Handler", (_JSONResponder,), {"body": body})
    with recording_server(handler) as server:
        yield _zalo_backend(server)


class TestZaloBugs:
    """Zalo bugs (G) send errors swallowed and (I) poll ignores channel_id."""

    def test_send_message_raises_on_api_error(self) -> None:
        """(G) send_message must raise when the Zalo API reports an error."""
        with _zalo_api(b'{"error": 1, "message": "token expired"}') as backend:
            with pytest.raises(RuntimeError, match="token expired"):
                backend.send_message("U1", "hello")

    def test_send_message_success_does_not_raise(self) -> None:
        """(G) send_message succeeds silently when the API accepts the message."""
        with _zalo_api(b'{"error": 0, "data": {"message_id": "m1"}}') as backend:
            backend.send_message("U1", "hello")

    def test_poll_messages_filters_by_channel_id(self) -> None:
        """(I) poll_messages must only return messages from the given user."""
        backend = _zalo_backend()
        backend._message_queue.put({"ts": "1", "user": "A", "text": "from A"})
        backend._message_queue.put({"ts": "2", "user": "B", "text": "from B"})
        messages, _ = backend.poll_messages("B", "")
        assert [m["user"] for m in messages] == ["B"]

    def test_poll_messages_empty_channel_returns_all(self) -> None:
        """(I) poll_messages with empty channel_id returns everything."""
        backend = _zalo_backend()
        backend._message_queue.put({"ts": "1", "user": "A", "text": "from A"})
        backend._message_queue.put({"ts": "2", "user": "B", "text": "from B"})
        messages, _ = backend.poll_messages("", "")
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

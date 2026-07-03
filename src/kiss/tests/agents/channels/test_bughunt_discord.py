# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing DiscordChannelBackend bugs — no mocks.

A real in-process HTTP server serves Discord-shaped JSON and records every
request (method, path, query params, body).  The backend is pointed at the
local server via its ``api_base`` constructor argument, so the tests verify
the actual HTTP traffic the backend produces:

1. REST paths must use ``/channels/`` (not ``/third_party_agents/``).
2. ``poll_messages`` must set ``ts`` to the message snowflake id.
3. ``poll_messages(oldest="0")`` must not send ``after=0`` (stale cursor).
4. ``connect()`` must store the bot user id so ``is_from_bot`` works.
5. Message ids must be sorted numerically, not lexicographically.
6. ``delete_message`` must propagate API errors instead of ``ok: true``.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from kiss.agents.third_party_agents.discord_agent import DiscordChannelBackend, _config


class _RecordingServer(ThreadingHTTPServer):
    """HTTP server that records every request it handles."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.requests: list[dict[str, Any]] = []


class _DiscordHandler(BaseHTTPRequestHandler):
    """Serves Discord-shaped JSON responses and records requests."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _respond(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, body: Any = None) -> None:
        parts = urlsplit(self.path)
        cast(_RecordingServer, self.server).requests.append(
            {
                "method": self.command,
                "path": parts.path,
                "params": {k: v[0] for k, v in parse_qs(parts.query).items()},
                "body": body,
            }
        )

    def do_GET(self) -> None:
        self._record()
        path = urlsplit(self.path).path
        if path == "/users/@me":
            self._respond(200, {"id": "BOT1", "username": "bot", "discriminator": "0001"})
        elif path == "/users/@me/guilds":
            self._respond(200, [{"id": "G1", "name": "guild"}])
        elif path == "/guilds/G1/channels":
            self._respond(200, [{"id": "111", "name": "general", "type": 0}])
        elif path == "/channels/111/messages":
            self._respond(
                200,
                [
                    {
                        "id": "10000",
                        "content": "second",
                        "author": {"id": "U1", "username": "alice"},
                        "timestamp": "2024-05-01T12:00:01+00:00",
                    },
                    {
                        "id": "9999",
                        "content": "first",
                        "author": {"id": "U1", "username": "alice"},
                        "timestamp": "2024-05-01T12:00:00+00:00",
                    },
                ],
            )
        else:
            self._respond(404, {"message": "404: Not Found", "code": 0})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw else None
        self._record(body)
        path = urlsplit(self.path).path
        if path.startswith("/channels/") and path.endswith("/messages"):
            self._respond(200, {"id": "M1", "channel_id": path.split("/")[2]})
        else:
            self._respond(404, {"message": "404: Not Found", "code": 0})

    def do_DELETE(self) -> None:
        self._record()
        self._respond(403, {"message": "Missing Permissions", "code": 50013})


class TestDiscordBackendBugs:
    """End-to-end tests against a local Discord-shaped HTTP server."""

    server: _RecordingServer
    api_base: str

    @classmethod
    def setup_class(cls) -> None:
        cls.server = _RecordingServer(("127.0.0.1", 0), _DiscordHandler)
        thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        thread.start()
        cls.api_base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def teardown_class(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setup_method(self) -> None:
        self.server.requests.clear()
        self._backup = _config.path.read_text() if _config.path.exists() else None
        _config.save({"bot_token": "test-token"})
        self.backend = DiscordChannelBackend(api_base=self.api_base)
        self.backend._bot_token = "test-token"

    def teardown_method(self) -> None:
        if self._backup is not None:
            _config.path.write_text(self._backup)
        elif _config.path.exists():
            _config.path.unlink()

    def _paths(self) -> list[str]:
        return [r["path"] for r in self.server.requests]

    def test_connect_stores_bot_id_for_is_from_bot(self) -> None:
        """connect() must remember the bot's own user id so is_from_bot works."""
        assert self.backend.connect() is True
        assert self.backend.is_from_bot({"user": "BOT1"}) is True
        assert self.backend.is_from_bot({"user": "U1"}) is False

    def test_find_channel_uses_channels_path(self) -> None:
        """find_channel must query /guilds/{id}/channels, not /third_party_agents."""
        assert self.backend.find_channel("general") == "111"
        assert "/guilds/G1/channels" in self._paths()
        assert not any("third_party_agents" in p for p in self._paths())

    def test_poll_messages_uses_channels_path(self) -> None:
        """poll_messages must GET /channels/{id}/messages."""
        self.backend.poll_messages("111", "0", limit=50)
        assert self._paths() == ["/channels/111/messages"]

    def test_poll_messages_treats_zero_as_no_cursor(self) -> None:
        """oldest='0' must not be forwarded as after=0 (oldest-history fetch)."""
        self.backend.poll_messages("111", "0", limit=50)
        params = self.server.requests[0]["params"]
        assert params.get("after") != "0"

    def test_poll_messages_forwards_real_cursor(self) -> None:
        """A genuine snowflake cursor must still be sent as the after param."""
        self.backend.poll_messages("111", "12345", limit=50)
        assert self.server.requests[0]["params"]["after"] == "12345"

    def test_poll_messages_ts_is_snowflake_id(self) -> None:
        """Each message's ts must be its snowflake id (usable as reply target)."""
        msgs, _ = self.backend.poll_messages("111", "0", limit=50)
        assert [m["ts"] for m in msgs] == ["9999", "10000"]

    def test_poll_messages_sorts_snowflakes_numerically(self) -> None:
        """Cursor must be the numerically largest id ('10000' > '9999')."""
        msgs, cursor = self.backend.poll_messages("111", "0", limit=50)
        assert [m["id"] for m in msgs] == ["9999", "10000"]
        assert cursor == "10000"

    def test_send_message_posts_to_channel_with_reply_reference(self) -> None:
        """send_message must POST to /channels/{channel}/messages with the
        thread_ts snowflake as a reply reference, not use it as channel id."""
        self.backend.send_message("111", "hello", thread_ts="9999")
        req = self.server.requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/channels/111/messages"
        assert req["body"]["content"] == "hello"
        assert req["body"]["message_reference"] == {"message_id": "9999"}

    def test_send_message_without_thread_posts_to_channel(self) -> None:
        """send_message without thread_ts posts a plain channel message."""
        self.backend.send_message("111", "hi")
        req = self.server.requests[0]
        assert req["path"] == "/channels/111/messages"
        assert req["body"] == {"content": "hi"}

    def test_delete_message_propagates_error(self) -> None:
        """delete_message must report the API error, not ok: true."""
        result = json.loads(self.backend.delete_message("111", "M9"))
        assert result["ok"] is False
        assert "Missing Permissions" in result["error"]
        assert self.server.requests[0]["path"] == "/channels/111/messages/M9"

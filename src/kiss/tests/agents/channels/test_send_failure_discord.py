# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Discord ``send_message`` failure propagation — no mocks.

The shared ChannelRunner records each outbound reply in a delivery ledger
before sending and removes it only when ``send_message`` returns without
raising. These tests verify against a real in-process HTTP server (pointed
at via the ``api_base`` constructor argument, following
``test_typing_discord.py``) that:

1. ``send_message`` raises on a non-2xx API response (500) so the ledger
   keeps the entry for retry.
2. ``send_message`` raises when the server is unreachable.
3. ``send_message`` succeeds without raising on a 200 response with a
   normal JSON body.
4. Tool-path callers of ``_post`` (e.g. ``post_message``) keep their
   return-an-error-string contract on API errors instead of raising.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import urlsplit

import pytest
import requests

from kiss.agents.third_party_agents.discord_agent import DiscordChannelBackend


class _RecordingServer(ThreadingHTTPServer):
    """HTTP server that records every request it handles."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.requests: list[dict[str, Any]] = []


class _SendHandler(BaseHTTPRequestHandler):
    """Records requests; 200 for channel OK, 500 for channel ERR."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _respond(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        path = urlsplit(self.path).path
        cast(_RecordingServer, self.server).requests.append(
            {
                "method": self.command,
                "path": path,
                "body": json.loads(raw) if raw else None,
                "authorization": self.headers.get("Authorization", ""),
            }
        )
        if path == "/channels/ERR/messages":
            self._respond(500, {"message": "Internal Server Error", "code": 0})
        else:
            self._respond(200, {"id": "M1", "channel_id": path.split("/")[2]})


class TestDiscordSendMessageFailures:
    """End-to-end tests against a local Discord-shaped HTTP server."""

    server: _RecordingServer
    api_base: str

    @classmethod
    def setup_class(cls) -> None:
        cls.server = _RecordingServer(("127.0.0.1", 0), _SendHandler)
        thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        thread.start()
        cls.api_base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def teardown_class(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setup_method(self) -> None:
        self.server.requests.clear()
        self.backend = DiscordChannelBackend(api_base=self.api_base)
        self.backend._bot_token = "test-token"

    def test_send_message_raises_on_http_500(self) -> None:
        """A 500 API response must raise so the ledger can retry the send."""
        with pytest.raises(requests.HTTPError):
            self.backend.send_message("ERR", "hello")
        req = self.server.requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/channels/ERR/messages"
        assert req["body"] == {"content": "hello"}

    def test_send_message_raises_on_unreachable_server(self) -> None:
        """An unreachable server (connection refused) must raise."""
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
        probe.close()
        backend = DiscordChannelBackend(api_base=f"http://127.0.0.1:{closed_port}")
        backend._bot_token = "test-token"
        with pytest.raises(requests.RequestException):
            backend.send_message("111", "hello")
        assert self.server.requests == []

    def test_send_message_succeeds_on_200(self) -> None:
        """A 200 response with a normal JSON body must not raise."""
        self.backend.send_message("111", "hello", thread_ts="9999")
        req = self.server.requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/channels/111/messages"
        assert req["authorization"] == "Bot test-token"
        assert req["body"]["content"] == "hello"
        assert req["body"]["message_reference"] == {"message_id": "9999"}

    def test_post_message_tool_returns_error_string_on_500(self) -> None:
        """The post_message tool must keep returning an error string, not raise."""
        result = json.loads(self.backend.post_message("ERR", "hello"))
        assert result["ok"] is False
        assert "Internal Server Error" in result["error"]

    def test_post_message_tool_succeeds_on_200(self) -> None:
        """The post_message tool must still report ok with the message id."""
        result = json.loads(self.backend.post_message("111", "hello"))
        assert result == {"ok": True, "id": "M1"}

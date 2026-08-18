# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``DiscordChannelBackend.send_typing`` — no mocks.

A real in-process HTTP server records every request (method, path, and
Authorization header) and serves Discord-shaped responses. The backend is
pointed at the local server via its ``api_base`` constructor argument, so
the tests verify the actual HTTP traffic the typing indicator produces:

1. ``send_typing`` must POST ``/channels/{channel_id}/typing`` with the
   bot Authorization header.
2. A non-empty ``thread_ts`` must not change the target channel, mirroring
   ``send_message`` which treats thread ids as reply references inside the
   channel rather than as separate channels.
3. Errors are best-effort: a 500 response or an unreachable server must
   never raise.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import urlsplit

from kiss.agents.third_party_agents.discord_agent import DiscordChannelBackend


class _RecordingServer(ThreadingHTTPServer):
    """HTTP server that records every request it handles."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.requests: list[dict[str, Any]] = []


class _TypingHandler(BaseHTTPRequestHandler):
    """Records requests; 204 for typing endpoints, 500 for channel ERR."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        cast(_RecordingServer, self.server).requests.append(
            {
                "method": self.command,
                "path": path,
                "authorization": self.headers.get("Authorization", ""),
            }
        )
        if path == "/channels/ERR/typing":
            body = json.dumps({"message": "Internal Server Error", "code": 0}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()


class TestDiscordSendTyping:
    """End-to-end tests against a local Discord-shaped HTTP server."""

    server: _RecordingServer
    api_base: str

    @classmethod
    def setup_class(cls) -> None:
        cls.server = _RecordingServer(("127.0.0.1", 0), _TypingHandler)
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

    def test_send_typing_posts_typing_endpoint_with_auth(self) -> None:
        """send_typing must POST /channels/{id}/typing with the bot header."""
        self.backend.send_typing("111")
        assert len(self.server.requests) == 1
        req = self.server.requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/channels/111/typing"
        assert req["authorization"] == "Bot test-token"

    def test_send_typing_with_thread_ts_targets_channel(self) -> None:
        """A reply-target message id must not change the typing channel."""
        self.backend.send_typing("111", thread_ts="9999")
        assert [r["path"] for r in self.server.requests] == ["/channels/111/typing"]

    def test_send_typing_swallows_http_500(self) -> None:
        """A 500 API response must be swallowed, never raised."""
        self.backend.send_typing("ERR")
        req = self.server.requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/channels/ERR/typing"

    def test_send_typing_swallows_unreachable_server(self) -> None:
        """An unreachable server (connection refused) must never raise."""
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
        probe.close()
        backend = DiscordChannelBackend(api_base=f"http://127.0.0.1:{closed_port}")
        backend._bot_token = "test-token"
        backend.send_typing("111")
        assert self.server.requests == []

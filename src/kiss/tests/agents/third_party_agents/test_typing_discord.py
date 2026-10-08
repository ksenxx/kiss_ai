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
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import urlsplit

import pytest

from kiss.agents.third_party_agents.discord.discord_sea import DiscordChannelBackend
from kiss.tests.agents.third_party_agents.recording_http import RecordingServer, serve_recording


class _TypingHandler(BaseHTTPRequestHandler):
    """Records requests; 204 for typing endpoints, 500 for channel ERR."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        cast(RecordingServer, self.server).requests.append(
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


@pytest.fixture(scope="module")
def server() -> Iterator[RecordingServer]:
    """The Discord-shaped recording server, shared by the module."""
    yield from serve_recording(_TypingHandler)


@pytest.fixture()
def backend(server: RecordingServer) -> DiscordChannelBackend:
    """A backend pointed at *server*; the request log starts empty."""
    server.requests.clear()
    backend = DiscordChannelBackend(api_base=f"http://127.0.0.1:{server.server_address[1]}")
    backend._token = "test-token"
    return backend


def test_send_typing_posts_typing_endpoint_with_auth(
    server: RecordingServer, backend: DiscordChannelBackend
) -> None:
    """send_typing must POST /channels/{id}/typing with the bot header."""
    backend.send_typing("111")
    assert len(server.requests) == 1
    req = server.requests[0]
    assert req["method"] == "POST"
    assert req["path"] == "/channels/111/typing"
    assert req["authorization"] == "Bot test-token"


def test_send_typing_with_thread_ts_targets_channel(
    server: RecordingServer, backend: DiscordChannelBackend
) -> None:
    """A reply-target message id must not change the typing channel."""
    backend.send_typing("111", thread_ts="9999")
    assert [r["path"] for r in server.requests] == ["/channels/111/typing"]


def test_send_typing_swallows_http_500(
    server: RecordingServer, backend: DiscordChannelBackend
) -> None:
    """A 500 API response must be swallowed, never raised."""
    backend.send_typing("ERR")
    req = server.requests[0]
    assert req["method"] == "POST"
    assert req["path"] == "/channels/ERR/typing"


def test_send_typing_swallows_unreachable_server(
    server: RecordingServer, backend: DiscordChannelBackend, refusing_port: int
) -> None:
    """An unreachable server (connection refused) must never raise."""
    backend = DiscordChannelBackend(api_base=f"http://127.0.0.1:{refusing_port}")
    backend._token = "test-token"
    backend.send_typing("111")
    assert server.requests == []

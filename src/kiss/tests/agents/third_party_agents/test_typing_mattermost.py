# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Mattermost typing indicator.

Uses a real in-process HTTP server on an ephemeral port that records
every request's method, path, body, and headers, so the tests verify
actual wire behavior with no mocks or test doubles.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, cast

import pytest

from kiss.agents.third_party_agents.mattermost.mattermost_sea import MattermostChannelBackend
from kiss.tests.agents.third_party_agents.recording_http import RecordingServer, serve_recording


class _MattermostServer(RecordingServer):
    """Recording server whose POST reply status the test can set."""

    response_status = 200


class _RecordingHandler(BaseHTTPRequestHandler):
    """Records request method/path/body/headers and replies with a set status."""

    def do_POST(self) -> None:
        """Record the POST request and respond with the configured status."""
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode() if length else ""
        server = cast(_MattermostServer, self.server)
        server.requests.append(
            {"method": "POST", "path": self.path, "body": body, "headers": dict(self.headers)}
        )
        status = server.response_status
        payload = json.dumps({"status": "OK" if status == 200 else "error"}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence request logging."""


@pytest.fixture()
def mm_server():
    """Start a recording HTTP server that mimics the Mattermost REST API."""
    yield from serve_recording(_RecordingHandler, _MattermostServer)


def _make_backend(server: RecordingServer) -> MattermostChannelBackend:
    """Create a Mattermost backend pointed at the local recording server."""
    return MattermostChannelBackend(base_url=server.base_url, token="test-token")


def test_send_typing_posts_typing_endpoint(mm_server) -> None:
    """send_typing must POST /api/v4/users/me/typing with channel_id and bearer auth."""
    backend = _make_backend(mm_server)
    backend.send_typing("chan1")
    assert len(mm_server.requests) == 1
    request = mm_server.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/api/v4/users/me/typing"
    assert json.loads(request["body"]) == {"channel_id": "chan1"}
    assert request["headers"].get("Authorization") == "Bearer test-token"
    assert request["headers"].get("Content-Type") == "application/json"


def test_send_typing_includes_parent_id_for_thread(mm_server) -> None:
    """A non-empty thread_ts must be sent as parent_id alongside channel_id."""
    backend = _make_backend(mm_server)
    backend.send_typing("chan1", thread_ts="root42")
    request = mm_server.requests[0]
    assert request["path"] == "/api/v4/users/me/typing"
    assert json.loads(request["body"]) == {"channel_id": "chan1", "parent_id": "root42"}


def test_send_typing_swallows_http_error(mm_server) -> None:
    """A 500 response from the server must not raise."""
    mm_server.response_status = 500
    backend = _make_backend(mm_server)
    backend.send_typing("chan1", thread_ts="root42")
    assert len(mm_server.requests) == 1


def test_send_typing_swallows_unreachable_server(refusing_port: int) -> None:
    """An unreachable server (closed port) must not raise."""
    backend = MattermostChannelBackend(
        base_url=f"http://127.0.0.1:{refusing_port}", token="test-token"
    )
    backend.send_typing("chan1")


def test_send_typing_without_base_url_is_noop(mm_server) -> None:
    """Without a configured base URL, send_typing must not raise or send anything."""
    backend = MattermostChannelBackend()
    backend.send_typing("chan1")
    assert mm_server.requests == []


def test_send_typing_without_channel_id_is_noop(mm_server) -> None:
    """An empty channel_id must not produce any HTTP request."""
    backend = _make_backend(mm_server)
    backend.send_typing("")
    assert mm_server.requests == []


def test_send_typing_strips_trailing_slash_in_base_url(mm_server) -> None:
    """A trailing slash in base_url must not produce a double slash in the path."""
    backend = MattermostChannelBackend(base_url=f"{mm_server.base_url}/", token="test-token")
    backend.send_typing("chan1")
    assert mm_server.requests[0]["path"] == "/api/v4/users/me/typing"

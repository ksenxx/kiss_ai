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
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.agents.third_party_agents.mattermost_agent import MattermostChannelBackend


class _RecordingHandler(BaseHTTPRequestHandler):
    """Records request method/path/body/headers and replies with a set status."""

    def do_POST(self) -> None:
        """Record the POST request and respond with the configured status."""
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode() if length else ""
        self.server.requests.append(  # type: ignore[attr-defined]
            ("POST", self.path, body, dict(self.headers))
        )
        status = self.server.response_status  # type: ignore[attr-defined]
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
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.response_status = 200  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _make_backend(server: ThreadingHTTPServer) -> MattermostChannelBackend:
    """Create a Mattermost backend pointed at the local recording server."""
    port = server.server_address[1]
    return MattermostChannelBackend(
        base_url=f"http://127.0.0.1:{port}", token="test-token"
    )


def test_send_typing_posts_typing_endpoint(mm_server) -> None:
    """send_typing must POST /api/v4/users/me/typing with channel_id and bearer auth."""
    backend = _make_backend(mm_server)
    backend.send_typing("chan1")
    requests_seen = mm_server.requests
    assert len(requests_seen) == 1
    method, path, body, headers = requests_seen[0]
    assert method == "POST"
    assert path == "/api/v4/users/me/typing"
    assert json.loads(body) == {"channel_id": "chan1"}
    assert headers.get("Authorization") == "Bearer test-token"
    assert headers.get("Content-Type") == "application/json"


def test_send_typing_includes_parent_id_for_thread(mm_server) -> None:
    """A non-empty thread_ts must be sent as parent_id alongside channel_id."""
    backend = _make_backend(mm_server)
    backend.send_typing("chan1", thread_ts="root42")
    _, path, body, _ = mm_server.requests[0]
    assert path == "/api/v4/users/me/typing"
    assert json.loads(body) == {"channel_id": "chan1", "parent_id": "root42"}


def test_send_typing_swallows_http_error(mm_server) -> None:
    """A 500 response from the server must not raise."""
    mm_server.response_status = 500
    backend = _make_backend(mm_server)
    backend.send_typing("chan1", thread_ts="root42")
    assert len(mm_server.requests) == 1


def test_send_typing_swallows_unreachable_server() -> None:
    """An unreachable server (closed port) must not raise."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    backend = MattermostChannelBackend(
        base_url=f"http://127.0.0.1:{port}", token="test-token"
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
    port = mm_server.server_address[1]
    backend = MattermostChannelBackend(
        base_url=f"http://127.0.0.1:{port}/", token="test-token"
    )
    backend.send_typing("chan1")
    _, path, _, _ = mm_server.requests[0]
    assert path == "/api/v4/users/me/typing"

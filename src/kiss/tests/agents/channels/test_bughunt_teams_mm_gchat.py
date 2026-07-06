# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing bugs in msteams/mattermost/googlechat backends.

Uses a real in-process HTTP server that returns Microsoft-Graph-shaped JSON and
records every request path, so the tests verify actual wire behavior with no
mocks or test doubles.
"""

from __future__ import annotations

import importlib.util
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.agents.third_party_agents.msteams_agent import MSTeamsChannelBackend

_GRAPH_MESSAGE = {
    "id": "MSG1",
    "lastModifiedDateTime": "2024-05-01T00:00:00Z",
    "createdDateTime": "2024-05-01T00:00:00Z",
    "from": {"user": {"id": "U1", "displayName": "Alice"}},
    "body": {"content": "hello", "contentType": "html"},
}


class _GraphHandler(BaseHTTPRequestHandler):
    """Records request paths and returns Graph-shaped JSON responses."""

    def _respond(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        """Serve a recorded GET with a Graph list payload."""
        self.server.requests.append(("GET", self.path))  # type: ignore[attr-defined]
        self._respond({"value": [dict(_GRAPH_MESSAGE)]})

    def do_POST(self) -> None:
        """Serve a recorded POST with a Graph create payload."""
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode() if length else ""
        self.server.requests.append(("POST", self.path, body))  # type: ignore[attr-defined]
        self._respond({"id": "NEW1"})

    def log_message(self, format: str, *args: Any) -> None:
        """Silence request logging."""


@pytest.fixture()
def graph_server():
    """Start a recording HTTP server that mimics Microsoft Graph."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GraphHandler)
    server.requests = []  # type: ignore[attr-defined]
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _make_backend(server: ThreadingHTTPServer) -> MSTeamsChannelBackend:
    """Create an MS Teams backend pointed at the local Graph server."""
    port = server.server_address[1]
    backend = MSTeamsChannelBackend(graph_base=f"http://127.0.0.1:{port}")
    backend._access_token = "test-token"
    backend._token_expiry = time.time() + 3600
    return backend


def _paths(server: ThreadingHTTPServer) -> list[str]:
    return [req[1] for req in server.requests]  # type: ignore[attr-defined]


def test_msteams_poll_uses_channels_url_and_message_ids(graph_server) -> None:
    """poll_messages must hit /channels/ and emit message ids as ts/thread_ts."""
    backend = _make_backend(graph_server)
    msgs, cursor = backend.poll_messages("team1:chan1", "0", limit=5)
    path = _paths(graph_server)[0]
    assert "/teams/team1/channels/chan1/messages" in path
    assert "third_party_agents" not in path
    assert "filter" not in path  # oldest sentinel "0" must not produce a $filter
    assert len(msgs) == 1
    assert msgs[0]["ts"] == "MSG1"
    assert msgs[0]["thread_ts"] == "MSG1"
    assert msgs[0]["user"] == "U1"
    assert cursor == "2024-05-01T00:00:00Z"


def test_msteams_poll_applies_filter_for_real_cursor(graph_server) -> None:
    """A genuine datetime cursor still produces a $filter query."""
    backend = _make_backend(graph_server)
    backend.poll_messages("team1:chan1", "2024-01-01T00:00:00Z", limit=5)
    path = _paths(graph_server)[0]
    assert "filter" in path
    assert "2024-01-01" in path


def test_msteams_send_message_reply_targets_message_id(graph_server) -> None:
    """Threaded replies must POST to /channels/<chan>/messages/<msg-id>/replies."""
    backend = _make_backend(graph_server)
    backend.send_message("team1:chan1", "hi there", thread_ts="MSG1")
    path = _paths(graph_server)[0]
    assert "/teams/team1/channels/chan1/messages/MSG1/replies" in path
    assert "third_party_agents" not in path


def test_msteams_send_message_without_thread(graph_server) -> None:
    """Non-threaded sends must POST to /channels/<chan>/messages."""
    backend = _make_backend(graph_server)
    backend.send_message("team1:chan1", "hi there")
    path = _paths(graph_server)[0]
    assert path.endswith("/teams/team1/channels/chan1/messages")


def test_msteams_tool_urls_use_channels(graph_server) -> None:
    """All channel tool methods must use the Graph 'channels' resource."""
    backend = _make_backend(graph_server)
    assert json.loads(backend.list_third_party_agents("team1"))["ok"]
    assert json.loads(backend.list_channel_messages("team1", "chan1"))["ok"]
    assert json.loads(backend.post_channel_message("team1", "chan1", "x"))["ok"]
    assert json.loads(backend.reply_to_message("team1", "chan1", "MSG1", "y"))["ok"]
    paths = _paths(graph_server)
    assert any(p.split("?")[0].endswith("/teams/team1/channels") for p in paths)
    assert any("/teams/team1/channels/chan1/messages" in p for p in paths)
    assert any("/teams/team1/channels/chan1/messages/MSG1/replies" in p for p in paths)
    assert all("third_party_agents" not in p for p in paths)


def test_msteams_poll_channel_runner_contract(graph_server) -> None:
    """ChannelRunner threads replies via thread_ts; verify the round trip works."""
    backend = _make_backend(graph_server)
    msgs, _ = backend.poll_messages("team1:chan1", "0", limit=50)
    msg = msgs[0]
    thread_ts = msg.get("thread_ts", msg.get("ts", ""))
    backend.send_message("team1:chan1", "reply", thread_ts)
    reply_path = _paths(graph_server)[-1]
    assert "/messages/MSG1/replies" in reply_path
    assert "2024-05-01" not in reply_path  # never use a datetime as message id


@pytest.mark.skipif(
    importlib.util.find_spec("mattermostdriver") is None,
    reason="mattermostdriver not installed",
)
def test_mattermost_driver_channels_endpoint_exists() -> None:
    """The fixed code calls driver.channels.*; verify those endpoints exist."""
    driver_cls = importlib.import_module("mattermostdriver").Driver

    driver = driver_cls({"url": "localhost", "token": "x", "port": 8065, "scheme": "http"})
    assert hasattr(driver, "channels")
    assert hasattr(driver.channels, "get_channels_for_user")
    assert hasattr(driver.channels, "get_channel")
    assert hasattr(driver.channels, "create_direct_message_channel")
    assert not hasattr(driver, "third_party_agents")

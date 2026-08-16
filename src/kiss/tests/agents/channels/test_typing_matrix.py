# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Matrix backend's Hermes-style typing indicator.

``MatrixChannelBackend.send_typing`` must PUT
``/_matrix/client/v3/rooms/{roomId}/typing/{userId}`` with body
``{"typing": true, "timeout": 15000}`` against the backend's homeserver,
authenticated with the stored access token, and must be best-effort:
server errors and unreachable servers are swallowed, and a missing user
id (or missing client) is a silent no-op.

No mock/patch libraries are used: every HTTP assertion runs against a
REAL ``http.server`` on an ephemeral port that records the request's
method, path, headers, and body. The backend is pointed at that server
via the same credential data contract (``homeserver`` / ``access_token``
/ ``user_id``) that nio's ``AsyncClient`` exposes at runtime.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kiss.agents.third_party_agents.matrix_agent import MatrixChannelBackend


class _MatrixClientCredentials:
    """Real credential holder with nio ``AsyncClient``'s data contract.

    ``send_typing`` reads exactly three attributes off the backend's
    stored client — ``homeserver``, ``access_token``, and ``user_id`` —
    which this class carries as plain values (matrix-nio is an optional
    dependency and is not installed in the test environment).
    """

    def __init__(self, homeserver: str, access_token: str, user_id: str) -> None:
        self.homeserver = homeserver
        self.access_token = access_token
        self.user_id = user_id


class _RecordingHandler(BaseHTTPRequestHandler):
    """HTTP handler that records every request and replies with a fixed status."""

    def _record_and_reply(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        server: Any = self.server
        server.recorded.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "content_type": self.headers.get("Content-Type", ""),
                "body": body,
            }
        )
        self.send_response(server.reply_status)
        payload = b"{}"
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_PUT(self) -> None:
        """Record a PUT request and reply."""
        self._record_and_reply()

    def do_GET(self) -> None:
        """Record a GET request and reply."""
        self._record_and_reply()

    def do_POST(self) -> None:
        """Record a POST request and reply."""
        self._record_and_reply()

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request logging."""


class _RecordingServer:
    """Context manager running a real recording HTTP server on an ephemeral port."""

    def __init__(self, reply_status: int = 200) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
        self._httpd.recorded = []  # type: ignore[attr-defined]
        self._httpd.reply_status = reply_status  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> _RecordingServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=10)

    @property
    def url(self) -> str:
        """Base URL of the running server."""
        port = self._httpd.server_address[1]
        return f"http://127.0.0.1:{port}"

    @property
    def recorded(self) -> list[dict[str, Any]]:
        """All requests recorded so far."""
        requests: list[dict[str, Any]] = self._httpd.recorded  # type: ignore[attr-defined]
        return requests


def _backend_for(server_url: str, user_id: str = "@bot:example.org") -> MatrixChannelBackend:
    """Build a Matrix backend whose client points at *server_url*."""
    backend = MatrixChannelBackend()
    backend._client = _MatrixClientCredentials(
        homeserver=server_url,
        access_token="syt_secret_token",
        user_id=user_id,
    )
    return backend


class TestSendTypingHappyPath:
    """send_typing must PUT the spec path with the spec body."""

    def test_put_arrives_with_typing_true(self) -> None:
        """A real PUT with {"typing": true, "timeout": 15000} reaches the server."""
        with _RecordingServer(reply_status=200) as server:
            backend = _backend_for(server.url)
            backend.send_typing("!room123:example.org")
            assert len(server.recorded) == 1
            req = server.recorded[0]
            assert req["method"] == "PUT"
            expected_path = (
                "/_matrix/client/v3/rooms/"
                + urllib.parse.quote("!room123:example.org", safe="")
                + "/typing/"
                + urllib.parse.quote("@bot:example.org", safe="")
            )
            assert req["path"] == expected_path
            assert json.loads(req["body"]) == {"typing": True, "timeout": 15000}

    def test_room_and_user_ids_are_url_encoded(self) -> None:
        """Room and user ids with reserved characters are fully percent-encoded."""
        with _RecordingServer(reply_status=200) as server:
            backend = _backend_for(server.url, user_id="@we ird/user:example.org")
            backend.send_typing("!ro om/1:example.org")
            assert len(server.recorded) == 1
            path = server.recorded[0]["path"]
            assert "/rooms/%21ro%20om%2F1%3Aexample.org/typing/" in path
            assert path.endswith("/%40we%20ird%2Fuser%3Aexample.org")
            assert " " not in path and "!" not in path and "@" not in path

    def test_bearer_token_and_content_type_sent(self) -> None:
        """The stored access token is sent as a Bearer Authorization header."""
        with _RecordingServer(reply_status=200) as server:
            backend = _backend_for(server.url)
            backend.send_typing("!room123:example.org")
            req = server.recorded[0]
            assert req["authorization"] == "Bearer syt_secret_token"
            assert req["content_type"] == "application/json"

    def test_trailing_slash_homeserver_normalized(self) -> None:
        """A homeserver URL with a trailing slash yields no double slash in the path."""
        with _RecordingServer(reply_status=200) as server:
            backend = _backend_for(server.url + "/")
            backend.send_typing("!room123:example.org")
            assert len(server.recorded) == 1
            assert server.recorded[0]["path"].startswith("/_matrix/client/v3/rooms/")


class TestSendTypingBestEffort:
    """send_typing must never raise, whatever the transport does."""

    def test_server_error_500_is_swallowed(self) -> None:
        """A 500 reply from the homeserver does not raise."""
        with _RecordingServer(reply_status=500) as server:
            backend = _backend_for(server.url)
            backend.send_typing("!room123:example.org")
            assert len(server.recorded) == 1
            assert server.recorded[0]["method"] == "PUT"

    def test_unreachable_server_is_swallowed(self) -> None:
        """A connection-refused homeserver does not raise."""
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
        probe.close()
        backend = _backend_for(f"http://127.0.0.1:{dead_port}")
        backend.send_typing("!room123:example.org")

    def test_missing_user_id_is_silent_noop(self) -> None:
        """No stored user id: return without raising and without any request."""
        with _RecordingServer(reply_status=200) as server:
            backend = _backend_for(server.url, user_id="")
            backend.send_typing("!room123:example.org")
            assert server.recorded == []

    def test_no_client_is_silent_noop(self) -> None:
        """No client at all: return without raising."""
        backend = MatrixChannelBackend()
        assert backend._client is None
        backend.send_typing("!room123:example.org")

    def test_thread_ts_argument_is_ignored(self) -> None:
        """The parity thread_ts argument does not alter the request."""
        with _RecordingServer(reply_status=200) as server:
            backend = _backend_for(server.url)
            backend.send_typing("!room123:example.org", thread_ts="1234.5678")
            assert len(server.recorded) == 1
            assert json.loads(server.recorded[0]["body"]) == {"typing": True, "timeout": 15000}

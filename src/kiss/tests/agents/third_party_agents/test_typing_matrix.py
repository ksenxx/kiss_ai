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
through a real ``nio.AsyncClient`` carrying the access token.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager

from nio import AsyncClient

from kiss.agents.third_party_agents.matrix.matrix_sea import MatrixChannelBackend
from kiss.tests.agents.third_party_agents.recording_http import (
    JsonWebhookHandler,
    JsonWebhookServer,
    recording_server,
)


@contextmanager
def _matrix_server(reply_status: int = 200) -> Iterator[JsonWebhookServer]:
    """Run a recording homeserver stand-in that answers every request with *reply_status*."""
    with recording_server(JsonWebhookHandler, JsonWebhookServer) as server:
        server.response_status = reply_status
        server.response_body = {}
        yield server


@contextmanager
def _backend_for(
    server_url: str, user_id: str = "@bot:example.org"
) -> Iterator[MatrixChannelBackend]:
    """Yield a Matrix backend whose client points at *server_url*; disconnect on exit.

    ``user_id`` is set the way ``_client_from_config`` sets it from the
    stored session: nio fills it from a login response, which never
    happens here.
    """
    backend = MatrixChannelBackend()
    client = AsyncClient(server_url, user_id)
    client.access_token = "syt_secret_token"
    client.user_id = user_id
    backend._client = client
    try:
        yield backend
    finally:
        backend.disconnect()


class TestSendTypingHappyPath:
    """send_typing must PUT the spec path with the spec body."""

    def test_put_arrives_with_typing_true(self) -> None:
        """One authenticated JSON PUT with {"typing": true, "timeout": 15000} reaches the server.

        The parity ``thread_ts`` argument does not alter the request.
        """
        with _matrix_server() as server, _backend_for(server.base_url) as backend:
            backend.send_typing("!room123:example.org", thread_ts="1234.5678")
            assert len(server.requests) == 1
            req = server.requests[0]
            assert req["method"] == "PUT"
            assert server.header("Authorization") == "Bearer syt_secret_token"
            assert server.header("Content-Type") == "application/json"
            expected_path = (
                "/_matrix/client/v3/rooms/"
                + urllib.parse.quote("!room123:example.org", safe="")
                + "/typing/"
                + urllib.parse.quote("@bot:example.org", safe="")
            )
            assert req["path"] == expected_path
            assert req["json"] == {"typing": True, "timeout": 15000}

    def test_room_and_user_ids_are_url_encoded(self) -> None:
        """Room and user ids with reserved characters are fully percent-encoded."""
        with (
            _matrix_server() as server,
            _backend_for(server.base_url, user_id="@we ird/user:example.org") as backend,
        ):
            backend.send_typing("!ro om/1:example.org")
            assert len(server.requests) == 1
            path = server.requests[0]["path"]
            assert "/rooms/%21ro%20om%2F1%3Aexample.org/typing/" in path
            assert path.endswith("/%40we%20ird%2Fuser%3Aexample.org")
            assert " " not in path and "!" not in path and "@" not in path

    def test_trailing_slash_homeserver_normalized(self) -> None:
        """A homeserver URL with a trailing slash yields no double slash in the path."""
        with _matrix_server() as server, _backend_for(server.base_url + "/") as backend:
            backend.send_typing("!room123:example.org")
            assert len(server.requests) == 1
            assert server.requests[0]["path"].startswith("/_matrix/client/v3/rooms/")


class TestSendTypingBestEffort:
    """send_typing must never raise, whatever the transport does."""

    def test_server_error_500_is_swallowed(self) -> None:
        """A 500 reply from the homeserver does not raise."""
        with _matrix_server(reply_status=500) as server, _backend_for(server.base_url) as backend:
            backend.send_typing("!room123:example.org")
            assert len(server.requests) == 1
            assert server.requests[0]["method"] == "PUT"

    def test_unreachable_server_is_swallowed(self, refusing_port: int) -> None:
        """A connection-refused homeserver does not raise."""
        with _backend_for(f"http://127.0.0.1:{refusing_port}") as backend:
            backend.send_typing("!room123:example.org")

    def test_missing_user_id_is_silent_noop(self) -> None:
        """No stored user id: return without raising and without any request."""
        with _matrix_server() as server, _backend_for(server.base_url, user_id="") as backend:
            backend.send_typing("!room123:example.org")
            assert server.requests == []

    def test_no_client_is_silent_noop(self) -> None:
        """No client at all: return without raising."""
        backend = MatrixChannelBackend()
        assert backend._client is None
        backend.send_typing("!room123:example.org")

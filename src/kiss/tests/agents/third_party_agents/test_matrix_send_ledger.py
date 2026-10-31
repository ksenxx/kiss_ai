# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Matrix send_message delivery-ledger contract.

The shared ``ChannelRunner`` treats a ``send_message`` that returns without
raising as a successful delivery and deletes the reply from its
at-least-once ledger.  nio's ``room_send`` reports a refusal it does not
retry (``M_FORBIDDEN``, an auth failure) by *returning* an error response
object rather than raising, so ``MatrixChannelBackend.send_message`` must
convert error responses into exceptions — otherwise the reply is silently
lost.

These tests exercise ``_raise_on_error`` (the extracted
response-checking helper) with nio's real response classes
(``RoomSendError`` / ``RoomSendResponse``), and ``send_message`` through
the real nio ``AsyncClient`` the backend builds (``_client_from_config``)
against a local HTTP server answering the canned homeserver reply;
``send_message`` runs on the backend's real persistent background loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import unquote

import pytest
from nio.responses import RoomSendError, RoomSendResponse

from kiss.agents.third_party_agents.matrix.matrix_sea import (
    MatrixChannelBackend,
    _client_from_config,
    _raise_on_error,
)
from kiss.tests.agents.third_party_agents.recording_http import (
    JsonWebhookServer,
    serve_json_webhook,
)


@pytest.fixture
def homeserver() -> Iterator[JsonWebhookServer]:
    """A local HTTP server answering every PUT with the test's canned JSON reply."""
    yield from serve_json_webhook()


def _backend_for(server: JsonWebhookServer) -> MatrixChannelBackend:
    """Build a backend whose real nio client talks to *server*."""
    backend = MatrixChannelBackend()
    backend._client = _client_from_config(
        {"homeserver_url": server.base_url, "access_token": "tok", "user_id": "@bot:example.org"}
    )
    return backend


class TestRaiseOnSendErrorHelper:
    """_raise_on_error must raise on error responses only."""

    def test_error_response_raises_with_details(self) -> None:
        """An error response raises RuntimeError carrying status and message."""
        resp = RoomSendError("Too Many Requests", "M_LIMIT_EXCEEDED")
        with pytest.raises(RuntimeError) as excinfo:
            _raise_on_error(resp, "send to !room:example.org")
        text = str(excinfo.value)
        assert "!room:example.org" in text
        assert "M_LIMIT_EXCEEDED" in text
        assert "Too Many Requests" in text

    def test_error_response_without_status_code_uses_unknown(self) -> None:
        """A None status_code is reported as 'unknown' instead of 'None'."""
        resp = RoomSendError("Invalid token", None)
        with pytest.raises(RuntimeError) as excinfo:
            _raise_on_error(resp, "send to !room:example.org")
        text = str(excinfo.value)
        assert "unknown" in text
        assert "Invalid token" in text
        assert "None" not in text

    def test_success_response_is_returned_unchanged(self) -> None:
        """A success response passes through without raising."""
        resp = RoomSendResponse("$event1", "!room:example.org")
        assert _raise_on_error(resp, "send to !room:example.org") is resp


class TestSendMessageLedgerContract:
    """send_message must raise on error responses and stay silent on success."""

    def test_send_message_raises_on_error_response(self, homeserver: JsonWebhookServer) -> None:
        """A refusal nio returns as RoomSendError surfaces as RuntimeError."""
        homeserver.response_status = 403
        homeserver.response_body = {"errcode": "M_FORBIDDEN", "error": "Not in the room."}
        backend = _backend_for(homeserver)
        try:
            with pytest.raises(RuntimeError) as excinfo:
                backend.send_message("!room:example.org", "hello")
        finally:
            backend.disconnect()
        assert "M_FORBIDDEN" in str(excinfo.value)
        assert "Not in the room." in str(excinfo.value)
        assert len(homeserver.requests) == 1

    def test_send_message_success_does_not_raise(self, homeserver: JsonWebhookServer) -> None:
        """A 200 carrying an event id completes silently with the message on the wire."""
        homeserver.response_body = {"event_id": "$event1"}
        backend = _backend_for(homeserver)
        try:
            backend.send_message("!room:example.org", "hello")
        finally:
            backend.disconnect()
        [request] = homeserver.requests
        assert request["method"] == "PUT"
        assert "/rooms/!room:example.org/send/m.room.message/" in unquote(request["path"])
        assert homeserver.header("Authorization") == "Bearer tok"
        assert request["json"] == {"msgtype": "m.text", "body": "hello"}

    def test_send_message_without_client_is_noop(self) -> None:
        """No client configured: send_message returns without raising."""
        backend = MatrixChannelBackend()
        assert backend._client is None
        backend.send_message("!room:example.org", "hello")

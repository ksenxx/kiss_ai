# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Matrix send_message delivery-ledger contract.

The shared ``ChannelRunner`` treats a ``send_message`` that returns without
raising as a successful delivery and deletes the reply from its
at-least-once ledger.  nio's ``room_send`` reports failures (rate limits,
auth failures) by *returning* an error response object rather than raising,
so ``MatrixChannelBackend.send_message`` must convert error responses into
exceptions — otherwise the reply is silently lost.

matrix-nio is an optional dependency and is not installed in the test
environment, so these tests exercise ``_raise_on_send_error`` (the extracted
response-checking helper) and ``send_message`` with real minimal stand-in
response objects that carry nio's data contract: error responses
(``nio.responses.ErrorResponse`` / ``RoomSendError``) have ``message`` and
``status_code`` attributes, success responses (``RoomSendResponse``) have
``event_id`` and neither error attribute.  The stand-ins are test data, not
mocks of KISS code; ``send_message`` runs on the backend's real persistent
background event loop.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.agents.third_party_agents.matrix_agent import (
    MatrixChannelBackend,
    _raise_on_send_error,
)


class _SendErrorResponse:
    """Stand-in for ``nio.responses.RoomSendError`` (nio's error data contract)."""

    def __init__(self, message: str, status_code: str | None) -> None:
        self.message = message
        self.status_code = status_code


class _SendSuccessResponse:
    """Stand-in for ``nio.responses.RoomSendResponse`` (nio's success contract)."""

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        self.room_id = "!room:example.org"


class _SendingClient:
    """Stand-in nio client whose ``room_send`` returns a canned response.

    Records the send arguments so the tests can assert real wire intent,
    and provides the ``close()`` coroutine used by ``disconnect()``.
    """

    def __init__(self, response: Any) -> None:
        self.response = response
        self.sent: list[dict[str, Any]] = []

    async def room_send(self, room_id: str, message_type: str, content: dict[str, Any]) -> Any:
        """Record the send and return the canned response."""
        self.sent.append({"room_id": room_id, "message_type": message_type, "content": content})
        return self.response

    async def close(self) -> None:
        """No-op session close for ``disconnect()``."""


def _backend_with(response: Any) -> tuple[MatrixChannelBackend, _SendingClient]:
    """Build a backend whose client returns *response* from ``room_send``."""
    backend = MatrixChannelBackend()
    client = _SendingClient(response)
    backend._client = client
    return backend, client


class TestRaiseOnSendErrorHelper:
    """_raise_on_send_error must raise on error responses only."""

    def test_error_response_raises_with_details(self) -> None:
        """An error response raises RuntimeError carrying status and message."""
        resp = _SendErrorResponse("Too Many Requests", "M_LIMIT_EXCEEDED")
        with pytest.raises(RuntimeError) as excinfo:
            _raise_on_send_error(resp, "!room:example.org")
        text = str(excinfo.value)
        assert "!room:example.org" in text
        assert "M_LIMIT_EXCEEDED" in text
        assert "Too Many Requests" in text

    def test_error_response_without_status_code_uses_unknown(self) -> None:
        """A None status_code is reported as 'unknown' instead of 'None'."""
        resp = _SendErrorResponse("Invalid token", None)
        with pytest.raises(RuntimeError) as excinfo:
            _raise_on_send_error(resp, "!room:example.org")
        text = str(excinfo.value)
        assert "unknown" in text
        assert "Invalid token" in text
        assert "None" not in text

    def test_success_response_is_returned_unchanged(self) -> None:
        """A success response passes through without raising."""
        resp = _SendSuccessResponse("$event1")
        assert _raise_on_send_error(resp, "!room:example.org") is resp


class TestSendMessageLedgerContract:
    """send_message must raise on error responses and stay silent on success."""

    def test_send_message_raises_on_error_response(self) -> None:
        """A returned RoomSendError-style response surfaces as RuntimeError."""
        backend, client = _backend_with(
            _SendErrorResponse("Too Many Requests", "M_LIMIT_EXCEEDED")
        )
        try:
            with pytest.raises(RuntimeError) as excinfo:
                backend.send_message("!room:example.org", "hello")
            assert "M_LIMIT_EXCEEDED" in str(excinfo.value)
            assert len(client.sent) == 1
        finally:
            backend.disconnect()

    def test_send_message_success_does_not_raise(self) -> None:
        """A success response completes silently with the message on the wire."""
        backend, client = _backend_with(_SendSuccessResponse("$event1"))
        try:
            backend.send_message("!room:example.org", "hello")
            assert client.sent == [
                {
                    "room_id": "!room:example.org",
                    "message_type": "m.room.message",
                    "content": {"msgtype": "m.text", "body": "hello"},
                }
            ]
        finally:
            backend.disconnect()

    def test_send_message_without_client_is_noop(self) -> None:
        """No client configured: send_message returns without raising."""
        backend = MatrixChannelBackend()
        assert backend._client is None
        backend.send_message("!room:example.org", "hello")

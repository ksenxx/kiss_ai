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
   normal JSON body, posting to the channel with ``thread_ts`` as the
   reply reference (not as the channel id).
4. Tool-path callers of ``_post`` (e.g. ``post_message``) keep their
   return-an-error-string contract on API errors instead of raising.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
import requests

from kiss.agents.third_party_agents.discord.discord_sea import DiscordChannelBackend
from kiss.tests.agents.third_party_agents.recording_http import (
    JsonWebhookHandler,
    RecordingServer,
    serve_recording,
)


class _SendHandler(JsonWebhookHandler):
    """Records requests; 200 for channel OK, 500 for channel ERR."""

    def do_POST(self) -> None:  # noqa: N802
        """Answer the create-message endpoint, failing channel ``ERR`` like the Discord API."""
        path = self.record()["path"]
        if path == "/channels/ERR/messages":
            self._json(500, {"message": "Internal Server Error", "code": 0})
        else:
            self._json(200, {"id": "M1", "channel_id": path.split("/")[2]})


@pytest.fixture(scope="module")
def send_server() -> Iterator[RecordingServer]:
    """One Discord-shaped recording server shared by the module's tests."""
    yield from serve_recording(_SendHandler)


class TestDiscordSendMessageFailures:
    """End-to-end tests against a local Discord-shaped HTTP server."""

    @pytest.fixture(autouse=True)
    def _fresh_backend(self, send_server: RecordingServer) -> None:
        self.server = send_server
        self.server.requests.clear()
        self.backend = DiscordChannelBackend(api_base=send_server.base_url)
        self.backend._token = "test-token"

    def test_send_message_raises_on_http_500(self) -> None:
        """A 500 API response must raise so the ledger can retry the send."""
        with pytest.raises(requests.HTTPError):
            self.backend.send_message("ERR", "hello")
        req = self.server.requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/channels/ERR/messages"
        assert req["json"] == {"content": "hello"}

    def test_send_message_raises_on_unreachable_server(self, refusing_port: int) -> None:
        """An unreachable server (connection refused) must raise."""
        backend = DiscordChannelBackend(api_base=f"http://127.0.0.1:{refusing_port}")
        backend._token = "test-token"
        with pytest.raises(requests.RequestException):
            backend.send_message("111", "hello")
        assert self.server.requests == []

    def test_send_message_succeeds_on_200(self) -> None:
        """A 200 response must not raise; ``thread_ts`` is the reply reference, not the channel."""
        self.backend.send_message("111", "hello", thread_ts="9999")
        req = self.server.requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/channels/111/messages"
        assert req["headers"]["Authorization"] == "Bot test-token"
        assert req["json"]["content"] == "hello"
        assert req["json"]["message_reference"] == {"message_id": "9999"}

    def test_post_message_tool_returns_error_string_on_500(self) -> None:
        """The post_message tool must keep returning an error string, not raise."""
        result = json.loads(self.backend.post_message("ERR", "hello"))
        assert result["ok"] is False
        assert "Internal Server Error" in result["error"]

    def test_post_message_tool_succeeds_on_200(self) -> None:
        """The post_message tool must still report ok with the message id."""
        result = json.loads(self.backend.post_message("111", "hello"))
        assert result == {"ok": True, "id": "M1"}

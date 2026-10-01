# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for SlackChannelBackend — no mocks or test doubles.

Tests the SlackChannelBackend class with invalid tokens to verify error
handling, method signatures, and protocol conformance. A real in-process
HTTP server emulates the Slack Web API answering ``ok:false`` /
``invalid_auth`` for every call (same local-server pattern as
``test_bughunt_slack_rename.py``), so a real ``slack_sdk.WebClient``
raises ``SlackApiError`` deterministically without network access.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from kiss.agents.third_party_agents.slack.slack_sea import SlackChannelBackend
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingServer,
    recording_server,
    serve_recording,
)
from kiss.tests.agents.third_party_agents.slack_invalid_auth import InvalidAuthHandler


@pytest.fixture(scope="module")
def invalid_auth_server() -> Iterator[RecordingServer]:
    """One loopback ``invalid_auth`` Slack API for the whole module."""
    yield from serve_recording(InvalidAuthHandler)


def _send_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    """Consume the request body and answer *payload* as JSON."""
    length = int(handler.headers.get("Content-Length") or 0)
    if length:
        handler.rfile.read(length)
    data = json.dumps(payload).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class TestSlackChannelBackendMethods:
    """Tests for SlackChannelBackend methods with invalid token."""

    server: RecordingServer
    backend: SlackChannelBackend

    @pytest.fixture(autouse=True)
    def _fresh_backend(self, invalid_auth_server: RecordingServer) -> None:
        """A backend whose client talks to the module's ``invalid_auth`` server."""
        self.server = invalid_auth_server
        self.server.requests.clear()
        self.backend = SlackChannelBackend()
        self.backend._bot_user_id = "U_BOT_TEST"
        self._use_server(invalid_auth_server)

    def test_find_channel_returns_none_on_api_error(self) -> None:
        """find_channel raises SlackApiError with invalid token."""
        with pytest.raises(SlackApiError):
            self.backend.find_channel("nonexistent")

    def test_find_user_returns_none_on_api_error(self) -> None:
        """find_user raises SlackApiError with invalid token."""
        with pytest.raises(SlackApiError):
            self.backend.find_user("nobody")

    def _use_server(self, server: RecordingServer) -> None:
        self.backend._client = WebClient(
            token="xoxb-invalid-test-token-for-methods",
            base_url=f"{server.base_url}/",
            retry_handlers=[],
        )

    def test_find_channel_verifies_conversation_id(self) -> None:
        """find_channel returns a conversation ID once conversations.info is ok.

        A dedicated local server answers ``ok:true`` to ``conversations.info``
        and ``invalid_auth`` to everything else, so getting the ID back
        proves the direct-by-ID path was taken without a name lookup.
        """
        requests: list[str] = []

        class _InfoOkHandler(InvalidAuthHandler):
            def _respond(self) -> None:
                method = self.path.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
                requests.append(method)
                if method != "conversations.info":
                    super()._respond()
                    return
                _send_json(self, {"ok": True, "channel": {"id": "x"}})

        with recording_server(_InfoOkHandler) as server:
            self._use_server(server)
            assert self.backend.find_channel("C0AKYSNLB7W") == "C0AKYSNLB7W"
            assert self.backend.find_channel("G012ABCDEFG") == "G012ABCDEFG"
            assert self.backend.find_channel("D012ABCDEFG") == "D012ABCDEFG"
        assert requests == ["conversations.info"] * 3

    def test_find_channel_unverifiable_id_falls_back_to_name_lookup(self) -> None:
        """An ID that conversations.info rejects is looked up by name instead.

        The local server rejects ``conversations.info`` (``channel_not_found``)
        but lists a channel literally named like the ID, so the returned ID
        and the recorded request order prove the fallback name lookup ran.
        """
        requests: list[str] = []

        class _InfoFailsListOkHandler(InvalidAuthHandler):
            def _respond(self) -> None:
                method = self.path.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
                requests.append(method)
                if method == "conversations.info":
                    _send_json(self, {"ok": False, "error": "channel_not_found"})
                elif method == "conversations.list":
                    _send_json(
                        self,
                        {
                            "ok": True,
                            "channels": [{"id": "C_BY_NAME", "name": "C0AKYSNLB7W"}],
                            "response_metadata": {"next_cursor": ""},
                        },
                    )
                else:
                    super()._respond()

        with recording_server(_InfoFailsListOkHandler) as server:
            self._use_server(server)
            assert self.backend.find_channel("C0AKYSNLB7W") == "C_BY_NAME"
        assert requests == ["conversations.info", "conversations.list"]

    def test_find_channel_unverifiable_id_raises_when_name_lookup_fails(self) -> None:
        """With every call rejected, the fallback name lookup raises.

        The recorded request sequence proves ``conversations.info`` was
        tried first and ``conversations.list`` second for each ID prefix.
        """
        for ident in ("C0AKYSNLB7W", "G012ABCDEFG", "D012ABCDEFG"):
            self.server.requests.clear()
            with pytest.raises(SlackApiError):
                self.backend.find_channel(ident)
            # slack_sdk's urllib transport sends every Web API call as POST and
            # carries GET-style parameters in the query string.
            assert self.server.requests == [
                {"method": "POST", "path": "/conversations.info"},
                {"method": "POST", "path": "/conversations.list"},
            ]

    def test_find_user_passes_user_id_through(self) -> None:
        """find_user returns a user ID as-is without any API call.

        The server answers invalid_auth to every call, so getting the ID
        back (instead of SlackApiError) proves no request was made.
        """
        assert self.backend.find_user("UD7PM70GG") == "UD7PM70GG"
        assert self.backend.find_user("W012ABCDEFG") == "W012ABCDEFG"

    def test_join_channel_swallows_api_error(self) -> None:
        """join_channel silently ignores SlackApiError."""
        self.backend.join_channel("C_FAKE_CHANNEL")

    def test_strip_bot_mention_no_mention(self) -> None:
        """strip_bot_mention returns text unchanged if no mention."""
        assert self.backend.strip_bot_mention("hello world") == "hello world"

    def test_strip_bot_mention_no_bot_id(self) -> None:
        """strip_bot_mention returns text when bot_user_id is empty."""
        self.backend._bot_user_id = ""
        assert self.backend.strip_bot_mention("<@U_OTHER> hello") == "<@U_OTHER> hello"

    def test_poll_messages_raises_on_api_error(self) -> None:
        """poll_messages raises SlackApiError with invalid token."""
        with pytest.raises(SlackApiError):
            self.backend.poll_messages("C_FAKE", "0.000000")

    def test_send_message_raises_on_api_error(self) -> None:
        """send_message raises SlackApiError with invalid token."""
        with pytest.raises(SlackApiError):
            self.backend.send_message("C_FAKE", "test message")

    def test_send_message_with_thread(self) -> None:
        """send_message with thread_ts raises SlackApiError."""
        with pytest.raises(SlackApiError):
            self.backend.send_message("C_FAKE", "reply", thread_ts="1234.5678")

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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from kiss.agents.third_party_agents.slack_sea import (
    SlackChannelBackend,
    _save_token,
    _token_path,
)


def _backup_and_clear() -> str | None:
    path = _token_path()
    backup = None
    if path.exists():
        backup = path.read_text()
        path.unlink()
    return backup


def _restore(backup: str | None) -> None:
    path = _token_path()
    if backup is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(backup)
    elif path.exists():
        path.unlink()


class _InvalidAuthHandler(BaseHTTPRequestHandler):
    """Slack Web API emulator answering ``invalid_auth`` to every call."""

    def _respond(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        data = json.dumps({"ok": False, "error": "invalid_auth"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests (e.g. conversations.list)."""
        self._respond()

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests (e.g. chat.postMessage)."""
        self._respond()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


class TestSlackChannelBackendMethods:
    """Tests for SlackChannelBackend methods with invalid token."""

    server: ClassVar[ThreadingHTTPServer]
    thread: ClassVar[threading.Thread]

    @classmethod
    def setup_class(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _InvalidAuthHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def teardown_class(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()
        _save_token("xoxb-invalid-test-token-for-methods")
        port = self.server.server_address[1]
        self.backend = SlackChannelBackend()
        self.backend._client = WebClient(
            token="xoxb-invalid-test-token-for-methods",
            base_url=f"http://127.0.0.1:{port}/",
            retry_handlers=[],
        )
        self.backend._bot_user_id = "U_BOT_TEST"

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_find_channel_returns_none_on_api_error(self) -> None:
        """find_channel raises SlackApiError with invalid token."""
        with pytest.raises(SlackApiError):
            self.backend.find_channel("nonexistent")

    def test_find_user_returns_none_on_api_error(self) -> None:
        """find_user raises SlackApiError with invalid token."""
        with pytest.raises(SlackApiError):
            self.backend.find_user("nobody")

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

"""Integration tests for SlackChannelBackend — no mocks or test doubles.

Tests the SlackChannelBackend class with invalid tokens to verify error
handling, method signatures, and protocol conformance.
"""

from __future__ import annotations

from kiss.channels.slack_agent import (
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


class TestSlackChannelBackendConnect:
    """Tests for SlackChannelBackend.connect()."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_connect_no_token(self) -> None:
        """connect() returns False when no token is stored."""
        backend = SlackChannelBackend()
        assert backend.connect() is False
        assert "No Slack token" in backend.connection_info

    def test_connect_invalid_token(self) -> None:
        """connect() returns False with an invalid token."""
        _save_token("xoxb-invalid-test-token")
        backend = SlackChannelBackend()
        assert backend.connect() is False
        info = backend.connection_info.lower()
        assert "auth failed" in info or "error" in info

class TestSlackChannelBackendMethods:
    """Tests for SlackChannelBackend methods with invalid token."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()
        _save_token("xoxb-invalid-test-token-for-methods")
        self.backend = SlackChannelBackend()
        from slack_sdk import WebClient
        self.backend._client = WebClient(token="xoxb-invalid-test-token-for-methods")
        self.backend._bot_user_id = "U_BOT_TEST"

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_find_channel_returns_none_on_api_error(self) -> None:
        """find_channel raises SlackApiError with invalid token."""
        from slack_sdk.errors import SlackApiError
        try:
            self.backend.find_channel("nonexistent")
            assert False, "Should have raised SlackApiError"
        except SlackApiError:
            pass

    def test_find_user_returns_none_on_api_error(self) -> None:
        """find_user raises SlackApiError with invalid token."""
        from slack_sdk.errors import SlackApiError
        try:
            self.backend.find_user("nobody")
            assert False, "Should have raised SlackApiError"
        except SlackApiError:
            pass

    def test_join_channel_swallows_api_error(self) -> None:
        """join_channel silently ignores SlackApiError."""
        self.backend.join_channel("C_FAKE_CHANNEL")

    def test_is_from_bot_with_bot_user_id(self) -> None:
        """is_from_bot returns True when user matches bot_user_id."""
        assert self.backend.is_from_bot({"user": "U_BOT_TEST"})

    def test_strip_bot_mention_no_mention(self) -> None:
        """strip_bot_mention returns text unchanged if no mention."""
        assert self.backend.strip_bot_mention("hello world") == "hello world"

    def test_strip_bot_mention_no_bot_id(self) -> None:
        """strip_bot_mention returns text when bot_user_id is empty."""
        self.backend._bot_user_id = ""
        assert self.backend.strip_bot_mention("<@U_OTHER> hello") == "<@U_OTHER> hello"

    def test_poll_messages_raises_on_api_error(self) -> None:
        """poll_messages raises SlackApiError with invalid token."""
        from slack_sdk.errors import SlackApiError
        try:
            self.backend.poll_messages("C_FAKE", "0.000000")
            assert False, "Should have raised SlackApiError"
        except SlackApiError:
            pass

    def test_send_message_raises_on_api_error(self) -> None:
        """send_message raises SlackApiError with invalid token."""
        from slack_sdk.errors import SlackApiError
        try:
            self.backend.send_message("C_FAKE", "test message")
            assert False, "Should have raised SlackApiError"
        except SlackApiError:
            pass

    def test_send_message_with_thread(self) -> None:
        """send_message with thread_ts raises SlackApiError."""
        from slack_sdk.errors import SlackApiError
        try:
            self.backend.send_message("C_FAKE", "reply", thread_ts="1234.5678")
            assert False, "Should have raised SlackApiError"
        except SlackApiError:
            pass



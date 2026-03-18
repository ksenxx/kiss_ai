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


# ---------------------------------------------------------------------------
# kiss/channels/slack_agent.py — SlackChannelBackend, _save_token
# ---------------------------------------------------------------------------

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

    def test_connection_info_before_connect(self) -> None:
        """connection_info is empty before connect() is called."""
        backend = SlackChannelBackend()
        assert backend.connection_info == ""


class TestSlackChannelBackendMethods:
    """Tests for SlackChannelBackend methods with invalid token."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()
        _save_token("xoxb-invalid-test-token-for-methods")
        self.backend = SlackChannelBackend()
        # Force the client to be set even though auth fails
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
            pass  # Expected

    def test_find_user_returns_none_on_api_error(self) -> None:
        """find_user raises SlackApiError with invalid token."""
        from slack_sdk.errors import SlackApiError
        try:
            self.backend.find_user("nobody")
            assert False, "Should have raised SlackApiError"
        except SlackApiError:
            pass  # Expected

    def test_join_channel_swallows_api_error(self) -> None:
        """join_channel silently ignores SlackApiError."""
        # Should not raise
        self.backend.join_channel("C_FAKE_CHANNEL")

    def test_is_from_bot_with_bot_id(self) -> None:
        """is_from_bot returns True for messages with bot_id."""
        assert self.backend.is_from_bot({"bot_id": "B123", "user": "U_OTHER"})

    def test_is_from_bot_with_bot_user_id(self) -> None:
        """is_from_bot returns True when user matches bot_user_id."""
        assert self.backend.is_from_bot({"user": "U_BOT_TEST"})

    def test_is_from_bot_false_for_regular_user(self) -> None:
        """is_from_bot returns False for regular user messages."""
        assert not self.backend.is_from_bot({"user": "U_REGULAR"})

    def test_strip_bot_mention(self) -> None:
        """strip_bot_mention removes the <@BOT_ID> pattern."""
        text = "<@U_BOT_TEST> hello world"
        assert self.backend.strip_bot_mention(text) == "hello world"

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


class TestSlackChannelBackendToolMethods:
    """Tests for tool methods (return JSON strings) with invalid token."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()
        _save_token("xoxb-invalid-test-token-for-tools")
        self.backend = SlackChannelBackend()
        from slack_sdk import WebClient
        self.backend._client = WebClient(token="xoxb-invalid-test-token-for-tools")

    def teardown_method(self) -> None:
        _restore(self._backup)

    def _assert_error_json(self, result: str) -> None:
        import json
        data = json.loads(result)
        assert data["ok"] is False
        assert "error" in data

    def test_list_channels_error(self) -> None:
        """list_channels returns JSON error with invalid token."""
        self._assert_error_json(self.backend.list_channels())

    def test_read_messages_error(self) -> None:
        """read_messages returns JSON error with invalid token."""
        self._assert_error_json(self.backend.read_messages("C_FAKE"))

    def test_read_thread_error(self) -> None:
        """read_thread returns JSON error with invalid token."""
        self._assert_error_json(self.backend.read_thread("C_FAKE", "1234.5678"))

    def test_post_message_error(self) -> None:
        """post_message returns JSON error with invalid token."""
        self._assert_error_json(self.backend.post_message("C_FAKE", "hello"))

    def test_update_message_error(self) -> None:
        """update_message returns JSON error with invalid token."""
        self._assert_error_json(self.backend.update_message("C_FAKE", "1234.5678", "new"))

    def test_delete_message_error(self) -> None:
        """delete_message returns JSON error with invalid token."""
        self._assert_error_json(self.backend.delete_message("C_FAKE", "1234.5678"))

    def test_list_users_error(self) -> None:
        """list_users returns JSON error with invalid token."""
        self._assert_error_json(self.backend.list_users())

    def test_get_user_info_error(self) -> None:
        """get_user_info returns JSON error with invalid token."""
        self._assert_error_json(self.backend.get_user_info("U_FAKE"))

    def test_create_channel_error(self) -> None:
        """create_channel returns JSON error with invalid token."""
        self._assert_error_json(self.backend.create_channel("test-channel"))

    def test_invite_to_channel_error(self) -> None:
        """invite_to_channel returns JSON error with invalid token."""
        self._assert_error_json(self.backend.invite_to_channel("C_FAKE", "U_FAKE"))

    def test_add_reaction_error(self) -> None:
        """add_reaction returns JSON error with invalid token."""
        self._assert_error_json(self.backend.add_reaction("C_FAKE", "1234.5678", "thumbsup"))

    def test_search_messages_error(self) -> None:
        """search_messages returns JSON error with invalid token."""
        self._assert_error_json(self.backend.search_messages("test query"))

    def test_set_channel_topic_error(self) -> None:
        """set_channel_topic returns JSON error with invalid token."""
        self._assert_error_json(self.backend.set_channel_topic("C_FAKE", "new topic"))

    def test_upload_file_error(self) -> None:
        """upload_file returns JSON error with invalid token."""
        self._assert_error_json(self.backend.upload_file("C_FAKE", "content", "test.txt"))

    def test_get_channel_info_error(self) -> None:
        """get_channel_info returns JSON error with invalid token."""
        self._assert_error_json(self.backend.get_channel_info("C_FAKE"))

    def test_get_tool_methods_returns_all_tools(self) -> None:
        """get_tool_methods returns 15 bound method callables."""
        tools = self.backend.get_tool_methods()
        assert len(tools) == 15
        names = {t.__name__ for t in tools}
        assert "list_channels" in names
        assert "post_message" in names
        assert "get_channel_info" in names
        for tool in tools:
            assert callable(tool)


class TestSlackChannelBackendProtocol:
    """Verify SlackChannelBackend conforms to ChannelBackend protocol."""

    def test_has_all_protocol_methods(self) -> None:
        """SlackChannelBackend has all methods required by ChannelBackend."""
        backend = SlackChannelBackend()
        assert hasattr(backend, "connect")
        assert hasattr(backend, "connection_info")
        assert hasattr(backend, "find_channel")
        assert hasattr(backend, "find_user")
        assert hasattr(backend, "join_channel")
        assert hasattr(backend, "poll_messages")
        assert hasattr(backend, "send_message")
        assert hasattr(backend, "wait_for_reply")
        assert hasattr(backend, "is_from_bot")
        assert hasattr(backend, "strip_bot_mention")

    def test_isinstance_check_structural(self) -> None:
        """SlackChannelBackend satisfies ChannelBackend protocol structurally."""
        # Protocol conformance is checked at type-check time, but we can
        # verify the methods exist and are callable at runtime
        backend = SlackChannelBackend()
        for method_name in [
            "connect", "find_channel", "find_user", "join_channel",
            "poll_messages", "send_message", "wait_for_reply",
            "is_from_bot", "strip_bot_mention",
        ]:
            assert callable(getattr(backend, method_name))

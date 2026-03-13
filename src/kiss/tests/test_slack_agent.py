"""Integration tests for slack_agent — no mocks or test doubles.

Tests token persistence, tool creation, SlackAgent construction,
authentication workflows, and tool function signatures.
"""

from __future__ import annotations

import json

from kiss.channels.slack_agent import (
    SlackAgent,
    _cli_ask_user_question,
    _cli_wait_for_user,
    _load_token,
    _make_slack_tools,
    _save_token,
    _token_path,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backup_and_clear() -> str | None:
    """Back up existing token file and remove it."""
    path = _token_path()
    backup = None
    if path.exists():
        backup = path.read_text()
        path.unlink()
    return backup


def _restore(backup: str | None) -> None:
    """Restore a previously backed-up token file."""
    path = _token_path()
    if backup is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(backup)
    elif path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


class TestTokenPersistence:
    """Tests for _load_token, _save_token, _clear_token."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_load_corrupt_json(self) -> None:
        path = _token_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json!!")
        assert _load_token() is None

    def test_load_non_dict_json(self) -> None:
        path = _token_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"just a string"')
        assert _load_token() is None

# ---------------------------------------------------------------------------
# Slack tools
# ---------------------------------------------------------------------------


class TestSlackTools:
    """Tests for _make_slack_tools tool creation."""

    def test_tools_return_error_on_invalid_token(self) -> None:
        """Tools should return JSON error rather than raising."""
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        # list_channels will fail with invalid token but shouldn't raise
        list_channels = next(t for t in tools if t.__name__ == "list_channels")
        result = json.loads(list_channels())
        assert result["ok"] is False
        assert "error" in result

    def test_read_messages_returns_error_on_invalid_token(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        read_messages = next(t for t in tools if t.__name__ == "read_messages")
        result = json.loads(read_messages(channel="C01234567"))
        assert result["ok"] is False

    def test_send_message_returns_error_on_invalid_token(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        send_message = next(t for t in tools if t.__name__ == "send_message")
        result = json.loads(send_message(channel="C01234567", text="test"))
        assert result["ok"] is False

    def test_list_users_returns_error_on_invalid_token(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        list_users = next(t for t in tools if t.__name__ == "list_users")
        result = json.loads(list_users())
        assert result["ok"] is False

    def test_get_user_info_returns_error_on_invalid_token(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        get_user_info = next(
            t for t in tools if t.__name__ == "get_user_info"
        )
        result = json.loads(get_user_info(user="U01234567"))
        assert result["ok"] is False

    def test_create_channel_returns_error_on_invalid_token(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        create_channel = next(
            t for t in tools if t.__name__ == "create_channel"
        )
        result = json.loads(create_channel(name="test-channel"))
        assert result["ok"] is False

    def test_delete_message_returns_error_on_invalid_token(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        delete_message = next(
            t for t in tools if t.__name__ == "delete_message"
        )
        result = json.loads(delete_message(channel="C01234567", ts="1234.5678"))
        assert result["ok"] is False

    def test_update_message_returns_error_on_invalid_token(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        update_message = next(
            t for t in tools if t.__name__ == "update_message"
        )
        result = json.loads(
            update_message(channel="C01234567", ts="1234.5678", text="new")
        )
        assert result["ok"] is False

    def test_read_thread_returns_error_on_invalid_token(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        read_thread = next(t for t in tools if t.__name__ == "read_thread")
        result = json.loads(
            read_thread(channel="C01234567", thread_ts="1234.5678")
        )
        assert result["ok"] is False

    def test_invite_to_channel_returns_error(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        invite = next(
            t for t in tools if t.__name__ == "invite_to_channel"
        )
        result = json.loads(invite(channel="C01234567", users="U01234567"))
        assert result["ok"] is False

    def test_add_reaction_returns_error(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        add_reaction = next(t for t in tools if t.__name__ == "add_reaction")
        result = json.loads(
            add_reaction(channel="C01234567", timestamp="1234.5678", name="thumbsup")
        )
        assert result["ok"] is False

    def test_search_messages_returns_error(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        search = next(t for t in tools if t.__name__ == "search_messages")
        result = json.loads(search(query="test"))
        assert result["ok"] is False

    def test_set_channel_topic_returns_error(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        set_topic = next(
            t for t in tools if t.__name__ == "set_channel_topic"
        )
        result = json.loads(
            set_topic(channel="C01234567", topic="new topic")
        )
        assert result["ok"] is False

    def test_upload_file_returns_error(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        upload = next(t for t in tools if t.__name__ == "upload_file")
        result = json.loads(
            upload(channels="C01234567", content="hello", filename="test.txt")
        )
        assert result["ok"] is False

    def test_get_channel_info_returns_error(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        get_info = next(t for t in tools if t.__name__ == "get_channel_info")
        result = json.loads(get_info(channel="C01234567"))
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# SlackAgent
# ---------------------------------------------------------------------------


class TestSlackAgent:
    """Tests for SlackAgent construction and tool integration."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_check_auth_unauthenticated(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_slack_auth")
        result = check()
        assert "Not authenticated" in result
        assert "xoxb-" in result

    def test_check_auth_with_invalid_token(self) -> None:
        _save_token("xoxb-invalid-token")
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_slack_auth")
        result = json.loads(check())
        assert result["ok"] is False

    def test_authenticate_whitespace_token(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth = next(t for t in tools if t.__name__ == "authenticate_slack")
        result = auth(token="   ")
        assert "empty" in result.lower()

    def test_authenticate_invalid_token(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth = next(t for t in tools if t.__name__ == "authenticate_slack")
        result = json.loads(auth(token="xoxb-invalid-test"))
        assert result["ok"] is False
        assert "error" in result
        # Token should not be saved
        assert _load_token() is None

    def test_clear_auth(self) -> None:
        _save_token("xoxb-to-clear")
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_slack_auth")
        result = clear()
        assert "cleared" in result.lower()
        assert _load_token() is None
        assert agent._slack_client is None

    def test_clear_auth_when_not_authenticated(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_slack_auth")
        result = clear()
        assert "cleared" in result.lower()

# ---------------------------------------------------------------------------
# CLI helpers and main
# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_main_is_callable(self) -> None:
        assert callable(main)

    def test_main_missing_task_exits(self) -> None:
        import sys

        original_argv = sys.argv
        sys.argv = ["slack_agent"]
        try:
            main()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 2  # argparse exits with 2 for missing required args
        finally:
            sys.argv = original_argv

    def test_cli_callbacks_are_callable(self) -> None:
        assert callable(_cli_wait_for_user)
        assert callable(_cli_ask_user_question)

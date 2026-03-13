"""Integration tests for slack_agent — no mocks or test doubles.

Tests token persistence, tool creation, SlackAgent construction,
authentication workflows, and tool function signatures.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.channels.slack_agent import (
    SlackAgent,
    _clear_token,
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

    def test_token_path_location(self) -> None:
        path = _token_path()
        assert path.parent.name == "slack"
        assert path.parent.parent.name == "channels"
        assert path.name == "token.json"
        assert Path.home() / ".kiss" in path.parents

    def test_load_missing_returns_none(self) -> None:
        assert _load_token() is None

    def test_save_and_load_roundtrip(self) -> None:
        _save_token("xoxb-test-token-123")
        loaded = _load_token()
        assert loaded == "xoxb-test-token-123"

    def test_save_strips_whitespace(self) -> None:
        _save_token("  xoxb-test  ")
        loaded = _load_token()
        assert loaded == "xoxb-test"

    def test_save_creates_directory(self) -> None:
        _save_token("xoxb-dir-test")
        assert _token_path().exists()

    def test_save_sets_permissions(self) -> None:
        _save_token("xoxb-perm-test")
        path = _token_path()
        mode = path.stat().st_mode
        # Owner read/write only
        assert mode & stat.S_IRWXG == 0
        assert mode & stat.S_IRWXO == 0

    def test_clear_removes_file(self) -> None:
        _save_token("xoxb-clear-test")
        _clear_token()
        assert _load_token() is None
        assert not _token_path().exists()

    def test_clear_missing_is_noop(self) -> None:
        _clear_token()  # should not raise

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

    def test_load_dict_without_access_token(self) -> None:
        path = _token_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"other_key": "val"}))
        assert _load_token() is None

    def test_load_dict_with_empty_access_token(self) -> None:
        path = _token_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"access_token": ""}))
        assert _load_token() is None

    def test_save_overwrites_existing(self) -> None:
        _save_token("xoxb-first")
        _save_token("xoxb-second")
        assert _load_token() == "xoxb-second"


# ---------------------------------------------------------------------------
# Slack tools
# ---------------------------------------------------------------------------


class TestSlackTools:
    """Tests for _make_slack_tools tool creation."""

    def test_tool_count(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-fake-for-test")
        tools = _make_slack_tools(client)
        assert len(tools) == 15

    def test_tool_names(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-fake-for-test")
        tools = _make_slack_tools(client)
        names = {t.__name__ for t in tools}
        expected = {
            "list_channels",
            "read_messages",
            "read_thread",
            "send_message",
            "update_message",
            "delete_message",
            "list_users",
            "get_user_info",
            "create_channel",
            "invite_to_channel",
            "add_reaction",
            "search_messages",
            "set_channel_topic",
            "upload_file",
            "get_channel_info",
        }
        assert names == expected

    def test_all_tools_callable_with_docstrings(self) -> None:
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-fake-for-test")
        tools = _make_slack_tools(client)
        for tool in tools:
            assert callable(tool)
            assert tool.__doc__, f"{tool.__name__} missing docstring"

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

    def test_inherits_sorcar_agent(self) -> None:
        agent = SlackAgent()
        assert isinstance(agent, SorcarAgent)

    def test_agent_name(self) -> None:
        agent = SlackAgent()
        assert agent.name == "Slack Agent"

    def test_unauthenticated_has_auth_tools(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        assert "check_slack_auth" in names
        assert "authenticate_slack" in names
        assert "clear_slack_auth" in names

    def test_unauthenticated_no_slack_api_tools(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        assert "list_channels" not in names
        assert "send_message" not in names

    def test_authenticated_has_slack_api_tools(self) -> None:
        _save_token("xoxb-test-token")
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        # Auth tools
        assert "check_slack_auth" in names
        assert "authenticate_slack" in names
        assert "clear_slack_auth" in names
        # Slack API tools
        assert "list_channels" in names
        assert "send_message" in names
        assert "read_messages" in names

    def test_authenticated_has_all_15_slack_tools(self) -> None:
        _save_token("xoxb-test-token")
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        slack_tools = {
            "list_channels",
            "read_messages",
            "read_thread",
            "send_message",
            "update_message",
            "delete_message",
            "list_users",
            "get_user_info",
            "create_channel",
            "invite_to_channel",
            "add_reaction",
            "search_messages",
            "set_channel_topic",
            "upload_file",
            "get_channel_info",
        }
        assert slack_tools.issubset(names)

    def test_has_base_sorcar_tools(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        # SorcarAgent base tools
        assert "Bash" in names or "Read" in names
        assert "ask_user_question" in names

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

    def test_authenticate_empty_token(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth = next(t for t in tools if t.__name__ == "authenticate_slack")
        result = auth(token="")
        assert "empty" in result.lower()

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

    def test_agent_with_callbacks(self) -> None:
        def wait_cb(a: str, b: str) -> None:
            pass

        def ask_cb(q: str) -> str:
            return "answer"

        agent = SlackAgent(
            wait_for_user_callback=wait_cb,
            ask_user_question_callback=ask_cb,
        )
        assert agent._wait_for_user_callback is wait_cb
        assert agent._ask_user_question_callback is ask_cb

    def test_loads_existing_token_on_init(self) -> None:
        _save_token("xoxb-existing-token")
        agent = SlackAgent()
        assert agent._slack_client is not None

    def test_no_client_when_no_token(self) -> None:
        agent = SlackAgent()
        assert agent._slack_client is None


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

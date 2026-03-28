"""Integration tests for slack_agent — no mocks or test doubles.

Tests token persistence, tool creation, SlackAgent construction,
authentication workflows, and tool function signatures.
"""

from __future__ import annotations

import json

import pytest

from kiss.channels.slack_agent import (
    SlackAgent,
    _load_token,
    _make_slack_tools,
    _save_token,
    _token_path,
    main,
)


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


_SLACK_TOOL_ERROR_CASES = [
    ("list_channels", {}),
    ("read_messages", {"channel": "C01234567"}),
    ("post_message", {"channel": "C01234567", "text": "test"}),
    ("list_users", {}),
    ("get_user_info", {"user": "U01234567"}),
    ("create_channel", {"name": "test-channel"}),
    ("delete_message", {"channel": "C01234567", "ts": "1234.5678"}),
    ("update_message", {"channel": "C01234567", "ts": "1234.5678", "text": "new"}),
    ("read_thread", {"channel": "C01234567", "thread_ts": "1234.5678"}),
    ("invite_to_channel", {"channel": "C01234567", "users": "U01234567"}),
    ("add_reaction", {"channel": "C01234567", "timestamp": "1234.5678", "name": "thumbsup"}),
    ("search_messages", {"query": "test"}),
    ("set_channel_topic", {"channel": "C01234567", "topic": "new topic"}),
    ("upload_file", {"channels": "C01234567", "content": "hello", "filename": "test.txt"}),
    ("get_channel_info", {"channel": "C01234567"}),
]


class TestSlackTools:
    """Tests for _make_slack_tools tool creation."""

    @pytest.mark.parametrize("tool_name,kwargs", _SLACK_TOOL_ERROR_CASES)
    def test_tool_returns_error_on_invalid_token(
        self, tool_name: str, kwargs: dict
    ) -> None:
        """Every Slack tool returns {ok: false, error: ...} with invalid token."""
        from slack_sdk import WebClient

        client = WebClient(token="xoxb-invalid-token-for-test")
        tools = _make_slack_tools(client)
        fn = next(t for t in tools if t.__name__ == tool_name)
        result = json.loads(fn(**kwargs))
        assert result["ok"] is False
        assert "error" in result


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
        assert agent._backend._client is None

    def test_clear_auth_when_not_authenticated(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_slack_auth")
        result = clear()
        assert "cleared" in result.lower()


class TestCLIMain:
    def test_main_missing_task_exits(self) -> None:
        import sys

        original_argv = sys.argv
        sys.argv = ["slack_agent"]
        try:
            main()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 2
        finally:
            sys.argv = original_argv

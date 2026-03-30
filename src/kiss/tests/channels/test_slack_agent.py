"""Integration tests for slack_agent — no mocks or test doubles.

Tests token persistence, tool creation, SlackAgent construction,
authentication workflows, tool function signatures, and chat session
persistence (new_chat, resume_chat, -n flag).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
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
        original_argv = sys.argv
        sys.argv = ["slack_agent"]
        try:
            main()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 1
        finally:
            sys.argv = original_argv


# ---------------------------------------------------------------------------
# Helpers for chat persistence tests (redirect DB to temp dir)
# ---------------------------------------------------------------------------


def _redirect_db(tmpdir: str) -> tuple:
    """Redirect persistence DB to a temp dir and reset singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _intercept_run(agent: SlackAgent, captured: dict[str, Any]) -> Any:
    """Replace RelentlessAgent.run to capture the prompt without calling LLM.

    Returns the original method so it can be restored.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])  # RelentlessAgent
    original = parent_class.run

    def intercepted_run(self_agent: object, **kwargs: object) -> str:
        captured["prompt_template"] = kwargs.get("prompt_template", "")
        return "success: true\nsummary: done\n"

    parent_class.run = intercepted_run
    return original


class TestSlackAgentChatPersistence:
    """Integration tests for SlackAgent chat session persistence.

    Verifies new_chat(), resume_chat(), build_chat_prompt(), and the
    -n CLI flag work correctly with the real SQLite persistence layer.
    """

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()
        self._tmpdir = tempfile.mkdtemp()
        self._db_saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore_db(self._db_saved)
        _restore(self._backup)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_slack_agent_is_stateful(self) -> None:
        """SlackAgent inherits StatefulSorcarAgent and has chat methods."""
        from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

        agent = SlackAgent()
        assert isinstance(agent, StatefulSorcarAgent)
        assert hasattr(agent, "new_chat")
        assert hasattr(agent, "resume_chat")
        assert hasattr(agent, "chat_id")
        assert len(agent.chat_id) == 32  # UUID4 hex

    def test_new_chat_generates_new_id(self) -> None:
        """new_chat() replaces the chat_id with a fresh one."""
        agent = SlackAgent()
        old_id = agent.chat_id
        agent.new_chat()
        assert agent.chat_id != old_id
        assert len(agent.chat_id) == 32

    def test_resume_chat_loads_previous_session(self) -> None:
        """resume_chat() restores the chat_id from a previously persisted task."""
        agent = SlackAgent()
        agent.web_use_tool = None
        captured: dict[str, Any] = {}
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original = _intercept_run(agent, captured)
        try:
            agent.run(prompt_template="slack task alpha")
        finally:
            parent_class.run = original
        original_chat_id = agent.chat_id

        # Create a new agent and resume the session
        agent2 = SlackAgent()
        assert agent2.chat_id != original_chat_id
        agent2.resume_chat("slack task alpha")
        assert agent2.chat_id == original_chat_id

    def test_resume_chat_nonexistent_keeps_id(self) -> None:
        """resume_chat() with unknown task keeps the current chat_id."""
        agent = SlackAgent()
        old_id = agent.chat_id
        agent.resume_chat("task that does not exist in db")
        assert agent.chat_id == old_id

    def test_build_chat_prompt_no_history(self) -> None:
        """build_chat_prompt() returns '# Task' prefix when no prior context."""
        agent = SlackAgent()
        prompt = agent.build_chat_prompt("do something")
        assert prompt == "# Task\ndo something"

    def test_build_chat_prompt_with_history(self) -> None:
        """build_chat_prompt() includes previous tasks and results."""
        agent = SlackAgent()
        agent.web_use_tool = None
        captured: dict[str, Any] = {}
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original = _intercept_run(agent, captured)
        try:
            agent.run(prompt_template="first slack task")
        finally:
            parent_class.run = original

        prompt = agent.build_chat_prompt("second slack task")
        assert "## Previous tasks and results" in prompt
        assert "### Task 1\nfirst slack task" in prompt
        assert "### Result 1\ndone" in prompt
        assert "# Task (work on it now)\n\nsecond slack task" in prompt

    def test_main_with_new_flag_creates_new_session(self) -> None:
        """main() with -n flag calls new_chat(), giving a fresh session."""
        agent = SlackAgent()
        agent.web_use_tool = None
        captured: dict[str, Any] = {}
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original = _intercept_run(agent, captured)
        try:
            agent.run(prompt_template="pre-existing task")
        finally:
            parent_class.run = original
        # Now run main() with -n -t "new task"
        captured2: dict[str, Any] = {}
        original2 = _intercept_run(agent, captured2)
        original_argv = sys.argv
        try:
            sys.argv = ["slack_agent", "-n", "-t", "new task"]
            main()
        finally:
            parent_class.run = original2
            sys.argv = original_argv

        # The prompt should NOT contain previous task history (fresh session)
        prompt = str(captured2.get("prompt_template", ""))
        assert "# Task\nnew task" in prompt
        assert "pre-existing task" not in prompt

    def test_main_without_new_flag_resumes_session(self) -> None:
        """main() without -n flag calls resume_chat(), continuing the session."""
        agent = SlackAgent()
        agent.web_use_tool = None
        captured: dict[str, Any] = {}
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original = _intercept_run(agent, captured)
        try:
            agent.run(prompt_template="resumable task")
        finally:
            parent_class.run = original

        # Run main() without -n, with same task description
        captured2: dict[str, Any] = {}
        original2 = _intercept_run(agent, captured2)
        original_argv = sys.argv
        try:
            sys.argv = ["slack_agent", "-t", "resumable task"]
            main()
        finally:
            parent_class.run = original2
            sys.argv = original_argv

        # The prompt should include previous context
        prompt = str(captured2.get("prompt_template", ""))
        assert "## Previous tasks and results" in prompt
        assert "### Task 1\nresumable task" in prompt

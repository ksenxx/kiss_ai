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
from kiss.agents.channels.slack_agent import (
    _SLACK_DIR,
    SlackAgent,
    SlackChannelBackend,
    _delete_workspace,
    _list_workspaces,
    _load_token,
    _migrate_legacy_token,
    _save_token,
    _token_path,
    main,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


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


class TestWorkspaceTokenPaths:
    """Tests for workspace-keyed token storage and legacy migration."""

    def setup_method(self) -> None:
        self._default_backup = _backup_and_clear()
        self._created_dirs: list[Path] = []

    def teardown_method(self) -> None:
        for d in self._created_dirs:
            shutil.rmtree(d, ignore_errors=True)
        _restore(self._default_backup)

    def test_migrate_legacy_token(self) -> None:
        """Legacy token at _SLACK_DIR/token.json migrates to default/."""
        legacy = _SLACK_DIR / "token.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text('{"access_token": "xoxb-legacy"}')
        try:
            _migrate_legacy_token()
            assert not legacy.exists()
            assert _load_token() == "xoxb-legacy"
        finally:
            if legacy.exists():
                legacy.unlink()


class TestWorkspaceSlackAgent:
    """Tests for SlackAgent and SlackChannelBackend with workspace parameter."""

    def setup_method(self) -> None:
        self._default_backup = _backup_and_clear()
        self._created_dirs: list[Path] = []

    def teardown_method(self) -> None:
        for d in self._created_dirs:
            shutil.rmtree(d, ignore_errors=True)
        _restore(self._default_backup)

    def test_clear_auth_uses_workspace(self) -> None:
        """clear_slack_auth clears only the agent's workspace token."""
        ws = "test-ws-clear-auth"
        ws_dir = _SLACK_DIR / ws
        self._created_dirs.append(ws_dir)
        _save_token("xoxb-to-clear-ws", workspace=ws)
        _save_token("xoxb-keep-default")
        agent = SlackAgent(workspace=ws)
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_slack_auth")
        clear()
        assert _load_token(workspace=ws) is None
        assert _load_token() == "xoxb-keep-default"

    def test_cli_workspace_flag_in_usage(self) -> None:
        """main() with no args shows --workspace in usage."""
        original_argv = sys.argv
        sys.argv = ["kiss-slack"]
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
        assert "--workspace" in buf.getvalue()


class TestListWorkspaces:
    """Tests for _list_workspaces() and --list-workspaces CLI flag."""

    def setup_method(self) -> None:
        self._default_backup = _backup_and_clear()
        self._created_dirs: list[Path] = []

    def teardown_method(self) -> None:
        for d in self._created_dirs:
            shutil.rmtree(d, ignore_errors=True)
        _restore(self._default_backup)

    def test_no_slack_dir(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_list_workspaces() prints 'No workspaces found.' when _SLACK_DIR missing."""
        import kiss.agents.channels.slack_agent as mod

        original = mod._SLACK_DIR
        mod._SLACK_DIR = Path(tempfile.mkdtemp()) / "nonexistent"
        try:
            _list_workspaces()
            out = capsys.readouterr().out
            assert "No workspaces found" in out
        finally:
            mod._SLACK_DIR = original

    def test_empty_slack_dir(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_list_workspaces() prints 'No workspaces found.' when no workspace dirs."""
        import kiss.agents.channels.slack_agent as mod

        original = mod._SLACK_DIR
        empty_dir = Path(tempfile.mkdtemp())
        mod._SLACK_DIR = empty_dir
        try:
            _list_workspaces()
            out = capsys.readouterr().out
            assert "No workspaces found" in out
        finally:
            mod._SLACK_DIR = original
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_workspace_with_no_token_value(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_list_workspaces() shows 'no token' for empty/malformed token file."""
        ws = "test-ws-list-notoken"
        ws_dir = _SLACK_DIR / ws
        self._created_dirs.append(ws_dir)
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "token.json").write_text("{}")
        _list_workspaces()
        out = capsys.readouterr().out
        assert ws in out
        assert "no token" in out

    def test_cli_list_workspaces_flag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() with --list-workspaces runs _list_workspaces() and returns."""
        ws = "test-ws-cli-list"
        ws_dir = _SLACK_DIR / ws
        self._created_dirs.append(ws_dir)
        _save_token("xoxb-cli-list-test", workspace=ws)
        original_argv = sys.argv
        sys.argv = ["kiss-slack", "--list-workspaces"]
        try:
            main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert ws in out


class TestDeleteWorkspace:
    """Tests for _delete_workspace() and --delete-workspace CLI flag."""

    def setup_method(self) -> None:
        self._default_backup = _backup_and_clear()
        self._created_dirs: list[Path] = []

    def teardown_method(self) -> None:
        for d in self._created_dirs:
            shutil.rmtree(d, ignore_errors=True)
        _restore(self._default_backup)

    def test_delete_nonexistent_workspace(self) -> None:
        """_delete_workspace() exits with code 1 for missing workspace."""
        with pytest.raises(SystemExit) as exc_info:
            _delete_workspace("no-such-workspace")
        assert exc_info.value.code == 1

    def test_cli_delete_workspace_flag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() with --delete-workspace removes the workspace."""
        ws = "test-ws-cli-del"
        ws_dir = _SLACK_DIR / ws
        self._created_dirs.append(ws_dir)
        _save_token("xoxb-cli-del", workspace=ws)
        original_argv = sys.argv
        sys.argv = ["kiss-slack", "--delete-workspace", ws]
        try:
            main()
        finally:
            sys.argv = original_argv
        assert not ws_dir.exists()
        out = capsys.readouterr().out
        assert "deleted" in out.lower()

    def test_cli_delete_workspace_missing_arg(self) -> None:
        """main() with --delete-workspace but no value exits with code 1."""
        original_argv = sys.argv
        sys.argv = ["kiss-slack", "--delete-workspace"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv


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
    """Tests for SlackChannelBackend tool methods."""

    @pytest.mark.parametrize("tool_name,kwargs", _SLACK_TOOL_ERROR_CASES)
    def test_tool_returns_error_on_invalid_token(
        self, tool_name: str, kwargs: dict
    ) -> None:
        """Every Slack tool returns {ok: false, error: ...} with invalid token."""
        from slack_sdk import WebClient

        backend = SlackChannelBackend()
        backend._client = WebClient(token="xoxb-invalid-token-for-test")
        tools = backend.get_tool_methods()
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

    def test_authenticate_invalid_token(self) -> None:
        agent = SlackAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth = next(t for t in tools if t.__name__ == "authenticate_slack")
        result = json.loads(auth(token="xoxb-invalid-test"))
        assert result["ok"] is False
        assert "error" in result
        assert _load_token() is None


def _redirect_db(tmpdir: str) -> tuple:
    """Redirect persistence DB to a temp dir and reset singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _intercept_run(agent: SlackAgent, captured: dict[str, Any]) -> Any:
    """Replace RelentlessAgent.run to capture the prompt without calling LLM.

    Returns the original method so it can be restored.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
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

    def test_resume_chat_nonexistent_keeps_id(self) -> None:
        """resume_chat() with unknown task keeps the current chat_id."""
        agent = SlackAgent()
        old_id = agent.chat_id
        agent.resume_chat("task that does not exist in db")
        assert agent.chat_id == old_id

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
        captured2: dict[str, Any] = {}
        original2 = _intercept_run(agent, captured2)
        original_argv = sys.argv
        try:
            sys.argv = ["slack_agent", "-n", "-t", "new task"]
            main()
        finally:
            parent_class.run = original2
            sys.argv = original_argv

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

        captured2: dict[str, Any] = {}
        original2 = _intercept_run(agent, captured2)
        original_argv = sys.argv
        try:
            sys.argv = ["slack_agent", "-t", "resumable task"]
            main()
        finally:
            parent_class.run = original2
            sys.argv = original_argv

        prompt = str(captured2.get("prompt_template", ""))
        assert "## Previous tasks and results" in prompt
        assert "### Task 1\nresumable task" in prompt

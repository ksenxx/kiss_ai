# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for slack_sea — no mocks or test doubles.

Tests token persistence, tool creation, SlackAgent construction,
authentication workflows, and tool function signatures.  Invalid-token
scenarios run against a loopback Slack Web API emulator that answers
``invalid_auth``, so no test contacts ``slack.com``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from slack_sdk import WebClient

from kiss.agents.third_party_agents.slack.slack_sea import (
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
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingServer,
    serve_recording,
)
from kiss.tests.agents.third_party_agents.slack_invalid_auth import InvalidAuthHandler


@pytest.fixture(scope="module")
def invalid_auth_server() -> Iterator[RecordingServer]:
    """One loopback ``invalid_auth`` Slack API for the whole module."""
    yield from serve_recording(InvalidAuthHandler)


class TestTokenPersistence:
    """Tests for _load_token, _save_token, _clear_token."""

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

    def test_migrate_legacy_token(self, _isolated_slack_dir: Path) -> None:
        """Legacy token at <slack dir>/token.json migrates to default/."""
        legacy = _isolated_slack_dir / "token.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text('{"access_token": "xoxb-legacy"}')
        _migrate_legacy_token()
        assert not legacy.exists()
        assert _load_token() == "xoxb-legacy"


class TestWorkspaceSlackAgent:
    """Tests for SlackAgent and SlackChannelBackend with workspace parameter."""

    def test_clear_auth_uses_workspace(self) -> None:
        """clear_slack_auth clears only the agent's workspace token."""
        ws = "test-ws-clear-auth"
        _save_token("xoxb-to-clear-ws", workspace=ws)
        _save_token("xoxb-keep-default")
        agent = SlackAgent(workspace=ws)
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

    def test_no_slack_dir(
        self, capsys: pytest.CaptureFixture[str], _isolated_slack_dir: Path
    ) -> None:
        """_list_workspaces() prints 'No workspaces found.' when the dir is missing."""
        assert not _isolated_slack_dir.exists()
        _list_workspaces()
        assert "No workspaces found" in capsys.readouterr().out

    def test_empty_slack_dir(
        self, capsys: pytest.CaptureFixture[str], _isolated_slack_dir: Path
    ) -> None:
        """_list_workspaces() prints 'No workspaces found.' when no workspace dirs."""
        _isolated_slack_dir.mkdir(parents=True)
        _list_workspaces()
        assert "No workspaces found" in capsys.readouterr().out

    def test_workspace_with_no_token_value(
        self, capsys: pytest.CaptureFixture[str], _isolated_slack_dir: Path
    ) -> None:
        """_list_workspaces() shows 'no token' for empty/malformed token file."""
        ws = "test-ws-list-notoken"
        ws_dir = _isolated_slack_dir / ws
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "token.json").write_text("{}")
        _list_workspaces()
        out = capsys.readouterr().out
        assert ws in out
        assert "no token" in out

    def test_cli_list_workspaces_flag(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        invalid_auth_server: RecordingServer,
    ) -> None:
        """main() with --list-workspaces runs _list_workspaces() and returns."""
        # Listing validates each stored token with auth.test; keep that call
        # on the loopback emulator.
        monkeypatch.setenv("KISS_SLACK_BASE_URL", invalid_auth_server.base_url)
        ws = "test-ws-cli-list"
        _save_token("xoxb-cli-list-test", workspace=ws)
        original_argv = sys.argv
        sys.argv = ["kiss-slack", "--list-workspaces"]
        try:
            main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert ws in out
        assert {"method": "POST", "path": "/api/auth.test"} in invalid_auth_server.requests


class TestDeleteWorkspace:
    """Tests for _delete_workspace() and --delete-workspace CLI flag."""

    def test_delete_nonexistent_workspace(self) -> None:
        """_delete_workspace() exits with code 1 for missing workspace."""
        with pytest.raises(SystemExit) as exc_info:
            _delete_workspace("no-such-workspace")
        assert exc_info.value.code == 1

    def test_cli_delete_workspace_flag(
        self, capsys: pytest.CaptureFixture[str], _isolated_slack_dir: Path
    ) -> None:
        """main() with --delete-workspace removes the workspace."""
        ws = "test-ws-cli-del"
        _save_token("xoxb-cli-del", workspace=ws)
        assert (_isolated_slack_dir / ws).is_dir()
        original_argv = sys.argv
        sys.argv = ["kiss-slack", "--delete-workspace", ws]
        try:
            main()
        finally:
            sys.argv = original_argv
        assert not (_isolated_slack_dir / ws).exists()
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
    ("list_third_party_agents", {}),
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
    ("upload_file", {
        "third_party_agents": "C01234567", "content": "hello", "filename": "test.txt",
    }),
    ("get_channel_info", {"channel": "C01234567"}),
]


class TestSlackTools:
    """Tests for SlackChannelBackend tool methods."""

    @pytest.mark.parametrize("tool_name,kwargs", _SLACK_TOOL_ERROR_CASES)
    def test_tool_returns_error_on_invalid_token(
        self, tool_name: str, kwargs: dict, invalid_auth_server: RecordingServer
    ) -> None:
        """Every Slack tool returns {ok: false, error: ...} with invalid token."""
        backend = SlackChannelBackend()
        backend._client = WebClient(
            token="xoxb-invalid-token-for-test",
            base_url=f"{invalid_auth_server.base_url}/",
            retry_handlers=[],
        )
        tools = backend.get_tool_methods()
        fn = next(t for t in tools if t.__name__ == tool_name)
        before = len(invalid_auth_server.requests)
        result = json.loads(fn(**kwargs))
        assert result["ok"] is False
        assert "error" in result
        assert len(invalid_auth_server.requests) > before


class TestSlackAgent:
    """Tests for SlackAgent construction and tool integration."""

    def test_check_auth_unauthenticated(self) -> None:
        agent = SlackAgent()
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_slack_auth")
        result = check()
        assert "Not authenticated" in result
        assert "authenticate_slack()" in result

    def test_check_auth_with_invalid_token(
        self, monkeypatch: pytest.MonkeyPatch, invalid_auth_server: RecordingServer
    ) -> None:
        monkeypatch.setenv("KISS_SLACK_BASE_URL", invalid_auth_server.base_url)
        _save_token("xoxb-invalid-token")
        agent = SlackAgent()
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_slack_auth")
        result = json.loads(check())
        assert result["ok"] is False
        assert {"method": "POST", "path": "/api/auth.test"} in invalid_auth_server.requests

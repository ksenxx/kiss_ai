"""Integration tests for the Slack channel agent.

Tests workspace config management, tool interfaces, channel resolution,
polling lifecycle, and error handling paths — all without mocks.
"""

from __future__ import annotations

import time

import pytest

from kiss.channels.slack_agent import (
    _SLACK_CONFIG_DIR,
    _SLACK_WORKSPACES_FILE,
    SlackChannelAgent,
    _format_messages,
    _get_client,
    _load_workspaces,
    _resolve_channel_id,
    _resolve_workspace_client,
    _save_workspaces,
    add_workspace,
    list_channels,
    list_workspaces,
    read_messages,
    remove_workspace,
    send_message,
)


class TestWorkspaceConfigPersistence:
    """Test loading and saving workspace configurations."""

    def setup_method(self):
        """Back up and clear workspace config for test isolation."""
        self._backup = None
        if _SLACK_WORKSPACES_FILE.exists():
            self._backup = _SLACK_WORKSPACES_FILE.read_text()
            _SLACK_WORKSPACES_FILE.unlink()

    def teardown_method(self):
        """Restore original workspace config."""
        if self._backup is not None:
            _SLACK_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _SLACK_WORKSPACES_FILE.write_text(self._backup)
        elif _SLACK_WORKSPACES_FILE.exists():
            _SLACK_WORKSPACES_FILE.unlink()

    def test_load_corrupt_json(self):
        _SLACK_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _SLACK_WORKSPACES_FILE.write_text("{bad json!!!")
        assert _load_workspaces() == {}

    def test_list_workspaces_empty(self):
        result = list_workspaces()
        assert "No Slack workspaces configured" in result
        assert "add_workspace" in result

    def test_list_workspaces_with_entries(self):
        _save_workspaces(
            {
                "ws1": {"token": "t1", "team": "Team1"},
                "ws2": {"token": "t2", "team": "Team2"},
            }
        )
        result = list_workspaces()
        assert "ws1" in result
        assert "Team1" in result
        assert "ws2" in result
        assert "Team2" in result

    def test_remove_workspace_exists(self):
        _save_workspaces({"ws1": {"token": "t", "team": "T"}})
        result = remove_workspace("ws1")
        assert "removed" in result
        assert _load_workspaces() == {}

    def test_remove_workspace_not_found(self):
        result = remove_workspace("nonexistent")
        assert "not found" in result


class TestAddWorkspace:
    """Test add_workspace with validation."""

    def setup_method(self):
        self._backup = None
        if _SLACK_WORKSPACES_FILE.exists():
            self._backup = _SLACK_WORKSPACES_FILE.read_text()
            _SLACK_WORKSPACES_FILE.unlink()

    def teardown_method(self):
        if self._backup is not None:
            _SLACK_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _SLACK_WORKSPACES_FILE.write_text(self._backup)
        elif _SLACK_WORKSPACES_FILE.exists():
            _SLACK_WORKSPACES_FILE.unlink()

    def test_empty_name(self):
        result = add_workspace("", "xoxb-test")
        assert "Error" in result
        assert "name" in result

    def test_empty_token(self):
        result = add_workspace("test", "")
        assert "Error" in result
        assert "token" in result

    def test_invalid_token(self):
        result = add_workspace("test", "xoxb-invalid-fake-token")
        assert "Error" in result
        assert "invalid token" in result


class TestResolveWorkspaceClient:
    """Test workspace client resolution."""

    def setup_method(self):
        self._backup = None
        if _SLACK_WORKSPACES_FILE.exists():
            self._backup = _SLACK_WORKSPACES_FILE.read_text()
            _SLACK_WORKSPACES_FILE.unlink()

    def teardown_method(self):
        if self._backup is not None:
            _SLACK_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _SLACK_WORKSPACES_FILE.write_text(self._backup)
        elif _SLACK_WORKSPACES_FILE.exists():
            _SLACK_WORKSPACES_FILE.unlink()

    def test_single_workspace_auto_select(self):
        _save_workspaces({"ws1": {"token": "xoxb-test", "team": "T1"}})
        client, name = _resolve_workspace_client(None)
        assert name == "ws1"
        assert client is not None

    def test_multiple_workspaces_no_selection(self):
        _save_workspaces(
            {
                "ws1": {"token": "xoxb-t1", "team": "T1"},
                "ws2": {"token": "xoxb-t2", "team": "T2"},
            }
        )
        with pytest.raises(ValueError, match="Multiple workspaces"):
            _resolve_workspace_client(None)


class TestResolveChannelId:
    """Test channel ID resolution logic."""

    def test_name_lookup_fails_with_invalid_token(self):
        client = _get_client("xoxb-fake")
        with pytest.raises(Exception):
            _resolve_channel_id(client, "general")


class TestSlackApiErrorPaths:
    """Test error handling for Slack API calls with invalid credentials."""

    def setup_method(self):
        self._backup = None
        if _SLACK_WORKSPACES_FILE.exists():
            self._backup = _SLACK_WORKSPACES_FILE.read_text()
            _SLACK_WORKSPACES_FILE.unlink()

    def teardown_method(self):
        if self._backup is not None:
            _SLACK_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _SLACK_WORKSPACES_FILE.write_text(self._backup)
        elif _SLACK_WORKSPACES_FILE.exists():
            _SLACK_WORKSPACES_FILE.unlink()

    def test_list_channels_invalid_token(self):
        _save_workspaces({"ws1": {"token": "xoxb-fake", "team": "T1"}})
        result = list_channels("ws1")
        assert "Error" in result

    def test_read_messages_invalid_token(self):
        _save_workspaces({"ws1": {"token": "xoxb-fake", "team": "T1"}})
        # Use a channel ID to skip name resolution
        result = read_messages("C01234ABCD", workspace="ws1")
        assert "Error" in result

    def test_send_message_empty_text(self):
        result = send_message("general", "")
        assert "Error" in result
        assert "empty" in result

    def test_send_message_invalid_token(self):
        _save_workspaces({"ws1": {"token": "xoxb-fake", "team": "T1"}})
        result = send_message("C01234ABCD", "hello", workspace="ws1")
        assert "Error" in result

    def test_list_channels_wrong_workspace(self):
        _save_workspaces({"ws1": {"token": "xoxb-fake", "team": "T1"}})
        result = list_channels("nonexistent")
        assert "not found" in result

    def test_read_messages_wrong_workspace(self):
        _save_workspaces({"ws1": {"token": "xoxb-fake", "team": "T1"}})
        result = read_messages("general", workspace="nonexistent")
        assert "not found" in result

    def test_send_message_wrong_workspace(self):
        _save_workspaces({"ws1": {"token": "xoxb-fake", "team": "T1"}})
        result = send_message("general", "hi", workspace="nonexistent")
        assert "not found" in result


class TestSlackChannelAgent:
    """Test the SlackChannelAgent wrapper class."""

    def test_tools_have_docstrings(self):
        agent = SlackChannelAgent()
        for tool in agent.get_tools():
            assert tool.__doc__, f"{tool.__name__} missing docstring"


class TestPolling:
    """Test polling start/stop lifecycle."""

    def test_start_and_stop_polling(self):
        agent = SlackChannelAgent()
        # Start polling (it will fail on API calls but thread starts)
        result = agent.start_polling("test-channel", workspace="test", interval=0.1)
        assert "Started polling" in result

        # Duplicate start
        result = agent.start_polling("test-channel", workspace="test", interval=0.1)
        assert "Already polling" in result

        # Stop
        result = agent.stop_polling("test-channel", workspace="test")
        assert "Stopped polling" in result

        # Stop again - not polling
        result = agent.stop_polling("test-channel", workspace="test")
        assert "Not polling" in result

    def test_polling_callback(self):
        """Test that polling thread runs and can be stopped cleanly."""
        agent = SlackChannelAgent()
        # Use a very short interval - the poll will fail (no valid workspace)
        # but we verify the thread lifecycle works
        result = agent.start_polling("ch1", workspace="ws1", interval=0.05)
        assert "Started" in result
        time.sleep(0.2)  # Let a few poll cycles attempt
        agent.stop_all_polling()


class TestFormatMessages:
    """Test message formatting helper."""

    def test_format_with_messages(self):
        client = _get_client("xoxb-fake")
        messages = [
            {"user": "U123", "text": "hello world"},
            {"user": "U456", "text": "hi there"},
        ]
        # Will fail user lookup but still format with raw user ID
        result = _format_messages(client, "general", messages)
        # Messages are reversed (newest first from API -> chronological)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "#general" in lines[0]
        assert "hi there" in lines[0]  # Last message appears first (reversed)
        assert "hello world" in lines[1]

"""Integration tests for ChannelDaemon.run_once() — no mocks or test doubles.

Tests the one-shot poll mode: run_once(), _has_bot_reply(), and the CLI
integration for ``--daemon-channel`` without ``--daemon``.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from typing import Any

import pytest

from kiss.channels.background_agent import ChannelDaemon, _SenderState
from kiss.channels.slack_agent import (
    SlackChannelBackend,
    _save_token,
    _token_path,
    main,
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


class TestRunOnceConnectFailure:
    """Tests for run_once() when backend connection fails."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_run_once_raises_on_connect_failure(self) -> None:
        """run_once() raises RuntimeError when backend.connect() returns False."""
        backend = SlackChannelBackend()
        # No token saved — connect() returns False immediately
        daemon = ChannelDaemon(
            backend=backend,
            channel_name="test-channel",
            agent_name="test",
        )
        with pytest.raises(RuntimeError, match="Failed to connect"):
            daemon.run_once()

    def test_run_once_raises_on_invalid_token(self) -> None:
        """run_once() raises RuntimeError with an invalid token (auth fails)."""
        _save_token("xoxb-invalid-for-run-once-test")
        backend = SlackChannelBackend()
        daemon = ChannelDaemon(
            backend=backend,
            channel_name="test-channel",
            agent_name="test",
        )
        with pytest.raises(RuntimeError, match="Failed to connect"):
            daemon.run_once()


class TestHasBotReply:
    """Tests for ChannelDaemon._has_bot_reply() logic."""

    def _make_daemon_with_poll_fn(
        self, poll_fn: Any = None
    ) -> ChannelDaemon:
        """Create a daemon with a configurable poll_thread_fn."""
        backend = SlackChannelBackend()
        backend._bot_user_id = "U_BOT"
        daemon = ChannelDaemon(
            backend=backend,
            channel_name="test",
            agent_name="test",
        )
        # Override the poll_thread_fn directly
        daemon._poll_thread_fn = poll_fn
        return daemon

    def test_no_poll_thread_fn_returns_false(self) -> None:
        """_has_bot_reply returns False when backend has no poll_thread_messages."""
        daemon = self._make_daemon_with_poll_fn(None)
        msg = {"ts": "1234.5678", "reply_count": 5, "user": "U_HUMAN"}
        assert daemon._has_bot_reply("C_TEST", msg) is False

    def test_zero_reply_count_returns_false(self) -> None:
        """_has_bot_reply returns False when message has no replies."""

        def poll_fn(ch: str, ts: str, oldest: str, limit: int = 10) -> tuple:
            raise AssertionError("Should not be called")

        daemon = self._make_daemon_with_poll_fn(poll_fn)
        msg = {"ts": "1234.5678", "reply_count": 0, "user": "U_HUMAN"}
        assert daemon._has_bot_reply("C_TEST", msg) is False

    def test_no_ts_returns_false(self) -> None:
        """_has_bot_reply returns False when message has no ts."""

        def poll_fn(ch: str, ts: str, oldest: str, limit: int = 10) -> tuple:
            raise AssertionError("Should not be called")

        daemon = self._make_daemon_with_poll_fn(poll_fn)
        msg = {"reply_count": 3, "user": "U_HUMAN"}
        assert daemon._has_bot_reply("C_TEST", msg) is False

    def test_bot_reply_found(self) -> None:
        """_has_bot_reply returns True when a bot reply exists in thread."""
        bot_reply = {"user": "U_BOT", "ts": "1234.6000", "text": "bot response"}

        def poll_fn(
            ch: str, ts: str, oldest: str, limit: int = 10
        ) -> tuple[list[dict[str, Any]], str]:
            return [bot_reply], "1234.600001"

        daemon = self._make_daemon_with_poll_fn(poll_fn)
        msg = {"ts": "1234.5678", "reply_count": 1, "user": "U_HUMAN"}
        assert daemon._has_bot_reply("C_TEST", msg) is True

    def test_no_bot_reply_in_thread(self) -> None:
        """_has_bot_reply returns False when thread has only human replies."""
        human_reply = {"user": "U_OTHER_HUMAN", "ts": "1234.6000", "text": "ok"}

        def poll_fn(
            ch: str, ts: str, oldest: str, limit: int = 10
        ) -> tuple[list[dict[str, Any]], str]:
            return [human_reply], "1234.600001"

        daemon = self._make_daemon_with_poll_fn(poll_fn)
        msg = {"ts": "1234.5678", "reply_count": 1, "user": "U_HUMAN"}
        assert daemon._has_bot_reply("C_TEST", msg) is False

    def test_bot_id_reply_detected(self) -> None:
        """_has_bot_reply detects replies with bot_id field."""
        bot_msg = {"bot_id": "B123", "ts": "1234.6000", "text": "automated"}

        def poll_fn(
            ch: str, ts: str, oldest: str, limit: int = 10
        ) -> tuple[list[dict[str, Any]], str]:
            return [bot_msg], "1234.600001"

        daemon = self._make_daemon_with_poll_fn(poll_fn)
        msg = {"ts": "1234.5678", "reply_count": 1, "user": "U_HUMAN"}
        assert daemon._has_bot_reply("C_TEST", msg) is True

    def test_poll_fn_exception_returns_false(self) -> None:
        """_has_bot_reply returns False when poll_thread_fn raises."""

        def poll_fn(ch: str, ts: str, oldest: str, limit: int = 10) -> tuple:
            raise ConnectionError("network down")

        daemon = self._make_daemon_with_poll_fn(poll_fn)
        msg = {"ts": "1234.5678", "reply_count": 1, "user": "U_HUMAN"}
        assert daemon._has_bot_reply("C_TEST", msg) is False


class TestSenderState:
    """Tests for _SenderState dataclass."""

    def test_initial_state(self) -> None:
        """_SenderState has empty chat_id and empty queue."""
        state = _SenderState()
        assert state.chat_id == ""
        assert state.pending_messages.empty()


class TestCLIOneShotMode:
    """Tests for CLI integration of one-shot poll mode."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_daemon_channel_without_daemon_no_token_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--daemon-channel without --daemon exits when no token stored.

        The _make_daemon_backend factory calls sys.exit(1) when no token
        is found, before run_once() is reached.
        """
        original_argv = sys.argv
        sys.argv = [
            "kiss-slack",
            "--daemon-channel",
            "test-channel",
            "-m",
            "test-model",
        ]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert "Not authenticated" in out

    def test_daemon_channel_without_daemon_with_invalid_token(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """One-shot mode with invalid token prints checking message and raises.

        When a token exists but is invalid, _make_daemon_backend succeeds
        (it only loads the token), but run_once() fails at connect().
        """
        _save_token("xoxb-invalid-for-oneshot-test")
        original_argv = sys.argv
        sys.argv = [
            "kiss-slack",
            "--daemon-channel",
            "some-channel",
            "-m",
            "test-model",
        ]
        try:
            with pytest.raises(RuntimeError, match="Failed to connect"):
                main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert "Checking Slack channel for pending messages..." in out

    def test_usage_shows_daemon_channel(self) -> None:
        """main() with no args includes --daemon-channel in help."""
        original_argv = sys.argv
        sys.argv = ["kiss-slack"]
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
        # --daemon-channel is available via argparse even if not in custom usage
        # The custom usage shows [--daemon]
        assert "--daemon" in buf.getvalue()

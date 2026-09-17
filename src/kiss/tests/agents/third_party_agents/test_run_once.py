# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ChannelRunner.run_once() — no mocks or test doubles.

Tests the one-shot poll mode: run_once(), _has_bot_reply(), and the CLI
integration for ``--channel``.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer
from kiss.agents.third_party_agents._channel_agent_utils import ChannelRunner
from kiss.agents.third_party_agents.slack_agent import (
    SlackChannelBackend,
    _save_token,
    _token_path,
    main,
)


class _InvalidAuthHandler(BaseHTTPRequestHandler):
    """Local Slack Web API stand-in answering every call with ``invalid_auth``.

    Returns exactly what ``https://slack.com/api/auth.test`` returns for a
    bad token, so ``SlackChannelBackend.connect()`` follows its real
    ``SlackApiError`` path without any network dependence.
    """

    def do_POST(self) -> None:
        """Reply 200 with Slack's invalid-token error body."""
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length:
            self.rfile.read(length)
        body = json.dumps({"ok": False, "error": "invalid_auth"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


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
        poller = ChannelRunner(
            backend=backend,
            channel_name="test-channel",
            agent_name="test",
        )
        with pytest.raises(RuntimeError, match="Failed to connect"):
            poller.run_once()


class TestHasBotReply:
    """Tests for ChannelRunner._has_bot_reply() logic."""

    def _make_poller_with_poll_fn(
        self, poll_fn: Any = None
    ) -> ChannelRunner:
        """Create a poller with a configurable poll_thread_fn."""
        backend = SlackChannelBackend()
        backend._bot_user_id = "U_BOT"
        poller = ChannelRunner(
            backend=backend,
            channel_name="test",
            agent_name="test",
        )
        poller._poll_thread_fn = poll_fn
        return poller

    def test_no_poll_thread_fn_returns_false(self) -> None:
        """_has_bot_reply returns False when backend has no poll_thread_messages."""
        poller = self._make_poller_with_poll_fn(None)
        msg = {"ts": "1234.5678", "reply_count": 5, "user": "U_HUMAN"}
        assert poller._has_bot_reply("C_TEST", msg) is False

    def test_zero_reply_count_returns_false(self) -> None:
        """_has_bot_reply returns False when message has no replies."""

        def poll_fn(ch: str, ts: str, oldest: str, limit: int = 10) -> tuple:
            raise AssertionError("Should not be called")

        poller = self._make_poller_with_poll_fn(poll_fn)
        msg = {"ts": "1234.5678", "reply_count": 0, "user": "U_HUMAN"}
        assert poller._has_bot_reply("C_TEST", msg) is False

    def test_no_ts_returns_false(self) -> None:
        """_has_bot_reply returns False when message has no ts."""

        def poll_fn(ch: str, ts: str, oldest: str, limit: int = 10) -> tuple:
            raise AssertionError("Should not be called")

        poller = self._make_poller_with_poll_fn(poll_fn)
        msg = {"reply_count": 3, "user": "U_HUMAN"}
        assert poller._has_bot_reply("C_TEST", msg) is False

    def test_no_bot_reply_in_thread(self) -> None:
        """_has_bot_reply returns False when thread has only human replies."""
        human_reply = {"user": "U_OTHER_HUMAN", "ts": "1234.6000", "text": "ok"}

        def poll_fn(
            ch: str, ts: str, oldest: str, limit: int = 10
        ) -> tuple[list[dict[str, Any]], str]:
            return [human_reply], "1234.600001"

        poller = self._make_poller_with_poll_fn(poll_fn)
        msg = {"ts": "1234.5678", "reply_count": 1, "user": "U_HUMAN"}
        assert poller._has_bot_reply("C_TEST", msg) is False

    def test_poll_fn_exception_returns_false(self) -> None:
        """_has_bot_reply returns False when poll_thread_fn raises."""

        def poll_fn(ch: str, ts: str, oldest: str, limit: int = 10) -> tuple:
            raise ConnectionError("network down")

        poller = self._make_poller_with_poll_fn(poll_fn)
        msg = {"ts": "1234.5678", "reply_count": 1, "user": "U_HUMAN"}
        assert poller._has_bot_reply("C_TEST", msg) is False


class TestCLIOneShotMode:
    """Tests for CLI integration of one-shot poll mode."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_channel_without_token_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--channel without token exits when no token stored.

        The _make_backend factory calls sys.exit(1) when no token
        is found, before run_once() is reached.
        """
        original_argv = sys.argv
        sys.argv = [
            "kiss-slack",
            "--channel",
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

    def test_channel_with_invalid_token(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """One-shot mode with invalid token prints checking message and raises.

        When a token exists but is invalid, _make_backend succeeds
        (it only loads the token), but run_once() fails at connect().
        The Web API is a local server answering ``invalid_auth`` exactly
        like slack.com does for a bad token, injected into the backend
        instance ``main()`` constructs via its ``_api_base_url``
        attribute, so the test is deterministic offline.
        """
        _save_token("xoxb-invalid-for-oneshot-test")
        server = ThreadedHTTPServer(("127.0.0.1", 0), _InvalidAuthHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}/api/"
        original_init = SlackChannelBackend.__init__

        def _init_with_local_api(
            self: SlackChannelBackend, workspace: str = "default"
        ) -> None:
            original_init(self, workspace=workspace)
            self._api_base_url = base_url

        original_argv = sys.argv
        sys.argv = [
            "kiss-slack",
            "--channel",
            "some-channel",
            "-m",
            "test-model",
        ]
        SlackChannelBackend.__init__ = _init_with_local_api  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="Failed to connect"):
                main()
        finally:
            SlackChannelBackend.__init__ = original_init  # type: ignore[method-assign]
            sys.argv = original_argv
            server.shutdown()
            server.server_close()
        out = capsys.readouterr().out
        assert "Checking Slack channel for pending messages..." in out

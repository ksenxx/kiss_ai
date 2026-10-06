# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ChannelRunner.run_once() — no mocks or test doubles.

Tests the one-shot poll mode: run_once(), _has_bot_reply(), and the CLI
integration for ``--channel``.  Every Slack call goes through the real
``SlackChannelBackend`` to a local Web API emulator selected with the
supported ``KISS_SLACK_BASE_URL`` setting.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import ChannelRunner
from kiss.agents.third_party_agents.irc.irc_sea import IRCChannelBackend
from kiss.agents.third_party_agents.slack.slack_sea import (
    SlackChannelBackend,
    _save_token,
    main,
)
from kiss.tests.agents.third_party_agents.recording_http import RecordingServer, recording_server
from kiss.tests.agents.third_party_agents.slack_invalid_auth import InvalidAuthHandler

BOT_USER_ID = "U_BOT"
PARENT_TS = "1234.5678"


class _SlackThreadApiHandler(InvalidAuthHandler):
    """Local Slack Web API: ``auth.test`` succeeds, ``conversations.replies`` is scripted.

    ``replies`` is the JSON body returned for ``conversations.replies``;
    every other method answers ``invalid_auth`` like the parent.
    """

    replies: dict[str, Any] = {}

    def _respond(self) -> None:
        method = self.path.split("?", 1)[0].rsplit("/", 1)[-1]
        if method not in ("auth.test", "conversations.replies"):
            super()._respond()
            return
        cast(RecordingServer, self.server).requests.append(
            {"method": self.command, "path": self.path.split("?", 1)[0]}
        )
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if method == "auth.test":
            body: dict[str, Any] = {"ok": True, "user_id": BOT_USER_ID, "user": "bot", "team": "T1"}
        else:
            body = type(self).replies
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def _connected_slack_runner(
    monkeypatch: pytest.MonkeyPatch, replies: dict[str, Any]
) -> Iterator[tuple[ChannelRunner, RecordingServer]]:
    """Yield a ChannelRunner over a real, connected SlackChannelBackend.

    The backend talks to a loopback emulator whose
    ``conversations.replies`` answer is *replies*.
    """
    handler = type("_Handler", (_SlackThreadApiHandler,), {"replies": replies})
    with recording_server(handler) as server:
        monkeypatch.setenv("KISS_SLACK_BASE_URL", server.base_url)
        _save_token("xoxb-thread-test")
        backend = SlackChannelBackend()
        assert backend.connect() is True, backend.connection_info
        yield ChannelRunner(backend=backend, channel_name="test", agent_name="test"), server


def _replies_paths(server: RecordingServer) -> list[str]:
    """Return the recorded request paths other than the connect-time ``auth.test``."""
    return [r["path"] for r in server.requests if r["path"] != "/api/auth.test"]


class TestRunOnceConnectFailure:
    """Tests for run_once() when backend connection fails."""

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
    """Tests for ChannelRunner._has_bot_reply() through the real Slack backend."""

    def test_backend_without_thread_polling_returns_false(self) -> None:
        """_has_bot_reply returns False when the backend has no poll_thread_messages.

        ``IRCChannelBackend`` is a real backend without thread support,
        so the runner's ``_poll_thread_fn`` is ``None``.
        """
        poller = ChannelRunner(backend=IRCChannelBackend(), channel_name="test", agent_name="test")
        assert poller._poll_thread_fn is None
        msg = {"ts": PARENT_TS, "reply_count": 5, "user": "U_HUMAN"}
        assert poller._has_bot_reply("C_TEST", msg) is False

    def test_zero_reply_count_does_not_poll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_has_bot_reply returns False without a Slack call when the message has no replies."""
        with _connected_slack_runner(monkeypatch, {"ok": True, "messages": []}) as (poller, server):
            msg = {"ts": PARENT_TS, "reply_count": 0, "user": "U_HUMAN"}
            assert poller._has_bot_reply("C_TEST", msg) is False
            assert _replies_paths(server) == []

    def test_no_ts_does_not_poll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_has_bot_reply returns False without a Slack call when the message has no ts."""
        with _connected_slack_runner(monkeypatch, {"ok": True, "messages": []}) as (poller, server):
            msg = {"reply_count": 3, "user": "U_HUMAN"}
            assert poller._has_bot_reply("C_TEST", msg) is False
            assert _replies_paths(server) == []

    def test_no_bot_reply_in_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_has_bot_reply returns False when the thread holds only human replies."""
        replies = {
            "ok": True,
            "messages": [
                {"user": "U_HUMAN", "ts": PARENT_TS, "text": "parent"},
                {"user": "U_OTHER_HUMAN", "ts": "1234.6000", "text": "ok"},
            ],
        }
        with _connected_slack_runner(monkeypatch, replies) as (poller, server):
            msg = {"ts": PARENT_TS, "reply_count": 1, "user": "U_HUMAN"}
            assert poller._has_bot_reply("C_TEST", msg) is False
            assert _replies_paths(server) == ["/api/conversations.replies"]

    def test_bot_reply_in_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_has_bot_reply returns True when the signed-in bot user posted in the thread."""
        replies = {
            "ok": True,
            "messages": [
                {"user": "U_HUMAN", "ts": PARENT_TS, "text": "parent"},
                {"user": BOT_USER_ID, "ts": "1234.7000", "text": "done"},
            ],
        }
        with _connected_slack_runner(monkeypatch, replies) as (poller, server):
            msg = {"ts": PARENT_TS, "reply_count": 1, "user": "U_HUMAN"}
            assert poller._has_bot_reply("C_TEST", msg) is True
            assert _replies_paths(server) == ["/api/conversations.replies"]

    def test_slack_error_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_has_bot_reply returns False when Slack answers conversations.replies with an error."""
        replies = {"ok": False, "error": "thread_not_found"}
        with _connected_slack_runner(monkeypatch, replies) as (poller, server):
            msg = {"ts": PARENT_TS, "reply_count": 1, "user": "U_HUMAN"}
            assert poller._has_bot_reply("C_TEST", msg) is False
            assert _replies_paths(server) == ["/api/conversations.replies"]


class TestCLIOneShotMode:
    """Tests for CLI integration of one-shot poll mode."""

    def test_channel_without_token_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
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
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One-shot mode with invalid token prints checking message and raises.

        When a token exists but is invalid, _make_backend succeeds
        (it only loads the token), but run_once() fails at connect().
        The Web API is a local server answering ``invalid_auth`` exactly
        like slack.com does for a bad token, selected through the
        supported ``KISS_SLACK_BASE_URL`` setting, so the test is
        deterministic offline.
        """
        _save_token("xoxb-invalid-for-oneshot-test")
        with recording_server(InvalidAuthHandler) as server:
            monkeypatch.setenv("KISS_SLACK_BASE_URL", server.base_url)
            monkeypatch.setattr(
                sys, "argv", ["kiss-slack", "--channel", "some-channel", "-m", "test-model"]
            )
            with pytest.raises(RuntimeError, match="Failed to connect"):
                main()
            assert [r["path"] for r in server.requests] == ["/api/auth.test"]
        out = capsys.readouterr().out
        assert "Checking Slack channel for pending messages..." in out

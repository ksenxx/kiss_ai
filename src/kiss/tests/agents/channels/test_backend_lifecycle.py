from __future__ import annotations

import queue
import threading
import time
from typing import Any, cast

from kiss.agents.third_party_agents.irc_agent import IRCChannelBackend
from kiss.agents.third_party_agents.line_agent import LineChannelBackend
from kiss.agents.third_party_agents.slack_agent import SlackChannelBackend
from kiss.agents.third_party_agents.synology_chat_agent import SynologyChatChannelBackend
from kiss.agents.third_party_agents.whatsapp_agent import WhatsAppChannelBackend
from kiss.agents.third_party_agents.zalo_agent import ZaloChannelBackend


class _FakeSlackClient:
    def __init__(self) -> None:
        self.calls = 0

    def conversations_replies(self, *, channel: str, ts: str, limit: int) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {"messages": [{"ts": "1", "user": "other", "text": "old"}]}
        if self.calls == 2:
            return {"messages": [{"ts": "1", "user": "other", "text": "old"}]}
        return {"messages": [{"ts": "2", "user": "u1", "text": "reply"}]}


class _FakeSocket:
    def __init__(self) -> None:
        self.timeout: float | None = None
        self.shutdown_called = False
        self.closed = False

    def settimeout(self, value: float | None) -> None:
        self.timeout = value

    def recv(self, size: int) -> bytes:
        raise OSError("closed")

    def shutdown(self, how: int) -> None:
        self.shutdown_called = True

    def close(self) -> None:
        self.closed = True


def test_slack_wait_for_reply_honors_timeout() -> None:
    backend = SlackChannelBackend()
    backend._client = cast(Any, _FakeSlackClient())
    assert backend.wait_for_reply("c", "t", "missing", timeout_seconds=0.01) is None


def test_slack_wait_for_reply_returns_new_matching_message() -> None:
    backend = SlackChannelBackend()
    backend._client = cast(Any, _FakeSlackClient())
    assert backend.wait_for_reply("c", "t", "u1", timeout_seconds=0.1) == "reply"


def test_whatsapp_disconnect_stops_server() -> None:
    backend = WhatsAppChannelBackend()
    assert backend._start_webhook_server(port=18080)
    assert backend._webhook_server is not None
    backend.disconnect()
    assert backend._webhook_server is None
    assert backend._webhook_thread is None


def test_webhook_connect_failure_is_reported() -> None:
    backend = LineChannelBackend()
    backend._message_queue = queue.Queue()
    assert backend._start_webhook_server(port=18083)
    conflict = LineChannelBackend()
    assert not conflict._start_webhook_server(port=18083)
    assert "bind failed" in conflict.connection_info.lower()
    backend.disconnect()


def test_synology_disconnect_stops_server() -> None:
    backend = SynologyChatChannelBackend()
    assert backend._start_webhook_server(port=18081)
    backend.disconnect()
    assert backend._webhook_server is None
    assert backend._webhook_thread is None


def test_zalo_disconnect_stops_server() -> None:
    backend = ZaloChannelBackend()
    assert backend._start_webhook_server(port=18082)
    backend.disconnect()
    assert backend._webhook_server is None
    assert backend._webhook_thread is None


def test_irc_disconnect_closes_socket_and_joins_thread() -> None:
    backend = IRCChannelBackend()
    fake_sock = _FakeSocket()
    backend._sock = cast(Any, fake_sock)
    thread = threading.Thread(target=lambda: time.sleep(0.01))
    thread.start()
    backend._reader_thread = thread
    backend.disconnect()
    assert fake_sock.shutdown_called
    assert fake_sock.closed
    assert backend._reader_thread is None

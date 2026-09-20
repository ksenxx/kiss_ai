# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
from __future__ import annotations

import queue
import threading
import time
from typing import Any, cast

from kiss.agents.third_party_agents.irc_sea import IRCChannelBackend
from kiss.agents.third_party_agents.line_sea import LineChannelBackend
from kiss.agents.third_party_agents.synology_sea import SynologyChatChannelBackend
from kiss.agents.third_party_agents.zalo_sea import ZaloChannelBackend


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


def test_webhook_connect_failure_is_reported() -> None:
    backend = LineChannelBackend()
    backend._message_queue = queue.Queue()
    assert backend._start_webhook_server(port=0)
    try:
        assert backend._webhook_server is not None
        port = int(backend._webhook_server.server_address[1])
        conflict = LineChannelBackend()
        assert not conflict._start_webhook_server(port=port)
        assert "bind failed" in conflict.connection_info.lower()
    finally:
        backend.disconnect()


def test_synology_disconnect_stops_server() -> None:
    backend = SynologyChatChannelBackend()
    assert backend._start_webhook_server(port=0)
    backend.disconnect()
    assert backend._webhook_server is None
    assert backend._webhook_thread is None


def test_zalo_disconnect_stops_server() -> None:
    backend = ZaloChannelBackend()
    assert backend._start_webhook_server(port=0)
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

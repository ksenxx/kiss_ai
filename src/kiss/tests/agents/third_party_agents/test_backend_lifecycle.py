# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
from __future__ import annotations

import socket
from pathlib import Path

from kiss.agents.third_party_agents.irc.irc_sea import IRCChannelBackend, _config
from kiss.agents.third_party_agents.line.line_sea import LineChannelBackend
from kiss.agents.third_party_agents.synology.synology_sea import SynologyChatChannelBackend
from kiss.agents.third_party_agents.zalo.zalo_sea import ZaloChannelBackend


def test_webhook_connect_failure_is_reported() -> None:
    backend = LineChannelBackend()
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


def _recv_until_eof(conn: socket.socket) -> bytes:
    """Read *conn* until the peer closes it and return everything received."""
    data = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return data
        data += chunk


def test_irc_disconnect_closes_socket_and_joins_thread(isolated_kiss_home: Path) -> None:
    """disconnect() closes the live IRC socket and joins the reader thread.

    The backend connects to a real loopback listener; after
    ``disconnect()`` the server side reads EOF (the socket was shut
    down and closed) and the reader thread the connect started is gone.
    """
    with socket.create_server(("127.0.0.1", 0)) as listener:
        listener.settimeout(5.0)
        _config.save({"server": "127.0.0.1", "port": str(listener.getsockname()[1]), "nick": "bot"})
        backend = IRCChannelBackend()
        try:
            assert backend.connect() is True, backend.connection_info
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(5.0)
                reader = backend._reader_thread
                assert reader is not None and reader.is_alive()
                backend.disconnect()
                assert backend._sock is None
                assert backend._reader_thread is None
                assert not reader.is_alive()
                assert b"NICK bot\r\n" in _recv_until_eof(conn)
        finally:
            backend.disconnect()

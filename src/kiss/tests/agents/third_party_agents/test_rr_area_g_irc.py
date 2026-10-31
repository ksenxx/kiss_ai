# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the IRC reader-thread race fixes (G-RC1).

The reader thread used to answer PING via ``_send_raw``, whose
connect-on-demand path calls ``disconnect()`` — joining the reader
thread from itself raises ``RuntimeError('cannot join current
thread')``, leaks the freshly opened socket, and kills the reader with
unprocessed buffer.  ``_read_loop`` also re-read ``self._sock``, so an
outliving reader could adopt the next connection's socket.

Fixes under test, driven against a real loopback TCP listener
(``irc_test_utils.IRCServer``; no mocks or test doubles):

- the reader owns the socket it was started with and exits once
  ``self._sock`` no longer refers to it;
- PING is answered with a direct PONG on the owned socket;
- ``disconnect()`` never joins from the reader thread itself.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from kiss.agents.third_party_agents.irc.irc_sea import IRCChannelBackend, _config
from kiss.tests.agents.third_party_agents.irc_test_utils import IRCServer, read_lines
from kiss.tests.agents.third_party_agents.muse_test_utils import wait_until

pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


@pytest.fixture()
def server() -> Iterator[IRCServer]:
    """A listening IRC server for one test."""
    srv = IRCServer()
    try:
        yield srv
    finally:
        srv.close()


def _connected_backend(server: IRCServer) -> IRCChannelBackend:
    """Save config for *server* and return a connected backend."""
    _config.save({"server": "127.0.0.1", "port": str(server.port), "nick": "bot"})
    backend = IRCChannelBackend()
    assert backend.connect() is True
    return backend


class TestPingPong:
    """PING is answered directly on the reader's own socket."""

    def test_pong_sent_on_same_connection(self, server: IRCServer) -> None:
        """The PONG arrives on the original connection; no reconnect happens."""
        backend = _connected_backend(server)
        try:
            with server.accept() as conn:
                read_lines(conn, "USER ")
                conn.sendall(b"PING :tok-123\r\n")
                lines = read_lines(conn, "PONG")
                assert lines[-1] == "PONG :tok-123"
                # No second connection was opened to answer the PING.
                server.sock.settimeout(0.3)
                with pytest.raises(TimeoutError):
                    server.accept()
                reader = backend._reader_thread
                assert reader is not None and reader.is_alive()
        finally:
            backend.disconnect()

    def test_messages_still_processed_after_ping(self, server: IRCServer) -> None:
        """The reader keeps parsing PRIVMSG lines after answering a PING."""
        backend = _connected_backend(server)
        try:
            with server.accept() as conn:
                read_lines(conn, "USER ")
                conn.sendall(b"PING :alive\r\n")
                read_lines(conn, "PONG")
                conn.sendall(b":alice!u@h PRIVMSG #chan :hello bot\r\n")
                assert wait_until(lambda: not backend._message_queue.empty(), timeout=6.0)
                messages, _ = backend.poll_messages("#chan", "")
                assert [m["text"] for m in messages] == ["hello bot"]
                assert messages[0]["user"] == "alice"
        finally:
            backend.disconnect()


class TestReconnect:
    """A reconnect retires the old reader instead of splitting the stream."""

    def test_old_reader_exits_and_new_socket_is_not_adopted(self, server: IRCServer) -> None:
        """After connect() twice, only the new reader consumes the new socket."""
        backend = _connected_backend(server)
        try:
            with server.accept() as conn1:
                read_lines(conn1, "USER ")
                reader1 = backend._reader_thread
                assert reader1 is not None

                assert backend.connect() is True
                with server.accept() as conn2:
                    read_lines(conn2, "USER ")
                    reader2 = backend._reader_thread
                    assert reader2 is not None and reader2 is not reader1
                    assert wait_until(lambda: not reader1.is_alive(), timeout=6.0)

                    conn2.sendall(b":carol!u@h PRIVMSG #chan :after reconnect\r\n")
                    assert wait_until(lambda: not backend._message_queue.empty(), timeout=6.0)
                    messages, _ = backend.poll_messages("", "")
                    assert [m["text"] for m in messages] == ["after reconnect"]
        finally:
            backend.disconnect()


class TestDisconnectGuard:
    """disconnect() must never join the calling reader thread."""

    def test_disconnect_on_reader_thread_does_not_raise(self) -> None:
        """A disconnect running on the registered reader thread is safe.

        Pre-fix this raised ``RuntimeError('cannot join current
        thread')``.  The guard is exercised on a real thread that
        registers itself as the reader before disconnecting.
        """
        backend = IRCChannelBackend()
        errors: list[BaseException] = []

        def run() -> None:
            backend._reader_thread = threading.current_thread()
            try:
                backend.disconnect()
            except BaseException as e:  # pragma: no cover - the pre-fix bug
                errors.append(e)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert errors == []
        # The guard skips the join; the reader slot is left for the
        # next connect() to overwrite.
        assert backend._reader_thread is thread

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

Fixes under test, driven against a REAL TCP IRC server in this file
(no mocks or test doubles):

- the reader owns the socket it was started with and exits once
  ``self._sock`` no longer refers to it;
- PING is answered with a direct PONG on the owned socket;
- ``disconnect()`` never joins from the reader thread itself.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.third_party_agents.irc_agent import IRCChannelBackend, _config


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path) -> Iterator[Path]:
    """Point KISS_HOME at a fresh temp dir so tests never touch ~/.kiss."""
    saved = os.environ.get("KISS_HOME")
    home = tmp_path / "kiss_home"
    os.environ["KISS_HOME"] = str(home)
    try:
        yield home
    finally:
        if saved is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = saved


def _read_lines(conn: socket.socket, until: str, timeout: float = 5.0) -> list[str]:
    """Read CRLF lines from *conn* until one starts with *until*."""
    conn.settimeout(timeout)
    buf = ""
    lines: list[str] = []
    while True:
        chunk = conn.recv(4096).decode("utf-8", errors="replace")
        if not chunk:
            raise AssertionError(f"connection closed before {until!r}; got {lines}")
        buf += chunk
        while "\r\n" in buf:
            line, buf = buf.split("\r\n", 1)
            lines.append(line)
            if line.startswith(until):
                return lines


def _wait_until(predicate, timeout: float = 6.0) -> bool:
    """Poll *predicate* until true or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class _IRCServer:
    """Minimal real TCP server accepting IRC client connections."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.sock.settimeout(5.0)
        self.port = self.sock.getsockname()[1]

    def accept(self) -> socket.socket:
        """Accept one client connection."""
        conn, _ = self.sock.accept()
        return conn

    def close(self) -> None:
        """Close the listening socket."""
        self.sock.close()


@pytest.fixture()
def server() -> Iterator[_IRCServer]:
    """A listening IRC server for one test."""
    srv = _IRCServer()
    try:
        yield srv
    finally:
        srv.close()


def _connected_backend(server: _IRCServer) -> IRCChannelBackend:
    """Save config for *server* and return a connected backend."""
    _config.save({"server": "127.0.0.1", "port": str(server.port), "nick": "bot"})
    backend = IRCChannelBackend()
    assert backend.connect() is True
    return backend


class TestPingPong:
    """PING is answered directly on the reader's own socket."""

    def test_pong_sent_on_same_connection(self, server: _IRCServer) -> None:
        """The PONG arrives on the original connection; no reconnect happens."""
        backend = _connected_backend(server)
        try:
            conn = server.accept()
            _read_lines(conn, "USER ")
            conn.sendall(b"PING :tok-123\r\n")
            lines = _read_lines(conn, "PONG")
            assert lines[-1] == "PONG :tok-123"
            # No second connection was opened to answer the PING.
            server.sock.settimeout(0.3)
            with pytest.raises(TimeoutError):
                server.accept()
            reader = backend._reader_thread
            assert reader is not None and reader.is_alive()
            conn.close()
        finally:
            backend.disconnect()

    def test_messages_still_processed_after_ping(self, server: _IRCServer) -> None:
        """The reader keeps parsing PRIVMSG lines after answering a PING."""
        backend = _connected_backend(server)
        try:
            conn = server.accept()
            _read_lines(conn, "USER ")
            conn.sendall(b"PING :alive\r\n")
            _read_lines(conn, "PONG")
            conn.sendall(b":alice!u@h PRIVMSG #chan :hello bot\r\n")
            assert _wait_until(lambda: not backend._message_queue.empty())
            messages, _ = backend.poll_messages("#chan", "")
            assert [m["text"] for m in messages] == ["hello bot"]
            assert messages[0]["user"] == "alice"
            conn.close()
        finally:
            backend.disconnect()


class TestReconnect:
    """A reconnect retires the old reader instead of splitting the stream."""

    def test_old_reader_exits_and_new_socket_is_not_adopted(
        self, server: _IRCServer
    ) -> None:
        """After connect() twice, only the new reader consumes the new socket."""
        backend = _connected_backend(server)
        try:
            conn1 = server.accept()
            _read_lines(conn1, "USER ")
            reader1 = backend._reader_thread
            assert reader1 is not None

            assert backend.connect() is True
            conn2 = server.accept()
            _read_lines(conn2, "USER ")
            reader2 = backend._reader_thread
            assert reader2 is not None and reader2 is not reader1
            assert _wait_until(lambda: not reader1.is_alive())

            conn2.sendall(b":carol!u@h PRIVMSG #chan :after reconnect\r\n")
            assert _wait_until(lambda: not backend._message_queue.empty())
            messages, _ = backend.poll_messages("", "")
            assert [m["text"] for m in messages] == ["after reconnect"]
            conn1.close()
            conn2.close()
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

    def test_disconnect_from_other_thread_joins_reader(
        self, server: _IRCServer
    ) -> None:
        """A normal disconnect still joins and clears the reader thread."""
        backend = _connected_backend(server)
        conn = server.accept()
        _read_lines(conn, "USER ")
        reader = backend._reader_thread
        assert reader is not None
        backend.disconnect()
        assert backend._reader_thread is None
        assert not reader.is_alive()
        conn.close()

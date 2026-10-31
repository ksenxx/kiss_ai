# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A loopback IRC listener the test thread serves itself.

``IRCChannelBackend.connect()`` returns once it has sent ``NICK`` and
``USER``; the kernel has queued the TCP connection on the listener by
then, so a test calls the backend first and then :meth:`IRCServer.accept`
and :func:`read_lines` on its own thread.  There is no server thread, so
nothing but a socket to close.
"""

from __future__ import annotations

import socket


def read_lines(conn: socket.socket, until: str, timeout: float = 5.0) -> list[str]:
    """Read CRLF lines from *conn* until one starts with *until*.

    Args:
        conn: An accepted client connection.
        until: Prefix of the last line to read (e.g. ``"USER "``).
        timeout: Bound on each ``recv``.

    Returns:
        Every line read, the matching one last.

    Raises:
        AssertionError: The client closed the connection first.
    """
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


class IRCServer:
    """A listening loopback TCP socket for IRC client connections."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.sock.settimeout(5.0)
        self.port: int = self.sock.getsockname()[1]

    def accept(self) -> socket.socket:
        """Accept one client connection, waiting at most 5 s."""
        conn, _ = self.sock.accept()
        return conn

    def close(self) -> None:
        """Close the listening socket."""
        self.sock.close()

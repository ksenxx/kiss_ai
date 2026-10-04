# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``hold_loopback_port`` waits until the fixed port can really be bound.

The OAuth fixtures bind a fixed loopback port (53682 / 53683) under an
inter-process lock.  The lock cannot keep unrelated connections off the
port: when a client connection of some other test happened to get that
ephemeral port, its TIME_WAIT (60 s, no ``SO_REUSEADDR``) makes even a
``SO_REUSEADDR`` bind fail with ``EADDRINUSE`` on Linux, which took all
six Slack PKCE tests down at once in a loaded run.  These tests build
that TIME_WAIT with real sockets and check the helper against the live
kernel: nothing is mocked.  Every port under test is read from a socket
that still holds it (no close-then-rebind gap another process could
slip into).
"""

import socket
import sys
import threading
import time

import pytest

from kiss.tests.conftest import hold_loopback_port, wait_until_loopback_port_bindable

LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux", reason="only Linux refuses a SO_REUSEADDR bind over a TIME_WAIT",
)


def _port(sock: socket.socket) -> int:
    return int(sock.getsockname()[1])


def _leave_client_time_wait() -> int:
    """Close a client connection first so its local port sits in TIME_WAIT; return that port."""
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.socket()
        client.connect(server.getsockname())
        port = _port(client)
        accepted, _ = server.accept()
        client.close()  # the active closer keeps the TIME_WAIT
        accepted.close()
        return port


def _bind_with_reuseaddr(port: int) -> None:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def _listener() -> socket.socket:
    """A listener on a fresh loopback port that denies every other bind.

    ``SO_REUSEADDR`` on POSIX is what the OAuth servers set; on Windows
    that flag would let another ``SO_REUSEADDR`` socket bind through, so
    the blocker takes ``SO_EXCLUSIVEADDRUSE`` there (as
    ``occupy_loopback_port`` does).
    """
    blocker = socket.socket()
    if sys.platform == "win32":
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    return blocker


@LINUX_ONLY
def test_a_client_time_wait_refuses_the_reuseaddr_bind() -> None:
    """The premise: the kernel refuses the bind while the TIME_WAIT lives."""
    port = _leave_client_time_wait()
    with pytest.raises(OSError):
        _bind_with_reuseaddr(port)
    assert wait_until_loopback_port_bindable(port, timeout=0.0) is False


def test_a_bindable_port_returns_at_once() -> None:
    """A port held only by a non-listening SO_REUSEADDR socket binds right away."""
    with socket.socket() as holder:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 0))
        started = time.monotonic()
        assert wait_until_loopback_port_bindable(_port(holder)) is True
        assert time.monotonic() - started < 1.0


def test_the_wait_ends_when_the_port_frees_up() -> None:
    """A listener released part-way through the wait lets the helper return True."""
    blocker = _listener()
    try:
        port = _port(blocker)
        assert wait_until_loopback_port_bindable(port, timeout=0.0) is False
        threading.Timer(1.5, blocker.close).start()
        started = time.monotonic()
        assert wait_until_loopback_port_bindable(port, timeout=10.0) is True
        assert 1.0 < time.monotonic() - started < 6.0
    finally:
        blocker.close()


def test_a_port_that_stays_taken_times_out_false() -> None:
    blocker = _listener()
    try:
        started = time.monotonic()
        assert wait_until_loopback_port_bindable(_port(blocker), timeout=1.2) is False
        assert 1.0 < time.monotonic() - started < 4.0
    finally:
        blocker.close()


def test_hold_loopback_port_enters_only_once_the_port_binds() -> None:
    """The context opens after the foreign listener goes away, with the port usable."""
    blocker = _listener()
    try:
        port = _port(blocker)
        threading.Timer(1.5, blocker.close).start()
        started = time.monotonic()
        with hold_loopback_port(port):
            assert time.monotonic() - started > 1.0
            _bind_with_reuseaddr(port)
    finally:
        blocker.close()

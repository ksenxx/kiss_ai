# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A daemon-down ``run()`` must fail as fast as the connect does.

``daemon_client.run`` sends a best-effort abort-cascade ``stop`` and a
``closeTab`` from its ``finally`` block.  Both used to run even when the
connect itself had failed: on macOS ``poll()`` never reports a
never-connected Unix socket writable, so each 5-second bounded send ran
to its timeout and every "daemon unavailable" error -- including the
OpenAI-compatible API's 502 -- surfaced only after 10 seconds.  Linux
hid the defect by failing such sends immediately with ``ENOTCONN``.
"""

from __future__ import annotations

import socket
import tempfile
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.daemon_client import run
from kiss.tests.conftest import posix_only

pytestmark = posix_only("Unix-domain sockets")


def test_missing_socket_raises_connection_error_immediately(tmp_path: Path) -> None:
    """No socket file: ``ConnectionError`` names the path and arrives well under a second."""
    sock_path = tmp_path / "no-such-daemon.sock"
    started = time.monotonic()
    with pytest.raises(ConnectionError, match=str(sock_path)):
        run("hello", sock_path=sock_path)
    assert time.monotonic() - started < 2.0


def test_stale_socket_file_raises_connection_error_immediately() -> None:
    """A socket path nobody listens on is refused, not waited on."""
    # Not pytest's tmp_path: macOS caps ``sun_path`` at 104 bytes.
    with tempfile.TemporaryDirectory(prefix="dc") as short_dir:
        sock_path = Path(short_dir) / "stale.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as leftover:
            leftover.bind(str(sock_path))
        # Bound-then-closed: the file remains, no listener behind it.
        assert sock_path.exists()
        started = time.monotonic()
        with pytest.raises(ConnectionError, match="Cannot connect to the sorcar daemon"):
            run("hello", sock_path=sock_path)
        assert time.monotonic() - started < 2.0

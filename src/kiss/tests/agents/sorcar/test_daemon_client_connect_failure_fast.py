# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A daemon-down ``run()`` must fail as fast as the connect does.

``daemon_client.run`` sends a best-effort abort-cascade ``stop`` and a
``closeTab`` from its ``finally`` block.  Both used to run even when the
connect itself had failed, so each 5-second bounded send ran to its
timeout and every "daemon unavailable" error -- including the
OpenAI-compatible API's 502 -- surfaced only after 10 seconds.  With the
local WSS channel the two daemon-down shapes are a missing endpoint file
and an endpoint file whose port nobody listens on; both must be reported
as soon as the connect fails.
"""

from __future__ import annotations

import os
import re
import socket
import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar.daemon_client import run

# "Immediately" is bounded by the OS: Windows reports a refused loopback
# connect only after ~2 s of SYN retransmits (POSIX: well under 10 ms).
_FAST_SECONDS = 4.0 if sys.platform == "win32" else 2.0


def test_missing_endpoint_file_raises_connection_error_immediately(tmp_path: Path) -> None:
    """No endpoint file: ``ConnectionError`` names the path and arrives well under a second."""
    endpoint_file = tmp_path / "no-such-daemon.json"
    started = time.monotonic()
    with pytest.raises(ConnectionError, match=re.escape(str(endpoint_file))):
        run("hello", endpoint_file=endpoint_file)
    assert time.monotonic() - started < _FAST_SECONDS


def test_stale_endpoint_file_raises_connection_error_immediately(tmp_path: Path) -> None:
    """An endpoint whose port nobody listens on is refused, not waited on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as leftover:
        leftover.bind(("127.0.0.1", 0))
        port = leftover.getsockname()[1]
    # Bound-then-closed: the endpoint file remains, no listener behind it.
    endpoint_file = tmp_path / "stale.json"
    local_endpoint.write_endpoint(
        endpoint_file,
        local_endpoint.LocalEndpoint(
            url=f"wss://127.0.0.1:{port}/ws", token="stale-token", ca=None,
            pid=os.getpid(),
        ),
    )
    assert endpoint_file.exists()
    started = time.monotonic()
    with pytest.raises(ConnectionError, match="Cannot connect to the sorcar daemon"):
        run("hello", endpoint_file=endpoint_file)
    assert time.monotonic() - started < _FAST_SECONDS

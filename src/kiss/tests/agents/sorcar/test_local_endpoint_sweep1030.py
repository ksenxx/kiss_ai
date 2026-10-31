# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-30 (S11): ``local_endpoint`` contracts its docstrings now state.

* ``connect(max_size=None)`` was documented as "the ``websockets``
  default" (1 MiB), but ``None`` is passed through and lifts the limit:
  the docstring now says so, and this pins it (the daemon client relies on
  a cap it passes explicitly).
* ``write_endpoint`` no longer creates the parent directory itself: the
  endpoint lock and the atomic write both do, which this pins.

A real ``websockets`` server on loopback plays the daemon's side of the
local ``auth`` handshake.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from kiss.agents.sorcar import local_endpoint
from kiss.tests.conftest import IS_WINDOWS

_BIG_FRAME = "x" * (2 * 1024 * 1024)  # twice the websockets default limit
_TOKEN = "t" * 64


def _daemon(ws: ServerConnection) -> None:
    """Accept the local token, then send one frame larger than 1 MiB."""
    auth = json.loads(ws.recv(timeout=5))
    ws.send(json.dumps({"type": "auth_ok", "local": auth.get("token") == _TOKEN}))
    ws.send(_BIG_FRAME)
    try:
        ws.recv(timeout=5)  # hold the connection until the client is done
    except Exception:  # noqa: BLE001 - the client closed it
        pass


@pytest.fixture
def endpoint_file(tmp_path: Path) -> Iterator[Path]:
    """An endpoint file, under a directory that does not exist yet, naming the daemon."""
    with serve(_daemon, "127.0.0.1", 0, max_size=None) as server:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.socket.getsockname()[1]
        path = tmp_path / "not" / "yet" / "sorcar-local.json"
        local_endpoint.write_endpoint(
            path,
            local_endpoint.LocalEndpoint(
                url=f"ws://127.0.0.1:{port}", token=_TOKEN, ca=None, pid=1,
            ),
        )
        yield path
    server_thread.join(5)


def test_write_endpoint_creates_the_directory_with_a_private_file(endpoint_file: Path) -> None:
    """The file lands in a fresh directory, mode 0600 (POSIX modes only), and reads back."""
    assert IS_WINDOWS or endpoint_file.stat().st_mode & 0o777 == 0o600
    endpoint = local_endpoint.read_endpoint(endpoint_file)
    assert endpoint is not None and endpoint.token == _TOKEN


def test_connect_without_max_size_accepts_any_frame_size(endpoint_file: Path) -> None:
    """``max_size=None`` (the default) receives a 2 MiB frame."""
    ws = local_endpoint.connect(endpoint_file, open_timeout=5.0)
    try:
        assert ws.recv(timeout=5) == _BIG_FRAME
    finally:
        ws.close()


def test_connect_with_max_size_caps_the_frame(endpoint_file: Path) -> None:
    """An explicit 1 MiB cap refuses the same frame (close code 1009)."""
    ws = local_endpoint.connect(endpoint_file, open_timeout=5.0, max_size=1024 * 1024)
    try:
        with pytest.raises(ConnectionClosed) as closed:
            ws.recv(timeout=5)
        assert closed.value.rcvd is None or closed.value.rcvd.code == 1009
        assert closed.value.sent is not None and closed.value.sent.code == 1009
    finally:
        ws.close()

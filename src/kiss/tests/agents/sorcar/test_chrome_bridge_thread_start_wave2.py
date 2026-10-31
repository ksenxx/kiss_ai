# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A bridge whose server thread cannot start fails as a ``ChromeBridgeError`` and cleans up.

``ChromeBridge.start`` binds the loopback socket, publishes
``bridge-config.json`` and then starts the thread that serves the
socket.  A ``RuntimeError`` from that thread start (thread exhaustion)
escaped ``start()`` and ``start_if_enabled()`` as a non-bridge error,
left the socket bound with nobody serving it, and left the extension
pointed at that dead port.  The thread is starved for real by lowering
``RLIMIT_NPROC`` (skipped where the limit does not apply).
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.chrome_bridge import (
    CONFIG_FILE_NAME,
    ChromeBridge,
    ChromeBridgeError,
    extension_dir,
)
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved

pytestmark = pytest.mark.skipif(
    not thread_start_can_be_starved(), reason="Thread.start cannot be starved on this host",
)


@pytest.fixture
def kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A private ``KISS_HOME`` (extension dir and config file); no singleton."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    ChromeBridge.shutdown()
    yield tmp_path
    ChromeBridge.shutdown()


def test_a_server_thread_that_cannot_start_frees_the_port_and_withdraws_the_config(
    kiss_home: Path,
) -> None:
    bridge = ChromeBridge(write_config=True)
    with nproc_limit_lowered_to_one(), pytest.raises(ChromeBridgeError, match="thread"):
        bridge.start(takeover=True)

    assert bridge.port != 0
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", bridge.port), timeout=2).close()
    assert not (extension_dir() / CONFIG_FILE_NAME).exists()
    # Nothing is left half-started: the next start serves normally.
    bridge.start(takeover=True)
    try:
        socket.create_connection(("127.0.0.1", bridge.port), timeout=2).close()
        assert (extension_dir() / CONFIG_FILE_NAME).exists()
    finally:
        bridge.stop()

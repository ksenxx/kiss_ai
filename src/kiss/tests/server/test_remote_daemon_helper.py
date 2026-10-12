# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the :mod:`kiss.tests.remote_daemon` helper itself.

The helper starts a real second daemon in a subprocess; these tests
check that the parent learns the port from the child's own endpoint
record (not from a stale one left by an earlier daemon of the same
home) and that a child dying before it seeds the database is reported.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from kiss.agents.sorcar import local_endpoint
from kiss.tests.remote_daemon import start_remote_daemon


def test_port_comes_from_the_child_endpoint_record(tmp_path: Path) -> None:
    """A stale endpoint file of another pid is ignored; the child's own port is used."""
    home = tmp_path / "home"
    home.mkdir()
    stale = home / local_endpoint.LOCAL_ENDPOINT_FILE
    stale.write_text(json.dumps(
        {"url": "wss://127.0.0.1:1/ws", "token": "stale", "ca": None, "pid": 1},
    ))
    remote = start_remote_daemon(home, password="pw", timeout=60)
    try:
        assert remote.port != 1
        assert remote.url == f"https://127.0.0.1:{remote.port}"
        endpoint = local_endpoint.read_endpoint(stale)
        assert endpoint is not None
        assert endpoint.pid == remote.process.pid
        assert endpoint.url == f"wss://127.0.0.1:{remote.port}/ws"
        with socket.create_connection(("127.0.0.1", remote.port), timeout=5):
            pass
    finally:
        remote.stop()
    assert remote.process.poll() is not None


def test_child_that_dies_before_seeding_is_reported(tmp_path: Path) -> None:
    """Two seed rows forcing the same task id break the child before it prints."""
    home = tmp_path / "home"
    seed = [
        {"chat_id": "c", "task": "a", "task_id": "dup"},
        {"chat_id": "c", "task": "b", "task_id": "dup"},
    ]
    with pytest.raises(RuntimeError, match="before seeding"):
        start_remote_daemon(home, seed=seed, timeout=60)

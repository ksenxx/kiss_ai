# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the :mod:`kiss.tests.remote_daemon` helper itself.

The helper starts a real second daemon in a subprocess; these tests
check that the parent learns the port from the child's own endpoint
record (not from a stale one left by an earlier daemon of the same
home), that a child dying before it seeds the database is reported and
that a child which does not start is reaped by the helper.
"""

from __future__ import annotations

import gc
import json
import os
import socket
import subprocess
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
        # The daemon's own pid; on Windows Popen.pid is the venv launcher's.
        assert endpoint.pid == remote.pid
        if os.name != "nt":
            assert remote.pid == remote.process.pid
        assert endpoint.url == f"ws://127.0.0.1:{remote.port}/ws"
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


def test_a_child_that_does_not_start_is_reaped(tmp_path: Path) -> None:
    """The port is taken: the child exits at its bind and the helper leaves nothing of it.

    The caller gets no handle to stop, so the child was left to the
    garbage collector, its pipe open.
    """
    home = tmp_path / "home"
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(RuntimeError, match="remote daemon exited with"):
            start_remote_daemon(home, port=taken.getsockname()[1], timeout=60)
    children = [
        child for child in gc.get_objects()
        if isinstance(child, subprocess.Popen) and isinstance(child.args, list)
        and str(home / "seed.json") in child.args
    ]
    assert children, "the test no longer finds the child it is about"
    for child in children:
        assert child.returncode is not None
        assert child.stdout is not None and child.stdout.closed

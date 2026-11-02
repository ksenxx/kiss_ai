"""The SimpleX test server fixture releases every descriptor it opened.

``websockets.sync.server.Server`` (15.x) allocates a POSIX shutdown pipe in
its constructor and ``shutdown()`` only writes to it, so a fixture that
merely shuts the server down and joins its thread leaked two pipe fds per
test.  Three fixture lifecycles here must leave the process's descriptor
table as it was.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest

from kiss.tests.agents.third_party_agents.test_simplex_agent import simplex_server

_FD_DIR = Path("/proc/self/fd")


def _open_fds() -> dict[int, str]:
    """Return ``{fd: target}`` for every descriptor of this process."""
    found: dict[int, str] = {}
    for entry in _FD_DIR.iterdir():
        try:
            found[int(entry.name)] = os.readlink(entry)
        except FileNotFoundError:
            pass
    return found


@pytest.mark.skipif(not _FD_DIR.is_dir(), reason="needs /proc/self/fd accounting")
def test_fixture_lifecycle_leaves_no_descriptor_behind() -> None:
    """Starting and stopping the fixture three times adds no fd, pipe or otherwise."""
    start_server = simplex_server.__wrapped__  # type: ignore[attr-defined]
    gc.collect()
    before = _open_fds()
    for _ in range(3):
        lifecycle = start_server()
        assert next(lifecycle).startswith("ws://127.0.0.1:")
        lifecycle.close()
    gc.collect()
    leaked = {fd: target for fd, target in _open_fds().items() if fd not in before}
    assert leaked == {}

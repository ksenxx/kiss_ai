# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``RemoteAccessServer.start()`` must disarm its stall watchdog on exit.

``start()`` arms :func:`kiss.server.stall_watchdog.start_stall_watchdog`
so a GIL-holding stall still leaves a stack dump on stderr.  Before the
fix the heartbeat thread outlived the server: every in-process
``start()`` in the test suite leaked one, and when pytest closed its
captured stderr at session end each leaked thread died with
``ValueError: I/O operation on closed file``.
"""

from __future__ import annotations

import threading
from pathlib import Path

from kiss.core.vscode_config import save_config
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server._blocking_start import close_leaked_listeners
from kiss.tests.server.test_blocking_start_releases_uds import (
    _make_server,
    _start_on_thread,
    _stop_thread,
)


def _watchdog_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "stall-watchdog"]


@requires_unix_sockets
def test_stopped_start_leaves_no_watchdog_thread(
    tmp_path: Path, uds_tmp_path: Path,
) -> None:
    save_config({"remote_password": ""})
    before = _watchdog_threads()
    server = _make_server(tmp_path, uds_tmp_path, "watched")
    thread = _start_on_thread(server)
    try:
        armed = [t for t in _watchdog_threads() if t not in before]
        assert len(armed) == 1, "start() did not arm exactly one stall watchdog"
    finally:
        _stop_thread(server, thread)
        close_leaked_listeners(server)

    assert not armed[0].is_alive(), "start() returned with its stall watchdog still armed"

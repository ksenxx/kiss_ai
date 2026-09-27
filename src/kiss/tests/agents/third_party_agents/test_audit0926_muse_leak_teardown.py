# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``teardown_muse_env`` returns only after the daemon process has exited.

The daemon unlinks its socket when its serve loop ends and then needs
tens of milliseconds more to shut its interpreter down.  The helper
used to return as soon as the socket vanished, so the subprocess
reaper's teardown sweep found the daemon still running, ``SIGTERM``-ed
it and reported a ``LeakedSubprocessWarning`` for every Muse test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.third_party_agents.muse_auth._common import socket_path
from kiss.agents.third_party_agents.muse_auth.client import ensure_daemon
from kiss.tests.agents.third_party_agents.muse_test_utils import (
    _daemon_pid,
    _process_running,
    setup_muse_env,
    teardown_muse_env,
)


def test_teardown_waits_for_daemon_exit(
    isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Right after teardown the socket is gone and the daemon has exited."""
    setup_muse_env(monkeypatch, {"defaults": {"read": "allow", "write": "ask"}})
    for _ in range(3):  # the exit lag is short: repeat to make a miss visible
        ensure_daemon()
        pid = _daemon_pid()
        assert pid is not None and _process_running(pid)
        teardown_muse_env()
        assert not socket_path().exists()
        assert not _process_running(pid)
    assert _daemon_pid() is None
    teardown_muse_env()  # no daemon running: a clean no-op
    assert not _process_running(2**22 + 1)  # above Linux's pid_max: no such process

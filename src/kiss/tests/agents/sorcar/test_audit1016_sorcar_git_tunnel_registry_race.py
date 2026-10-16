# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``tunnel_port`` vs. ``close_tunnels``: an ssh forward opened after the
registry change must not outlive the cache (audit 2026-10-16, sorcar-git).

``tunnel_port`` takes the tunnel out of ``_tunnels`` under
``_tunnels_lock`` and opens it outside the lock.  When ``close_tunnels``
(a ``/machine add`` or ``/machine remove`` through ``save_machines``)
runs in that window it clears the cache and closes a tunnel that has no
process yet; the opener then spawns ``ssh -N`` on a tunnel nothing
references any more, so the process ran until the daemon exited.  The
fix checks the cache again after the open and closes the orphan; the
window is widened with the suite's ``KISS_RACE_DELAY`` hook so the race
is deterministic.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.tests.agents.sorcar.test_machines import fake_ssh_only

_FAKE_SSH = '''#!__PYTHON__
"""Stand-in for ssh: ``-N -L`` binds the local port and serves nothing."""
import socket
import sys
import time

args = sys.argv[1:]
_host, local, _rhost, _remote = args[args.index("-L") + 1].split(":")
server = socket.socket()
server.bind(("127.0.0.1", int(local)))
server.listen()
while True:
    time.sleep(1)
'''


def _wait_for(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@fake_ssh_only
def test_registry_change_during_tunnel_open_closes_the_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ssh"
    script.write_text(_FAKE_SSH.replace("__PYTHON__", sys.executable), encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv(mach.SSH_CONFIG_ENV, str(tmp_path / "no-such-config"))
    monkeypatch.setenv("KISS_RACE_DELAY", "0.1")
    login = "ksen@far.example.org"
    errors: list[BaseException] = []
    ports: list[int] = []

    def opener() -> None:
        try:
            ports.append(mach.tunnel_port(login, 8787))
        except BaseException as exc:
            errors.append(exc)

    mach.close_tunnels()
    thread = threading.Thread(target=opener, name="opener")
    thread.start()
    tunnel: mach.SshTunnel | None = None
    try:
        assert _wait_for(lambda: login in mach._tunnels, 5.0)
        tunnel = mach._tunnels[login]
        # Inside the opener's window: the tunnel is cached but its ssh
        # is not started yet, so this close finds nothing to kill.
        assert tunnel.process is None
        mach.close_tunnels()
        assert mach._tunnels == {}
        thread.join(timeout=30.0)
        assert not thread.is_alive()
        # The forward opened after the registry change was closed again
        # and the caller told to start over; no ssh process is left.
        assert ports == []
        assert len(errors) == 1 and isinstance(errors[0], mach.MachineError)
        assert "the machine registry changed, try again" in str(errors[0])
        assert tunnel.process is None and tunnel.local_port == 0 and not tunnel.alive()
        assert mach._tunnels == {}
    finally:
        thread.join(timeout=30.0)
        mach.close_tunnels()
        if tunnel is not None:
            tunnel.close()

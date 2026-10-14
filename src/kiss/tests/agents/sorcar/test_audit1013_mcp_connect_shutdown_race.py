# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``MCPManager.connect`` never returns a connection that is still pending.

``connect`` registers the connection under the manager lock, then
checks ``_shut_down`` outside it.  A shutdown that began in between
(after ``_connection_for`` returned, before the check) used to make
``connect`` return the brand-new record at once, with ``ready`` clear,
``error`` empty and ``session`` ``None``: the caller saw neither a
session nor a failure.  ``disconnect_all`` stamps every registered
connection a moment later, so ``connect`` must keep waiting on
``ready`` and let that stamp (or the handshake) decide; only the
shut-down manager's never-started failure record returns at once.

The interleaving is forced with a profile hook on the two real
threads: the connecting thread parks as ``_connection_for`` returns
until the shutdown thread has raised the flag, and the shutdown thread
parks at ``disconnect_all`` long enough for the connecting thread to
reach its check.  The MCP child is a real stdio subprocess that never
speaks MCP, so the handshake cannot finish on its own.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import mcp_servers
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig


def _silent_config(pidfile: Path, name: str) -> MCPServerConfig:
    """A real stdio child that records its pid, then hangs without speaking MCP."""
    code = f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(300)"
    return MCPServerConfig(
        name=name, transport="stdio", command=sys.executable, args=("-c", code),
    )


def _kill_child(pidfile: Path) -> None:
    """Kill the silent child outright: a cancelled mid-handshake task may not reap it."""
    try:
        pid = int(pidfile.read_text())
    except (OSError, ValueError):
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, check=False)
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


class _Interleaver:
    """Profile hook that sequences the connecting and the shutdown thread."""

    def __init__(self) -> None:
        self.connection_registered = threading.Event()
        self.shutdown_flagged = threading.Event()
        self.connect_returned = threading.Event()

    def __call__(self, frame: FrameType, event: str, _arg: Any) -> None:
        code = frame.f_code
        if event == "return" and code is MCPManager._connection_for.__code__:
            self.connection_registered.set()
            self.shutdown_flagged.wait(10)
        elif event == "call" and code is MCPManager.disconnect_all.__code__:
            self.shutdown_flagged.set()
            # Long enough for the parked connecting thread to run its
            # shut-down check; short enough not to stall the test.
            self.connect_returned.wait(0.5)


def test_connect_during_shutdown_returns_a_settled_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pidfile = tmp_path / "child.pid"
    monkeypatch.setattr(mcp_servers, "CONNECT_TIMEOUT", 30.0)
    monkeypatch.setattr(mcp_servers, "_DISCONNECT_WAIT_S", 1.0)
    manager = MCPManager()  # its loop thread starts before the hook is installed
    gates = _Interleaver()
    result: dict[str, Any] = {}

    def connect() -> None:
        conn = manager.connect(_silent_config(pidfile, "silent-race"))
        # Snapshot at the moment of return: disconnect_all stamps the
        # record a little later, which would hide a premature return.
        result["ready"] = conn.ready.is_set()
        result["error"] = conn.error
        result["session"] = conn.session
        gates.connect_returned.set()

    threading.setprofile(gates)
    try:
        connector = threading.Thread(target=connect, name="connector")
        connector.start()
        assert gates.connection_registered.wait(10), "connect never registered"
        shutter = threading.Thread(target=manager.shutdown, name="shutter")
        shutter.start()
        connector.join(30)
        shutter.join(30)
    finally:
        threading.setprofile(None)
        manager.shutdown()
        _kill_child(pidfile)
    assert not connector.is_alive() and not shutter.is_alive()
    assert result["ready"], "connect returned a still-pending connection"
    assert result["error"] and result["session"] is None


if __name__ == "__main__":
    pytest.main([__file__])

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review round: a timed-out connect() must not leak the server child.

``MCPManager.connect`` removes a connection that missed CONNECT_TIMEOUT
from ``_connections`` and sets its ``stop`` event.  But a task stuck
mid-handshake (a stdio child that never speaks MCP) is not parked on
``stop.wait()``, so setting the event never unwinds it — and because
the record was just deleted, ``disconnect_all`` / ``shutdown`` can no
longer see it either.  The frozen task keeps the transport's child
process alive FOREVER: even after ``manager.shutdown()`` the silent
MCP server subprocess survives, leaking a process per timed-out
connect.

The fix tracks such stragglers in an orphan list that
``disconnect_all`` also tears down, and schedules a cancellation of
the stuck task after a short grace period so the child dies even
without an explicit shutdown.

Real end-to-end tests: a real subprocess child is spawned through the
real MCP stdio transport; no mocks.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import mcp_servers
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover — foreign-owned pid
        return True
    return True


def _wait_pid_dead(pid: int, deadline_s: float) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    return not _pid_alive(pid)


def _spawn_silent_config(pidfile: Path, name: str) -> MCPServerConfig:
    """A real child that records its pid, then hangs without speaking MCP."""
    code = (
        "import os,time; "
        f"open({str(pidfile)!r},'w').write(str(os.getpid())); "
        "time.sleep(300)"
    )
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=("-c", code),
    )


def _read_child_pid(pidfile: Path, deadline_s: float = 10.0) -> int:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            text = pidfile.read_text()
            if text:
                return int(text)
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    raise AssertionError("silent MCP child never wrote its pidfile")


def test_connect_timeout_straggler_killed_on_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shutdown() must reap a child orphaned by a connect() timeout."""
    monkeypatch.setattr(mcp_servers, "CONNECT_TIMEOUT", 1.0)
    pidfile = tmp_path / "child.pid"
    manager = MCPManager()
    child_pid = 0
    try:
        conn = manager.connect(_spawn_silent_config(pidfile, "silent-shutdown"))
        assert conn.error, "timed-out connect must stamp an error"
        child_pid = _read_child_pid(pidfile)
        assert _pid_alive(child_pid), "child should still be mid-handshake"
        manager.shutdown()
        assert _wait_pid_dead(child_pid, 20.0), (
            "silent MCP child survived manager.shutdown() — the "
            "timed-out connection's task leaked out of disconnect_all"
        )
    finally:
        manager.shutdown()
        if child_pid and _pid_alive(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except OSError:
                pass


def test_connect_timeout_straggler_killed_after_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even without shutdown, the stuck task is cancelled after a grace."""
    monkeypatch.setattr(mcp_servers, "CONNECT_TIMEOUT", 1.0)
    pidfile = tmp_path / "child.pid"
    manager = MCPManager()
    child_pid = 0
    try:
        conn = manager.connect(_spawn_silent_config(pidfile, "silent-grace"))
        assert conn.error, "timed-out connect must stamp an error"
        child_pid = _read_child_pid(pidfile)
        grace = mcp_servers._CONNECT_STRAGGLER_GRACE_S
        assert _wait_pid_dead(child_pid, grace + 15.0), (
            "silent MCP child survived past the straggler grace period — "
            "the stuck handshake task was never cancelled"
        )
    finally:
        manager.shutdown()
        if child_pid and _pid_alive(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except OSError:
                pass

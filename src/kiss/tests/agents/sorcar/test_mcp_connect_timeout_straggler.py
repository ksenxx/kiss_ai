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
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import mcp_servers
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig
from kiss.core import tool_interrupt
from kiss.core.processes import pid_alive as _pid_alive
from kiss.tests.agents.sorcar.test_mcp_connection_lifecycle import _run_under_hook


def _force_kill(pid: int) -> None:
    """Kill a leaked child outright (``SIGKILL``; ``taskkill`` on Windows)."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, check=False)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


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


@pytest.mark.slow
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
            _force_kill(child_pid)


@pytest.mark.slow
def test_stopped_connect_waiter_leaves_the_handshake_its_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent child whose connect() waiter was stopped still dies at the deadline.

    CONNECT_TIMEOUT was enforced by the waiting ``connect()`` alone: a
    Stop injected into that thread took the only deadline with it, and
    the silent child sat in the pool for ever, unready and so never
    evicted, its task never cancelled.  The handshake now carries its
    own deadline on the manager loop.  The deadline leaves the Stop time
    to land while ``connect()`` still waits: under load the child can take
    over a second to write its pid, and a waiter that timed out first
    returns without the Stop.
    """
    timeout = 5.0
    monkeypatch.setattr(mcp_servers, "CONNECT_TIMEOUT", timeout)
    pidfile = tmp_path / "child.pid"
    config = _spawn_silent_config(pidfile, "silent-stopped-waiter")
    manager = MCPManager(max_connections=1)
    child_pid = 0
    errors: list[BaseException] = []
    try:
        waiter = threading.Thread(
            target=_run_under_hook, args=(None, partial(manager.connect, config), errors),
        )
        waiter.start()
        child_pid = _read_child_pid(pidfile)
        assert waiter.ident is not None
        tool_interrupt.inject_async_exception(waiter.ident, KeyboardInterrupt)
        waiter.join(10)
        assert not waiter.is_alive() and isinstance(errors[0], KeyboardInterrupt), errors
        grace = mcp_servers._CONNECT_STRAGGLER_GRACE_S
        assert _wait_pid_dead(child_pid, timeout + grace + 15.0), (
            "silent MCP child survived its stopped connect() waiter — the "
            "handshake had no deadline of its own"
        )
        conn = manager._connections[mcp_servers._connection_key(config)]
        assert conn.finished.wait(10) and conn.error and conn.session is None
        # The failed record is evictable again: a healthy connection takes
        # its place under the pool cap instead of joining it above the cap.
        # A real server's start (importing FastMCP) can take well over the
        # 1 s used for the silent one on a loaded machine.
        monkeypatch.setattr(mcp_servers, "CONNECT_TIMEOUT", 60.0)
        healthy = MCPServerConfig(
            name="healthy", transport="stdio", command=sys.executable,
            args=("-c", "from mcp.server.fastmcp import FastMCP; FastMCP('healthy').run()"),
        )
        good = manager.connect(healthy)
        assert good.session is not None, good.error
        assert list(manager._connections.values()) == [good]
    finally:
        manager.shutdown()
        if child_pid and _pid_alive(child_pid):
            _force_kill(child_pid)


def _spawn_gated_config(pidfile: Path, go: Path, name: str) -> MCPServerConfig:
    """A real FastMCP child that records its pid, then serves only once *go* exists."""
    code = (
        "import os, time\n"
        "from mcp.server.fastmcp import FastMCP\n"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
        f"while not os.path.exists({str(go)!r}):\n"
        "    time.sleep(0.02)\n"
        "FastMCP('gated').run()\n"
    )
    return MCPServerConfig(
        name=name, transport="stdio", command=sys.executable, args=("-c", code),
    )


def _stop_as_the_orphaning_returns(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: inject the task Stop as ``_orphan_timed_out`` has just orphaned the record."""
    if event == "return" and frame.f_code is MCPManager._orphan_timed_out.__code__ and arg:
        sys.setprofile(None)
        tool_interrupt.inject_async_exception(threading.get_ident(), KeyboardInterrupt)


@pytest.mark.slow
def test_stop_as_the_timed_out_connection_is_orphaned_still_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Stop landing right after the orphaning must not leave a live, never-stopped connection.

    ``_await_ready`` orphaned the timed-out record under the lock and
    only then, on the task thread, told it to stop.  A Stop delivered
    in between left a connection in no pool whose ``stop`` was never
    set: a server slower than CONNECT_TIMEOUT but faster than the
    handshake's own deadline then connected and parked for ever, its
    child alive until the daemon's shutdown, since no eviction looks at
    the orphans.  The stop is now signalled under the lock, as the
    eviction does.
    """
    monkeypatch.setattr(mcp_servers, "CONNECT_TIMEOUT", 1.0)
    pidfile, go = tmp_path / "child.pid", tmp_path / "go"
    config = _spawn_gated_config(pidfile, go, "gated-orphan")
    manager = MCPManager()
    child_pid = 0
    errors: list[BaseException] = []
    try:
        waiter = threading.Thread(
            target=_run_under_hook,
            args=(_stop_as_the_orphaning_returns, partial(manager.connect, config), errors),
        )
        waiter.start()
        waiter.join(30)
        assert not waiter.is_alive() and isinstance(errors[0], KeyboardInterrupt), errors
        child_pid = _read_child_pid(pidfile)
        assert _pid_alive(child_pid), "child should still be waiting to serve"
        [conn] = manager._orphans
        assert conn.error == "connection timed out"
        # The handshake lands now, before the maintainer's own deadline.
        go.write_text("go", encoding="utf-8")
        assert conn.finished.wait(20), (
            "the orphaned connection parked for ever: its stop was never signalled"
        )
        assert _wait_pid_dead(child_pid, 20.0), "the orphaned connection's child survived"
    finally:
        manager.shutdown()
        if child_pid and _pid_alive(child_pid):
            _force_kill(child_pid)


@pytest.mark.slow
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
            _force_kill(child_pid)

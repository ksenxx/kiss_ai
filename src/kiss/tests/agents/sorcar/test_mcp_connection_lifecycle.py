# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""MCP connection lifecycle: pooling, eviction, reconnect and locking.

F7 (a) ``connect()`` only stopped a connection under the *same* key, and the
key is a digest of the whole config, so any config change spawned a fresh
stdio child and orphaned the previous one for the life of the process.
``shutdown()`` is reachable only from ``atexit``, so the long-lived
``kiss-web`` daemon leaked one child per (project x config revision).
(c) ``call_tool`` never reconnected, so after a mid-task server crash every
remaining call to that server returned "is not connected" for the rest of
the (possibly hour-long) task.

F8 ``FileTokenStorage.set_tokens`` performed a blocking inter-process
``flock`` from inside a coroutine on the manager's single shared event loop,
so another process holding that lock (an interactive OAuth login) stalled
every other MCP tool call in this process for up to the 305 s call timeout.

Real stdio MCP servers (real FastMCP subprocesses over the real protocol),
real signals, real file locks held by a real second process.  No mocks.
"""

from __future__ import annotations

import os
import pty
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import (
    FileTokenStorage,
    MCPManager,
    MCPServerConfig,
    _connection_key,
)

_SERVER_SCRIPT = '''
import os, sys, time

from mcp.server.fastmcp import FastMCP

open(sys.argv[1], "w").write(str(os.getpid()))

mcp = FastMCP("gsrv")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool()
def block_until(started: str, release: str) -> str:
    """Announce *started*, wait for *release* to appear, then answer."""
    open(started, "w").close()
    deadline = time.time() + 120
    while time.time() < deadline:
        if os.path.exists(release):
            return "released"
        time.sleep(0.02)
    return "timeout"


if __name__ == "__main__":
    mcp.run()
'''

#: Holds the shared token lock for 5 s so the parent can measure whether an
#: unrelated MCP tool call is blocked behind it.
_LOCK_HOLDER = '''
import sys, time
from pathlib import Path

from kiss.agents.sorcar.useful_tools import _file_lock

lock_path, ready = Path(sys.argv[1]), Path(sys.argv[2])
with _file_lock(lock_path):
    ready.write_text("held")
    time.sleep(5)
'''


@pytest.fixture
def real_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Give ``sys.stdin`` and the MCP errlog real file descriptors.

    Under pytest the std streams are capture objects whose ``fileno()``
    raises, which stops the MCP stdio transport from spawning a server at
    all.  A real pty and a real file restore them for the test.

    Only ``sys.stderr`` needs replacing: ``_enter_transport`` resolves the
    child's stderr through ``_child_errlog()`` on every spawn, so it picks
    this file up without the transport's import-time default being touched.
    """
    master_fd, slave_fd = pty.openpty()
    stdin_stream = os.fdopen(slave_fd, "r", closefd=True)
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()
        os.close(master_fd)


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    """Write the real FastMCP stdio server used by these tests."""
    script = tmp_path / "gsrv.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return script


def _config(script: Path, pid_file: Path, *extra: str) -> MCPServerConfig:
    """Build a stdio config that records the server's pid in *pid_file*."""
    return MCPServerConfig(
        name="gsrv",
        transport="stdio",
        command=sys.executable,
        args=(str(script), str(pid_file), *extra),
    )


def _wait_for_pid_file(pid_file: Path, timeout: float = 30) -> int:
    """Return the pid the freshly spawned server wrote, waiting for it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return int(pid_file.read_text())
        except (OSError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"server never wrote {pid_file}")


def _alive(pid: int) -> bool:
    """Return True while *pid* still exists."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_until_dead(pid: int, timeout: float = 20) -> bool:
    """Poll until *pid* exits, returning whether it did."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def manager() -> Iterator[MCPManager]:
    """A private manager (never the process-wide singleton), always closed."""
    mgr = MCPManager(idle_timeout_s=0.3, max_connections=2, health_interval_s=0.5)
    try:
        yield mgr
    finally:
        mgr.shutdown()


def test_superseded_config_does_not_orphan_its_child(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F7a: editing the config must not leave the old stdio child running."""
    first_pid_file = tmp_path / "pid1"
    conn = manager.connect(_config(server_script, first_pid_file))
    assert conn.session is not None, conn.error
    first_pid = _wait_for_pid_file(first_pid_file)

    time.sleep(0.5)  # exceed the idle timeout
    second_pid_file = tmp_path / "pid2"
    conn2 = manager.connect(_config(server_script, second_pid_file, "--verbose"))
    assert conn2.session is not None, conn2.error

    assert _wait_until_dead(first_pid), (
        "the superseded server process was orphaned and is still running"
    )
    assert len(manager._connections) == 1


def test_pool_is_capped(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F7a: the live pool never grows past the cap, oldest evicted first."""
    pids = []
    for i in range(3):
        pid_file = tmp_path / f"cap{i}"
        conn = manager.connect(_config(server_script, pid_file, f"--n={i}"))
        assert conn.session is not None, conn.error
        pids.append(_wait_for_pid_file(pid_file))

    assert len(manager._connections) <= 2
    assert _wait_until_dead(pids[0]), "least recently used child was not reaped"
    assert _alive(pids[2])


def test_least_recently_used_is_evicted_at_the_cap(
    server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F7a: with nothing idle, the cap still bounds the pool by LRU."""
    mgr = MCPManager(idle_timeout_s=3600, max_connections=2, health_interval_s=30)
    try:
        pids = []
        for i in range(3):
            pid_file = tmp_path / f"lru{i}"
            conn = mgr.connect(_config(server_script, pid_file, f"--n={i}"))
            assert conn.session is not None, conn.error
            pids.append(_wait_for_pid_file(pid_file))

        assert len(mgr._connections) == 2
        assert _wait_until_dead(pids[0]), "least recently used child survived"
        assert _alive(pids[1]) and _alive(pids[2])
    finally:
        mgr.shutdown()


def test_a_running_tool_call_is_not_evicted_by_the_cap(
    server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """A server executing a tool survives other agents connecting.

    The pool is capped, and every ``connect()`` evicts the oldest
    connection past the cap.  ``call_tool`` blocks for as long as the
    tool runs — up to five minutes — so the connection it is using is
    frequently the oldest one.  Tearing it down turns a valid, still
    running call into an error the agent can do nothing about.
    """
    mgr = MCPManager(idle_timeout_s=3600, max_connections=2, health_interval_s=30)
    started = tmp_path / "tool-started"
    release = tmp_path / "tool-release"
    answer: list[str] = []
    try:
        busy_pid_file = tmp_path / "busy_pid"
        busy = _config(server_script, busy_pid_file, "--busy")
        busy_key = _connection_key(busy)
        assert mgr.connect(busy).session is not None
        busy_pid = _wait_for_pid_file(busy_pid_file)

        caller = threading.Thread(
            target=lambda: answer.append(
                mgr.call_tool(
                    busy_key,
                    "block_until",
                    {"started": str(started), "release": str(release)},
                )
            ),
            name="mcp-busy-caller",
            daemon=True,
        )
        caller.start()
        deadline = time.time() + 60
        while time.time() < deadline and not started.exists():
            time.sleep(0.02)
        assert started.exists(), "the tool call never reached the server"

        # Two more agents connect, taking the pool past its cap while
        # the first server is still executing its tool.
        for i in range(2):
            other = _config(server_script, tmp_path / f"other{i}", f"--o={i}")
            assert mgr.connect(other).session is not None

        assert _alive(busy_pid), (
            "the server executing a tool call was torn down to make "
            "room in the pool"
        )
        release.write_text("go", encoding="utf-8")
        caller.join(timeout=120)
        assert not caller.is_alive()
        assert answer == ["released"], (
            f"the in-flight tool call was broken by pool eviction: {answer}"
        )

        # Once the call is done the lease is gone and the connection is
        # an ordinary eviction candidate again — the freshest one, since
        # it was in use most recently, so it takes two more connects to
        # push it out.  Without that the cap would leak a slot forever.
        for name in ("last-a", "last-b"):
            nxt = _config(server_script, tmp_path / name, f"--{name}")
            assert mgr.connect(nxt).session is not None
        assert _wait_until_dead(busy_pid), (
            "a finished connection was never evicted, so the cap leaks"
        )
        assert len(mgr._connections) <= 2
    finally:
        release.write_text("go", encoding="utf-8")
        mgr.shutdown()


def test_call_tool_on_an_unconfigured_server_reports_it(
    manager: MCPManager,
) -> None:
    """A server the manager never saw cannot be rebuilt, and says so."""
    result = manager.call_tool("nosuchserver", "add", {"a": 1, "b": 2})
    assert "is not connected" in result
    assert "nosuchserver" in result


def test_call_tool_reconnects_after_the_server_dies(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F7c: a crashed stdio server is reconnected on the next tool call."""
    pid_file = tmp_path / "crash_pid"
    config = _config(server_script, pid_file)
    key = _connection_key(config)
    conn = manager.connect(config)
    assert conn.session is not None, conn.error
    assert manager.call_tool(key, "add", {"a": 1, "b": 2}) == "3"

    pid = _wait_for_pid_file(pid_file)
    pid_file.unlink()
    os.kill(pid, 9)
    assert _wait_until_dead(pid)

    deadline = time.time() + 20
    while time.time() < deadline and conn.session is not None:
        time.sleep(0.05)
    assert conn.session is None, "connection did not notice the dead server"

    assert manager.call_tool(key, "add", {"a": 2, "b": 3}) == "5"
    assert _wait_for_pid_file(pid_file) != pid
    assert len(manager._connections) == 1


def test_token_lock_does_not_stall_other_mcp_calls(
    manager: MCPManager, server_script: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """F8: a foreign process holding the token lock must not block the loop."""
    import asyncio

    from mcp.shared.auth import OAuthToken

    config = _config(server_script, tmp_path / "tok_pid")
    key = _connection_key(config)
    assert manager.connect(config).session is not None

    storage = FileTokenStorage("gsrv")
    holder_script = tmp_path / "holder.py"
    holder_script.write_text(_LOCK_HOLDER, encoding="utf-8")
    ready = tmp_path / "held"
    holder = subprocess.Popen(
        [sys.executable, str(holder_script), str(storage._lock_path), str(ready)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not ready.exists():
            time.sleep(0.02)
        assert ready.exists(), "lock holder never started"

        future = asyncio.run_coroutine_threadsafe(
            storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer")),
            manager._loop,
        )
        time.sleep(0.3)

        result: list[str] = []
        start = time.monotonic()
        thread = threading.Thread(
            target=lambda: result.append(
                manager.call_tool(key, "add", {"a": 4, "b": 4}),
            ),
        )
        thread.start()
        thread.join(timeout=30)
        elapsed = time.monotonic() - start
        assert result == ["8"], result
        assert elapsed < 2.5, (
            f"an unrelated MCP call waited {elapsed:.1f}s behind a foreign "
            f"process's token lock"
        )
        future.result(timeout=30)
    finally:
        holder.wait(timeout=30)


def test_spawn_survives_a_closed_stderr(
    manager: MCPManager,
    server_script: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed ``sys.stderr`` must not stop an MCP server from starting.

    ``mcp.client.stdio.stdio_client`` declares ``errlog: TextIO =
    sys.stderr``.  A default argument is evaluated once, when the module
    is imported, and the SDK passes it straight to the child's
    ``stderr=``.  So the very first import pins one stream object for the
    life of the process, and anything that later closes or replaces it --
    a log redirect in the long-lived ``kiss-web`` daemon, or pytest's
    per-test capture being torn down -- makes the next spawn die with
    ``ValueError: I/O operation on closed file``.

    This drives the real transport with a genuinely closed
    ``sys.stderr`` and requires a real FastMCP child to come up anyway.
    """
    master_fd, slave_fd = pty.openpty()
    stdin_stream = os.fdopen(slave_fd, "r", closefd=True)
    dead = (tmp_path / "dead_errlog.txt").open("w", encoding="utf-8")
    dead.close()
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", dead)
    try:
        pid_file = tmp_path / "closed-stderr-pid"
        conn = manager.connect(_config(server_script, pid_file))
        assert conn.session is not None, conn.error
        assert _alive(_wait_for_pid_file(pid_file))
    finally:
        stdin_stream.close()
        os.close(master_fd)

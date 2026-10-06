# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-05 scope E: the MCP health ping and in-flight tool calls.

``_park_until_stopped`` skips the idle ping while a call is in flight,
but a call that starts AFTER the ping was sent and before the server
answers it is not protected: a server busy in that call's sync handler
cannot answer, the ping times out, and the session is unwound under the
live call.  A ping that times out while a call is in flight is now
inconclusive (the server is busy, not dead) and the task keeps parking;
a dead server is caught by the call's own timeout.

Real FastMCP stdio subprocess, real signals.  The server is paused with
SIGSTOP and its stdin is copied to a file by ``tee``, so the test SEES
the ping request go out (unanswerable) before it starts the call, and
sees the call request go out before the ping deadline: the interleaving
is observed, not assumed.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig, _connection_key
from kiss.tests.conftest import posix_only

_SERVER_SCRIPT = '''
import os, sys

from mcp.server.fastmcp import FastMCP

open(sys.argv[1], "w").write(str(os.getpid()))

mcp = FastMCP("psrv")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


if __name__ == "__main__":
    mcp.run()
'''

_HEALTH_INTERVAL = 0.5


@pytest.fixture
def real_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give ``sys.stdin`` and the MCP errlog real descriptors (as test_mcp_connection_lifecycle)."""
    stdin_stream = open(os.devnull, encoding="utf-8")
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()


def _wait_for_pid_file(pid_file: Path, timeout: float = 30) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return int(pid_file.read_text())
        except (OSError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"server never wrote {pid_file}")


def _wait_for_request(stdin_copy: Path, method: str, timeout: float = 30) -> None:
    """Block until a JSON-RPC request with *method* was written to the server's stdin."""
    needle = f'"method":"{method}"'
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if needle in stdin_copy.read_text(encoding="utf-8").replace(" ", ""):
                return
        except OSError:
            pass
        time.sleep(0.01)
    raise AssertionError(f"no {method} request reached the server's stdin within {timeout}s")


@posix_only("the server is paused with SIGSTOP; stdin is copied with tee")
def test_a_ping_outstanding_when_a_call_starts_does_not_tear_down_the_session(
    tmp_path: Path, real_stdin: None,
) -> None:
    """Observed order: ping sent to a paused server; call sent; ping deadline; server resumed."""
    script = tmp_path / "psrv.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    pid_file = tmp_path / "pid"
    stdin_copy = tmp_path / "stdin_copy.jsonl"
    config = MCPServerConfig(
        name="psrv", transport="stdio", command="sh",
        args=("-c", 'exec tee "$1" | exec "$2" "$3" "$4"', "psrv",
              str(stdin_copy), sys.executable, str(script), str(pid_file)),
    )
    key = _connection_key(config)
    manager = MCPManager(
        idle_timeout_s=3600, max_connections=2, health_interval_s=_HEALTH_INTERVAL,
    )
    pid = None
    try:
        conn = manager.connect(config)
        assert conn.session is not None, conn.error
        pid = _wait_for_pid_file(pid_file)
        os.kill(pid, signal.SIGSTOP)  # from here on no request is answered
        _wait_for_request(stdin_copy, "ping")  # the idle ping is outstanding
        answer: list[str] = []
        caller = threading.Thread(
            target=lambda: answer.append(manager.call_tool(key, "add", {"a": 1, "b": 2})),
            daemon=True,
        )
        caller.start()
        _wait_for_request(stdin_copy, "tools/call")  # the call is in flight
        time.sleep(_HEALTH_INTERVAL + 0.5)  # the outstanding ping has timed out by now
        os.kill(pid, signal.SIGCONT)
        caller.join(timeout=60)
        assert not caller.is_alive(), "the in-flight call was stranded by the ping timeout"
        assert answer == ["3"], f"the in-flight call was broken by the ping: {answer}"
        assert not conn.error, conn.error
        assert conn.session is not None, "the busy connection was torn down"
        assert manager._connections.get(key) is conn
        assert manager.call_tool(key, "add", {"a": 2, "b": 2}) == "4"
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        manager.shutdown()

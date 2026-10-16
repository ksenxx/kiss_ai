# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-16 (sorcar-tools): ``make_mcp_tools`` must not connect serially.

``make_mcp_tools`` runs on the task thread while the agent builds its
tools.  It used to call ``MCPManager.connect`` once per configured
server, and each call blocks for up to ``CONNECT_TIMEOUT`` (60 s) when
the server accepts the transport but never finishes the MCP handshake
(a stdio child that does not speak MCP, a URL that accepts TCP and
never answers).  With N such servers every task start stalled for
N x 60 s even though the handshakes are independent tasks on the
manager's loop.  ``MCPManager.connect_all`` now starts every
connection first and waits for all of them under one shared deadline.

Real end-to-end: two real stdio children are spawned through the real
MCP transport; ``CONNECT_TIMEOUT`` is shortened so the serial cost is
measurable (>= 2 x timeout) against the fixed one (~1 x timeout).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import mcp_servers
from kiss.agents.sorcar.mcp_servers import (
    MCPManager,
    MCPServerConfig,
    make_mcp_tools,
    save_mcp_server,
)

_TIMEOUT_S = 2.0


def _silent_config(name: str) -> MCPServerConfig:
    """A real child that stays alive without ever speaking MCP."""
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=("-c", "import time; time.sleep(300)"),
    )


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every user-level config location into *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claudehome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    (tmp_path / "project").mkdir()
    return tmp_path


def test_make_mcp_tools_waits_for_silent_servers_concurrently(
    isolated_homes: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two servers stuck mid-handshake cost one CONNECT_TIMEOUT, not two."""
    monkeypatch.setattr(mcp_servers, "CONNECT_TIMEOUT", _TIMEOUT_S)
    project = isolated_homes / "project"
    save_mcp_server(_silent_config("silent_one"), "user", str(project))
    save_mcp_server(_silent_config("silent_two"), "user", str(project))
    manager = MCPManager.instance()
    try:
        started = time.monotonic()
        tools = make_mcp_tools(str(project))
        elapsed = time.monotonic() - started
    finally:
        manager.shutdown()
    assert tools == []
    assert elapsed >= _TIMEOUT_S, "a silent server must still cost its handshake deadline"
    assert elapsed < 2 * _TIMEOUT_S - 0.5, (
        f"make_mcp_tools took {elapsed:.1f}s for two silent servers with a "
        f"{_TIMEOUT_S:.0f}s CONNECT_TIMEOUT: the connections were awaited one after another"
    )


def test_connect_all_returns_one_connection_per_config(
    isolated_homes: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``connect_all`` keeps the input order and stamps every timed-out connection."""
    monkeypatch.setattr(mcp_servers, "CONNECT_TIMEOUT", _TIMEOUT_S)
    manager = MCPManager()
    try:
        configs = [_silent_config("silent_a"), _silent_config("silent_b")]
        conns = manager.connect_all(configs)
        assert [conn.config.name for conn in conns] == ["silent_a", "silent_b"]
        assert all(conn.session is None for conn in conns)
        assert all(conn.error == "connection timed out" for conn in conns)
        # A shut-down manager answers at once with failed records.
        manager.shutdown()
        conns = manager.connect_all(configs)
        assert [conn.error for conn in conns] == ["manager shut down"] * 2
    finally:
        manager.shutdown()


_FASTMCP_SCRIPT = '''
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(sys.argv[1])


@mcp.tool()
def ping() -> str:
    """Return this server's name."""
    return sys.argv[1]


if __name__ == "__main__":
    mcp.run()
'''


def _fastmcp_config(script: Path, name: str) -> MCPServerConfig:
    """A real FastMCP stdio server exposing one ``ping`` tool."""
    return MCPServerConfig(
        name=name, transport="stdio", command=sys.executable, args=(str(script), name),
    )


def test_batch_larger_than_the_pool_cap_keeps_every_member_until_its_tools_are_read(
    isolated_homes: Path,
) -> None:
    """Nine healthy servers (cap 8) all yield their tools; none is evicted mid-batch.

    Regression: ``connect_all`` evicted once per started connection,
    sparing only that connection, so once nine connections existed the
    already-ready ``srv0`` was torn down before ``make_mcp_tools`` read
    its tools and the run lost a configured server ("evicted").
    """
    project = isolated_homes / "project"
    script = project / "srv.py"
    script.write_text(_FASTMCP_SCRIPT, encoding="utf-8")
    names = [f"srv{i}" for i in range(9)]
    for name in names:
        save_mcp_server(_fastmcp_config(script, name), "user", str(project))
    manager = MCPManager.instance()
    try:
        # srv0 is ready (and therefore evictable) before the batch starts.
        first = manager.connect(_fastmcp_config(script, "srv0"))
        assert first.session is not None, first.error
        tools = make_mcp_tools(str(project))
        assert sorted(tool.__name__ for tool in tools) == sorted(f"{n}_ping" for n in names)
        assert first.error == "", first.error
        srv0_ping = next(tool for tool in tools if tool.__name__ == "srv0_ping")
        assert "srv0" in str(srv0_ping())
    finally:
        manager.shutdown()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The reserved MCP tool-name list must describe reality.

``_RESERVED_TOOL_NAMES`` exists so a synthesized ``<server>_<tool>`` name that
collides with one of the agent's built-in tools gets a numeric suffix instead
of aborting the whole tool loop.  It listed ``"web_search"``, which is not a
tool anywhere in the repo, so a perfectly ordinary MCP server named ``web``
with a tool named ``search`` was needlessly renamed to ``web_search_2``.

Driven end to end through a real FastMCP stdio server and the real
``make_mcp_tools`` wiring; the tool is then actually invoked.
"""

from __future__ import annotations

import os
import pty
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import (
    MCPManager,
    MCPServerConfig,
    make_mcp_tools,
    save_mcp_server,
)

_WEB_SERVER = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web")


@mcp.tool()
def search(query: str) -> str:
    """Search the web for a query."""
    return "results for " + query


if __name__ == "__main__":
    mcp.run()
'''


@pytest.fixture
def real_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Give ``sys.stdin`` and the MCP errlog real file descriptors."""
    # Imported *before* sys.stderr is redirected: this module binds
    # ``errlog=sys.stderr`` as a default argument at import time, and a
    # first import from inside this fixture would capture (and outlive)
    # the file closed below.
    from mcp.client.stdio import stdio_client

    master_fd, slave_fd = pty.openpty()
    stdin_stream = os.fdopen(slave_fd, "r", closefd=True)
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)

    wrapped = stdio_client.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(wrapped, "__defaults__", (errlog,))
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()
        os.close(master_fd)


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every user-level MCP location into *tmp_path*."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "project").mkdir()
    return tmp_path


def test_web_search_tool_keeps_its_natural_name(
    isolated_homes: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """A ``web`` server's ``search`` tool is exposed as ``web_search``."""
    script = tmp_path / "websrv.py"
    script.write_text(_WEB_SERVER, encoding="utf-8")
    project = isolated_homes / "project"
    save_mcp_server(
        MCPServerConfig(
            name="web",
            transport="stdio",
            command=sys.executable,
            args=(str(script),),
        ),
        "user",
        str(project),
    )
    try:
        tools = make_mcp_tools(str(project))
        names = [tool.__name__ for tool in tools]
        assert names == ["web_search"], names
        assert "results for kiss" in tools[0](query="kiss")
    finally:
        MCPManager.instance().shutdown()

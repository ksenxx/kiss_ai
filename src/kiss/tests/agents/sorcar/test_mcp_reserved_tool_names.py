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
The opposite drift happened too: ``create_worktree`` and the ``memory_*``
tools were missing, so a server ``create`` with a tool ``worktree`` shadowed
the built-in and aborted the agent's tool setup.

Driven end to end through a real FastMCP stdio server and the real
``make_mcp_tools`` wiring; the tool is then actually invoked.

The agent's own ``_get_tools`` must reserve its REAL toolset, not the
static list: a tool the caller (a SEA's ``tools=[...]``) adds under a name
an MCP server also synthesizes used to reach ``KISSAgent._add_functions``
twice and abort the run.
"""

from __future__ import annotations

import os
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

    # ``os.devnull`` has a real descriptor on every platform (Windows has
    # no pty) and nothing reads stdin while the client talks to the child.
    stdin_stream = open(os.devnull, encoding="utf-8")
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


_CREATE_SERVER = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("create")


@mcp.tool()
def worktree(name: str) -> str:
    """Create a named worktree on the remote side."""
    return "created " + name


if __name__ == "__main__":
    mcp.run()
'''


def _save_create_server(tmp_path: Path, project: Path) -> None:
    """Register a user-level stdio server ``create`` exposing the tool ``worktree``."""
    script = tmp_path / "createsrv.py"
    script.write_text(_CREATE_SERVER, encoding="utf-8")
    save_mcp_server(
        MCPServerConfig(
            name="create",
            transport="stdio",
            command=sys.executable,
            args=(str(script),),
        ),
        "user",
        str(project),
    )


def test_mcp_tool_colliding_with_a_built_in_gets_a_suffix_in_the_agent_toolset(
    isolated_homes: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """``create`` + ``worktree`` must not shadow the agent's ``create_worktree``.

    The reserved list once lacked ``create_worktree``, ``commit_and_merge``
    and the ``memory_*`` tools, so the agent's toolset carried two tools
    named ``create_worktree`` and ``KISSAgent._add_functions`` aborted the
    run's tool setup.
    """
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    project = isolated_homes / "project"
    _save_create_server(tmp_path, project)
    agent = SorcarAgent("test-reserved-names")
    agent._use_web_tools = False
    agent.work_dir = str(project)
    try:
        names = [tool.__name__ for tool in agent._get_tools()]
        assert len(names) == len(set(names)), names
        assert "create_worktree" in names
        assert "create_worktree_2" in names
    finally:
        MCPManager.instance().shutdown()


_SEA_SERVER = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sea")


@mcp.tool()
def probe(name: str) -> str:
    """Probe something on the remote side."""
    return "mcp probed " + name


if __name__ == "__main__":
    mcp.run()
'''


def sea_probe(name: str) -> str:
    """A SEA-added tool whose name server ``sea``'s ``probe`` also synthesizes.

    Args:
        name: What to probe.

    Returns:
        A fixed marker string.
    """
    return "sea tool probed " + name


def test_mcp_tool_colliding_with_a_caller_tool_gets_a_suffix_in_a_run(
    isolated_homes: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """A caller's ``sea_probe`` and server ``sea``'s ``probe`` both survive a run."""
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent
    from kiss.tests.core.test_default_max_steps import _all_text, _ScriptedServer

    project = isolated_homes / "project"
    script = tmp_path / "seasrv.py"
    script.write_text(_SEA_SERVER, encoding="utf-8")
    save_mcp_server(
        MCPServerConfig(
            name="sea", transport="stdio", command=sys.executable, args=(str(script),),
        ),
        "user",
        str(project),
    )
    finish = {"success": True, "is_continue": False, "summary_in_html": "<p>done</p>"}
    server = _ScriptedServer([
        ("sea_probe", {"name": "a"}),
        ("sea_probe_2", {"name": "b"}),
        ("finish", finish),
    ])
    agent = SorcarAgent("test-caller-tool-collision")
    try:
        agent.run(
            model_name="gpt-4o-mini",
            prompt_template="Call both probe tools, then finish.",
            tools=[sea_probe],
            work_dir=str(project),
            model_config=server.model_config(),
            web_tools=False,
            verbose=False,
        )
        assert {"sea_probe", "sea_probe_2"} <= agent._run_tool_names
        seen = _all_text(server.request_bodies)
        assert "sea tool probed a" in seen
        assert "mcp probed b" in seen
    finally:
        server.stop()
        MCPManager.instance().shutdown()


def test_make_mcp_tools_reserves_the_names_the_caller_passes(
    isolated_homes: Path, tmp_path: Path, real_stdin: None,
) -> None:
    """*taken_names* replaces the built-in default: it is the caller's toolset that counts."""
    project = isolated_homes / "project"
    _save_create_server(tmp_path, project)
    try:
        natural = make_mcp_tools(str(project), taken_names=())
        assert [tool.__name__ for tool in natural] == ["create_worktree"]
        assert "created wt" in natural[0](name="wt")
        shifted = make_mcp_tools(
            str(project), taken_names={"create_worktree", "create_worktree_2"},
        )
        assert [tool.__name__ for tool in shifted] == ["create_worktree_3"]
    finally:
        MCPManager.instance().shutdown()

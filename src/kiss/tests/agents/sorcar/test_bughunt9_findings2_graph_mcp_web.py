# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9 (findings-2 audit): mcp_servers, web_use_tool.

End-to-end tests (no mocks/patches/fakes) covering:

* S2-13 — two servers sharing a name must have isolated connections.
* S2-14 — distinct server names must never share a token file.
* S2-15 — colliding sanitized tool names must be disambiguated.
* S2-16 — a Unicode JSON property (``½``) must not abort tool wrapping.
* S2-20 — clicking the second of two same-named buttons targets it.
* S2-27 — browser startup failures return string errors, not raises.
"""

from __future__ import annotations

import json
import os
import pty
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiss.agents.sorcar.mcp_servers import (
    FileTokenStorage,
    MCPManager,
    MCPServerConfig,
    _python_param_name,
    make_mcp_tool_wrapper,
    make_mcp_tools,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool

_SERVER_TEMPLATE = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("{server_name}")


@mcp.tool()
def {tool_name}() -> str:
    """Return this server's identity marker."""
    return "{marker}"


if __name__ == "__main__":
    mcp.run()
'''


def _write_server(tmp_path: Path, file_stem: str, server_name: str,
                  tool_name: str, marker: str) -> Path:
    script = tmp_path / f"{file_stem}.py"
    script.write_text(
        _SERVER_TEMPLATE.format(
            server_name=server_name, tool_name=tool_name, marker=marker,
        ),
        encoding="utf-8",
    )
    return script


@pytest.fixture
def real_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Give ``sys.stdin`` (and the MCP errlog) a real file descriptor.

    Same environment plumbing as in ``test_sorcar_mcp.py``: pytest's
    captured std streams have no ``fileno()``, which the MCP stdio
    transport requires to spawn the real server subprocess.  This is
    I/O plumbing for running the *real* servers, not a test double.
    """
    master_fd, slave_fd = pty.openpty()
    stdin_stream = os.fdopen(slave_fd, "r", closefd=True)
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)

    from mcp.client.stdio import stdio_client

    wrapped = stdio_client.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(wrapped, "__defaults__", (errlog,))
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()
        os.close(master_fd)


class TestMcpIsolationAndNaming:
    """S2-13 / S2-14 / S2-15 / S2-16."""

    def test_token_files_are_injective(self) -> None:
        paths = {
            FileTokenStorage(name).path
            for name in ("a/b", "a?b", "a b", "a_b")
        }
        assert len(paths) == 4, f"token paths collide: {paths}"
        # A plain name keeps its historical file name.
        assert FileTokenStorage("github").path.name == "github.json"

    def test_unicode_property_does_not_break_wrapping(self) -> None:
        name = _python_param_name("½", set())
        assert name.isidentifier()
        tool = SimpleNamespace(
            name="frac",
            description="tool with unicode prop",
            inputSchema={
                "type": "object",
                "properties": {"½": {"type": "string"}},
                "required": [],
            },
        )
        wrapper = make_mcp_tool_wrapper(MCPManager.instance(), "srv", tool)
        assert callable(wrapper)
        assert wrapper.__name__ == "srv_frac"

    def test_same_named_servers_are_isolated(
        self, tmp_path: Path, real_stdin: object,
    ) -> None:
        script_a = _write_server(tmp_path, "srv_a", "same", "whoami", "A")
        script_b = _write_server(tmp_path, "srv_b", "same", "whoami", "B")
        cfg_a = MCPServerConfig(
            name="same", transport="stdio",
            command=sys.executable, args=(str(script_a),),
        )
        cfg_b = MCPServerConfig(
            name="same", transport="stdio",
            command=sys.executable, args=(str(script_b),),
        )
        manager = MCPManager.instance()
        conn_a = manager.connect(cfg_a)
        assert conn_a.session is not None, conn_a.error
        from kiss.agents.sorcar.mcp_servers import _connection_key

        tool_a = next(t for t in conn_a.tools if str(t.name) == "whoami")
        wrapper_a = make_mcp_tool_wrapper(
            manager, "same", tool_a, connection_key=_connection_key(cfg_a),
        )
        assert wrapper_a() == "A"

        conn_b = manager.connect(cfg_b)
        assert conn_b.session is not None, conn_b.error
        tool_b = next(t for t in conn_b.tools if str(t.name) == "whoami")
        wrapper_b = make_mcp_tool_wrapper(
            manager, "same", tool_b, connection_key=_connection_key(cfg_b),
        )

        # Project B's connection must not hijack project A's wrapper.
        assert wrapper_a() == "A", (
            "connecting a same-named server rerouted the first wrapper"
        )
        assert wrapper_b() == "B"

    def test_colliding_tool_names_are_disambiguated(
        self, tmp_path: Path, real_stdin: object,
    ) -> None:
        work_dir = tmp_path / "project"
        (work_dir / ".kiss").mkdir(parents=True)
        script_1 = _write_server(tmp_path, "s1", "a", "b_c", "ONE")
        script_2 = _write_server(tmp_path, "s2", "a_b", "c", "TWO")
        config = {
            "mcpServers": {
                "a": {
                    "type": "stdio", "command": sys.executable,
                    "args": [str(script_1)],
                },
                "a_b": {
                    "type": "stdio", "command": sys.executable,
                    "args": [str(script_2)],
                },
            },
        }
        (work_dir / ".kiss" / "mcp.json").write_text(json.dumps(config))

        tools = make_mcp_tools(str(work_dir))

        names = [t.__name__ for t in tools]
        collided = [n for n in names if n.startswith("a_b_c")]
        assert len(names) == len(set(names)), f"duplicate tool names: {names}"
        assert len(collided) == 2, f"expected both a_b_c tools, got {names}"
        by_name = {t.__name__: t for t in tools}
        results = {by_name[n]() for n in collided}
        assert results == {"ONE", "TWO"}


class TestWebUseTool:
    """S2-20 / S2-27."""

    def test_click_second_duplicate_named_button(self) -> None:
        html = (
            "<html><title>none</title><body>"
            "<button onclick=\"document.title='first'\">Delete</button>"
            "<button onclick=\"document.title='second'\">Delete</button>"
            "</body></html>"
        )
        tool = WebUseTool(headless=True, ephemeral=True)
        try:
            tree = tool.go_to_url("data:text/html," + html)
            assert "Error" not in tree.splitlines()[0]
            ids = []
            for line in tree.splitlines():
                if "Delete" in line and "[" in line:
                    ids.append(int(line.split("[", 1)[1].split("]", 1)[0]))
            assert len(ids) == 2, f"expected two Delete buttons in:\n{tree}"
            tool.click(ids[1])
            content = tool.get_page_content(text_only=True)
            assert "Page: second" in content, (
                f"clicking the second ID hit the wrong button:\n{content}"
            )
        finally:
            tool.close()

    def test_startup_failure_returns_error_string(self, tmp_path: Path) -> None:
        not_a_dir = tmp_path / "profile"
        not_a_dir.write_text("this is a file, not a directory")
        tool = WebUseTool(headless=True, user_data_dir=str(not_a_dir))
        try:
            result = tool.go_to_url("https://example.com")
        finally:
            tool.close()
        assert isinstance(result, str)
        assert result.startswith("Error navigating to https://example.com"), result

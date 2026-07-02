# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 8 (SORCAR-EXT): malformed MCP inputSchema crashes tool building.

``make_mcp_tools`` promises that "a broken server never breaks agent
startup", yet ``make_mcp_tool_wrapper`` indexed ``schema["properties"]``
assuming a dict.  The MCP SDK only requires ``inputSchema`` to be a JSON
object (``dict[str, Any]``); nothing constrains the *value* of its
``properties`` key.  A real server whose tool declares
``"properties": ["query"]`` (a list — seen in the wild from schema
generators) made ``props[prop_name]`` raise ``TypeError: list indices
must be integers``, aborting ``make_mcp_tools`` for every server and
therefore agent startup.  Similarly ``"required": "query"`` (a string
instead of a list) was fed to ``set()``, silently turning the single
required property into the bogus character set ``{'q','u','e','r','y'}``.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

from kiss.agents.sorcar.mcp_servers import (
    MCPServerConfig,
    make_mcp_tools,
    save_mcp_server,
)
from kiss.tests.agents.sorcar.test_sorcar_mcp import (  # noqa: F401
    isolated_homes,
)

# A real stdio MCP server (low-level API) whose tool schemas are
# malformed-but-legal MCP data: ``properties`` is a list on one tool,
# and ``required`` is a bare string on the other.
_MALFORMED_SERVER_SCRIPT = '''
import json

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

server = Server("badschema")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="listprops",
            description="properties is a list, not a dict.",
            inputSchema={"type": "object", "properties": ["query"]},
        ),
        types.Tool(
            name="strreq",
            description="required is a string, not a list.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": "query",
            },
        ),
    ]


@server.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(arguments))]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
'''


def test_malformed_schema_does_not_break_tool_building(
    isolated_homes: Path, tmp_path: Path,  # noqa: F811
) -> None:
    """Tools with non-dict properties / non-list required still register."""
    script = tmp_path / "badschema.py"
    script.write_text(_MALFORMED_SERVER_SCRIPT, encoding="utf-8")
    project = isolated_homes / "project"
    save_mcp_server(
        MCPServerConfig(
            name="badschema",
            transport="stdio",
            command=sys.executable,
            args=(str(script),),
        ),
        "user",
        str(project),
    )
    tools = make_mcp_tools(str(project))
    names = {t.__name__ for t in tools}
    assert "badschema_listprops" in names
    assert "badschema_strreq" in names

    # The list-properties tool degrades to a parameterless wrapper that
    # still round-trips a live call.
    listprops = next(t for t in tools if t.__name__ == "badschema_listprops")
    assert list(inspect.signature(listprops).parameters) == []
    assert json.loads(listprops()) == {}

    # The string-required tool must not treat the characters
    # 'q','u','e','r','y' as required property names: with no usable
    # required list, ``query`` is optional (default None) and a live
    # call still delivers it under its original name.
    strreq = next(t for t in tools if t.__name__ == "badschema_strreq")
    params = inspect.signature(strreq).parameters
    assert list(params) == ["query"]
    assert params["query"].default is None
    assert json.loads(strreq(query="hi")) == {"query": "hi"}

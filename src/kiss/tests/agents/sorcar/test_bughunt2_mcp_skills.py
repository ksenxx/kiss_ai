# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt round 2: MCP tool wrappers, OAuth callback server, skills.

End-to-end regression tests (no mocks) for genuine bugs found in
``kiss.agents.sorcar.mcp_servers``, ``kiss.ui.cli.mcp_cli``, and
``kiss.agents.sorcar.skills``:

1. ``make_mcp_tool_wrapper`` crashed with ``ValueError`` when an
   optional inputSchema property was listed before a required one
   (invalid Python signature: non-default parameter after default).
2. ``make_mcp_tool_wrapper`` crashed with ``ValueError`` when a
   property name was not a valid Python identifier (``max-results``)
   or was a Python keyword (``from``) — taking down ``make_mcp_tools``
   and therefore agent startup for the whole server.
3. ``_OAuthCallbackServer`` treated *any* GET (e.g. the browser's
   ``/favicon.ico`` probe) as the OAuth redirect: it clobbered a
   previously captured authorization code with ``None`` and set the
   done event, making ``wait()`` raise ``TimeoutError`` even though
   authentication succeeded.
4. ``load_skill_content`` appended ``<note>listing truncated</note>``
   when a skill had *exactly* ``_MAX_RESOURCE_LISTING`` resource files,
   although nothing was omitted.
"""

from __future__ import annotations

import inspect
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from kiss.agents.sorcar.mcp_servers import (
    MCPManager,
    MCPServerConfig,
    make_mcp_tool_wrapper,
    make_mcp_tools,
    save_mcp_server,
)
from kiss.agents.sorcar.skills import (
    _MAX_RESOURCE_LISTING,
    Skill,
    load_skill_content,
)
from kiss.tests.agents.sorcar.test_sorcar_mcp import (  # noqa: F401
    isolated_homes,
)
from kiss.ui.cli.mcp_cli import _OAuthCallbackServer

# A real stdio MCP server (low-level API) whose tool schema uses a
# hyphenated property name, a Python keyword, and lists an optional
# property before the required one — all legal in JSON Schema / MCP.
_WEIRD_SERVER_SCRIPT = '''
import json

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

server = Server("weirdsrv")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch",
            description="Echo the received arguments as JSON.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max-results": {
                        "type": "integer",
                        "description": "Result cap",
                    },
                    "from": {"type": "string", "description": "Sender"},
                },
                "required": ["from"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(arguments))]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
'''


def test_wrapper_optional_before_required_property() -> None:
    """An optional property listed before a required one must not crash.

    JSON object properties are unordered; real MCP servers freely list
    optional properties first.  The synthesized Python signature must
    reorder required parameters first instead of raising ``ValueError``
    ("parameter without a default follows parameter with a default").
    """
    from mcp.types import Tool

    tool = Tool(
        name="search",
        description="Search.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "query": {"type": "string", "description": "The query"},
            },
            "required": ["query"],
        },
    )
    wrapper = make_mcp_tool_wrapper(MCPManager.instance(), "ordsrv", tool)
    params = list(inspect.signature(wrapper).parameters.values())
    assert [p.name for p in params] == ["query", "limit"]
    assert params[0].default is inspect.Parameter.empty
    assert params[1].default is None


def test_non_identifier_and_keyword_properties_live_call(
    isolated_homes: Path, tmp_path: Path,  # noqa: F811
) -> None:
    """Hyphenated/keyword property names work end to end.

    ``make_mcp_tools`` must not crash on them (that would break agent
    startup for the whole server), and the wrapper must map the
    sanitized Python parameter names back to the original JSON property
    names when calling the server.
    """
    script = tmp_path / "weirdsrv.py"
    script.write_text(_WEIRD_SERVER_SCRIPT, encoding="utf-8")
    project = isolated_homes / "project"
    save_mcp_server(
        MCPServerConfig(
            name="weirdsrv",
            transport="stdio",
            command=sys.executable,
            args=(str(script),),
        ),
        "user",
        str(project),
    )
    tools = make_mcp_tools(str(project))
    names = {t.__name__ for t in tools}
    assert "weirdsrv_fetch" in names

    fetch = next(t for t in tools if t.__name__ == "weirdsrv_fetch")
    params = inspect.signature(fetch).parameters
    # Every synthesized parameter is a usable Python identifier.
    assert all(name.isidentifier() for name in params)
    # Required-first ordering holds after sanitization too.
    defaults = [p.default is inspect.Parameter.empty for p in params.values()]
    assert defaults == sorted(defaults, reverse=True)

    required = [n for n, p in params.items()
                if p.default is inspect.Parameter.empty]
    optional = [n for n, p in params.items()
                if p.default is not inspect.Parameter.empty]
    assert len(required) == 1 and len(optional) == 1

    # The server must receive the ORIGINAL JSON property names.
    received = json.loads(fetch(**{required[0]: "alice", optional[0]: 3}))
    assert received == {"from": "alice", "max-results": 3}
    # Omitted optional arguments are not sent at all.
    received = json.loads(fetch(**{required[0]: "bob"}))
    assert received == {"from": "bob"}


def test_oauth_callback_ignores_non_callback_requests() -> None:
    """A stray browser request must not clobber the captured OAuth code.

    Browsers routinely fetch ``/favicon.ico`` from the redirect origin.
    Sequence: the real ``/callback`` redirect arrives, then a favicon
    probe.  ``wait()`` must still return the captured code instead of
    raising ``TimeoutError`` because the probe overwrote it with None.
    """
    server = _OAuthCallbackServer()
    try:
        url = f"http://localhost:{server.port}/callback?code=abc123&state=st9"
        with urllib.request.urlopen(url, timeout=10) as resp:
            assert resp.status == 200
        # Stray request without an authorization code (e.g. favicon).
        try:
            with urllib.request.urlopen(
                f"http://localhost:{server.port}/favicon.ico", timeout=10,
            ) as resp:
                pass
        except urllib.error.HTTPError:
            pass  # A non-200 answer for the stray request is fine.
        code, state = server.wait(timeout=10)
        assert code == "abc123"
        assert state == "st9"
    finally:
        server.close()


def test_no_truncation_note_at_exact_resource_cap(tmp_path: Path) -> None:
    """Exactly _MAX_RESOURCE_LISTING resources → no false truncation note."""
    skill_dir = tmp_path / "big-skill"
    (skill_dir / "assets").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: big-skill\ndescription: Big.\n---\nBody.\n",
        encoding="utf-8",
    )
    for i in range(_MAX_RESOURCE_LISTING):
        (skill_dir / "assets" / f"r{i:03d}.txt").write_text("x")
    skill = Skill(
        name="big-skill", description="Big.",
        path=str(skill_dir / "SKILL.md"), source="project",
    )
    content = load_skill_content(skill)
    assert content.count("<file>") == _MAX_RESOURCE_LISTING
    assert "listing truncated" not in content

    # One file beyond the cap → the note appears and the list stays capped.
    (skill_dir / "assets" / "zzz-extra.txt").write_text("x")
    content = load_skill_content(skill)
    assert content.count("<file>") == _MAX_RESOURCE_LISTING
    assert "listing truncated" in content

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``MCPManager``: every page of ``tools/list``, and a Stop right after a call's lease.

Audit 2026-10-30 (S9).  The MCP spec lets a server paginate
``tools/list``; the SDK's ``ClientSession.list_tools`` sends one request
and returns one page with its ``nextCursor``.  The manager loaded only
that first page, so every tool after it silently never became an agent
tool.

``call_tool`` leased the connection (``in_flight += 1``) and then
scheduled the call (several Python-level calls) before entering the
``try`` whose ``finally`` releases the lease.  A task Stop is a
``KeyboardInterrupt`` injected at any check point of the task thread;
one landing there left ``in_flight`` at 1 for good, so the connection
was never evicted nor health-pinged again.  The lease is now taken
inside the ``try``.  The interrupt is landed there deterministically:
a :class:`Pause` holds the calling thread at the entry of the loop's
``call_soon_threadsafe`` (instrumentation; nothing is replaced) while
it is injected.

Review 1 (R3): a Stop landing after the call was queued on the loop but
before ``asyncio.run_coroutine_threadsafe`` returned its future left
the ``finally`` nothing to cancel, so the call ran on with its lease
returned.  The caller now creates the call's future before submitting
it; the Stop is held at ``_write_to_self``, which
``call_soon_threadsafe`` calls once the call is queued.

The tests run a REAL low-level MCP stdio server that pages its tools;
no mocks.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import CodeType
from typing import Any

import pytest

from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig, _connection_key
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.server.pause import Pause

_SERVER_SCRIPT = '''
import sys

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

# cursor -> (tool names on that page, the next cursor)
PAGES = {
    "paged": {None: (["alpha", "beta"], "page-2"), "page-2": (["gamma"], None)},
    # A buggy server that hands out the same cursor for ever.
    "stuck": {None: (["alpha"], "again"), "again": (["beta"], "again")},
}[sys.argv[1]]
server = Server("pager")


@server.list_tools()
async def list_tools(request: types.ListToolsRequest) -> types.ListToolsResult:
    cursor = request.params.cursor if request.params else None
    names, next_cursor = PAGES[cursor]
    tools = [types.Tool(name=n, description=n, inputSchema={"type": "object"}) for n in names]
    return types.ListToolsResult(tools=tools, nextCursor=next_cursor)


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=f"called {name}")]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
'''

# A server whose ``beta`` call takes a minute: a call nobody cancels is
# unfinished on the manager's loop for that long.
_SLOW_BETA_SCRIPT = _SERVER_SCRIPT.replace(
    '    return [types.TextContent(type="text", text=f"called {name}")]',
    '    if name == "beta":\n        await anyio.sleep(60)\n'
    '    return [types.TextContent(type="text", text=f"called {name}")]',
)


@pytest.fixture
def real_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give ``sys.stdin`` (and the MCP errlog) a real file descriptor.

    Under pytest the std streams are in-memory capture objects whose
    ``.fileno()`` raises, so the stdio transport cannot spawn its child.
    """
    stdin_stream = open(os.devnull, encoding="utf-8")
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()


def _config(tmp_path: Path, mode: str, source: str = _SERVER_SCRIPT) -> MCPServerConfig:
    script = tmp_path / "pager.py"
    script.write_text(source, encoding="utf-8")
    return MCPServerConfig(
        name="pager", transport="stdio", command=sys.executable, args=(str(script), mode),
    )


@pytest.mark.usefixtures("real_stdin")
def test_tools_of_every_page_are_loaded_and_callable(tmp_path: Path) -> None:
    manager = MCPManager()
    config = _config(tmp_path, "paged")
    try:
        conn = manager.connect(config)
        assert conn.error == "", conn.error
        assert [tool.name for tool in conn.tools] == ["alpha", "beta", "gamma"]
        assert manager.call_tool(_connection_key(config), "gamma", {}) == "called gamma"
    finally:
        manager.shutdown()


@pytest.mark.usefixtures("real_stdin")
def test_a_cursor_handed_out_again_ends_the_listing(tmp_path: Path) -> None:
    manager = MCPManager()
    config = _config(tmp_path, "stuck")
    try:
        conn = manager.connect(config)
        assert conn.error == "", conn.error
        assert [tool.name for tool in conn.tools] == ["alpha", "beta"]
    finally:
        manager.shutdown()


def _on_the_caller(_local: dict[str, Any]) -> bool:
    """Pause matcher: hold only the test's calling thread."""
    return threading.current_thread().name == "mcp-caller"


def _loop_code(loop: asyncio.AbstractEventLoop, name: str) -> CodeType:
    """Return the code of the method *name* as the class of *loop* implements it.

    :class:`Pause` matches a code object, and the manager's loop is the
    platform's default: a selector loop on POSIX, a proactor loop (with
    its own ``_write_to_self``) on Windows.
    """
    code: CodeType = getattr(loop, name).__func__.__code__
    return code


def _call(
    manager: MCPManager, key: str, outcome: dict[str, object], tool: str = "gamma",
) -> None:
    """Run one ``call_tool``, recording its result or the interrupt it ended with."""
    try:
        outcome["result"] = manager.call_tool(key, tool, {})
    except KeyboardInterrupt:
        outcome["interrupted"] = True


@pytest.mark.usefixtures("real_stdin")
def test_a_stop_between_the_lease_and_the_call_releases_the_lease(tmp_path: Path) -> None:
    manager = MCPManager()
    config = _config(tmp_path, "paged")
    key = _connection_key(config)
    outcome: dict[str, object] = {}
    try:
        conn = manager.connect(config)
        assert conn.error == "", conn.error
        caller = threading.Thread(target=_call, args=(manager, key, outcome), name="mcp-caller")
        with Pause(_loop_code(manager._loop, "call_soon_threadsafe"), _on_the_caller) as held:
            caller.start()
            assert held.paused.wait(30), "the call never reached its scheduling"
            assert conn.in_flight == 1, "leased, not scheduled yet"
            assert caller.ident is not None
            assert inject_keyboard_interrupt(caller.ident) == 1
        caller.join(30)
        assert not caller.is_alive()
        assert outcome == {"interrupted": True}
        assert conn.in_flight == 0, "the interrupted call kept its lease"
        # The connection still serves calls; an unknown server leases nothing.
        assert manager.call_tool(key, "gamma", {}) == "called gamma"
        assert conn.in_flight == 0
        assert manager.call_tool("nosuch", "gamma", {}) == (
            "Error: MCP server 'nosuch' is not connected (never connected)"
        )
    finally:
        manager.shutdown()


async def _unfinished_calls() -> int:
    """Return how many ``ClientSession.call_tool`` tasks of the running loop are unfinished."""
    return sum(
        getattr(task.get_coro(), "__qualname__", "") == "ClientSession.call_tool"
        for task in asyncio.all_tasks()
    )


@pytest.mark.usefixtures("real_stdin")
def test_a_stop_right_after_the_call_was_queued_cancels_the_call(tmp_path: Path) -> None:
    manager = MCPManager()
    config = _config(tmp_path, "paged", _SLOW_BETA_SCRIPT)
    key = _connection_key(config)
    outcome: dict[str, object] = {}
    try:
        conn = manager.connect(config)
        assert conn.error == "", conn.error
        caller = threading.Thread(
            target=_call, args=(manager, key, outcome, "beta"), name="mcp-caller",
        )
        with Pause(_loop_code(manager._loop, "_write_to_self"), _on_the_caller) as held:
            caller.start()
            assert held.paused.wait(30), "the call was never queued"
            assert caller.ident is not None
            assert inject_keyboard_interrupt(caller.ident) == 1
        caller.join(30)
        assert not caller.is_alive()
        assert outcome == {"interrupted": True}
        assert conn.in_flight == 0, "the interrupted call kept its lease"
        deadline = time.monotonic() + 10
        while asyncio.run_coroutine_threadsafe(_unfinished_calls(), manager._loop).result(5):
            assert time.monotonic() < deadline, "the interrupted call runs on without its lease"
            time.sleep(0.05)
        assert manager.call_tool(key, "gamma", {}) == "called gamma"
    finally:
        manager.shutdown()

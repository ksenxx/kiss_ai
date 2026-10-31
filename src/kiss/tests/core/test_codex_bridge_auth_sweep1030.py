# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: the Codex MCP tool bridge serves only the proxy Codex spawned.

The bridge listens on a loopback port, which any local user can reach
(``ss -ltn`` lists it).  Its tools are the agent's (``Bash``, ``Write``,
...) run with the agent's privileges, so a request must carry the
bridge's secret, which reaches the proxy through Codex's environment
(``env_vars``), never through a command line.  Without it a plain socket
ran any tool it named.
"""

import asyncio
import json
import socket
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kiss.core.models.codex_mcp_bridge import BRIDGE_TOKEN_ENV, ToolBridge

BASH_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]

UNAUTHENTICATED = {"error": "KISSError: unauthenticated KISS tool bridge request"}


class _Executor:
    """Tool executor that records every call it is asked to run."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, call: dict[str, Any]) -> tuple[str, str]:
        self.calls.append(call)
        return call["name"], f"ran {call['arguments']['command']}"


def _ask(port: int, request: dict[str, Any]) -> dict[str, Any]:
    """Send *request* to the bridge on *port* as a plain local client; return the reply."""
    with socket.create_connection(("127.0.0.1", port), timeout=15) as client:
        client.sendall(json.dumps(request).encode() + b"\n")
        with client.makefile("rb") as reply:
            return dict(json.loads(reply.readline()))


def test_a_client_without_the_secret_runs_no_tool_and_sees_no_tool_list() -> None:
    """A call or tool list request without the right token is refused before anything runs."""
    executor = _Executor()
    call = {"op": "call", "name": "Bash", "arguments": {"command": "id"}}
    with ToolBridge(BASH_SCHEMA, executor) as bridge:
        token = bridge.codex_env()[BRIDGE_TOKEN_ENV]
        assert _ask(bridge.port, call) == UNAUTHENTICATED
        assert _ask(bridge.port, {**call, "token": "0" * len(token)}) == UNAUTHENTICATED
        assert _ask(bridge.port, {"op": "tools"}) == UNAUTHENTICATED
        assert executor.calls == []
        assert _ask(bridge.port, {**call, "token": token}) == {"text": "ran id"}
    assert executor.calls == [{"name": "Bash", "arguments": {"command": "id"}}]
    assert [served.name for served in bridge.take_served()] == ["Bash"]


def test_two_bridges_have_different_secrets() -> None:
    """A secret is per bridge: one turn's token opens no other turn's bridge."""
    executor = _Executor()
    with ToolBridge(BASH_SCHEMA, executor) as first, ToolBridge(BASH_SCHEMA, executor) as second:
        first_token = first.codex_env()[BRIDGE_TOKEN_ENV]
        assert first_token != second.codex_env()[BRIDGE_TOKEN_ENV]
        call = {"op": "call", "name": "Bash", "arguments": {"command": "id"}, "token": first_token}
        assert _ask(second.port, call) == UNAUTHENTICATED
    assert executor.calls == []


def test_codex_args_forward_the_secret_by_name_only() -> None:
    """The overrides name the variable for Codex to forward; the secret is not on argv."""
    with ToolBridge(BASH_SCHEMA, _Executor()) as bridge:
        args = bridge.codex_args(60)
        token = bridge.codex_env()[BRIDGE_TOKEN_ENV]
    assert f'mcp_servers.kiss.env_vars=["{BRIDGE_TOKEN_ENV}"]' in args
    assert not any(token in arg for arg in args)


async def _drive_proxy(bridge: ToolBridge, env: dict[str, str]) -> tuple[str, Any]:
    """Spawn the proxy of *bridge* with *env*; return the tool-list error and a call's result."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kiss.core.models.codex_mcp_bridge", str(bridge.port)],
        env=env,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        try:
            await session.list_tools()
            listing_error = ""
        except Exception as e:  # the MCP error the proxy's list_tools raised
            listing_error = str(e)
        result = await session.call_tool("Bash", {"command": "id"})
    return listing_error, result


def test_a_proxy_started_without_the_secret_is_refused() -> None:
    """The proxy run without Codex's forwarded variable gets errors, and no tool runs."""
    executor = _Executor()
    with ToolBridge(BASH_SCHEMA, executor) as bridge:
        listing_error, result = asyncio.run(_drive_proxy(bridge, {}))
    assert "unauthenticated" in listing_error
    assert result.isError and "unauthenticated" in result.content[0].text
    assert executor.calls == []


def test_a_proxy_holding_the_secret_is_served() -> None:
    """With the variable Codex forwards, the same proxy lists and calls the tools."""
    executor = _Executor()
    with ToolBridge(BASH_SCHEMA, executor) as bridge:
        listing_error, result = asyncio.run(_drive_proxy(bridge, bridge.codex_env()))
    assert listing_error == ""
    assert not result.isError and result.content[0].text == "ran id"
    assert executor.calls == [{"name": "Bash", "arguments": {"command": "id"}}]

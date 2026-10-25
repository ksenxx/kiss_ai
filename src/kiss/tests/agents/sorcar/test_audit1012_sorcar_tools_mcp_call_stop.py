# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop must end an MCP tool call (or connect) promptly.

Audit 2026-10-12 (sorcar-tools).  ``MCPManager.call_tool`` waited for the
result with one ``future.result(timeout=CALL_TIMEOUT + 5)``: a single
blocking C call of up to 305 s.  A tool-panel Stop is delivered to the
tool thread as an injected ``ToolCallInterrupted`` (a task Stop as an
injected ``KeyboardInterrupt``), which only lands at a bytecode boundary,
so the Stop was ignored until the MCP tool returned on its own or the
timeout expired.  ``connect()``'s ``ready.wait(CONNECT_TIMEOUT)`` had the
same shape (60 s against a server that is slow to start).

Both tests spawn a REAL FastMCP stdio server and deliver the Stop through
the real :mod:`kiss.core.tool_interrupt` machinery; no mocks.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig, _connection_key
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    end_tool_call,
    interrupt_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)

_SERVER_SCRIPT = """
import sys
import time
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stopsrv")


@mcp.tool()
def slow(seconds: float) -> str:
    \"\"\"Sleep for the given number of seconds, then return "done".\"\"\"
    time.sleep(seconds)
    return "done"


if __name__ == "__main__":
    time.sleep(float(sys.argv[1]))
    mcp.run()
"""


@pytest.fixture
def real_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give ``sys.stdin`` (and the MCP errlog) a real file descriptor.

    Same plumbing as ``test_sorcar_mcp.py``: under pytest the std
    streams are in-memory capture objects whose ``.fileno()`` raises,
    so the stdio transport cannot spawn its child.
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


def _config(tmp_path: Path, start_delay: float) -> MCPServerConfig:
    script = tmp_path / "stopsrv.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return MCPServerConfig(
        name="stopsrv",
        transport="stdio",
        command=sys.executable,
        args=(str(script), str(start_delay)),
    )


def _run_interrupted(name: str, work: object) -> tuple[dict[str, object], float]:
    """Run *work* as tool call *name* on a thread, Stop it after 1 s.

    Returns:
        The outcome (``{"interrupted": True}`` when the Stop surfaced as
        ``ToolCallInterrupted``) and the seconds between the Stop and the
        call's return.
    """
    outcome: dict[str, object] = {}

    def run() -> None:
        token = new_tool_call(name)
        register_tool_call(token)
        try:
            try:
                outcome["result"] = work()  # type: ignore[operator]
                end_tool_call(token)
            except ToolCallInterrupted:
                outcome["interrupted"] = True
        finally:
            unregister_tool_call(token)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(1.0)
    assert thread.ident is not None
    started = time.monotonic()
    assert interrupt_tool_call(thread.ident, name)
    thread.join(timeout=30)
    assert not thread.is_alive(), f"{name} ignored the Stop for 30 s"
    return outcome, time.monotonic() - started


@pytest.mark.usefixtures("real_stdin")
def test_call_tool_stops_promptly(tmp_path: Path) -> None:
    manager = MCPManager()
    config = _config(tmp_path, 0.0)
    try:
        conn = manager.connect(config)
        assert conn.error == "", conn.error
        key = _connection_key(config)
        outcome, elapsed = _run_interrupted(
            "slow", lambda: manager.call_tool(key, "slow", {"seconds": 20})
        )
        assert outcome == {"interrupted": True}
        assert elapsed < 10, elapsed
        # The lease was released: the connection is evictable again and
        # a later call on it still works.
        assert conn.in_flight == 0
        assert manager.call_tool(key, "slow", {"seconds": 0}) == "done"
    finally:
        manager.shutdown()


@pytest.mark.usefixtures("real_stdin")
def test_connect_stops_promptly(tmp_path: Path) -> None:
    manager = MCPManager()
    config = _config(tmp_path, 20.0)
    try:
        outcome, elapsed = _run_interrupted("connect_mcp_server", lambda: manager.connect(config))
        assert outcome == {"interrupted": True}
        assert elapsed < 10, elapsed
    finally:
        manager.shutdown()

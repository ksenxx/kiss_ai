# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``MCPLoginSession.wait`` is sliced and honours the tool-panel Stop.

Audit 2026-10-15 (sorcar-tools).  ``finish_mcp_server_connect`` waited
for the sign-in with one ``Event.wait(timeout=5.0)`` on the task
thread, and ``MCPLoginSession.start`` waited the same way for the
session it replaces.  A single C-level wait defers both an injected
Stop and the tool-panel interrupt by up to the whole 5 s; the repo's
rule is 0.2 s slices with a cooperative check.  Real sessions, real
tool-call registration; no mocks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.mcp_oauth import MCPLoginSession, make_mcp_auth_tools
from kiss.agents.sorcar.mcp_servers import MCPServerConfig
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    end_tool_call,
    interrupt_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)


@pytest.fixture
def pending_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MCPLoginSession]:
    """A session whose sign-in never finishes, published as the active one."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    cfg = MCPServerConfig(name="slow", transport="http", url="http://127.0.0.1:9/mcp")
    session = MCPLoginSession(cfg)
    previous = MCPLoginSession._active
    MCPLoginSession._active = session
    yield session
    MCPLoginSession._active = previous


def _run_interrupted(fn: Callable[[], object], tool_name: str) -> tuple[dict[str, object], float]:
    """Run *fn* as a registered tool call, Stop it after 0.3 s, time the exit."""
    outcome: dict[str, object] = {}

    def run() -> None:
        token = new_tool_call(tool_name)
        register_tool_call(token)
        try:
            try:
                outcome["result"] = fn()
                end_tool_call(token)
            except ToolCallInterrupted:
                outcome["interrupted"] = True
        finally:
            unregister_tool_call(token)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(0.3)
    assert thread.ident is not None
    started = time.monotonic()
    assert interrupt_tool_call(thread.ident, tool_name)
    thread.join(timeout=10)
    assert not thread.is_alive(), "the wait ignored the Stop"
    return outcome, time.monotonic() - started


def test_wait_times_out_and_returns_true_once_finished(pending_session: MCPLoginSession) -> None:
    started = time.monotonic()
    assert pending_session.wait(0.5) is False
    assert 0.4 <= time.monotonic() - started < 2.0
    threading.Timer(0.3, pending_session._finished.set).start()
    started = time.monotonic()
    assert pending_session.wait(10.0) is True
    assert time.monotonic() - started < 2.0


def test_finish_tool_stops_within_a_slice(pending_session: MCPLoginSession) -> None:
    finish = make_mcp_auth_tools(".")[1]
    outcome, elapsed = _run_interrupted(lambda: finish("slow"), "finish_mcp_server_connect")
    assert outcome == {"interrupted": True}
    assert elapsed < 2.0, elapsed


def test_replacing_a_stuck_session_stops_within_a_slice(
    pending_session: MCPLoginSession, mcp_redirect_port: int,
) -> None:
    """``start`` waits for the session it replaces through the sliced wait."""
    cfg = MCPServerConfig(name="next", transport="http", url="http://127.0.0.1:9/mcp")
    outcome, elapsed = _run_interrupted(
        lambda: MCPLoginSession.start(cfg, wait_seconds=1.0), "connect_mcp_server",
    )
    assert outcome == {"interrupted": True}
    assert elapsed < 2.0, elapsed
    # The Stop landed inside ``_replace_active``; the stuck session is cancelled
    # and the replacement was never published or started.
    assert pending_session._cancelled
    assert MCPLoginSession._active is pending_session

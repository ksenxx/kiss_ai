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

import concurrent.futures
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig, _connection_key
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    end_tool_call,
    inject_async_exception,
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


class _PauseInFutureCancel:
    """Profile hook parking the thread as it enters ``Future.cancel``, until resumed."""

    def __init__(self) -> None:
        self.future: concurrent.futures.Future[Any] | None = None
        self.reached = threading.Event()
        self.resume = threading.Event()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is concurrent.futures.Future.cancel.__code__:
            sys.setprofile(None)
            self.future = frame.f_locals["self"]
            self.reached.set()
            self.resume.wait(60)


def _call_under_hook(
    hook: Any, manager: MCPManager, key: str, outcome: list[BaseException | str],
) -> None:
    """Thread body: a 1 s ``slow`` tool call under the profile *hook*; record its outcome."""
    token = new_tool_call("slow")
    register_tool_call(token)
    sys.setprofile(hook)
    try:
        outcome.append(manager.call_tool(key, "slow", {"seconds": 1}))
        end_tool_call(token)
    except BaseException as exc:  # noqa: BLE001 — the thread records any outcome
        outcome.append(exc)
    finally:
        sys.setprofile(None)
        unregister_tool_call(token)


def _blocked_in(thread: threading.Thread, code: Any, timeout: float) -> bool:
    """Whether *thread* reaches (and stays in) a frame running *code* within *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert thread.ident is not None
        frame = sys._current_frames().get(thread.ident)
        if frame is not None and frame.f_code is code:
            return True
        time.sleep(0.01)
    return False


@pytest.mark.usefixtures("real_stdin")
def test_stop_inside_the_calls_cancel_does_not_wedge_the_manager_loop(tmp_path: Path) -> None:
    """A Stop landing in ``call_tool``'s cancellation must not leave the loop blocked for good.

    The cleanup cancelled the call's future on the task thread.
    ``Future.cancel`` takes the future's condition with a plain
    ``with``, and a Stop delivered right after its ``__enter__``
    (CPython gh-74174, here made certain by a contended acquisition)
    leaves that condition held by a thread that then dies.  When the
    call later completed, the loop's ``_settle`` blocked on it for
    ever: every MCP connection of the daemon shares that loop.  The
    cancellation now runs on the loop thread, which receives no
    injections.
    """
    manager = MCPManager()
    config = _config(tmp_path, 0.0)
    pause = _PauseInFutureCancel()
    outcome: list[BaseException | str] = []
    try:
        conn = manager.connect(config)
        assert conn.error == "", conn.error
        key = _connection_key(config)
        caller = threading.Thread(target=_call_under_hook, args=(pause, manager, key, outcome))
        caller.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and conn.in_flight == 0:
            time.sleep(0.01)
        assert conn.in_flight == 1, "the call never leased the connection"
        assert caller.ident is not None
        # The panel Stop: the call's wait raises and the cleanup runs
        # while the tool is still executing on the server.
        assert interrupt_tool_call(caller.ident, "slow")
        while not pause.reached.is_set() and caller.is_alive():
            pause.reached.wait(0.05)
        if pause.reached.is_set():
            # The cleanup cancels on the task thread: hold the future's
            # condition so that thread blocks in ``__enter__``, deliver
            # the Stop there, then let it acquire.
            future = pause.future
            assert future is not None
            with future._condition:  # type: ignore[attr-defined]
                pause.resume.set()
                assert _blocked_in(caller, threading.Condition.__enter__.__code__, 10)
                inject_async_exception(caller.ident, KeyboardInterrupt)
        caller.join(30)
        assert not caller.is_alive(), "call_tool() did not return"
        assert len(outcome) == 1 and not isinstance(outcome[0], str), outcome
        # The server finishes the 1 s tool: the loop settles the call.
        time.sleep(1.5)
        heartbeat = threading.Event()
        manager._loop.call_soon_threadsafe(heartbeat.set)
        assert heartbeat.wait(30), "the manager loop is wedged on the call's future"
        assert conn.in_flight == 0, "the Stop left the call's lease on the connection"
        assert manager.call_tool(key, "slow", {"seconds": 0}) == "done"
    finally:
        pause.resume.set()
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

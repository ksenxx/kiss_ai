# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a forced Stop does not cut the bridge's wait for a running tool.

On Stop the daemon sets the task's stop event and, if the task thread is
still alive a second later, injects ``KeyboardInterrupt`` into it
(``_TaskRunnerMixin._force_stop_thread``).  Leaving the MCP bridge's
block waits up to ``STOP_WAIT_SECONDS`` for the interrupted tool calls,
so no tool keeps working after the run returned; the injected exception
used to end that wait after about one second, with the tool still
running.  These tests run the daemon's own watchdog (and the raw
injection a Ctrl+C amounts to) against a real bridge.
"""

import json
import socket
import threading
import time
from typing import Any

from kiss.core import stop_signal
from kiss.core.models.codex_mcp_bridge import BRIDGE_TOKEN_ENV, STOP_WAIT_SECONDS, ToolBridge
from kiss.core.tool_interrupt import inject_async_exception
from kiss.server.task_runner import _TaskRunnerMixin

SCHEMA = [
    {
        "type": "function",
        "function": {"name": "slow", "description": "Work.", "parameters": {"type": "object"}},
    }
]

# Longer than the watchdog's one-second grace, well inside STOP_WAIT_SECONDS.
TOOL_SECONDS = 2.5


class _SlowTool:
    """Executor whose tool ignores interrupts and runs for TOOL_SECONDS."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.ended_at: list[float] = []

    def __call__(self, call: dict[str, Any]) -> tuple[str, str]:
        self.started.set()
        try:
            time.sleep(TOOL_SECONDS)
        finally:
            self.ended_at.append(time.monotonic())
        return call["name"], "done"


class _TaskThread(threading.Thread):
    """The task thread: calls the tool over the bridge, then leaves the block on Stop.

    ``interrupted_at`` records when the forced Stop's ``KeyboardInterrupt``
    left the block; ``done`` is set once the thread stops owning the run.
    """

    def __init__(self, tool: _SlowTool, request_stop: bool) -> None:
        super().__init__(daemon=True)
        self.tool = tool
        self.request_stop = request_stop
        self.in_block = threading.Event()
        self.done = threading.Event()
        self.interrupted_at: float | None = None
        self.bridge: ToolBridge | None = None

    def run(self) -> None:
        stop = threading.Event()
        stop_signal.set_thread_stop_event(stop)
        client: socket.socket | None = None
        try:
            with ToolBridge(SCHEMA, self.tool) as bridge:
                self.bridge = bridge
                client = socket.create_connection(("127.0.0.1", bridge.port), timeout=15)
                token = bridge.codex_env()[BRIDGE_TOKEN_ENV]
                request = {"op": "call", "name": "slow", "arguments": {}, "token": token}
                client.sendall(json.dumps(request).encode() + b"\n")
                assert self.tool.started.wait(5)
                if self.request_stop:
                    stop.set()
                self.in_block.set()
        except KeyboardInterrupt:
            self.interrupted_at = time.monotonic()
        finally:
            self.done.set()
            stop_signal.set_thread_stop_event(None)
            if client is not None:
                client.close()

    def still_owns(self) -> bool:
        """Ownership guard for the watchdog: the run is not over yet."""
        return not self.done.is_set()


def test_the_daemons_forced_stop_waits_for_the_interrupted_tool() -> None:
    """The watchdog's injection lands in the wait; the wait still ends with the tool."""
    tool = _SlowTool()
    task = _TaskThread(tool, request_stop=True)
    task.start()
    assert task.in_block.wait(10)
    stopped_at = time.monotonic()
    watchdog = threading.Thread(
        target=_TaskRunnerMixin._force_stop_thread, args=(task, task.still_owns), daemon=True
    )
    watchdog.start()
    task.join(STOP_WAIT_SECONDS + 10)
    watchdog.join(10)

    assert task.interrupted_at is not None, "the forced Stop must still surface"
    assert tool.ended_at, "the tool never ended"
    assert tool.ended_at[0] <= task.interrupted_at, (
        f"the run unwound {tool.ended_at[0] - task.interrupted_at:.2f}s before its tool ended"
    )
    assert task.interrupted_at - stopped_at < STOP_WAIT_SECONDS + 1


def test_a_keyboard_interrupt_without_a_stop_event_bounds_the_wait_too() -> None:
    """A Ctrl+C (no stop event) interrupts the wait like a Stop: held, then re-raised."""
    tool = _SlowTool()
    task = _TaskThread(tool, request_stop=False)
    task.start()
    assert task.in_block.wait(10)
    assert task.bridge is not None
    # Once the listener thread is gone, the task thread is in the open-ended wait.
    deadline = time.monotonic() + 5
    while task.bridge._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert inject_async_exception(task.ident or 0, KeyboardInterrupt) == 1
    task.join(STOP_WAIT_SECONDS + 10)

    assert task.interrupted_at is not None
    assert tool.ended_at and tool.ended_at[0] <= task.interrupted_at

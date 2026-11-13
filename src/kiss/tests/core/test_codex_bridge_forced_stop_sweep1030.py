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
import socketserver
import threading
import time
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.models.codex_mcp_bridge import BRIDGE_TOKEN_ENV, STOP_WAIT_SECONDS, ToolBridge
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    end_tool_call,
    inject_async_exception,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
    wait_interruptible,
)
from kiss.server.task_runner import _TaskRunnerMixin
from kiss.tests.server.pause import Pause

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


class _PatientTool(_SlowTool):
    """Executor whose tool works for four times STOP_WAIT_SECONDS unless it is interrupted.

    Registered as the agent registers a tool call (``kiss_agent``), so the
    bridge's interrupt reaches it: cooperatively through
    ``wait_interruptible``, or injected after the grace period.
    """

    def __call__(self, call: dict[str, Any]) -> tuple[str, str]:
        self.started.set()
        token = new_tool_call(call["name"])
        register_tool_call(token)
        try:
            wait_interruptible(threading.Event(), 4 * STOP_WAIT_SECONDS)
            end_tool_call(token)
        except ToolCallInterrupted:
            unregister_tool_call(token)
            return call["name"], "interrupted"
        finally:
            self.ended_at.append(time.monotonic())
        return call["name"], "done"


class _TaskThread(threading.Thread):
    """The task thread: calls the tool over the bridge, then leaves the block on Stop.

    ``interrupted_at`` records when the forced Stop's ``KeyboardInterrupt``
    left the block; ``done`` is set once the thread stops owning the run.
    ``body_stop``, when set, is the Stop the block's body ends with, and
    ``listener_closed`` whether the listener was closed when the Stop left.
    """

    def __init__(self, tool: _SlowTool, request_stop: bool) -> None:
        super().__init__(daemon=True)
        self.tool = tool
        self.request_stop = request_stop
        self.in_block = threading.Event()
        self.done = threading.Event()
        self.interrupted_at: float | None = None
        self.bridge: ToolBridge | None = None
        self.body_stop: BaseException | None = None
        self.listener_closed = False

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
                if self.body_stop is not None:
                    raise self.body_stop
        except (KeyboardInterrupt, ToolCallInterrupted):
            self.interrupted_at = time.monotonic()
            closed = self.bridge is not None and self.bridge._server.socket.fileno() == -1
            self.listener_closed = closed
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


def test_a_keyboard_interrupt_during_the_listener_shutdown_still_bounds_the_wait() -> None:
    """A Ctrl+C landing while the block's exit stops the listener, before the wait for the
    running tool began, is held like one landing in the wait: the tool is interrupted and
    the exit returns within STOP_WAIT_SECONDS, not once the tool would have ended by itself
    (and not at once, with the tool still working)."""
    tool = _PatientTool()
    task = _TaskThread(tool, request_stop=False)
    # The task thread is held as it enters the listener's shutdown, so the interrupt
    # lands in that step for sure (released, the thread raises it right there).
    shutdown = Pause(ToolBridge._stop_listener.__code__, lambda loc: True)
    with shutdown:
        task.start()
        assert task.in_block.wait(10)
        assert shutdown.paused.wait(10), "the block's exit never reached the listener shutdown"
        injected_at = time.monotonic()
        assert inject_async_exception(task.ident or 0, KeyboardInterrupt) == 1
        shutdown.release()
        task.join(STOP_WAIT_SECONDS + 10)

    assert task.interrupted_at is not None
    assert tool.ended_at and tool.ended_at[0] <= task.interrupted_at, "the tool kept working"
    assert task.interrupted_at - injected_at < STOP_WAIT_SECONDS + 1, "the wait was open-ended"


def test_a_keyboard_interrupt_at_the_entry_of_the_wait_still_bounds_the_wait() -> None:
    """A Ctrl+C landing as the block's exit enters the wait for the running tool (the
    listener already stopped, new calls already refused) is held like one landing inside
    the wait: the tool is interrupted and the exit returns once it ended, within
    STOP_WAIT_SECONDS, instead of at once with the tool still working on its thread."""
    tool = _PatientTool()
    task = _TaskThread(tool, request_stop=False)
    entry = Pause(ToolBridge._wait_for_calls.__code__, lambda loc: True)
    with entry:
        task.start()
        assert task.in_block.wait(10)
        assert entry.paused.wait(10), "the block's exit never reached the wait"
        injected_at = time.monotonic()
        assert inject_async_exception(task.ident or 0, KeyboardInterrupt) == 1
        entry.release()
        task.join(STOP_WAIT_SECONDS + 10)

    assert task.interrupted_at is not None
    assert task.bridge is not None and task.bridge.calls_in_flight == 0, "the call was left"
    assert tool.ended_at and tool.ended_at[0] <= task.interrupted_at, "the tool kept working"
    assert task.interrupted_at - injected_at < STOP_WAIT_SECONDS + 1, "the wait was open-ended"


@pytest.mark.parametrize(
    ("stop", "step"),
    [
        (ToolCallInterrupted, None),
        (KeyboardInterrupt, None),
        (ToolCallInterrupted, ToolBridge._refuse_new_calls),
        (ToolCallInterrupted, ToolBridge._stop_listener),
        (ToolCallInterrupted, socketserver.ThreadingMixIn.server_close),
        (ToolCallInterrupted, ToolBridge._wait_for_calls),
    ],
)
def test_a_stop_ending_the_block_or_landing_in_its_exit_bounds_the_wait(
    stop: type[BaseException], step: Any
) -> None:
    """A per-panel Stop (``ToolCallInterrupted``) or a Ctrl+C without a stop event, ending
    the block's body (*step* is None) or landing in a step of its exit, interrupts the running
    tool and leaves once it ended with the listener closed: not once the tool would have ended
    by itself (the body's Stop was dropped), and not at once with the rest of the exit skipped."""
    tool = _PatientTool()
    task = _TaskThread(tool, request_stop=False)
    if step is None:
        task.body_stop = stop()
    with Pause((step or ToolBridge._close).__code__, lambda loc: True) as held:
        task.start()
        assert task.in_block.wait(10)
        assert held.paused.wait(10), "the block's exit never got there"
        stopped_at = time.monotonic()
        if step is not None:
            assert inject_async_exception(task.ident or 0, stop) == 1
        held.release()
        task.join(STOP_WAIT_SECONDS + 10)

    assert task.interrupted_at is not None, "the Stop must still surface"
    assert tool.ended_at and tool.ended_at[0] <= task.interrupted_at, "the tool kept working"
    assert task.interrupted_at - stopped_at < STOP_WAIT_SECONDS + 1, "the wait was open-ended"
    assert task.listener_closed, "the listener outlived the block"

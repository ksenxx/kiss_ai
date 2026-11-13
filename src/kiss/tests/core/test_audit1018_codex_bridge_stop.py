# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Codex tool bridge under a task Stop, a fatal limit and lock injection.

Findings of the 2026-10-18 core audit, each reproduced end to end:

* a plain ``with state_lock:`` on the task thread leaked the lock when
  the runner's Stop landed on the acquire's return (gh-74174): the
  bridge's task-thread lock sites now go through ``locked_call``, and
  so does ``interrupt_tool_call``, which the task thread now calls;
* a Stop landing inside ``ToolBridge.__enter__``'s ``Thread.start()``
  left the listener serving for good;
* the per-panel Stop could not reach a tool Codex runs over the bridge,
  because the token is registered under the bridge thread's ident while
  the runner looks it up by the task thread's;
* leaving the bridge joined the handler threads unbounded, so a task
  Stop hung behind a tool that ignores the stop event;
* a fatal limit hit on a bridge thread did not end the Codex turn: the
  run waited for Codex to talk past the error reply;
* the CLI stall deadline was re-armed only once it had expired, so Codex
  got whatever was left of a deadline re-armed mid-tool, not the full
  timeout of silence after the tool returned;
* a call queued behind another tool when the Stop ended the block had
  nothing to interrupt, so once the running tool let go it started its
  own tool after the bridge had returned;
* only ``lines()`` polled the bridge's fatal error: a limit hit while the
  caller was writing the prompt or waiting for the child to wind down
  ended as a stall timeout or a successful turn.
"""

from __future__ import annotations

import _thread
import functools
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.core import stop_signal, tool_interrupt
from kiss.core.kiss_agent import KISSAgent, NativeTurn
from kiss.core.kiss_error import BudgetExceededError
from kiss.core.models.codex_mcp_bridge import STOP_WAIT_SECONDS, ServedCall, ToolBridge
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.model import _CLIProcess, _StreamReadTimeoutError
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    current_tool_interrupt_event,
    end_tool_call,
    interrupt_tool_call,
    new_tool_call,
    raise_if_interrupted,
    register_tool_call,
)
from kiss.core.tool_verdict import ALLOW, ToolRedirect, Verdict, refusal_result
from kiss.core.utils import finish as structured_finish
from kiss.tests.core.test_codex_mcp_bridge import (
    AGENT_UNDER_TEST,
    TASK,
    _install_fake_codex,
    _RecordingPrinter,
    exhaust,
)
from kiss.tests.lock_injection import hammer


def _schema(*names: str) -> list[dict[str, Any]]:
    """An OpenAI tools schema listing one argument-less tool per name."""
    return [
        {
            "type": "function",
            "function": {"name": name, "description": "Nothing.", "parameters": {"type": "object"}},
        }
        for name in names
    ]


NOOP_SCHEMA = _schema("noop")


def _echo(call: dict[str, Any]) -> tuple[str, str]:
    """Executor returning the call's name."""
    return call["name"], call["name"]


# --- (1) task-thread lock sites under injection ------------------------------


class _BridgeReads:
    """``calls_in_flight`` and ``take_served``: the task thread's two lock sites."""

    def __init__(self, bridge: ToolBridge) -> None:
        self.bridge = bridge

    def __call__(self) -> None:
        self.bridge.take_served()
        assert self.bridge.calls_in_flight >= 0


class _InterruptLookup:
    """``interrupt_tool_call`` on a thread running no tool: the lock, no match."""

    def __call__(self) -> None:
        assert interrupt_tool_call(threading.get_ident()) is False


def run_hammer(name: str) -> None:
    """Child-process body: hammer the path *name*; its lock must stay free."""
    if name == "bridge":
        with ToolBridge(NOOP_SCHEMA, _echo) as bridge:
            lock = bridge._server.state_lock
            hammer(_BridgeReads(bridge), {"state_lock": lock}, contend=lock)
    else:
        lock = tool_interrupt._INJECT_LOCK
        hammer(_InterruptLookup(), {"_INJECT_LOCK": lock}, contend=lock)


def test_task_thread_lock_sites_survive_injection() -> None:
    """Neither the bridge's state lock nor the interrupt lock leaks under a Stop."""
    for name in ("bridge", "interrupt"):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({name!r})"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, f"{name}: {proc.stderr[-3000:]}"


# --- (2) a Stop inside __enter__ ---------------------------------------------


def _interrupt_after_spawn(frame: FrameType, event: str, arg: Any) -> None:
    """Raise ``KeyboardInterrupt`` once the OS thread has been spawned, then unhook."""
    if event == "c_return" and arg is _thread.start_joinable_thread:
        sys.setprofile(None)
        raise KeyboardInterrupt


def test_stop_inside_enter_closes_the_listener() -> None:
    """The listener spawned by an interrupted ``__enter__`` is shut down, not left serving."""
    bridge = ToolBridge(NOOP_SCHEMA, _echo)
    port = bridge.port
    sys.setprofile(_interrupt_after_spawn)
    try:
        with pytest.raises(KeyboardInterrupt), bridge:
            raise AssertionError("the body must not run")
    finally:
        sys.setprofile(None)
    bridge._thread.join(5.0)
    assert not bridge._thread.is_alive(), "the listener outlived its interrupted __enter__"
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)


def test_enter_that_never_spawned_the_thread_still_closes() -> None:
    """``Thread.start`` failing before the spawn (thread exhaustion) leaves no listener."""
    bridge = ToolBridge(NOOP_SCHEMA, _echo)
    bridge._thread.start = _refuse_to_start  # type: ignore[method-assign]
    port = bridge.port
    with pytest.raises(RuntimeError, match="can't start new thread"), bridge:
        raise AssertionError("the body must not run")
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)


def _refuse_to_start() -> None:
    """Stand in for ``Thread.start`` when the OS refuses a new thread."""
    raise RuntimeError("can't start new thread")


# --- (3) the Stop button finds a call by id on another thread ----------------


def test_interrupt_by_call_id_reaches_a_call_on_another_thread() -> None:
    """A call id names the call wherever it runs; the thread ident alone does not."""
    tokens: list[Any] = []
    started = threading.Event()
    release = threading.Event()

    def run_tool() -> None:
        token = new_tool_call("bridged")
        register_tool_call(token)
        tokens.append(token)
        started.set()
        release.wait(10)
        end_tool_call(token)

    worker = threading.Thread(target=run_tool, daemon=True)
    worker.start()
    assert started.wait(5)
    token = tokens[0]
    task_ident = threading.get_ident()  # the task thread runs no tool
    try:
        assert interrupt_tool_call(task_ident, "bridged") is False
        assert interrupt_tool_call(task_ident, "bridged", token.call_id + 1) is False
        assert interrupt_tool_call(task_ident, "other", token.call_id) is False
        assert interrupt_tool_call(task_ident, "bridged", token.call_id) is True
        assert token.interrupted and token.event.is_set()
    finally:
        release.set()
        worker.join(5)


# --- (4) and (5): the bridge inside a real codex/* run -----------------------

TOOL_STARTED = threading.Event()


def wait_for_interrupt(text: str) -> str:
    """Block until this call is interrupted, ignoring the task's stop event.

    Args:
        text: Ignored.

    Returns:
        ``"not interrupted"`` if the 30 s wait ran out.
    """
    TOOL_STARTED.set()
    event = current_tool_interrupt_event()
    assert event is not None
    event.wait(30)
    raise_if_interrupted()
    return "not interrupted"


def _run_until_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, turns: list[dict[str, Any]], tools: list[Any]
) -> tuple[KISSAgent, _RecordingPrinter, float]:
    """Run a codex/* agent whose stop event is set once a tool has started.

    Returns:
        The agent, the printer and the seconds from the Stop to the run's end.
    """
    printer = _RecordingPrinter()
    _install_fake_codex(tmp_path, monkeypatch, turns)
    agent = KISSAgent("codex bridge stop")
    stop_event = threading.Event()
    stop_signal.set_thread_stop_event(stop_event)
    TOOL_STARTED.clear()
    stopped_at: list[float] = []

    def stop_once_started() -> None:
        TOOL_STARTED.wait(30)
        stopped_at.append(time.monotonic())
        stop_event.set()

    threading.Thread(target=stop_once_started, daemon=True).start()
    try:
        with pytest.raises(KeyboardInterrupt):
            agent.run(
                model_name="codex/gpt-5.5",
                printer=printer,
                prompt_template=TASK,
                tools=[structured_finish, *tools],
                is_agentic=True,
                verbose=False,
            )
    finally:
        stop_signal.set_thread_stop_event(None)
    assert stopped_at, "the tool never started"
    return agent, printer, time.monotonic() - stopped_at[0]


def test_task_stop_interrupts_a_bridged_tool_and_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Stop while a bridged tool ignores the stop event: the tool is interrupted
    the way the Stop button interrupts one and the run ends promptly."""
    turns = [{"mcp_calls": [["wait_for_interrupt", {"text": "x"}]], "text": "never"}]
    agent, printer, elapsed = _run_until_stopped(tmp_path, monkeypatch, turns, [wait_for_interrupt])
    assert elapsed < STOP_WAIT_SECONDS - 1, f"the run hung {elapsed:.1f}s behind the tool"
    results = [c for t, c, _ in printer.events if t == "tool_result"]
    assert results == [USER_INTERRUPTED_MESSAGE]
    # The interrupted call still reaches the transcript and the trajectory.
    assert ("user", f"[wait_for_interrupt]: {USER_INTERRUPTED_MESSAGE}") in [
        (m["role"], m["content"]) for m in agent.messages
    ]


def stuck(text: str) -> str:
    """Sleep in C for 20 s: deaf to the stop event, the interrupt event and the injection.

    Args:
        text: Ignored.

    Returns:
        ``"woke"``.
    """
    TOOL_STARTED.set()
    time.sleep(20)
    return "woke"


def test_task_stop_abandons_a_tool_that_survives_the_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool nothing can interrupt is abandoned after the bounded wait, not joined for good."""
    turns = [{"mcp_calls": [["stuck", {"text": "x"}]], "text": "never"}]
    _agent, printer, elapsed = _run_until_stopped(tmp_path, monkeypatch, turns, [stuck])
    assert STOP_WAIT_SECONDS <= elapsed < STOP_WAIT_SECONDS + 5, f"the run took {elapsed:.1f}s"
    assert [c for t, c, _ in printer.events if t == "tool_result"] == []


def test_fatal_limit_on_the_bridge_ends_the_turn_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex hangs after a tool spent the whole budget: the turn is ended by the
    limit error without waiting for Codex (or its stall timeout)."""
    printer = _RecordingPrinter()
    turns = [{"mcp_calls": [["exhaust", {"text": "all"}]], "text": "never", "hang": 30}]
    _install_fake_codex(tmp_path, monkeypatch, turns)
    agent = KISSAgent("codex bridge limit")
    AGENT_UNDER_TEST.append(agent)
    began = time.monotonic()
    with pytest.raises(BudgetExceededError, match="budget exceeded"):
        agent.run(
            model_name="codex/gpt-5.5",
            printer=printer,
            prompt_template=TASK,
            tools=[structured_finish, exhaust],
            is_agentic=True,
            verbose=False,
        )
    assert time.monotonic() - began < 10
    # The call that ran is in the trajectory and replayed into the model
    # conversation (with the limit error as its recorded outcome).
    assert ("user", "[exhaust]: spent") in [(m["role"], m["content"]) for m in agent.messages]
    tool_msgs = [m["content"] for m in agent.model.conversation if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and "budget exceeded" in tool_msgs[0]


def test_a_call_still_being_read_when_the_block_ends_is_refused() -> None:
    """A handler accepted but not yet registered when the block ends runs no tool after it.

    socketserver's own close joined every handler, reader included; the
    bridge now joins only registered calls, so the pre-registration
    window is closed by refusing the call instead.
    """
    executed: list[dict[str, Any]] = []

    def execute(call: dict[str, Any]) -> tuple[str, str]:
        executed.append(call)
        return call["name"], "done"

    with ToolBridge(NOOP_SCHEMA, execute) as bridge:
        port = bridge.port
        client = socket.create_connection(("127.0.0.1", port))
        request = {"op": "call", "name": "noop", "arguments": {}, "token": bridge._server.token}
        client.sendall(json.dumps(request)[:-1].encode())  # no "}" and newline yet
        time.sleep(0.3)  # the handler is accepted and blocked in readline
    client.sendall(b"}\n")
    reply = json.loads(client.makefile("rb").readline())
    client.close()
    assert reply == {"error": "KISSError: the KISS tool bridge is closed"}
    assert executed == []


def test_bridge_records_the_first_executor_error() -> None:
    """``error`` is the first exception the executor raised; ordinary calls leave it ``None``."""
    with ToolBridge(NOOP_SCHEMA, _echo) as bridge:
        assert bridge.error is None
    seen: list[dict[str, Any]] = []

    def explode(call: dict[str, Any]) -> tuple[str, str]:
        seen.append(call)
        raise RuntimeError(f"boom {len(seen)}")

    with ToolBridge(NOOP_SCHEMA, explode) as bridge:
        for _ in range(2):
            with _call(bridge, "noop") as client:
                assert b"RuntimeError: boom" in client.makefile("rb").readline()
        assert str(bridge.error) == "boom 1"
        assert bridge.take_served() == [
            ServedCall("noop", {}, "RuntimeError: boom 1"),
            ServedCall("noop", {}, "RuntimeError: boom 2"),
        ]


# --- (6) the stall deadline while a tool runs ---------------------------------


class _ToolUntil:
    """``waiting_on_caller`` that reports a tool running until *until* (monotonic)."""

    def __init__(self, until: float) -> None:
        self.until = until

    def __call__(self) -> bool:
        return time.monotonic() < self.until


def test_child_gets_the_full_timeout_of_silence_after_a_tool_returned() -> None:
    """A line 1.2 s after the tool returned, under a 2 s timeout, is not a stall.

    Before the fix the deadline was re-armed only once expired (at 2 s,
    to 4 s): the tool returned at 3 s and the child, printing at 4.2 s,
    was killed as stalled at 4 s.
    """
    began = time.monotonic()
    with _CLIProcess(
        [sys.executable, "-c", "import time; time.sleep(4.2); print('late', flush=True)"],
        "fake",
        timeout=2.0,
        waiting_on_caller=_ToolUntil(began + 3.0),
    ) as proc:
        assert list(proc.lines()) == ["late\n"]


def test_caller_error_ends_the_wait_at_once() -> None:
    """The error the caller reports is raised from ``lines()`` within a poll, not a timeout."""
    error = BudgetExceededError("spent")

    def report() -> Exception | None:
        return error

    began = time.monotonic()
    with (
        _CLIProcess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            "fake",
            timeout=60.0,
            caller_error=report,
        ) as proc,
        pytest.raises(BudgetExceededError, match="spent"),
    ):
        list(proc.lines())
    assert time.monotonic() - began < 5


def test_silence_without_a_tool_is_still_a_stall() -> None:
    """``waiting_on_caller`` false: the timeout is the timeout."""
    with (
        _CLIProcess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            "fake",
            timeout=0.5,
            waiting_on_caller=_ToolUntil(0.0),
        ) as proc,
        pytest.raises(_StreamReadTimeoutError),
    ):
        list(proc.lines())


# --- (7) a call queued behind another tool when the Stop ends the block -------

HOLD_STARTED = threading.Event()
RELEASE_HOLD = threading.Event()
LATE_CALLS: list[dict[str, Any]] = []


def hold_until_interrupted() -> str:
    """Block until this call is interrupted (a Stop reaches it at once).

    Returns:
        ``"not interrupted"`` if the 30 s wait ran out.
    """
    HOLD_STARTED.set()
    event = current_tool_interrupt_event()
    assert event is not None
    event.wait(30)
    raise_if_interrupted()
    return "not interrupted"


def hold_until_released() -> str:
    """Block until the test releases it, deaf to the interrupt (the Stop abandons it).

    Returns:
        ``"released"``.
    """
    HOLD_STARTED.set()
    RELEASE_HOLD.wait(30)
    return "released"


def later() -> str:
    """The queued tool: must never run once the Stop ended the block.

    Returns:
        ``"late side effect"``.
    """
    LATE_CALLS.append({"stop_set": stop_signal.stop_requested()})
    return "late side effect"


def _call(bridge: ToolBridge, name: str) -> socket.socket:
    """Send one tool call to *bridge* and return the open connection."""
    client = socket.create_connection(("127.0.0.1", bridge.port), timeout=15)
    request = {"op": "call", "name": name, "arguments": {}, "token": bridge._server.token}
    client.sendall(json.dumps(request).encode() + b"\n")
    return client


def _reply(client: socket.socket) -> dict[str, Any]:
    """Read the bridge's reply line from *client* and close it."""
    with client, client.makefile("rb") as reply:
        return dict(json.loads(reply.readline()))


def _end_hold(calls: list[socket.socket], frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: the held tool returns and both calls are answered before the listener stops.
    Each reply is awaited on its socket, within the socket's timeout, and left there to read."""
    if event == "call" and frame.f_code is ToolBridge._stop_listener.__code__:
        sys.setprofile(None)
        RELEASE_HOLD.set()
        answered = [call.recv(1, socket.MSG_PEEK) for call in calls]
        assert all(answered), "a call was never answered"


@pytest.mark.parametrize(
    ("hold", "at_close"),
    [(hold_until_interrupted, None), (hold_until_released, None), (hold_until_released, _end_hold)],
)
def test_a_call_queued_behind_a_tool_when_the_stop_ends_the_block_is_refused(
    hold: Any, at_close: Any
) -> None:
    """Two calls through the agent's real executor: a Stop ends the block while the second
    waits behind the first.  Whether the first obeys the interrupt, is abandoned after
    :data:`STOP_WAIT_SECONDS` or returns as the listener stops, the second never starts its tool.
    """
    HOLD_STARTED.clear()
    RELEASE_HOLD.clear()
    LATE_CALLS.clear()
    stop = threading.Event()
    agent = KISSAgent("stopped bridge queue")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {hold.__name__: hold, "later": later}
    execute = functools.partial(
        agent._execute_tool_on_helper_thread,
        stop_event=stop,
        printer_binding=None,
        turn=NativeTurn(),
    )
    stop_signal.set_thread_stop_event(stop)
    try:
        with ToolBridge(_schema(hold.__name__, "later"), execute) as bridge:
            first = _call(bridge, hold.__name__)
            assert HOLD_STARTED.wait(5)
            second = _call(bridge, "later")
            deadline = time.monotonic() + 5
            while bridge.calls_in_flight != 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert bridge.calls_in_flight == 2
            stopped_at = time.monotonic()
            stop.set()
            if at_close:
                sys.setprofile(functools.partial(at_close, [second, first]))
        elapsed = time.monotonic() - stopped_at
        if hold is hold_until_interrupted or at_close:
            assert elapsed < STOP_WAIT_SECONDS - 1, f"the block took {elapsed:.1f}s to end"
        else:
            assert elapsed >= STOP_WAIT_SECONDS, f"the deaf tool was not abandoned ({elapsed:.1f}s)"
        RELEASE_HOLD.set()
        assert _reply(second) == {"error": "KISSError: the KISS tool bridge is closed"}
        _reply(first)
        assert LATE_CALLS == [], "the queued tool ran after the bridge returned"
        assert bridge.calls_in_flight == 0
    finally:
        sys.setprofile(None)
        RELEASE_HOLD.set()
        stop_signal.set_thread_stop_event(None)


class _ResultHook:
    """A run's ``tool_result_hook``: records every result it is given."""

    def __init__(self) -> None:
        self.results: list[tuple[str, dict[str, Any], str, bool]] = []

    def __call__(self, name: str, args: dict[str, Any], result: str, refused: bool) -> None:
        self.results.append((name, args, result, refused))


def redirect_to_hold() -> str:
    """Redirect to :func:`hold_until_released`: the deaf tool runs nested in this call.

    Raises:
        ToolRedirect: Always.
    """
    raise ToolRedirect("hold_until_released", {}, "redirected:")


@pytest.mark.parametrize("tool", [hold_until_released, redirect_to_hold])
def test_a_tool_the_stop_abandoned_settles_nothing_once_its_turn_is_closed(tool: Any) -> None:
    """A deaf tool the Stop abandoned returns after the agent was reset for the
    next run, which installed its own ``tool_result_hook`` and owes a re-issue:
    the abandoned call's result reaches neither (the turn it belongs to is
    closed), exactly as it stays out of the conversation.  The same holds when
    the deaf tool is the one a served call redirected to (a nested call of the
    same turn): neither its result nor the count it restores is the next run's.
    The call's late ``tool_result`` event goes to the printer of the run the
    call belongs to, never to the printer the reset installed for the next run."""
    HOLD_STARTED.clear()
    RELEASE_HOLD.clear()
    stop = threading.Event()
    agent = KISSAgent("abandoned native call")
    old_printer = _RecordingPrinter()
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, printer=old_printer, verbose=False)
    agent.function_map = {"hold_until_released": hold_until_released, tool.__name__: tool}
    agent._cached_tools_schema = _schema(tool.__name__)
    old_run = _ResultHook()
    agent.tool_result_hook = old_run
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            assert isinstance(agent.model, CodexModel)
            bridge = agent.model._tool_bridge
            assert bridge is not None
            first = _call(bridge, tool.__name__)
            assert HOLD_STARTED.wait(5)
            stop.set()
        assert agent._native_turn.closed.is_set()
        new_printer = _RecordingPrinter()
        agent._reset("codex/gpt-5.5", True, 10, 10.0, None, printer=new_printer, verbose=False)
        new_run = _ResultHook()
        agent.tool_result_hook = new_run
        agent._refused_call = "Bash"
        RELEASE_HOLD.set()
        _reply(first)
        assert old_run.results == []
        assert new_run.results == [], "the abandoned call reached the next run's hook"
        assert agent._refused_call == "Bash", "the abandoned call settled the next run's refusal"
        assert agent.messages == [] and agent.tool_calls_made == 0
        assert new_printer.events == [], "the abandoned call's event reached the next run's printer"
        nested = [] if tool is hold_until_released else ["tool_call", "tool_result"]
        assert [t for t, _, _ in old_printer.events] == ["tool_call", *nested, "tool_result"]
        # The Stop's interrupt, drained when the deaf tool returned, is the result.
        assert old_printer.events[-1][1].endswith(USER_INTERRUPTED_MESSAGE)
    finally:
        RELEASE_HOLD.set()
        stop_signal.set_thread_stop_event(None)


HOOK_STARTED = threading.Event()
RELEASE_HOOK = threading.Event()
PROBE_RAN: list[str] = []


def probe() -> str:
    """The tool behind the blocked hook: must never run once its turn is closed.

    Returns:
        ``"probed"``.
    """
    PROBE_RAN.append("probe")
    return "probed"


def hold_hook_until_released(name: str, args: dict[str, Any]) -> Verdict:
    """A ``tool_call_hook`` deaf to the Stop: allows once the test releases it."""
    HOOK_STARTED.set()
    RELEASE_HOOK.wait(30)
    return ALLOW


class _Guard:
    """A run's ``tool_call_guard``: records every call it is asked about and allows it."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, name: str, args: dict[str, Any]) -> None:
        self.asked.append(name)


def test_a_hook_the_stop_abandoned_consults_no_guard_once_its_turn_is_closed() -> None:
    """A served call blocked in its ``tool_call_hook`` is abandoned by the Stop;
    the agent is reset for the next run, which installs its own ``tool_call_guard``.
    When the old hook finally allows, neither that guard (Sorcar's mutates the
    coding gate) nor the tool is called: the call is refused as the turn's."""
    HOOK_STARTED.clear()
    RELEASE_HOOK.clear()
    PROBE_RAN.clear()
    stop = threading.Event()
    agent = KISSAgent("abandoned native hook")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"probe": probe}
    agent._cached_tools_schema = _schema("probe")
    agent.tool_call_hook = hold_hook_until_released
    old_guard = _Guard()
    agent.tool_call_guard = old_guard
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            assert isinstance(agent.model, CodexModel)
            bridge = agent.model._tool_bridge
            assert bridge is not None
            first = _call(bridge, "probe")
            assert HOOK_STARTED.wait(5)
            stop.set()
        assert agent._native_turn.closed.is_set()
        agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
        new_guard = _Guard()
        agent.tool_call_guard = new_guard
        RELEASE_HOOK.set()
        reply = _reply(first)
        assert reply == {"text": refusal_result("probe", "The turn has ended.")}
        assert new_guard.asked == [], "the abandoned call consulted the next run's guard"
        assert old_guard.asked == [] and PROBE_RAN == []
        assert agent.messages == [] and agent.tool_calls_made == 0
    finally:
        RELEASE_HOOK.set()
        stop_signal.set_thread_stop_event(None)


def test_a_call_reaching_the_executor_after_its_turn_closed_counts_nothing() -> None:
    """A served call handed to the executor once its turn is closed is refused and
    leaves the agent's call count alone, like the rest of its books: the count is
    the next run's by then, and so are the ``tool_call_hook`` and the
    ``tool_call_guard``, which are not asked about it (the root SEA's hook takes a
    ``summary`` call for the summary its run owes).  The executor is called
    directly with a closed turn."""
    PROBE_RAN.clear()
    agent = KISSAgent("closed turn count")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"summary": probe}
    sea = BaseSea()
    sea.summary_due = True
    agent.tool_call_hook = sea.tool_call_hook
    guard = _Guard()
    agent.tool_call_guard = guard
    turn = NativeTurn()
    turn.closed.set()
    try:
        name, response = agent._execute_tool_on_helper_thread(
            {"name": "summary", "arguments": {}}, stop_event=None, printer_binding=None, turn=turn
        )
        assert (name, response) == ("summary", refusal_result("summary", "The turn has ended."))
        assert PROBE_RAN == [] and agent.messages == []
        assert agent.tool_calls_made == 0
        assert sea.summary_due is True, "the next run's tool_call_hook was asked about the call"
        assert guard.asked == []
    finally:
        stop_signal.set_thread_stop_event(None)


PAUSED_BEFORE_TOOL = threading.Event()
RESUME_TOOL = threading.Event()


def _pause_before_the_tool(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook of a bridge thread: hold an admitted call at ``_execute_tool``'s entry."""
    if event == "call" and frame.f_code is KISSAgent._execute_tool.__code__:
        PAUSED_BEFORE_TOOL.set()
        RESUME_TOOL.wait(30)


def test_a_call_admitted_before_its_turn_closed_runs_no_tool_after_it() -> None:
    """A served call the Stop abandoned between its admission and its tool (nothing to
    interrupt yet) resumes once the turn is closed and the agent reset for the next run,
    which registered another tool under the same name: the call is refused as the
    turn's, and runs neither tool."""
    PAUSED_BEFORE_TOOL.clear()
    RESUME_TOOL.clear()
    PROBE_RAN.clear()
    LATE_CALLS.clear()
    stop = threading.Event()
    agent = KISSAgent("abandoned native dispatch")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"probe": probe}
    agent._cached_tools_schema = _schema("probe")
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            assert isinstance(agent.model, CodexModel)
            bridge = agent.model._tool_bridge
            assert bridge is not None
            threading.setprofile(_pause_before_the_tool)
            first = _call(bridge, "probe")
            assert PAUSED_BEFORE_TOOL.wait(5)
            threading.setprofile(None)
            stop.set()
        assert agent._native_turn.closed.is_set()
        agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
        agent.function_map = {"probe": later}
        RESUME_TOOL.set()
        assert _reply(first) == {"text": refusal_result("probe", "The turn has ended.")}
        assert LATE_CALLS == [], "the abandoned call ran the next run's tool"
        assert PROBE_RAN == []
        assert agent.messages == [] and agent.tool_calls_made == 0
    finally:
        threading.setprofile(None)
        RESUME_TOOL.set()
        stop_signal.set_thread_stop_event(None)


def _allow_as_the_bridge_gives_up(frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook of the agent's thread: the held hook allows once the bridge has given up
    on its call (the served calls are being replayed) and the tool starts before the turn
    is closed."""
    if event == "call" and frame.f_code is CodexModel._replay_served_calls.__code__:
        sys.setprofile(None)
        RELEASE_HOOK.set()
        assert HOLD_STARTED.wait(5), "the tool never started"


def test_a_tool_started_after_the_stop_gave_up_is_interrupted_when_its_turn_closes() -> None:
    """A served call sits in its ``tool_call_hook`` through the Stop's bounded wait, which
    finds nothing to interrupt; the hook allows right after, with the turn still open, and
    the tool starts.  Closing the turn interrupts it: nobody else is left to."""
    HOOK_STARTED.clear()
    RELEASE_HOOK.clear()
    HOLD_STARTED.clear()
    stop = threading.Event()
    agent = KISSAgent("late native tool")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"hold_until_interrupted": hold_until_interrupted}
    agent._cached_tools_schema = _schema("hold_until_interrupted")
    agent.tool_call_hook = hold_hook_until_released
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            assert isinstance(agent.model, CodexModel)
            bridge = agent.model._tool_bridge
            assert bridge is not None
            first = _call(bridge, "hold_until_interrupted")
            assert HOOK_STARTED.wait(5)
            stop.set()
            sys.setprofile(_allow_as_the_bridge_gives_up)
        assert agent._native_turn.closed.is_set() and HOLD_STARTED.is_set()
        closed_at = time.monotonic()
        assert _reply(first) == {"text": USER_INTERRUPTED_MESSAGE}
        assert time.monotonic() - closed_at < 5, "the tool ran on after its turn was closed"
    finally:
        sys.setprofile(None)
        RELEASE_HOOK.set()
        stop_signal.set_thread_stop_event(None)


# --- (8) the bridge's fatal error in the other two CLI waits -------------------


class _BridgeError:
    """``caller_error`` reporting the bridge's first fatal executor error."""

    def __init__(self, bridge: ToolBridge) -> None:
        self.bridge = bridge

    def __call__(self) -> Exception | None:
        return self.bridge.error


def _spend_slowly(call: dict[str, Any]) -> tuple[str, str]:
    """Executor that hits the budget 0.3 s into every call."""
    time.sleep(0.3)
    raise BudgetExceededError("spent on the bridge")


def _child_calling_the_bridge(bridge: ToolBridge, then: str) -> list[str]:
    """A child that asks *bridge* for one tool call, then runs *then*."""
    request = {"op": "call", "name": "noop", "arguments": {}, "token": bridge._server.token}
    line = json.dumps(request).encode() + b"\n"
    script = (
        "import os, socket, time\n"
        f"s = socket.create_connection(('127.0.0.1', {bridge.port}))\n"
        f"s.sendall({line!r})\n" + then
    )
    return [sys.executable, "-c", script]


def test_fatal_tool_error_ends_the_wind_down_wait() -> None:
    """The child closed stdout with a tool call in flight and lingers: the error the
    call raises ends ``raise_for_exit`` at once instead of a normal return."""
    with ToolBridge(NOOP_SCHEMA, _spend_slowly) as bridge:
        child = _child_calling_the_bridge(bridge, "os.close(1)\ns.recv(4096)\ntime.sleep(2)\n")
        began = time.monotonic()
        with _CLIProcess(child, "fake", timeout=20.0, caller_error=_BridgeError(bridge)) as proc:
            assert list(proc.lines()) == []
            with pytest.raises(BudgetExceededError, match="spent on the bridge"):
                proc.raise_for_exit()
        assert time.monotonic() - began < 1.5


def test_fatal_tool_error_ends_a_blocked_prompt_write() -> None:
    """The child never reads its prompt while its tool call hits the budget: the
    error ends ``send_prompt`` at once, and its writer with it, not a stall timeout."""
    with ToolBridge(NOOP_SCHEMA, _spend_slowly) as bridge:
        child = _child_calling_the_bridge(bridge, "s.recv(4096)\ntime.sleep(8)\n")
        began = time.monotonic()
        with (
            _CLIProcess(child, "fake", timeout=1.5, caller_error=_BridgeError(bridge)) as proc,
            pytest.raises(BudgetExceededError, match="spent on the bridge"),
        ):
            proc.send_prompt("x" * 1_000_000)
        assert time.monotonic() - began < 1.2
        assert proc._writer is not None and not proc._writer.is_alive()

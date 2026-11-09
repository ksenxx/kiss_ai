# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Codex tool bridge: unknown tools, a Stop on the handler thread, late calls.

Three findings of the 2026-10 core audit, each reproduced end to end:

* the bridge forwarded any tool name to the executor, so Codex could
  call ``finish`` (kept off the bridge so a turn ends when the task
  does) or a between-turns tool through the agent's executor;
* a task Stop lands on the bridge's handler thread as the
  ``KeyboardInterrupt`` the daemon's printer raises, which the bridge's
  ``except Exception`` did not catch: the handler thread died with it,
  Codex got no reply and the call was never recorded;
* a tool abandoned by the Stop (``_wait_for_calls`` gave up on it) still
  wrote its call into the conversation when it finally returned, while
  the agent thread was already past the turn;
* an abandoned call that finally *raised* still published its exception
  as the turn's fatal error, overwriting the reset the next turn had made;
* the guard added for that ("publish unless the bridge has closed") and
  the publication were not atomic: a handler past the guard when the Stop
  closed the bridge still overwrote the next turn's reset.  The error now
  lives on the turn's own record (``NativeTurn``), which the next turn
  replaces, so a stale handler has nothing shared left to write to.
"""

from __future__ import annotations

import inspect
import json
import socket
import sys
import threading
import time
from typing import Any

from kiss.core import stop_signal
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent, NativeTurn
from kiss.core.kiss_error import KISSError
from kiss.core.locked import ACQUIRE_SLICE_SECONDS, locked_call
from kiss.core.models.codex_mcp_bridge import STOP_WAIT_SECONDS, ToolBridge
from kiss.core.models.codex_model import CodexModel
from kiss.core.tool_interrupt import ToolCallInterrupted, inject_async_exception
from kiss.core.tool_verdict import ALLOW, Verdict, refusal_result, refuse
from kiss.server.json_printer import JsonPrinter
from kiss.server.task_runner import _TaskRunnerMixin
from kiss.tests.core.test_audit1018_codex_bridge_stop import (
    HOLD_STARTED,
    RELEASE_HOLD,
    _call,
    _reply,
    _schema,
    hold_until_released,
)

EXECUTED: list[str] = []


def _record(call: dict[str, Any]) -> tuple[str, str]:
    """Executor that records every call it is asked to run."""
    EXECUTED.append(call["name"])
    return call["name"], "ran"


def _bridge_of(agent: KISSAgent) -> ToolBridge:
    """The tool bridge the agent's Codex model has open."""
    assert isinstance(agent.model, CodexModel)
    assert agent.model._tool_bridge is not None
    return agent.model._tool_bridge


def test_a_tool_the_bridge_does_not_serve_is_refused_before_the_executor() -> None:
    """``finish`` is not on the bridge: the call is answered with an error, not run."""
    EXECUTED.clear()
    with ToolBridge(_schema("noop"), _record) as bridge:
        assert bridge.tool_names == frozenset({"noop"})
        assert _reply(_call(bridge, "finish")) == {"error": "KISSError: unknown tool 'finish'"}
        assert _reply(_call(bridge, "noop")) == {"text": "ran"}
    assert EXECUTED == ["noop"]
    assert [served.name for served in bridge.take_served()] == ["noop"]


def noop() -> str:
    """A tool that returns at once."""
    return "noop ran"


def test_a_stop_raised_by_the_printer_is_reported_to_codex_and_recorded() -> None:
    """The daemon's printer raises ``KeyboardInterrupt`` under a Stop; on the bridge's
    thread that must become an error reply and the turn's fatal error, not a dead
    handler thread and a connection closed without a reply."""
    stop = threading.Event()
    stop.set()
    agent = KISSAgent("stopped printer on the bridge")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, printer=JsonPrinter(), verbose=False)
    agent.function_map = {"noop": noop}
    agent._cached_tools_schema = _schema("noop")
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            bridge = _bridge_of(agent)
            reply = _reply(_call(bridge, "noop"))
    finally:
        stop_signal.set_thread_stop_event(None)
    assert reply == {"error": "KISSError: KISS Error: Tool noop stopped: Agent stop requested"}
    assert isinstance(agent._native_turn.error, KISSError)
    assert agent._native_turn.calls == 0, "a stopped call must not count as a served step"


def test_a_call_abandoned_by_the_stop_leaves_the_conversation_alone_when_it_returns() -> None:
    """A tool deaf to the interrupt is abandoned after ``STOP_WAIT_SECONDS``; when it
    finally returns, the turn is long over and its record must not be written then."""
    HOLD_STARTED.clear()
    RELEASE_HOLD.clear()
    stop = threading.Event()
    agent = KISSAgent("abandoned bridge call")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {hold_until_released.__name__: hold_until_released}
    agent._cached_tools_schema = _schema(hold_until_released.__name__)
    messages_before = list(agent.messages)
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            bridge = _bridge_of(agent)
            client = _call(bridge, hold_until_released.__name__)
            assert HOLD_STARTED.wait(5)
            stopped_at = time.monotonic()
            stop.set()
        assert time.monotonic() - stopped_at >= STOP_WAIT_SECONDS, "the deaf tool was not abandoned"
        RELEASE_HOLD.set()
        # The executor returns before the reply is written, so once the
        # reply is read the bookkeeping has either happened or been skipped.
        # (The late injection lands in the tool, so the text is the
        # interrupted-call message rather than the tool's own.)
        assert "text" in _reply(client)
    finally:
        RELEASE_HOLD.set()
        stop_signal.set_thread_stop_event(None)
    assert agent._native_turn.calls == 0, "the abandoned call was recorded after the turn"
    assert agent.messages == messages_before


class _HeldPastTheGuard:
    """A trace hook holding the bridge's handler at its first ``_add_message``.

    The handler is past the "publish unless the turn has closed" check
    there; it waits until the turn closes (the interleaving under test)
    or the test lets go, bounded, and records whether the turn was
    closed as each message was appended.  Installed with
    ``threading.settrace`` for the threads started after it.

    Args:
        turn: The turn record the handler publishes for.
        owner: The thread running the turn; its own appends are not held.
    """

    def __init__(self, turn: NativeTurn, owner: threading.Thread) -> None:
        self.turn = turn
        self.owner = owner
        self.paused = threading.Event()
        self.release = threading.Event()
        self.closed_at_append: list[bool] = []

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if (
            event != "call"
            or frame.f_code is not Base._add_message.__code__
            or threading.current_thread() is self.owner
        ):
            return None
        self.paused.set()
        deadline = time.monotonic() + 30
        while not (self.turn.closed.is_set() or self.release.is_set()):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        self.closed_at_append.append(self.turn.closed.is_set())
        return None


def test_a_call_past_the_closed_check_when_the_stop_closes_the_turn_is_published_first() -> None:
    """The check of ``closed`` and the publication (the routine-turn note, the two
    transcript messages, the limit check) were not one step: a handler past the check
    when the Stop closed the turn published afterwards, into the transcript the agent
    had moved on to — a later run's, once the agent was reused.  Closing the turn now
    waits for a publication in progress, so nothing lands on the agent after it; and
    the daemon's forced Stop (the task runner's watchdog injects ``KeyboardInterrupt``
    twice: the bridge's wait for the call takes the first, the closing wait the second)
    cannot cut that wait short, or the publication would land in the next run after
    all.  The owner thread reuses the agent once the run returns, as a tab's next task
    does, and the publication is let go only once both injections have landed."""
    stop = threading.Event()
    agent = KISSAgent("abandoned bridge call past the check")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"noop": noop}
    agent._cached_tools_schema = _schema("noop")
    turn = agent._native_turn
    entered = threading.Event()
    leave = threading.Event()
    bridges: list[ToolBridge] = []
    outcome: list[str] = []
    closed_at_exit: list[bool] = []

    def own_the_turn() -> None:
        stop_signal.set_thread_stop_event(stop)
        try:
            with agent._native_tools_exposed():
                bridges.append(_bridge_of(agent))
                entered.set()
                leave.wait(30)
        except KeyboardInterrupt:
            outcome.append("KeyboardInterrupt")
        finally:
            closed_at_exit.append(turn.closed.is_set())
            # The run has returned: the agent is reused for its next run.
            agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
            agent._add_message("user", "NEW RUN")
            stop_signal.set_thread_stop_event(None)

    owner = threading.Thread(target=own_the_turn, daemon=True)
    watchdog = threading.Thread(
        target=_TaskRunnerMixin._force_stop_thread, args=(owner, owner.is_alive), daemon=True
    )
    held = _HeldPastTheGuard(turn, owner)
    threading.settrace(held)
    try:
        owner.start()
        assert entered.wait(5)
        client = _call(bridges[0], "noop")
        assert held.paused.wait(5), "the handler never reached the publication"
        stop.set()
        leave.set()  # the Stop waits STOP_WAIT_SECONDS for the call, abandons it, closes
        watchdog.start()
        watchdog.join(STOP_WAIT_SECONDS * 2 + 5)  # both injections landed, or the owner died
        assert not watchdog.is_alive()
        held.release.set()
        assert _reply(client) == {"text": "noop ran"}
        owner.join(10)
    finally:
        held.release.set()
        leave.set()
        threading.settrace(None)
    assert not owner.is_alive()
    assert outcome == ["KeyboardInterrupt"], "the forced Stop must still end the run"
    assert closed_at_exit == [True], "the owner left the turn before it was closed"
    assert held.closed_at_append == [False, False], "a message was appended after the turn closed"
    assert turn.calls == 1
    assert [m["content"] for m in agent.messages] == ["NEW RUN"], "the old call reached the new run"


def _hold_then_raise(name: str, args: dict[str, Any]) -> Verdict:
    """A tool-call hook that blocks until released, then raises (a hook's error is fatal)."""
    HOLD_STARTED.set()
    RELEASE_HOLD.wait(30)
    raise RuntimeError("late hook failure")


def test_a_call_abandoned_by_the_stop_leaves_the_fatal_error_alone_when_it_raises() -> None:
    """The next turn replaces the turn record before its bridge opens; an abandoned
    call of the previous turn that raises after that must not touch the new record, or
    the new turn ends with the old turn's error."""
    HOLD_STARTED.clear()
    RELEASE_HOLD.clear()
    stop = threading.Event()
    agent = KISSAgent("abandoned bridge call that raises")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"noop": noop}
    agent._cached_tools_schema = _schema("noop")
    agent.tool_call_hook = _hold_then_raise
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            bridge = _bridge_of(agent)
            client = _call(bridge, "noop")
            assert HOLD_STARTED.wait(5)
            stop.set()
        agent._native_turn = NativeTurn()  # what the next turn's _execute_step does
        RELEASE_HOLD.set()
        assert "error" in _reply(client)
    finally:
        RELEASE_HOLD.set()
        stop_signal.set_thread_stop_event(None)
    assert agent._native_turn.error is None, "the abandoned call's error leaked into the next turn"


def test_an_abandoned_call_that_raises_records_its_error_on_its_own_turn_only() -> None:
    """The race a "publish unless the bridge has closed" check cannot rule out: the
    handler passes the check, the Stop closes the bridge and the next turn resets the
    error, then the handler publishes.  With the error on the turn's own record there
    is no shared slot to race for: the abandoned turn keeps the error, the running
    turn stays clean, whenever the handler gets to publish."""
    HOLD_STARTED.clear()
    RELEASE_HOLD.clear()
    stop = threading.Event()
    agent = KISSAgent("abandoned bridge call publishing late")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"noop": noop}
    agent._cached_tools_schema = _schema("noop")
    agent.tool_call_hook = _hold_then_raise
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            abandoned = agent._native_turn
            bridge = _bridge_of(agent)
            client = _call(bridge, "noop")
            assert HOLD_STARTED.wait(5)
            stop.set()
        assert abandoned.closed.is_set()
        agent._native_turn = NativeTurn()  # the next turn begins
        RELEASE_HOLD.set()  # the handler raises now, long after the bridge closed
        assert "error" in _reply(client)
    finally:
        RELEASE_HOLD.set()
        stop_signal.set_thread_stop_event(None)
    assert isinstance(abandoned.error, RuntimeError), "the error must be recorded on its turn"
    assert agent._native_turn.error is None, "the abandoned call's error reached the next turn"
    assert agent._native_turn is not abandoned


HOOK_STARTED = threading.Event()
RELEASE_HOOK = threading.Event()


def _hold_result_hook(name: str, args: dict[str, Any], result: str, refused: bool) -> None:
    """A tool-result hook that blocks until released (a gate waiting on git, say)."""
    HOOK_STARTED.set()
    RELEASE_HOOK.wait(30)


def test_a_stop_closes_the_turn_while_a_result_hook_is_still_running() -> None:
    """The handler called the ``tool_result_hook`` under ``turn.lock``, the lock the
    Stop closes the turn under: a hook that blocks (Sorcar's gate runs git in it)
    is outside the tool's registered lifetime, so the bridge's interrupt never reaches
    it, the bridge abandons the handler after ``STOP_WAIT_SECONDS`` and the owner then
    waited for the lock with no bound at all, retrying even the forced Stop's
    ``KeyboardInterrupt``.  The hook now runs before the lock is taken, so closing the
    turn waits only for the in-memory publication; the hook of an abandoned call that
    returns after the close publishes nothing."""
    HOOK_STARTED.clear()
    RELEASE_HOOK.clear()
    stop = threading.Event()
    agent = KISSAgent("result hook under the turn lock")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"noop": noop}
    agent._cached_tools_schema = _schema("noop")
    agent.tool_result_hook = _hold_result_hook
    turn = agent._native_turn
    messages_before = list(agent.messages)
    entered = threading.Event()
    leave = threading.Event()
    bridges: list[ToolBridge] = []

    def own_the_turn() -> None:
        stop_signal.set_thread_stop_event(stop)
        try:
            with agent._native_tools_exposed():
                bridges.append(_bridge_of(agent))
                entered.set()
                leave.wait(30)
        finally:
            stop_signal.set_thread_stop_event(None)

    owner = threading.Thread(target=own_the_turn, daemon=True)
    owner.start()
    try:
        assert entered.wait(5)
        client = _call(bridges[0], "noop")
        assert HOOK_STARTED.wait(5), "the result hook never ran"
        stop.set()
        leave.set()  # the Stop waits STOP_WAIT_SECONDS for the handler, abandons it, closes
        owner.join(STOP_WAIT_SECONDS + 10)
        assert not owner.is_alive(), "closing the turn waited for the result hook"
        assert turn.closed.is_set()
        RELEASE_HOOK.set()
        assert _reply(client) == {"text": "noop ran"}
    finally:
        RELEASE_HOOK.set()
        leave.set()
        owner.join(STOP_WAIT_SECONDS + 10)
    assert turn.calls == 0, "the abandoned call was recorded after the turn closed"
    assert agent.messages == messages_before


def _hold_then_allow(name: str, args: dict[str, Any]) -> Verdict:
    """A tool-call hook that blocks until released, then allows the call."""
    HOLD_STARTED.set()
    RELEASE_HOLD.wait(30)
    return ALLOW


def _hold_then_refuse(name: str, args: dict[str, Any]) -> Verdict:
    """A tool-call hook that blocks until released, then refuses the call."""
    HOLD_STARTED.set()
    RELEASE_HOLD.wait(30)
    return refuse("not now")


NEW_RUN_CALLS: list[str] = []


def new_run_noop() -> str:
    """The next run's ``noop``: an abandoned call of the previous run must never reach it."""
    NEW_RUN_CALLS.append("noop")
    return "new run's noop ran"


def _abandon_a_call_in_its_hook(hook: Any) -> tuple[KISSAgent, socket.socket, Verdict]:
    """Serve one ``noop`` call whose *hook* holds it, Stop, and let the bridge abandon it.

    Returns the agent, reset for its next run (as a tab's next task does) with
    ``NEW RUN`` sentinels in every slot the abandoned call could write to, the
    abandoned call's open connection, and the next run's pending redirect verdict.
    """
    HOLD_STARTED.clear()
    RELEASE_HOLD.clear()
    NEW_RUN_CALLS.clear()
    stop = threading.Event()
    agent = KISSAgent("abandoned call hook returning late")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"noop": noop}
    agent._cached_tools_schema = _schema("noop")
    agent.tool_call_hook = hook
    stop_signal.set_thread_stop_event(stop)
    try:
        with agent._native_tools_exposed():
            abandoned = agent._native_turn
            client = _call(_bridge_of(agent), "noop")
            assert HOLD_STARTED.wait(5)
            stop.set()
    finally:
        stop_signal.set_thread_stop_event(None)
    assert abandoned.closed.is_set()
    # The run has returned: the agent is reused for its next run.
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.tool_call_hook = None
    agent.function_map = {"noop": new_run_noop}
    agent._add_message("user", "NEW RUN")
    agent._refused_call = "new-run-pending-tool"
    pending = refuse("new run's pending verdict")
    agent._redirect_ended = pending
    return agent, client, pending


def test_a_call_hook_that_allows_after_the_turn_closed_runs_no_tool() -> None:
    """A handler held in the ``tool_call_hook`` has no registered tool call for the
    bridge's interrupt to reach: the Stop abandons it and the turn closes.  When the
    hook finally allowed the call, the handler went on through the agent's CURRENT
    ``function_map`` and cleared the current run's pending redirect verdict: the
    previous task's call ran the next task's tool, outside any record.  The closed
    turn is now checked right after the hook, so the call is refused instead."""
    agent, client, pending = _abandon_a_call_in_its_hook(_hold_then_allow)
    RELEASE_HOLD.set()
    reply = _reply(client)
    assert NEW_RUN_CALLS == [], "the abandoned call ran the next run's tool"
    assert agent._redirect_ended is pending, "the abandoned call cleared the next run's verdict"
    assert agent._refused_call == "new-run-pending-tool"
    assert [m["content"] for m in agent.messages] == ["NEW RUN"]
    assert reply == {"text": refusal_result("noop", "The turn has ended.")}


def test_a_call_hook_that_refuses_after_the_turn_closed_settles_no_refusal() -> None:
    """The refusing twin: the late refusal was recorded as the CURRENT run's pending
    re-issue, replacing the tool the next run actually owed."""
    agent, client, pending = _abandon_a_call_in_its_hook(_hold_then_refuse)
    RELEASE_HOLD.set()
    reply = _reply(client)
    assert agent._refused_call == "new-run-pending-tool", "the late refusal reached the next run"
    assert agent._redirect_ended is pending
    assert NEW_RUN_CALLS == []
    assert reply == {"text": refusal_result("noop", "The turn has ended.")}


def _blocked_on_the_turn_lock(owner: threading.Thread) -> bool:
    """Whether *owner* is inside ``locked_call`` called from ``_native_tools_exposed``."""
    exposed_code = inspect.unwrap(KISSAgent._native_tools_exposed).__code__
    frame = sys._current_frames().get(owner.ident or -1)
    while frame is not None:
        if frame.f_code is locked_call.__code__ and (
            frame.f_back is not None and frame.f_back.f_code is exposed_code
        ):
            return True
        frame = frame.f_back
    return False


def test_a_per_panel_stop_landing_in_the_closing_wait_still_closes_the_turn() -> None:
    """An agent run inline by a tool (``commit_and_merge`` runs the merge-conflict
    resolver on the tool's thread) with a Codex model builds this turn; the per-panel
    Stop injects ``ToolCallInterrupted`` into that thread, which can land in the
    closing wait for ``turn.lock``.  Only ``KeyboardInterrupt`` was held there: the
    injected Stop escaped with the turn still open, so a handler of the turn could
    publish into the agent's next run.  Both injected Stops are now held until the
    turn is closed, then re-raised."""
    agent = KISSAgent("per-panel stop in the closing wait")
    agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
    agent.function_map = {"noop": noop}
    agent._cached_tools_schema = _schema("noop")
    turn = agent._native_turn
    outcome: list[str] = []
    closed_at_exit: list[bool] = []

    def own_the_turn() -> None:
        try:
            with agent._native_tools_exposed():
                pass
        except ToolCallInterrupted:
            outcome.append("ToolCallInterrupted")
        finally:
            closed_at_exit.append(turn.closed.is_set())

    assert turn.lock.acquire(timeout=5)  # a handler publishing: the owner waits for it
    owner = threading.Thread(target=own_the_turn, daemon=True)
    try:
        owner.start()
        deadline = time.monotonic() + 10
        while not _blocked_on_the_turn_lock(owner):
            assert time.monotonic() < deadline, "the owner never waited for the turn lock"
            time.sleep(0.01)
        assert owner.ident is not None
        assert inject_async_exception(owner.ident, ToolCallInterrupted) == 1
        time.sleep(ACQUIRE_SLICE_SECONDS * 3)  # the injection lands at the end of a slice
    finally:
        turn.lock.release()
        owner.join(10)
    assert not owner.is_alive()
    assert outcome == ["ToolCallInterrupted"], "the injected Stop must still end the run"
    assert closed_at_exit == [True], "the owner left the turn open"


def _probe(bridge: ToolBridge, payload: dict[str, Any]) -> dict[str, Any]:
    """Send *payload* with the secret to *bridge* and return its reply line."""
    with socket.create_connection(("127.0.0.1", bridge.port), timeout=15) as client:
        client.sendall(json.dumps({**payload, "token": bridge._server.token}).encode() + b"\n")
        with client.makefile("rb") as reply:
            return dict(json.loads(reply.readline()))


def test_the_tool_list_is_still_served_to_the_proxy() -> None:
    """Control: the ``tools`` request is unaffected by the name check."""
    with ToolBridge(_schema("noop"), _record) as bridge:
        assert [tool["name"] for tool in _probe(bridge, {"op": "tools"})["tools"]] == ["noop"]

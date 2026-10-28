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

import json
import socket
import threading
import time
from typing import Any

from kiss.core import stop_signal
from kiss.core.kiss_agent import KISSAgent, NativeTurn
from kiss.core.kiss_error import KISSError
from kiss.core.models.codex_mcp_bridge import STOP_WAIT_SECONDS, ToolBridge
from kiss.core.models.codex_model import CodexModel
from kiss.core.tool_verdict import Verdict
from kiss.server.json_printer import JsonPrinter
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
        assert _reply(_call(bridge.port, "finish")) == {"error": "KISSError: unknown tool 'finish'"}
        assert _reply(_call(bridge.port, "noop")) == {"text": "ran"}
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
            reply = _reply(_call(bridge.port, "noop"))
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
            client = _call(bridge.port, hold_until_released.__name__)
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
            client = _call(bridge.port, "noop")
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
            client = _call(bridge.port, "noop")
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


def _probe(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Send *payload* to the bridge on *port* and return its reply line."""
    with socket.create_connection(("127.0.0.1", port), timeout=15) as client:
        client.sendall(json.dumps(payload).encode() + b"\n")
        with client.makefile("rb") as reply:
            return dict(json.loads(reply.readline()))


def test_the_tool_list_is_still_served_to_the_proxy() -> None:
    """Control: the ``tools`` request is unaffected by the name check."""
    with ToolBridge(_schema("noop"), _record) as bridge:
        assert [tool["name"] for tool in _probe(bridge.port, {"op": "tools"})["tools"]] == ["noop"]

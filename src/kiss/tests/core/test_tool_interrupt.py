# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for interrupting ONE tool call (the tool panel's Stop).

``kiss.core.tool_interrupt`` lets the daemon interrupt the tool call an
agent thread is running without stopping the task: the tool returns
``"User interrupted the tool call."`` and the agent loop continues.

The agent-level tests drive a real ``KISSAgent.run()`` against a real
local HTTP server speaking the OpenAI chat-completions protocol (no
mocks, no paid calls): the first turn asks for a slow tool, the second
turn — which only happens because the interrupted tool RETURNED — hands
the tool's result to ``finish``.  The Bash test kills a real ``sleep``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core import tool_interrupt
from kiss.core.kiss_agent import KISSAgent
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    ToolCallInterrupted,
    begin_tool_call,
    end_tool_call,
    interrupt_tool_call,
    running_tool_name,
    unregister_tool_call,
)

_MODEL = "gpt-4o-mini"
_SLOW_SECONDS = 30.0

tools = UsefulTools()


def slow_wait(seconds: float) -> str:
    """Spin in Python for *seconds* (a tool that never blocks in C).

    Args:
        seconds: How long to spin.

    Returns:
        A completion marker.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pass
    return "slow_wait completed"


def swallowing_wait(seconds: float) -> str:
    """Spin like :func:`slow_wait` but swallow EVERY exception.

    A badly behaved tool: it catches ``BaseException`` and returns
    normally, so the injected interrupt never reaches the agent by
    itself.

    Args:
        seconds: How long to spin.

    Returns:
        What happened.
    """
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            pass
    except BaseException:  # noqa: BLE001 — deliberately misbehaving
        return "swallowed the interrupt"
    return "swallowing_wait completed"


def _chat_completion(tool_call: dict[str, object]) -> bytes:
    return json.dumps({
        "id": "chatcmpl-interrupt",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }).encode()


class _ScriptedModelHandler(BaseHTTPRequestHandler):
    """First turn: call the slow tool; then: finish with the last tool result.

    The class attribute ``first_call`` names the tool the first turn
    asks for (set per test); every later turn reads the previous tool
    result back out of the request's ``tool`` message and calls
    ``finish`` with it, so the agent's final result IS what the tool
    returned to the model.
    """

    first_call: dict[str, object] = {}
    requests: list[dict[str, object]] = []

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default stderr access log."""

    def do_POST(self) -> None:  # noqa: N802
        """Answer a chat-completions request per the script above."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        type(self).requests.append(body)
        messages = body.get("messages", [])
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if not tool_results:
            tool_call: dict[str, object] = {
                "id": "call_1",
                "type": "function",
                "function": dict(type(self).first_call),
            }
        else:
            last = str(tool_results[-1].get("content", ""))
            tool_call = {
                "id": "call_finish",
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": json.dumps({"result": last}),
                },
            }
        payload = _chat_completion(tool_call)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def scripted_model() -> Iterator[str]:
    """Run the scripted local model endpoint; yields its base URL."""
    _ScriptedModelHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)


def _run_agent_in_thread(
    base_url: str, tool: Callable[..., Any], name: str,
) -> tuple[threading.Thread, dict[str, object]]:
    """Start ``KISSAgent.run`` on its own thread; returns thread + outcome box."""
    outcome: dict[str, object] = {}

    def target() -> None:
        agent = KISSAgent(name)
        try:
            outcome["result"] = agent.run(
                _MODEL,
                "Run the tool.",
                tools=[tool],
                max_steps=6,
                model_config={"base_url": base_url, "api_key": "local"},
                verbose=False,
            )
        except BaseException as exc:  # noqa: BLE001 — the test reports it
            outcome["error"] = exc
        outcome["agent"] = agent

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome


def _first_line(text: object) -> str:
    """The tool's own text: the model layer appends a usage-info footer."""
    return str(text).splitlines()[0] if str(text).strip() else ""


def _wait_for_tool(thread: threading.Thread, tool_name: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ident = thread.ident
        if ident is not None and running_tool_name(ident) == tool_name:
            return
        time.sleep(0.02)
    raise AssertionError(f"{tool_name} never started running")


class TestAgentToolInterrupt:
    """The agent loop survives the interrupt and sees the interrupt message."""

    def test_interrupted_python_tool_returns_message_and_task_continues(
        self, scripted_model: str,
    ) -> None:
        _ScriptedModelHandler.first_call = {
            "name": "slow_wait",
            "arguments": json.dumps({"seconds": _SLOW_SECONDS}),
        }
        thread, outcome = _run_agent_in_thread(scripted_model, slow_wait, "interrupt-py")
        _wait_for_tool(thread, "slow_wait")
        assert thread.ident is not None
        t0 = time.monotonic()
        # A click on a different tool's panel must not land on this one.
        assert interrupt_tool_call(thread.ident, "Bash") is False
        assert running_tool_name(thread.ident) == "slow_wait"
        assert interrupt_tool_call(thread.ident, "slow_wait") is True
        # A second click while the first is in flight is absorbed.
        assert interrupt_tool_call(thread.ident, "slow_wait") is True
        thread.join(timeout=30)
        assert not thread.is_alive(), "the agent did not finish"
        assert time.monotonic() - t0 < _SLOW_SECONDS / 2, "the tool was not cut short"
        assert "error" not in outcome, f"agent failed: {outcome.get('error')!r}"
        # The model called finish with the tool result it was given.
        assert _first_line(outcome["result"]) == USER_INTERRUPTED_MESSAGE
        # The tool result message the model received.
        tool_msgs = [
            m for req in _ScriptedModelHandler.requests
            for m in cast("list[dict[str, Any]]", req.get("messages", []))
            if m.get("role") == "tool"
        ]
        assert tool_msgs, "the model never received a tool result"
        assert _first_line(tool_msgs[0]["content"]) == USER_INTERRUPTED_MESSAGE
        # Nothing is left registered for the thread once the run ends.
        assert running_tool_name(thread.ident) is None

    def test_uninterrupted_tool_completes_normally(self, scripted_model: str) -> None:
        _ScriptedModelHandler.first_call = {
            "name": "slow_wait",
            "arguments": json.dumps({"seconds": 0.2}),
        }
        thread, outcome = _run_agent_in_thread(scripted_model, slow_wait, "no-interrupt")
        thread.join(timeout=30)
        assert "error" not in outcome, f"agent failed: {outcome.get('error')!r}"
        assert _first_line(outcome["result"]) == "slow_wait completed"

    def test_tool_that_swallows_the_interrupt_is_still_interrupted(
        self, scripted_model: str,
    ) -> None:
        """``end_tool_call`` drains an injected interrupt the tool ate."""
        _ScriptedModelHandler.first_call = {
            "name": "swallowing_wait",
            "arguments": json.dumps({"seconds": _SLOW_SECONDS}),
        }
        thread, outcome = _run_agent_in_thread(
            scripted_model, swallowing_wait, "interrupt-swallow",
        )
        _wait_for_tool(thread, "swallowing_wait")
        assert thread.ident is not None
        assert interrupt_tool_call(thread.ident, "swallowing_wait") is True
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert "error" not in outcome, f"agent failed: {outcome.get('error')!r}"
        assert _first_line(outcome["result"]) == USER_INTERRUPTED_MESSAGE

    def test_interrupted_bash_kills_the_process_promptly(
        self, scripted_model: str,
    ) -> None:
        _ScriptedModelHandler.first_call = {
            "name": "Bash",
            "arguments": json.dumps({
                "command": f"echo started; sleep {int(_SLOW_SECONDS)}; echo finished",
                "description": "sleep",
            }),
        }
        thread, outcome = _run_agent_in_thread(scripted_model, tools.Bash, "interrupt-bash")
        _wait_for_tool(thread, "Bash")
        assert thread.ident is not None
        # Let the shell actually start sleeping.
        time.sleep(0.5)
        t0 = time.monotonic()
        assert interrupt_tool_call(thread.ident, "Bash") is True
        thread.join(timeout=30)
        assert not thread.is_alive(), "the agent did not finish"
        assert time.monotonic() - t0 < 10, "the sleeping shell was not killed promptly"
        assert "error" not in outcome, f"agent failed: {outcome.get('error')!r}"
        assert _first_line(outcome["result"]) == USER_INTERRUPTED_MESSAGE


class TestRegistry:
    """The running-tool registry and its guards."""

    def test_no_tool_call_on_thread_is_rejected(self) -> None:
        assert interrupt_tool_call(threading.get_ident()) is False
        assert running_tool_name(threading.get_ident()) is None
        assert tool_interrupt.current_tool_call() is None
        assert tool_interrupt.current_tool_interrupt_event() is None

    def test_dead_thread_cannot_be_injected(self) -> None:
        tokens: list[tool_interrupt.ToolCallToken] = []
        gate = threading.Event()

        def target() -> None:
            tokens.append(begin_tool_call("ghost"))
            gate.wait(5)
            # Left registered on purpose: the thread dies with its token.

        thread = threading.Thread(target=target)
        thread.start()
        while not tokens:
            time.sleep(0.005)
        gate.set()
        thread.join(5)
        assert not thread.is_alive()
        token = tokens[0]
        # The cooperative signal is accepted (the registry still lists
        # the call) ...
        assert interrupt_tool_call(token.thread_ident, "ghost") is True
        assert token.event.is_set()
        # ... but the forced injection finds no such thread.
        assert tool_interrupt._inject_unless_closing(token) is False
        assert token.injected is False
        # A foreign token must not unregister another thread's call.
        foreign = tool_interrupt.ToolCallToken("ghost")
        foreign.thread_ident = token.thread_ident
        tool_interrupt.unregister_tool_call(foreign)
        assert running_tool_name(token.thread_ident) == "ghost"
        # Clean the registry for later tests.
        tool_interrupt.unregister_tool_call(token)
        assert running_tool_name(token.thread_ident) is None
        assert interrupt_tool_call(token.thread_ident, "ghost") is False

    def test_call_id_must_match_when_given(self) -> None:
        token = begin_tool_call("Read")
        try:
            assert token.call_id > 0
            assert interrupt_tool_call(token.thread_ident, "Read", token.call_id + 1) is False
            assert not token.interrupted
            assert interrupt_tool_call(token.thread_ident, "Read", token.call_id) is True
            assert token.interrupted
        finally:
            unregister_tool_call(token)
        # A closing call is no longer a target.
        assert interrupt_tool_call(token.thread_ident) is False
        # Its watchdog finds it closed and never injects.
        time.sleep(tool_interrupt._INJECT_AFTER_SECONDS + 0.3)
        assert token.injected is False

    def test_raise_if_interrupted_is_cooperative(self) -> None:
        tool_interrupt.raise_if_interrupted()  # no tool call: no-op
        token = begin_tool_call("Wait")
        try:
            tool_interrupt.raise_if_interrupted()  # not interrupted: no-op
            assert interrupt_tool_call(token.thread_ident, "Wait") is True
            with pytest.raises(ToolCallInterrupted):
                tool_interrupt.raise_if_interrupted()
            assert token.closing is True
            assert tool_interrupt.current_tool_call() is None
        finally:
            unregister_tool_call(token)
        time.sleep(tool_interrupt._INJECT_AFTER_SECONDS + 0.3)
        assert token.injected is False, "a call that raised itself is never injected"

    def test_event_and_current_call_follow_the_thread(self) -> None:
        token = begin_tool_call("probe")
        try:
            assert tool_interrupt.current_tool_call() is token
            assert tool_interrupt.current_tool_interrupt_event() is token.event
            assert not token.event.is_set()
            assert running_tool_name(threading.get_ident()) == "probe"
        finally:
            end_tool_call(token)
        assert tool_interrupt.current_tool_call() is None
        # Idempotent on a token that is already gone.
        unregister_tool_call(token)
        end_tool_call(token)

    def test_end_tool_call_drains_an_injection_that_raced_the_return(self) -> None:
        token = begin_tool_call("drainer")
        main_ident = threading.get_ident()
        gate = threading.Event()
        try:
            try:
                # Force the injection directly, as the watchdog would
                # after the grace period, while this thread is inside a
                # C-level wait where it cannot land.
                def inject_now() -> None:
                    interrupt_tool_call(main_ident, "drainer")
                    tool_interrupt._inject_unless_closing(token)
                    gate.set()

                injector = threading.Thread(target=inject_now, daemon=True)
                injector.start()
                gate.wait(5)
                # Whether the interrupt already landed or is still
                # pending, it must surface here and nowhere later.
                end_tool_call(token)
            except ToolCallInterrupted:
                pass
            else:
                raise AssertionError("the interrupt never surfaced")
        finally:
            unregister_tool_call(token)
        assert token.interrupted is True
        assert token.injected is True
        assert token.event.is_set()
        assert running_tool_name(main_ident) is None

    def test_cooperative_signal_alone_leaves_a_returned_result_alone(self) -> None:
        """A tool that returned on its own before the grace period keeps its result."""
        token = begin_tool_call("quick")
        assert interrupt_tool_call(token.thread_ident, "quick") is True
        end_tool_call(token)  # no injection yet: nothing to raise
        assert token.closing is True
        time.sleep(tool_interrupt._INJECT_AFTER_SECONDS + 0.3)
        assert token.injected is False

    def test_return_racing_an_injection_never_leaks_the_exception(self) -> None:
        """Thousands of Stop clicks racing tool returns: none escapes, no lock sticks."""
        outcome = {"calls": 0, "interrupted": 0, "escaped": 0}
        stop = threading.Event()
        saved_grace = tool_interrupt._INJECT_AFTER_SECONDS
        tool_interrupt._INJECT_AFTER_SECONDS = 0.0  # inject at once
        try:

            def worker() -> None:
                while not stop.is_set():
                    token = tool_interrupt.new_tool_call("racer")
                    try:
                        try:
                            tool_interrupt.register_tool_call(token)
                            for _ in range(200):
                                pass
                            end_tool_call(token)
                        except ToolCallInterrupted:
                            outcome["interrupted"] += 1
                        finally:
                            unregister_tool_call(token)
                        outcome["calls"] += 1
                    except ToolCallInterrupted:
                        outcome["escaped"] += 1

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            deadline = time.monotonic() + 3.0
            clicks = 0
            while time.monotonic() < deadline and thread.ident is not None:
                interrupt_tool_call(thread.ident, "racer")
                clicks += 1
            stop.set()
            thread.join(10)
        finally:
            tool_interrupt._INJECT_AFTER_SECONDS = saved_grace
        assert not thread.is_alive(), "the worker wedged"
        assert outcome["escaped"] == 0, outcome
        assert outcome["interrupted"] > 0, outcome
        assert clicks > 0
        # Every watchdog thread has exited and the injector lock is free.
        time.sleep(0.2)
        assert tool_interrupt._INJECT_LOCK.acquire(timeout=1)
        tool_interrupt._INJECT_LOCK.release()


class TestBashMonitor:
    """The Bash tool's process-group monitor observes the tool's event."""

    def test_bash_without_a_task_stop_event_is_killed_by_the_interrupt(self) -> None:
        results: dict[str, object] = {}

        def target() -> None:
            token = begin_tool_call("Bash")
            try:
                try:
                    results["out"] = UsefulTools().Bash(
                        "sleep 30; echo late", description="sleep",
                    )
                    end_tool_call(token)
                except ToolCallInterrupted:
                    results["out"] = USER_INTERRUPTED_MESSAGE
            finally:
                unregister_tool_call(token)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        time.sleep(0.5)
        assert thread.ident is not None
        t0 = time.monotonic()
        assert interrupt_tool_call(thread.ident, "Bash") is True
        thread.join(15)
        assert not thread.is_alive()
        assert time.monotonic() - t0 < 10
        assert results["out"] == USER_INTERRUPTED_MESSAGE

    def test_bash_with_both_events_still_honours_the_task_stop(self) -> None:
        stop = threading.Event()
        results: dict[str, object] = {}

        def target() -> None:
            token = begin_tool_call("Bash")
            try:
                results["out"] = UsefulTools(stop_event=stop).Bash(
                    "sleep 30; echo late", description="sleep",
                )
            finally:
                unregister_tool_call(token)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        time.sleep(0.5)
        stop.set()
        thread.join(15)
        assert not thread.is_alive()
        assert "late" not in str(results["out"])

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
import random
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core import tool_interrupt
from kiss.core.base import collecting_trajectories
from kiss.core.kiss_agent import KISSAgent
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    ToolCallInterrupted,
    end_tool_call,
    interrupt_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)

_MODEL = "gpt-4o-mini"
_SLOW_SECONDS = 30.0

tools = UsefulTools()


def running_tool_name(thread_ident: int) -> str | None:
    """Name of the tool thread *thread_ident* is inside, or ``None``."""
    token = tool_interrupt._RUNNING.get(thread_ident)
    return token.name if token is not None else None


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

    def test_a_stop_on_the_enclosing_panel_ends_an_inline_helper_agent(
        self, scripted_model: str
    ) -> None:
        """A tool runs a helper ``KISSAgent.run`` on its own thread (the merge-conflict
        resolver under ``commit_and_merge``).  A Stop accepted on the tool's panel
        reaches the helper's running tool; it must end the helper too — no further
        model request, no result — and the enclosing call ends as interrupted."""
        _ScriptedModelHandler.first_call = {"name": "nested_wait", "arguments": "{}"}
        accepted: list[bool] = []
        helper = KISSAgent("inline helper")

        def nested_wait() -> str:
            """Click the enclosing panel's Stop, then wait as a blocking tool does."""
            call = tool_interrupt.current_tool_call()
            assert call is not None and call.parent is not None
            enclosing = call.parent
            accepted.append(
                interrupt_tool_call(enclosing.thread_ident, enclosing.name, enclosing.call_id)
            )
            # Interrupted at the first slice; the bound only keeps a failure finite.
            tool_interrupt.wait_interruptible(threading.Event(), 5.0)
            return "nested_wait completed"

        def enclosing() -> str:
            """Run the helper inline, as commit_and_merge's conflict resolver does."""
            helper.run(
                _MODEL,
                "Run the tool.",
                tools=[nested_wait],
                max_steps=4,
                model_config={"base_url": scripted_model, "api_key": "local"},
                verbose=False,
            )
            return "enclosing completed"

        with collecting_trajectories(save=False):
            agent = KISSAgent("enclosing agent")
            agent.function_map = {"enclosing": enclosing}
            _, result, blocked = agent._guarded_execute_tool(
                {"name": "enclosing", "arguments": {}}
            )
        assert accepted == [True] and blocked is None
        assert result == USER_INTERRUPTED_MESSAGE, "the accepted Stop must end the enclosing call"
        assert len(_ScriptedModelHandler.requests) == 1, "a model request after the Stop"
        # The helper's interrupted call is still in its trajectory.
        assert ("user", f"[nested_wait]: {USER_INTERRUPTED_MESSAGE}") in [
            (m["role"], m["content"]) for m in helper.messages
        ]
        assert running_tool_name(threading.get_ident()) is None

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


class _ReturningTool(threading.Thread):
    """A tool call that returns when released, then runs "the agent loop" (a wait)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.token: tool_interrupt.ToolCallToken | None = None
        self.registered = threading.Event()
        self.release = threading.Event()
        self.ended = threading.Event()
        self.after = threading.Event()
        self.outcome = "never ran"

    def run(self) -> None:
        token = new_tool_call("paused injector")
        register_tool_call(token)
        self.token = token
        self.registered.set()
        self.release.wait(30)
        try:
            end_tool_call(token)
            self.ended.set()
            self.after.wait(30)
            self.outcome = "clean"
        except ToolCallInterrupted:
            self.outcome = "late" if self.ended.is_set() else "in end_tool_call"
        finally:
            unregister_tool_call(token)


class _PauseBeforeInjecting:
    """A profile hook for the injector's thread: pause on the call of
    ``inject_async_exception``, i.e. after the ``closing`` check, before the injection."""

    def __init__(self) -> None:
        self.paused = threading.Event()
        self.resume = threading.Event()

    def __call__(self, frame: Any, event: str, arg: object) -> None:
        if event == "call" and frame.f_code is tool_interrupt.inject_async_exception.__code__:
            self.paused.set()
            assert self.resume.wait(30), "the test never resumed the injector"


class _Injector(threading.Thread):
    """Force the injection on a thread of its own, under the pausing hook."""

    def __init__(self, token: tool_interrupt.ToolCallToken, hook: _PauseBeforeInjecting) -> None:
        super().__init__(daemon=True)
        self.token = token
        self.hook = hook
        self.injected: bool | None = None

    def run(self) -> None:
        sys.setprofile(self.hook)
        try:
            self.injected = tool_interrupt._inject_unless_closing(self.token)
        finally:
            sys.setprofile(None)


def _stop_as_the_watchdog_is_created(frame: Any, event: str, arg: object) -> None:
    """A profile hook for a signalling thread: a task Stop (``KeyboardInterrupt``) landing
    after the call was marked and its event set, before its watchdog starts."""
    if event == "call" and frame.f_code is threading.Thread.__init__.__code__:
        raise KeyboardInterrupt


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
            token = new_tool_call("ghost")
            register_tool_call(token)
            tokens.append(token)
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
        token = new_tool_call("Read")
        register_tool_call(token)
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
        token = new_tool_call("Wait")
        register_tool_call(token)
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
        token = new_tool_call("probe")
        register_tool_call(token)
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
        token = new_tool_call("drainer")
        register_tool_call(token)
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

    def test_a_return_waits_out_an_injector_paused_inside_its_verdict(self) -> None:
        """The injector has passed its ``closing`` check and is about to inject when
        it is descheduled for longer than the drain deadline.  The tool's return
        must wait for the verdict, however long: the interrupt lands inside
        ``end_tool_call``, never in whatever the thread runs next."""
        tool = _ReturningTool()
        tool.start()
        assert tool.registered.wait(5)
        token = tool.token
        assert token is not None
        assert interrupt_tool_call(token.thread_ident, token.name) is True
        hook = _PauseBeforeInjecting()
        injector = _Injector(token, hook)
        injector.start()
        try:
            assert hook.paused.wait(5), "the injector never reached the injection"
            tool.release.set()  # the tool returns while the injector is paused
            assert not tool.ended.wait(tool_interrupt._DRAIN_SECONDS + 0.5), (
                "the tool returned with an injection pending"
            )
        finally:
            hook.resume.set()
            injector.join(5)
            tool.after.set()
            tool.join(5)
        assert injector.injected is True and token.injected is True
        assert tool.outcome == "in end_tool_call"
        assert running_tool_name(token.thread_ident) is None

    def test_cooperative_signal_alone_leaves_a_returned_result_alone(self) -> None:
        """A tool that returned on its own before the grace period keeps its result."""
        token = new_tool_call("quick")
        register_tool_call(token)
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
                            # The call blocks like a real tool (I/O,
                            # sleep) for a random 0-2 ms.  A busy spin
                            # instead never let the watchdog thread take
                            # the GIL on Windows before the call closed
                            # (0 injections in 350k calls), so the race
                            # was never exercised there; with a blocking
                            # body both outcomes -- injected, and
                            # returned first -- occur on every platform
                            # (measured: hundreds of each per run).
                            time.sleep(random.uniform(0, 0.002))
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
                # Paced like a human: a tight loop would keep the GIL
                # from the tool thread and turn the race into a stall.
                time.sleep(0.001)
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

    def test_a_stop_cut_short_before_its_watchdog_is_completed_by_the_next(self) -> None:
        """A Stop whose sender was itself stopped between the mark and the watchdog's start
        left the call marked with nothing to force it; every later Stop returned at the mark."""
        tokens: list[tool_interrupt.ToolCallToken] = []
        outcomes: list[str] = []
        over: list[bool] = []  # no Event: the injected thread must hold no lock

        def deaf_tool() -> None:
            token = new_tool_call("deaf")
            tokens.append(token)
            try:
                register_tool_call(token)
                for _ in range(6000):  # never looks at its own event
                    if over:
                        break
                    time.sleep(0.005)
                end_tool_call(token)
            except ToolCallInterrupted:
                outcomes.append("interrupted")
            finally:
                unregister_tool_call(token)

        def cut_short_stop() -> None:
            sys.setprofile(_stop_as_the_watchdog_is_created)
            try:
                interrupt_tool_call(tokens[0].thread_ident)
            except KeyboardInterrupt:
                outcomes.append("sender stopped")
            finally:
                sys.setprofile(None)

        worker = threading.Thread(target=deaf_tool, daemon=True)
        worker.start()
        try:
            _wait_for_tool(worker, "deaf")
            sender = threading.Thread(target=cut_short_stop)
            sender.start()
            sender.join(10)
            token = tokens[0]
            assert outcomes == ["sender stopped"] and token.interrupted and token.event.is_set()
            assert interrupt_tool_call(token.thread_ident) is True
            worker.join(tool_interrupt._INJECT_AFTER_SECONDS + 10)
            assert outcomes == ["sender stopped", "interrupted"], "the next Stop armed no watchdog"
            assert running_tool_name(token.thread_ident) is None
        finally:
            over.append(True)
            worker.join(10)


def _hold_without_share_delete(path: Path, seconds: float) -> None:
    """Keep *path* open for *seconds* as a dying process or a virus scanner does on Windows.

    The handle allows reading and writing by others but not deletion, so
    ``DeleteFile`` on *path* fails with a sharing violation meanwhile.
    """
    if sys.platform == "win32":
        import _winapi

        handle = _winapi.CreateFile(
            str(path), _winapi.GENERIC_READ, 1 | 2, 0, _winapi.OPEN_EXISTING, 0, 0,
        )
        time.sleep(seconds)
        _winapi.CloseHandle(handle)


class TestBashMonitor:
    """The Bash tool's process-group monitor observes the tool's event."""

    def test_bash_without_a_task_stop_event_is_killed_by_the_interrupt(self) -> None:
        results: dict[str, object] = {}

        def target() -> None:
            token = new_tool_call("Bash")
            register_tool_call(token)
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

    @pytest.mark.skipif(sys.platform != "win32", reason="POSIX unlinks a file others hold open")
    def test_the_interrupt_wins_over_a_holder_of_the_job_log(self) -> None:
        """A job log held open for a moment after the kill is removed once let go.

        On Windows a process of the killed tree, or a virus scanner, can
        hold the log past ``taskkill``; removing it then failed with
        ``PermissionError``, which replaced the interrupt.
        """
        tools = UsefulTools()
        results: dict[str, object] = {}

        def target() -> None:
            token = new_tool_call("Bash")
            register_tool_call(token)
            try:
                results["out"] = tools.Bash("sleep 30; echo late", description="sleep")
            except BaseException as e:
                results["out"] = e
            finally:
                unregister_tool_call(token)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        time.sleep(0.5)
        (job,) = tools._jobs.values()
        holder = threading.Thread(
            target=_hold_without_share_delete, args=(job.log_path, 1.0), daemon=True,
        )
        holder.start()
        time.sleep(0.1)
        assert thread.ident is not None
        assert interrupt_tool_call(thread.ident, "Bash") is True
        thread.join(15)
        holder.join(5)
        assert isinstance(results["out"], ToolCallInterrupted), results["out"]
        assert not job.log_path.exists()

    def test_bash_with_both_events_still_honours_the_task_stop(self) -> None:
        stop = threading.Event()
        results: dict[str, object] = {}

        def target() -> None:
            token = new_tool_call("Bash")
            register_tool_call(token)
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

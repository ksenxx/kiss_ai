# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``interruptTool`` server command.

The tool-call panel's Stop button sends ``{"type": "interruptTool",
"tabId", "toolName"}``.  The daemon resolves the tab to its running
task (own state or viewer subscription), interrupts ONLY the tool call
that task's thread is running, and answers with ``tool_interrupt_ack``.
The task itself keeps running.

Real ``VSCodeServer``, real agent-state registry, real task thread
through ``_run_task`` with a scripted ``agent.run`` that performs a
registered tool call exactly the way ``KISSAgent._execute_tool`` does.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import unittest
from collections.abc import Callable
from typing import Any

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core import tool_interrupt
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    ToolCallInterrupted,
    begin_tool_call,
    end_tool_call,
    unregister_tool_call,
)
from kiss.server import agent_state
from kiss.server.sorcar import API


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.server.server import VSCodeServer

    return VSCodeServer()


def _capture(server: Any) -> tuple[list[dict[str, Any]], threading.Lock]:
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    orig = server.printer.broadcast

    def capture(e: dict[str, Any]) -> None:
        with lock:
            events.append(dict(e))
        orig(e)

    server.printer.broadcast = capture
    return events, lock


def _run_tool(name: str, body: Callable[[], object]) -> str:
    """Run *body* as a registered tool call, the way ``_execute_tool`` does."""
    token = begin_tool_call(name)
    try:
        try:
            result = str(body())
            end_tool_call(token)
            return result
        except ToolCallInterrupted:
            return USER_INTERRUPTED_MESSAGE
    finally:
        unregister_tool_call(token)


def _start_task(
    server: Any,
    task_id: str,
    tab_id: str,
    tool_name: str,
    body: Callable[[], object],
) -> tuple[agent_state.AgentState, threading.Thread, threading.Event, dict[str, Any]]:
    """Run a task whose scripted agent performs one tool call *body*.

    The stubbed ``run`` performs the tool call (recording its result),
    then returns a normal result string — so a successful interrupt
    shows up as the task FINISHING with the interrupt message, not as a
    stopped task.
    """
    agent = WorktreeSorcarAgent("Sorcar VS Code")
    state = agent_state.AgentState(
        task_id,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
        stop_event=threading.Event(),
    )
    state.user_answer_queue = queue.Queue()
    tool_started = threading.Event()
    box: dict[str, Any] = {}

    def scripted_run(**kwargs: Any) -> str:
        # What ChatSorcarAgent.run does once its task row exists: bind
        # the printer's thread-local task id so the task's state (and
        # its user-answer queue) resolve from this thread.
        server.printer._thread_local.task_id = task_id

        def guarded_body() -> object:
            tool_started.set()
            return body()

        try:
            box["tool_result"] = _run_tool(tool_name, guarded_body)
        finally:
            server.printer._thread_local.task_id = ""
        return "success: true\nsummary: " + str(box["tool_result"])

    agent.run = scripted_run  # type: ignore[assignment]
    agent_state.register(state)
    thread = threading.Thread(
        target=server._run_task,
        args=({"type": "run", "prompt": "tool task", "tabId": tab_id},),
        daemon=True,
    )
    state.task_thread = thread
    thread.start()
    return state, thread, tool_started, box


def _spin(seconds: float) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pass
    return "spin completed"


def _acks(events: list[dict[str, Any]], lock: threading.Lock) -> list[dict[str, Any]]:
    with lock:
        return [e for e in events if e.get("type") == "tool_interrupt_ack"]


class TestInterruptToolCommand(unittest.TestCase):
    """``interruptTool`` stops the tool call, not the task."""

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_command_is_in_the_api_catalog(self) -> None:
        self.assertIn("interruptTool", API)
        self.assertEqual(API["interruptTool"].required, ("tabId",))
        self.assertEqual(API["interruptTool"].handler, "forward")

    def test_interrupt_from_own_tab_returns_message_and_task_finishes(self) -> None:
        server = _make_server()
        events, lock = _capture(server)
        state, thread, started, box = _start_task(
            server, "it-task-1", "it-tab-1", "slow", lambda: _spin(30),
        )
        self.assertTrue(started.wait(10), "tool never started")
        t0 = time.monotonic()
        server._cmd_interrupt_tool({"tabId": "it-tab-1", "toolName": "slow"})
        thread.join(30)
        self.assertFalse(thread.is_alive(), "task thread still running")
        self.assertLess(time.monotonic() - t0, 15, "tool call was not cut short")
        self.assertEqual(box["tool_result"], USER_INTERRUPTED_MESSAGE)
        acks = _acks(events, lock)
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0]["accepted"], True)
        self.assertEqual(acks[0]["tabId"], "it-tab-1")
        with lock:
            types = [e.get("type") for e in events]
        # The TASK was not stopped: it ran to its own result.
        self.assertNotIn("task_stopped", types)
        self.assertIn("task_done", types)

    def test_wrong_tool_name_is_rejected_and_tool_runs_on(self) -> None:
        server = _make_server()
        events, lock = _capture(server)
        _state, thread, started, box = _start_task(
            server, "it-task-2", "it-tab-2", "quick", lambda: _spin(1.0),
        )
        self.assertTrue(started.wait(10))
        server._cmd_interrupt_tool({"tabId": "it-tab-2", "toolName": "Bash"})
        acks = _acks(events, lock)
        self.assertEqual([a["accepted"] for a in acks], [False])
        thread.join(30)
        self.assertEqual(box["tool_result"], "spin completed")

    def test_non_string_tool_name_matches_any_running_tool(self) -> None:
        server = _make_server()
        events, lock = _capture(server)
        _state, thread, started, box = _start_task(
            server, "it-task-3", "it-tab-3", "anything", lambda: _spin(30),
        )
        self.assertTrue(started.wait(10))
        server._cmd_interrupt_tool({"tabId": "it-tab-3", "toolName": 42})
        thread.join(30)
        self.assertEqual(box["tool_result"], USER_INTERRUPTED_MESSAGE)
        self.assertEqual([a["accepted"] for a in _acks(events, lock)], [True])

    def test_call_id_names_exactly_one_call(self) -> None:
        server = _make_server()
        events, lock = _capture(server)
        seen: dict[str, Any] = {}

        def body() -> str:
            seen["token"] = tool_interrupt.current_tool_call()
            return _spin(30)

        _state, thread, started, box = _start_task(
            server, "it-task-8", "it-tab-8", "slow", body,
        )
        self.assertTrue(started.wait(10))
        token = seen["token"]
        self.assertIsNotNone(token)
        # The wrong call id (a click on an earlier panel of the same
        # tool) is rejected ...
        server._cmd_interrupt_tool(
            {"tabId": "it-tab-8", "toolName": "slow", "callId": token.call_id - 1},
        )
        # ... and so is a malformed one that names no call.
        self.assertEqual([a["accepted"] for a in _acks(events, lock)], [False])
        server._cmd_interrupt_tool(
            {"tabId": "it-tab-8", "toolName": "slow", "callId": token.call_id},
        )
        thread.join(30)
        self.assertEqual(box["tool_result"], USER_INTERRUPTED_MESSAGE)
        self.assertEqual([a["accepted"] for a in _acks(events, lock)], [False, True])

    def test_interrupt_from_a_viewer_tab_resolves_the_source_task(self) -> None:
        server = _make_server()
        events, lock = _capture(server)
        _state, thread, started, box = _start_task(
            server, "it-task-4", "it-src-4", "slow", lambda: _spin(30),
        )
        self.assertTrue(started.wait(10))
        self.assertIsNone(agent_state.find_by_tab("it-viewer-4"))
        server.printer.subscribe_tab("it-task-4", "it-src-4")
        server.printer.subscribe_tab("it-task-4", "it-viewer-4")
        server._cmd_interrupt_tool({"tabId": "it-viewer-4", "toolName": "slow"})
        thread.join(30)
        self.assertEqual(box["tool_result"], USER_INTERRUPTED_MESSAGE)
        acks = _acks(events, lock)
        self.assertEqual(len(acks), 1)
        self.assertTrue(acks[0]["accepted"])
        self.assertEqual(acks[0]["tabId"], "it-viewer-4")

    def test_no_running_task_and_missing_tab_id(self) -> None:
        server = _make_server()
        events, lock = _capture(server)
        server._cmd_interrupt_tool({"tabId": "nobody-home", "toolName": "Bash"})
        acks = _acks(events, lock)
        self.assertEqual(len(acks), 1)
        self.assertFalse(acks[0]["accepted"])
        self.assertEqual(acks[0]["tabId"], "nobody-home")
        server._cmd_interrupt_tool({"toolName": "Bash"})
        self.assertEqual(len(_acks(events, lock)), 1, "a tab-less command is ignored")

    def test_task_between_tool_calls_is_rejected(self) -> None:
        """A running task that is NOT inside a tool call cannot be hit."""
        server = _make_server()
        events, lock = _capture(server)
        release = threading.Event()
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        state = agent_state.AgentState(
            "it-task-5", agent=agent, tab_id="it-tab-5", server_owned=True,
            stop_event=threading.Event(),
        )
        state.user_answer_queue = queue.Queue()
        entered = threading.Event()

        def idle_run(**kwargs: Any) -> str:
            entered.set()
            release.wait(10)
            return "success: true\nsummary: idle"

        agent.run = idle_run  # type: ignore[assignment]
        agent_state.register(state)
        thread = threading.Thread(
            target=server._run_task,
            args=({"type": "run", "prompt": "idle", "tabId": "it-tab-5"},),
            daemon=True,
        )
        state.task_thread = thread
        thread.start()
        self.assertTrue(entered.wait(10))
        server._cmd_interrupt_tool({"tabId": "it-tab-5", "toolName": ""})
        self.assertEqual([a["accepted"] for a in _acks(events, lock)], [False])
        release.set()
        thread.join(30)
        with lock:
            types = [e.get("type") for e in events]
        self.assertIn("task_done", types)
        self.assertNotIn("task_stopped", types)

    def test_interrupting_ask_user_question_closes_the_modal(self) -> None:
        """The answer wait is a C-level ``queue.get``: the tool's event wakes it."""
        server = _make_server()
        events, lock = _capture(server)

        def ask() -> str:
            return str(server._ask_user_question("Which colour?"))

        state, thread, started, box = _start_task(
            server, "it-task-6", "it-tab-6", "ask_user_question", ask,
        )
        self.assertTrue(started.wait(10))
        # Wait until the question is really pending (the agent thread is
        # inside _await_user_response).
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not state.pending_ask_question:
            time.sleep(0.01)
        self.assertEqual(state.pending_ask_question, "Which colour?")
        with lock:
            self.assertTrue(any(e.get("type") == "askUser" for e in events))
        server.printer.subscribe_tab("it-task-6", "it-tab-6")
        server._cmd_interrupt_tool({"tabId": "it-tab-6", "toolName": "ask_user_question"})
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(box["tool_result"], USER_INTERRUPTED_MESSAGE)
        self.assertEqual(state.pending_ask_question, "")
        with lock:
            done = [e for e in events if e.get("type") == "askUserDone"]
            types = [e.get("type") for e in events]
        self.assertEqual([d["tabId"] for d in done], ["it-tab-6"])
        self.assertIn("task_done", types)
        self.assertNotIn("task_stopped", types)

    def test_task_stop_still_wins_while_waiting_for_an_answer(self) -> None:
        """The whole-task Stop keeps its KeyboardInterrupt semantics."""
        server = _make_server()
        events, lock = _capture(server)

        def ask() -> str:
            return str(server._ask_user_question("Still there?"))

        state, thread, started, _box = _start_task(
            server, "it-task-7", "it-tab-7", "ask_user_question", ask,
        )
        self.assertTrue(started.wait(10))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not state.pending_ask_question:
            time.sleep(0.01)
        server._stop_task("it-tab-7")
        thread.join(30)
        self.assertFalse(thread.is_alive())
        with lock:
            types = [e.get("type") for e in events]
        self.assertIn("task_stopped", types)
        self.assertNotIn("askUserDone", types)


class TestInterruptedToolResultEvent(unittest.TestCase):
    """``JsonPrinter`` stamps call ids and flags interrupted results."""

    def test_tool_call_event_carries_the_call_id(self) -> None:
        server = _make_server()
        events, lock = _capture(server)
        printer = server.printer
        printer._thread_local.task_id = "it-printer-0"
        printer.print("Read", type="tool_call", tool_input={"file_path": "/x"}, call_id=41)
        printer.print("Read", type="tool_call", tool_input={"file_path": "/x"})
        printer.print("Read", type="tool_call", tool_input={"file_path": "/x"}, call_id="7")
        printer.print("Read", type="tool_call", tool_input={"file_path": "/x"}, call_id=True)
        printer._thread_local.task_id = ""
        with lock:
            calls = [e for e in events if e.get("type") == "tool_call"]
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0]["callId"], 41)
        for event in calls[1:]:
            self.assertNotIn("callId", event)

    def test_streamed_bash_result_keeps_content_when_interrupted(self) -> None:
        server = _make_server()
        events, lock = _capture(server)
        printer = server.printer
        printer._thread_local.task_id = "it-printer-1"
        printer.print("Bash", type="tool_call", tool_input={"command": "sleep 30"})
        printer.print("partial\n", type="system_output")
        printer.print(
            USER_INTERRUPTED_MESSAGE,
            type="tool_result",
            tool_name="Bash",
            tool_input={"command": "sleep 30"},
            is_error=False,
            interrupted=True,
        )
        with lock:
            results = [e for e in events if e.get("type") == "tool_result"]
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["interrupted"])
        self.assertEqual(results[0]["content"], USER_INTERRUPTED_MESSAGE)
        # An ordinary streamed Bash result is still blanked and unflagged.
        printer.print("Bash", type="tool_call", tool_input={"command": "echo x"})
        printer.print("x\n", type="system_output")
        printer.print(
            "exit 0\nx\n", type="tool_result", tool_name="Bash",
            tool_input={"command": "echo x"}, is_error=False,
        )
        with lock:
            results = [e for e in events if e.get("type") == "tool_result"]
        self.assertEqual(len(results), 2)
        self.assertNotIn("interrupted", results[1])
        printer._thread_local.task_id = ""


class TestInjectKeyboardInterruptStillWorks(unittest.TestCase):
    """The whole-task injector now shares the tool interrupt's primitive."""

    def test_inject_keyboard_interrupt_delegates(self) -> None:
        from kiss.server.task_runner import inject_keyboard_interrupt

        caught: list[BaseException] = []
        ready = threading.Event()

        def target() -> None:
            ready.set()
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    pass
            except KeyboardInterrupt as exc:
                caught.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(5))
        assert thread.ident is not None
        self.assertEqual(inject_keyboard_interrupt(thread.ident), 1)
        thread.join(10)
        self.assertEqual(len(caught), 1)
        self.assertEqual(
            tool_interrupt.inject_async_exception(thread.ident, KeyboardInterrupt), 0,
        )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the "queue follow-up prompt while task is running" feature.

Covers:

* :meth:`_CommandsMixin._cmd_append_user_message` — appends to
  ``AgentState.pending_user_messages`` only when the tab has a live
  task, broadcasts a ``prompt`` echo, and rejects empty /
  whitespace-only / non-string prompts.
* :meth:`SorcarAgent._drain_pending_user_messages` — drains the
  printer bridge (``JsonPrinter.drain_pending_user_messages``) and calls
  ``model.add_message_to_conversation("user", ...)`` for each queued
  entry, wrapping it as ``User says: <msg>. Take the message into
  account and finish your task.`` (then leaves the queue empty so
  the same message is never injected twice).
* Hook lifecycle: ``SorcarAgent.perform_task`` always installs the
  drain as ``pre_step_hook`` (and the finish guard as
  ``tool_call_guard``); both are self-guarding no-ops when the
  follow-up channel (the printer's duck-typed
  ``drain_pending_user_messages`` bridge) has nothing queued.
* ``KISSAgent.pre_step_hook`` runs BEFORE each model call.
* End-to-end: ``_run_task`` clears any leftover queued messages.

These tests exercise the production handler and drain code paths
directly; the only stand-in is a tiny recording object playing the
role of ``model`` for the drain (it just records each
``add_message_to_conversation`` call).
"""

from __future__ import annotations

import threading
from typing import Any

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer


def _clear_registry() -> None:
    agent_state.agent_states.clear()


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Spin up a real :class:`VSCodeServer` whose broadcasts land in a list."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _register_active_state(task_id: str, tab_id: str) -> AgentState:
    """Register a server-owned state with a live task on *tab_id*."""
    st = AgentState(task_id, tab_id=tab_id, server_owned=True)
    st.is_task_active = True
    agent_state.register(st)
    return st


class _RecordingModel:
    """Captures every ``add_message_to_conversation`` call.

    Stands in for a live LLM model object for the drain test — the
    drain hook only needs an object exposing
    ``add_message_to_conversation(role, content)``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.calls.append((role, content))


def _make_drain_agent(printer: Any = None) -> SorcarAgent:
    """Bare ``SorcarAgent`` carrying only what the drain hook reads."""
    agent = SorcarAgent.__new__(SorcarAgent)
    if printer is not None:
        agent.printer = printer
    return agent


def _bridge_printer(
    task_id: str, messages: list[str],
) -> tuple[Any, AgentState]:
    """A real ``JsonPrinter`` bound to *task_id* with queued *messages*.

    Registers an :class:`AgentState` for *task_id* holding *messages*
    so the printer bridge (the only steering channel) can drain them.
    The per-test registry cleanup unregisters the state.
    """
    from kiss.server.json_printer import JsonPrinter

    printer = JsonPrinter()
    printer._thread_local.task_id = task_id
    st = AgentState(task_id)
    st.pending_user_messages.extend(messages)
    agent_state.register(st)
    return printer, st


class TestAppendUserMessageHandler:
    """``_cmd_append_user_message`` queues prompts for active tabs."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_queues_prompt_and_echoes_when_task_active(self) -> None:
        server, events = _make_server()
        st = _register_active_state("task-1", "tab-1")

        server._cmd_append_user_message(
            {"tabId": "tab-1", "prompt": "follow up A"},
        )
        server._cmd_append_user_message(
            {"tabId": "tab-1", "prompt": "follow up B"},
        )

        assert st.pending_user_messages == ["follow up A", "follow up B"]
        assert st.unattributed_prompt_echoes == [
            "follow up A", "follow up B",
        ]
        echoes = [e for e in events if e.get("type") == "prompt"]
        assert echoes == [
            {"type": "prompt", "text": "follow up A", "tabId": "tab-1"},
            {"type": "prompt", "text": "follow up B", "tabId": "tab-1"},
        ]

    def test_dropped_when_no_live_task(self) -> None:
        """An idle tab has no drain hook, so queueing would leak forever."""
        server, events = _make_server()
        st = AgentState("task-idle", tab_id="tab-idle", server_owned=True)
        agent_state.register(st)

        server._cmd_append_user_message(
            {"tabId": "tab-idle", "prompt": "ignored"},
        )

        assert st.pending_user_messages == []
        assert [e for e in events if e.get("type") == "prompt"] == []

    def test_dropped_when_tab_missing(self) -> None:
        server, events = _make_server()
        server._cmd_append_user_message(
            {"tabId": "ghost-tab", "prompt": "ignored"},
        )
        assert [e for e in events if e.get("type") == "prompt"] == []

    def test_empty_prompt_ignored(self) -> None:
        server, events = _make_server()
        st = _register_active_state("task-2", "tab-2")

        for blank in ("", "   ", "\n\t  \n"):
            server._cmd_append_user_message(
                {"tabId": "tab-2", "prompt": blank},
            )

        assert st.pending_user_messages == []
        assert [e for e in events if e.get("type") == "prompt"] == []

    def test_non_string_prompt_ignored(self) -> None:
        server, _events = _make_server()
        st = _register_active_state("task-3", "tab-3")

        for bad in (None, 42, ["list"], {"prompt": "x"}):
            server._cmd_append_user_message(
                {"tabId": "tab-3", "prompt": bad},
            )

        assert st.pending_user_messages == []

    def test_handler_registered_in_dispatch_table(self) -> None:
        """``appendUserMessage`` must be wired into ``_HANDLERS``."""
        from kiss.server.commands import _CommandsMixin

        assert "appendUserMessage" in _CommandsMixin._HANDLERS
        handler = _CommandsMixin._HANDLERS["appendUserMessage"]
        assert handler is _CommandsMixin._cmd_append_user_message


class TestDrainPendingUserMessages:
    """``SorcarAgent._drain_pending_user_messages`` injects + clears."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_drain_injects_each_message_and_clears_queue(self) -> None:
        printer, st = _bridge_printer("task-d1", ["msg1", "msg2", "msg3"])
        agent = _make_drain_agent(printer=printer)

        model = _RecordingModel()
        agent._drain_pending_user_messages(model)

        assert model.calls == [
            (
                "user",
                "User says: msg1. "
                "Take the message into account and finish your task.",
            ),
            (
                "user",
                "User says: msg2. "
                "Take the message into account and finish your task.",
            ),
            (
                "user",
                "User says: msg3. "
                "Take the message into account and finish your task.",
            ),
        ]
        assert st.pending_user_messages == []

    def test_drain_noop_without_printer_bridge(self) -> None:
        agent = _make_drain_agent()
        model = _RecordingModel()
        agent._drain_pending_user_messages(model)
        assert model.calls == []

    def test_drain_via_printer_bridge_for_unknown_task_is_noop(self) -> None:
        """A printer thread with no bound task drains nothing."""
        server, _events = _make_server()
        server.printer._thread_local.task_id = "ghost-task"
        try:
            agent = _make_drain_agent(printer=server.printer)
            model = _RecordingModel()
            agent._drain_pending_user_messages(model)
            assert model.calls == []
        finally:
            server.printer._thread_local.task_id = None

    def test_queue_then_drain_roundtrip(self) -> None:
        """End-to-end: command handler queues, printer bridge drains."""
        server, _events = _make_server()
        st = _register_active_state("task-rt", "tab-rt")

        server._cmd_append_user_message(
            {"tabId": "tab-rt", "prompt": "first follow-up"},
        )
        server._cmd_append_user_message(
            {"tabId": "tab-rt", "prompt": "second follow-up"},
        )
        assert st.pending_user_messages == [
            "first follow-up", "second follow-up",
        ]

        server.printer._thread_local.task_id = "task-rt"
        try:
            agent = _make_drain_agent(printer=server.printer)
            model = _RecordingModel()
            agent._drain_pending_user_messages(model)

            expected = [
                (
                    "user",
                    "User says: first follow-up. "
                    "Take the message into account and finish your task.",
                ),
                (
                    "user",
                    "User says: second follow-up. "
                    "Take the message into account and finish your task.",
                ),
            ]
            assert model.calls == expected
            assert st.pending_user_messages == []
            agent._drain_pending_user_messages(model)
            assert model.calls == expected
        finally:
            server.printer._thread_local.task_id = None


class _PreStepHookRecordingModel:
    """A model that records whether the pre-step hook was called first.

    ``generate_and_process_with_tools`` immediately returns a synthetic
    "finish" tool call so ``_execute_step`` returns without invoking
    any real LLM.  Before returning it snapshots the current
    ``conversation`` so the test can verify the hook ran first.
    """

    def __init__(self) -> None:
        self.model_name = "gpt-4o-mini"
        self.conversation: list[dict[str, Any]] = []
        self.conversation_before_generate: list[dict[str, Any]] = []

    def initialize(self, prompt: str, attachments: list[Any] | None = None) -> None:
        self.conversation.append({"role": "user", "content": prompt})

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Any],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        self.conversation_before_generate = list(self.conversation)
        return (
            [{"name": "finish", "arguments": {"result": "ok"}}],
            "done",
            object(),
        )

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        pass

    def add_function_results_to_conversation_and_return(
        self,
        function_results: list[tuple[str, dict[str, Any]]],
    ) -> None:
        pass

    def extract_input_output_token_counts_from_response(
        self,
        response: Any,
    ) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)


class TestPreStepHookIntegration:
    """``KISSAgent.pre_step_hook`` invokes the drain before each model call."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def _make_kiss_agent(self, model_obj: Any) -> Any:
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent("HookTest")
        agent.model = model_obj
        agent.model_name = model_obj.model_name
        agent.verbose = False
        agent.printer = None
        agent.is_agentic = True
        agent.max_steps = 5
        agent.max_budget = 1.0
        agent.function_map = {"finish": agent.finish}
        agent.messages = []
        agent.step_count = 0
        agent.total_tokens_used = 0
        agent.budget_used = 0.0
        agent.run_start_timestamp = 0
        agent._cached_tools_schema = None
        return agent

    def test_default_pre_step_hook_is_none(self) -> None:
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent("Plain")
        assert agent.pre_step_hook is None

    def test_pre_step_hook_runs_before_model_call(self) -> None:
        """The hook is called BEFORE ``generate_and_process_with_tools``.

        Wires the production
        :meth:`SorcarAgent._drain_pending_user_messages` as the hook
        with two pending user messages, then runs one
        ``_execute_step``.  The model records its conversation
        immediately before generating — both pending messages must
        already be in it.
        """
        model = _PreStepHookRecordingModel()
        agent = self._make_kiss_agent(model)

        printer, _st = _bridge_printer("task-hook", ["queued 1", "queued 2"])
        sa = _make_drain_agent(printer=printer)
        agent.pre_step_hook = sa._drain_pending_user_messages
        result = agent._execute_step()

        assert result == "ok"
        roles_and_contents = [
            (m["role"], m["content"])
            for m in model.conversation_before_generate
        ]
        assert (
            "user",
            "User says: queued 1. "
            "Take the message into account and finish your task.",
        ) in roles_and_contents
        assert (
            "user",
            "User says: queued 2. "
            "Take the message into account and finish your task.",
        ) in roles_and_contents
        assert _st.pending_user_messages == []

    def test_pre_step_hook_none_does_not_break_step(self) -> None:
        model = _PreStepHookRecordingModel()
        agent = self._make_kiss_agent(model)
        assert agent.pre_step_hook is None
        result = agent._execute_step()
        assert result == "ok"


class TestPreStepHookInstalledByPerformTask:
    """``SorcarAgent.perform_task`` wires the drain hook + finish guard.

    Regression guard for the original steering bug:
    ``RelentlessAgent.run`` calls ``_reset`` which sets
    ``pre_step_hook``/``tool_call_guard`` to ``None``, so the hook
    MUST be (re-)installed inside ``perform_task`` (which runs after
    ``_reset``), never before ``super().run()``.  Captures the hook
    values the moment the parent ``perform_task`` takes over.
    """

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def _run_and_capture(self, printer: Any) -> dict[str, Any]:
        from kiss.agents.sorcar.relentless_agent import RelentlessAgent

        agent = SorcarAgent("Sorcar Regression")

        captured: dict[str, Any] = {}
        original_perform_task = RelentlessAgent.perform_task

        def capture_perform_task(
            self: RelentlessAgent,
            tools: list[Any],
            attachments: list[Any] | None = None,
        ) -> str:
            captured["hook"] = self.pre_step_hook
            captured["guard"] = self.tool_call_guard
            return '"success": true\n"summary": "stub"\n'

        RelentlessAgent.perform_task = capture_perform_task  # type: ignore[method-assign]
        try:
            agent.run(
                model_name="claude-opus-4-8",
                prompt_template="do nothing",
                web_tools=False,
                max_steps=2,
                max_budget=1.0,
                verbose=False,
                printer=printer,
            )
        finally:
            RelentlessAgent.perform_task = original_perform_task  # type: ignore[method-assign]
        captured["agent"] = agent
        return captured

    def test_executor_receives_live_drain_hook(self) -> None:
        from kiss.server.json_printer import JsonPrinter

        captured = self._run_and_capture(JsonPrinter())
        agent = captured["agent"]
        assert captured.get("hook") is not None
        assert captured["hook"] == agent._drain_pending_user_messages
        assert captured.get("guard") is not None
        assert (
            captured["guard"]
            == agent._block_finish_when_user_message_pending
        )

    def test_hooks_installed_without_drain_capable_printer(self) -> None:
        """Steering hooks are installed regardless of printer capability.

        An agent driven outside the daemon (programmatically, or with
        the default :class:`ConsolePrinter`) has no printer with the
        duck-typed ``drain_pending_user_messages`` bridge, yet the
        drain hook and the finish guard must still be installed: they
        degrade to no-ops without the bridge, keeping the wiring
        identical to a daemon run (whose ``WebPrinter`` supplies the
        bridge from the task's registered agent state)."""
        captured = self._run_and_capture(None)
        agent = captured["agent"]
        assert captured.get("hook") == agent._drain_pending_user_messages
        assert (
            captured.get("guard")
            == agent._block_finish_when_user_message_pending
        )


class TestSteeringMessageWrappedForModel:
    """Steering messages must be wrapped, not injected verbatim.

    A steering message queued by the user while the agent is running
    must reach the model's conversation as::

        User says: <message>. Take the message into account and
        finish your task.

    (prefix ``"User says: "``, suffix ``". Take the message into
    account and finish your task."``) — never as the bare message
    text.  The queue itself keeps the raw text (that is what the UI
    echoes back); the wrapping happens at drain time.
    """

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_queued_steering_message_is_wrapped_end_to_end(self) -> None:
        """End-to-end: frontend command → queue → drain → wrapped."""
        server, _events = _make_server()
        st = _register_active_state("task-wrap", "tab-wrap")

        server._cmd_append_user_message(
            {"tabId": "tab-wrap", "prompt": "focus on the login bug"},
        )
        assert st.pending_user_messages == ["focus on the login bug"]

        server.printer._thread_local.task_id = "task-wrap"
        try:
            agent = _make_drain_agent(printer=server.printer)
            model = _RecordingModel()
            agent._drain_pending_user_messages(model)
        finally:
            server.printer._thread_local.task_id = None

        assert model.calls == [
            (
                "user",
                "User says: focus on the login bug. "
                "Take the message into account and finish your task.",
            ),
        ]
        assert st.pending_user_messages == []

    def test_every_drained_message_is_wrapped(self) -> None:
        printer, _st = _bridge_printer(
            "task-wrap2", ["first steer", "second steer"],
        )
        agent = _make_drain_agent(printer=printer)

        model = _RecordingModel()
        agent._drain_pending_user_messages(model)

        assert model.calls == [
            (
                "user",
                "User says: first steer. "
                "Take the message into account and finish your task.",
            ),
            (
                "user",
                "User says: second steer. "
                "Take the message into account and finish your task.",
            ),
        ]


class TestFinishBlockedWhilePending:
    """``finish`` is rejected while a queued follow-up is undrained."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_finish_blocked_when_message_pending(self) -> None:
        printer, _st = _bridge_printer("task-fb", ["steer me"])
        agent = _make_drain_agent(printer=printer)
        verdict = agent._block_finish_when_user_message_pending("finish", {})
        assert verdict is not None
        assert "finish rejected" in verdict

    def test_finish_allowed_when_nothing_pending(self) -> None:
        agent = _make_drain_agent()
        assert (
            agent._block_finish_when_user_message_pending("finish", {})
            is None
        )

    def test_non_finish_tools_never_blocked(self) -> None:
        printer, _st = _bridge_printer("task-nf", ["steer me"])
        agent = _make_drain_agent(printer=printer)
        assert (
            agent._block_finish_when_user_message_pending("Bash", {}) is None
        )

    def test_finish_blocked_when_bridge_message_pending(self) -> None:
        server, _events = _make_server()
        st = _register_active_state("task-guard", "tab-guard")
        st.pending_user_messages.append("bridge steer")

        server.printer._thread_local.task_id = "task-guard"
        try:
            agent = _make_drain_agent(printer=server.printer)
            verdict = agent._block_finish_when_user_message_pending(
                "finish", {},
            )
            assert verdict is not None
        finally:
            server.printer._thread_local.task_id = None


class TestPendingMessagesClearedOnTaskFinish:
    """Lingering queued messages must not leak across successive tasks."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_agent_state_default_is_empty_list(self) -> None:
        st = AgentState("task-x", tab_id="tab-x")
        assert st.pending_user_messages == []
        other = AgentState("task-y", tab_id="tab-y")
        other.pending_user_messages.append("only-on-other")
        assert st.pending_user_messages == []

    def test_task_runner_clears_pending_messages_after_run(self) -> None:
        """End-to-end: pending_user_messages must be empty after _run_task.

        Drives the real :meth:`_TaskRunnerMixin._run_task` with a stub
        ``state.agent.run`` that queues two follow-ups mid-flight (the
        same path the frontend would take while a task is running).
        After ``_run_task`` returns, the ``finally`` block must have
        cleared the queue so the next task starts fresh.
        """
        import os
        import queue

        from kiss.agents.sorcar.worktree_sorcar_agent import (
            WorktreeSorcarAgent,
        )

        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        server = VSCodeServer()
        tab_id = "tab-clear-after-run"
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        st = AgentState(
            "task-clear-after-run",
            agent=agent,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
        )
        st.user_answer_queue = queue.Queue(maxsize=1)
        st.is_task_active = True
        agent_state.register(st)

        def fake_run(**_kwargs: Any) -> None:
            st.pending_user_messages.append("queued during task")
            st.pending_user_messages.append("also queued")

        agent.run = fake_run  # type: ignore[method-assign, assignment]

        task_thread = threading.Thread(
            target=server._run_task,
            args=({
                "type": "run",
                "prompt": "test prompt",
                "tabId": tab_id,
                "workDir": "/tmp",
                "useParallel": False,
                "useWorktree": False,
                "autoCommit": False,
                "_state_key": "task-clear-after-run",
            },),
            daemon=True,
        )
        st.task_thread = task_thread
        task_thread.start()
        task_thread.join(timeout=15)
        assert not task_thread.is_alive()

        post = agent_state.get("task-clear-after-run")
        assert post is not None
        assert post.pending_user_messages == []
        assert post.is_task_active is False

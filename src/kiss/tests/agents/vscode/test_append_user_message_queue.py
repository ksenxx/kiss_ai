"""Integration tests for the "queue follow-up prompt while task is running" feature.

Covers:

* :meth:`_CommandsMixin._cmd_append_user_message` — appends to
  ``_RunningAgentState.pending_user_messages`` only when a task is
  active on the tab, broadcasts a ``prompt`` echo, and rejects
  empty / whitespace-only / non-string prompts.
* :meth:`SorcarAgent._drain_pending_user_messages` — drains the
  list under ``_registry_lock`` and calls
  ``model.add_message_to_conversation("user", msg)`` for each
  queued entry (then leaves the list empty so the same message is
  never injected twice).
* Lifecycle: the drain hook is a no-op when ``_tab_id`` is unset
  or points at no live tab.

These tests exercise the production handler and drain code paths
directly; the only stand-in is a tiny recording object playing the
role of ``model`` for the drain (it just records each
``add_message_to_conversation`` call).
"""

from __future__ import annotations

import threading
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer


def _clear_registry() -> None:
    _RunningAgentState.running_agent_states.clear()


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


class TestAppendUserMessageHandler:
    """``_cmd_append_user_message`` queues prompts for active tabs."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_queues_prompt_and_echoes_when_task_active(self) -> None:
        server, events = _make_server()
        tab = _RunningAgentState("tab-1", "gemini")
        tab.is_task_active = True
        _RunningAgentState.running_agent_states["tab-1"] = tab

        server._cmd_append_user_message(
            {"tabId": "tab-1", "prompt": "follow up A"},
        )
        server._cmd_append_user_message(
            {"tabId": "tab-1", "prompt": "follow up B"},
        )

        assert tab.pending_user_messages == ["follow up A", "follow up B"]
        prompt_echoes = [e for e in events if e.get("type") == "prompt"]
        assert prompt_echoes == [
            {"type": "prompt", "text": "follow up A", "tabId": "tab-1"},
            {"type": "prompt", "text": "follow up B", "tabId": "tab-1"},
        ]

    def test_dropped_when_no_live_task(self) -> None:
        """An idle tab has no drain hook, so queueing would leak forever."""
        server, events = _make_server()
        tab = _RunningAgentState("tab-idle", "gemini")
        tab.is_task_active = False
        _RunningAgentState.running_agent_states["tab-idle"] = tab

        server._cmd_append_user_message(
            {"tabId": "tab-idle", "prompt": "ignored"},
        )

        assert tab.pending_user_messages == []
        assert [e for e in events if e.get("type") == "prompt"] == []

    def test_dropped_when_tab_missing(self) -> None:
        server, events = _make_server()
        server._cmd_append_user_message(
            {"tabId": "ghost-tab", "prompt": "ignored"},
        )
        assert [e for e in events if e.get("type") == "prompt"] == []

    def test_empty_prompt_ignored(self) -> None:
        server, events = _make_server()
        tab = _RunningAgentState("tab-2", "gemini")
        tab.is_task_active = True
        _RunningAgentState.running_agent_states["tab-2"] = tab

        for blank in ("", "   ", "\n\t  \n"):
            server._cmd_append_user_message(
                {"tabId": "tab-2", "prompt": blank},
            )

        assert tab.pending_user_messages == []
        assert [e for e in events if e.get("type") == "prompt"] == []

    def test_non_string_prompt_ignored(self) -> None:
        server, _events = _make_server()
        tab = _RunningAgentState("tab-3", "gemini")
        tab.is_task_active = True
        _RunningAgentState.running_agent_states["tab-3"] = tab

        for bad in (None, 42, ["list"], {"prompt": "x"}):
            server._cmd_append_user_message(
                {"tabId": "tab-3", "prompt": bad},
            )

        assert tab.pending_user_messages == []

    def test_handler_registered_in_dispatch_table(self) -> None:
        """``appendUserMessage`` must be wired into ``_HANDLERS``."""
        from kiss.agents.vscode.commands import _CommandsMixin

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
        agent = SorcarAgent.__new__(SorcarAgent)
        agent._tab_id = "tab-drain"  # type: ignore[attr-defined]

        tab = _RunningAgentState("tab-drain", "gemini")
        tab.is_task_active = True
        tab.pending_user_messages = ["msg1", "msg2", "msg3"]
        _RunningAgentState.running_agent_states["tab-drain"] = tab

        model = _RecordingModel()
        agent._drain_pending_user_messages(model)

        assert model.calls == [
            ("user", "msg1"),
            ("user", "msg2"),
            ("user", "msg3"),
        ]
        assert tab.pending_user_messages == []

    def test_drain_noop_when_no_tab_id(self) -> None:
        agent = SorcarAgent.__new__(SorcarAgent)
        agent._tab_id = ""  # type: ignore[attr-defined]
        model = _RecordingModel()
        agent._drain_pending_user_messages(model)
        assert model.calls == []

    def test_drain_noop_when_tab_id_missing_from_registry(self) -> None:
        agent = SorcarAgent.__new__(SorcarAgent)
        agent._tab_id = "ghost-tab"  # type: ignore[attr-defined]
        model = _RecordingModel()
        agent._drain_pending_user_messages(model)
        assert model.calls == []

    def test_drain_noop_when_queue_empty(self) -> None:
        agent = SorcarAgent.__new__(SorcarAgent)
        agent._tab_id = "tab-empty"  # type: ignore[attr-defined]

        tab = _RunningAgentState("tab-empty", "gemini")
        tab.pending_user_messages = []
        _RunningAgentState.running_agent_states["tab-empty"] = tab

        model = _RecordingModel()
        agent._drain_pending_user_messages(model)
        assert model.calls == []

    def test_queue_then_drain_roundtrip(self) -> None:
        """End-to-end: command handler queues, drain consumes."""
        server, _events = _make_server()
        tab = _RunningAgentState("tab-rt", "gemini")
        tab.is_task_active = True
        _RunningAgentState.running_agent_states["tab-rt"] = tab

        server._cmd_append_user_message(
            {"tabId": "tab-rt", "prompt": "first follow-up"},
        )
        server._cmd_append_user_message(
            {"tabId": "tab-rt", "prompt": "second follow-up"},
        )

        agent = SorcarAgent.__new__(SorcarAgent)
        agent._tab_id = "tab-rt"  # type: ignore[attr-defined]
        model = _RecordingModel()
        agent._drain_pending_user_messages(model)

        assert model.calls == [
            ("user", "first follow-up"),
            ("user", "second follow-up"),
        ]
        # Second drain (next model step) must be a no-op.
        agent._drain_pending_user_messages(model)
        assert model.calls == [
            ("user", "first follow-up"),
            ("user", "second follow-up"),
        ]


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
        :meth:`SorcarAgent._drain_pending_user_messages` as the hook,
        registers a tab with two pending user messages, then runs one
        ``_execute_step``.  The model records its conversation
        immediately before generating — both pending messages must
        already be in it.
        """
        model = _PreStepHookRecordingModel()
        agent = self._make_kiss_agent(model)

        sa = SorcarAgent.__new__(SorcarAgent)
        sa._tab_id = "tab-hook"  # type: ignore[attr-defined]

        tab = _RunningAgentState("tab-hook", "gemini")
        tab.pending_user_messages = ["queued 1", "queued 2"]
        _RunningAgentState.running_agent_states["tab-hook"] = tab

        agent.pre_step_hook = sa._drain_pending_user_messages
        result = agent._execute_step()

        assert result == "ok"
        roles_and_contents = [
            (m["role"], m["content"])
            for m in model.conversation_before_generate
        ]
        assert ("user", "queued 1") in roles_and_contents
        assert ("user", "queued 2") in roles_and_contents
        assert tab.pending_user_messages == []

    def test_pre_step_hook_none_does_not_break_step(self) -> None:
        model = _PreStepHookRecordingModel()
        agent = self._make_kiss_agent(model)
        assert agent.pre_step_hook is None
        result = agent._execute_step()
        assert result == "ok"


class TestPendingMessagesClearedOnTaskFinish:
    """Lingering queued messages must not leak across successive tasks."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_running_agent_state_default_is_empty_list(self) -> None:
        tab = _RunningAgentState("tab-x", "gemini")
        assert tab.pending_user_messages == []
        # And it must be a real list (mutable, separate per instance).
        other = _RunningAgentState("tab-y", "gemini")
        other.pending_user_messages.append("only-on-other")
        assert tab.pending_user_messages == []

    def test_task_runner_clears_pending_messages_after_run(self) -> None:
        """End-to-end: pending_user_messages must be empty after _run_task.

        Drives the real :meth:`_TaskRunnerMixin._run_task` with a stub
        ``tab.agent.run`` that queues two follow-ups mid-flight (the
        same path the frontend would take while a task is running).
        After ``_run_task`` returns, the outer ``finally`` block must
        have cleared the queue so the next task starts fresh.
        """
        import os
        import queue

        from kiss.agents.sorcar.worktree_sorcar_agent import (
            WorktreeSorcarAgent,
        )

        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        server = VSCodeServer()
        tab_id = "tab-clear-after-run"
        tab = server._get_tab(tab_id)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.agent = agent
        tab.is_task_active = True
        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue()

        def fake_run(**_kwargs: Any) -> None:
            tab.pending_user_messages.append("queued during task")
            tab.pending_user_messages.append("also queued")

        agent.run = fake_run  # type: ignore[assignment]

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
            },),
            daemon=True,
        )
        tab.task_thread = task_thread
        task_thread.start()
        task_thread.join(timeout=15)
        assert not task_thread.is_alive()

        # The same tab object is still in the registry — fetch the
        # post-finally state and confirm the queue was drained.
        post_tab = _RunningAgentState.running_agent_states.get(tab_id)
        assert post_tab is not None
        assert post_tab.pending_user_messages == []
        assert post_tab.is_task_active is False

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end backend wiring for the "interact with a RUNNING sub-agent"
feature: the input textbox shown on a running sub-agent's chat tab must
be able to

* STOP ONLY that sub-agent's task (parent and sibling sub-agents keep
  running), and
* INJECT follow-up prompts into that sub-agent's live conversation.

Wiring under test (all production code, no mocks of it):

* ``ChatSorcarAgent._run_tasks_parallel`` registers each sub-agent
  under its deterministic tab id with its OWN per-sub-agent
  ``stop_event`` (chained to the parent's) and stamps
  ``agent._tab_id`` so the pre-step drain hook targets the sub-agent's
  ``pending_user_messages`` queue.
* A ``KeyboardInterrupt`` raised inside one worker by the cooperative
  stop is absorbed by ``_run_single`` (reported as that one task's
  failure) unless the PARENT is being stopped.
* ``VSCodeServer._find_source_tab_for_viewer`` resolves a frontend
  viewer tab (subscribed to the sub-agent's task stream) to the
  sub-agent's registry entry, which makes both
  ``_stop_task(viewer_tab)`` and ``_cmd_append_user_message`` reach
  ONLY that sub-agent.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import (
    ChatSorcarAgent,
    _SubagentStopEvent,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer


def _clear_registry() -> None:
    _RunningAgentState.running_agent_states.clear()


class _RecordingPrinter(JsonPrinter):
    """``JsonPrinter`` that records every broadcast event in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._ev_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._ev_lock:
            self.events.append(event)


class _RecordingModel:
    """Records ``add_message_to_conversation`` calls (stands in for the
    LLM model object the drain hook writes into)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.calls.append((role, content))


def _wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    """Poll *predicate* until it returns truthy or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class TestSubagentOnlyStop:
    """Stopping ONE sub-agent leaves the parent and siblings running."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_stop_one_subagent_leaves_sibling_and_parent_alive(
        self, monkeypatch: Any,
    ) -> None:
        """Set the registry-published per-sub-agent stop event of ONE
        worker; that worker aborts via the cooperative
        ``KeyboardInterrupt`` path while its sibling completes and the
        parent's stop event stays unset."""
        release_sibling = threading.Event()

        def _stub_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
            printer = kwargs.get("printer")
            assert printer is not None
            self.printer = printer
            task_key = kwargs.get("prompt_template", "")
            stop = getattr(printer._thread_local, "stop_event", None)
            assert stop is not None
            if task_key == "victim":
                # Emulate the agent's step loop: every printer call
                # polls the cooperative stop signal (the production
                # poll lives in ``JsonPrinter._check_stop``).
                assert _wait_until(stop.is_set), (
                    "the victim sub-agent never observed its own "
                    "stop event"
                )
                raise KeyboardInterrupt("Agent stop requested")
            release_sibling.wait(10)
            assert not stop.is_set(), (
                "the sibling sub-agent's stop event must stay unset "
                "when only the victim is stopped"
            )
            return "success: true\nsummary: sibling done\n"

        monkeypatch.setattr(ChatSorcarAgent, "run", _stub_run)

        printer = _RecordingPrinter()
        printer._thread_local.task_id = "parent-task"
        parent_stop = threading.Event()
        printer._thread_local.stop_event = parent_stop
        parent = ChatSorcarAgent("parent")
        parent.printer = printer

        results: list[str] = []

        def _run_parent() -> None:
            results.extend(
                parent._run_tasks_parallel(["victim", "sibling"]),
            )

        runner = threading.Thread(target=_run_parent, daemon=True)
        runner.start()

        # Wait for the victim's registry entry, then stop ONLY it —
        # exactly what ``VSCodeServer._stop_task`` does once it
        # resolves the sub-agent's state.
        def _victim_state() -> _RunningAgentState | None:
            with _RunningAgentState._registry_lock:
                for tid, st in (
                    _RunningAgentState.running_agent_states.items()
                ):
                    if tid.endswith("__sub_0"):
                        return st
            return None

        def _victim_ready() -> bool:
            st = _victim_state()
            return st is not None and st.stop_event is not None

        assert _wait_until(_victim_ready), (
            "the victim sub-agent never registered its state"
        )
        victim_state = _victim_state()
        assert victim_state is not None
        assert isinstance(victim_state.stop_event, _SubagentStopEvent), (
            "each sub-agent must publish its own per-sub-agent stop "
            "event in the registry"
        )
        victim_state.stop_event.set()

        release_sibling.set()
        runner.join(timeout=15)
        assert not runner.is_alive(), "parent fan-out never finished"

        assert len(results) == 2, results
        assert "stopped" in results[0].lower(), (
            "the stopped sub-agent must report a stopped-by-user "
            f"failure result; got: {results[0]!r}"
        )
        assert "false" in results[0], results[0]
        assert "sibling done" in results[1], results[1]
        assert not parent_stop.is_set(), (
            "stopping one sub-agent must never stop the parent task"
        )

    def test_parent_stop_still_kills_whole_fanout(
        self, monkeypatch: Any,
    ) -> None:
        """A PARENT stop must keep propagating out of the fan-out as
        ``KeyboardInterrupt`` (the pre-existing whole-tree stop path)."""

        def _stub_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
            printer = kwargs.get("printer")
            assert printer is not None
            self.printer = printer
            printer._check_stop()
            return "success: true\nsummary: unreachable\n"

        monkeypatch.setattr(ChatSorcarAgent, "run", _stub_run)

        printer = _RecordingPrinter()
        printer._thread_local.task_id = "parent-task"
        parent_stop = threading.Event()
        parent_stop.set()
        printer._thread_local.stop_event = parent_stop
        parent = ChatSorcarAgent("parent")
        parent.printer = printer

        raised = False
        try:
            parent._run_tasks_parallel(["t1"])
        except KeyboardInterrupt:
            raised = True
        assert raised, (
            "a parent-task stop must propagate KeyboardInterrupt out "
            "of _run_tasks_parallel so the task runner reports "
            "task_stopped"
        )


class TestSubagentPromptInjectionWiring:
    """Injected prompts must reach the SUB-AGENT's pending queue and
    be drained into the sub-agent's own conversation."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_subagent_gets_tab_id_and_drains_pending_messages(
        self, monkeypatch: Any,
    ) -> None:
        captured: dict[str, Any] = {}
        queued = threading.Event()

        def _stub_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
            printer = kwargs.get("printer")
            assert printer is not None
            self.printer = printer
            captured["tab_id"] = getattr(self, "_tab_id", "")
            # Wait for the test to queue a message on this sub-agent's
            # registry entry, then run the REAL drain hook exactly like
            # the pre-step hook does before each model call.
            assert queued.wait(10)
            model = _RecordingModel()
            SorcarAgent._drain_pending_user_messages(self, model)
            captured["model_calls"] = list(model.calls)
            return "success: true\nsummary: drained\n"

        monkeypatch.setattr(ChatSorcarAgent, "run", _stub_run)

        printer = _RecordingPrinter()
        printer._thread_local.task_id = "parent-task"
        printer._thread_local.stop_event = threading.Event()
        parent = ChatSorcarAgent("parent")
        parent.printer = printer

        results: list[str] = []
        runner = threading.Thread(
            target=lambda: results.extend(
                parent._run_tasks_parallel(["solo"]),
            ),
            daemon=True,
        )
        runner.start()

        def _sub_entry() -> tuple[str, _RunningAgentState] | None:
            with _RunningAgentState._registry_lock:
                for tid, st in (
                    _RunningAgentState.running_agent_states.items()
                ):
                    if tid.endswith("__sub_0"):
                        return tid, st
            return None

        assert _wait_until(lambda: _sub_entry() is not None)
        entry = _sub_entry()
        assert entry is not None
        sub_tab_id, sub_state = entry
        # Queue a follow-up exactly like _cmd_append_user_message does.
        with _RunningAgentState._registry_lock:
            sub_state.pending_user_messages.append("focus on tests")
        queued.set()
        runner.join(timeout=15)
        assert not runner.is_alive()

        assert captured["tab_id"] == sub_tab_id, (
            "the sub-agent's _tab_id must be its registry tab id so "
            "the pre-step drain hook targets the sub-agent's own "
            "pending_user_messages queue"
        )
        calls = captured["model_calls"]
        assert len(calls) == 1, calls
        role, content = calls[0]
        assert role == "user"
        assert "focus on tests" in content
        assert sub_state.pending_user_messages == [], (
            "the drain must empty the queue so the same message is "
            "never injected twice"
        )


class TestViewerTabResolvesToSubagent:
    """A frontend viewer tab subscribed to a sub-agent's task stream
    must resolve to the sub-agent's registry entry for both Stop and
    prompt injection."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def _make_server(self) -> tuple[VSCodeServer, list[dict[str, Any]]]:
        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with lock:
                events.append(event)

        server.printer.broadcast = capture  # type: ignore[assignment]
        return server, events

    def _register_subagent(
        self, server: VSCodeServer, *, task_id: str, viewer_tab: str,
    ) -> _RunningAgentState:
        """Register a live sub-agent state (as ``_run_tasks_parallel``
        does) and subscribe *viewer_tab* to its task stream (as
        ``_reattach_running_chat`` does for the frontend tab)."""
        agent = ChatSorcarAgent("sub")
        agent._last_task_id = task_id
        sub_tab_id = "task-parent__sub_0"
        state = _RunningAgentState(
            sub_tab_id,
            "",
            agent=agent,  # type: ignore[arg-type]
            chat_id="chat-1",
            is_subagent=True,
            parent_task_id="parent",
            is_task_active=True,
            stop_event=_SubagentStopEvent(threading.Event()),
        )
        _RunningAgentState.register(sub_tab_id, state)
        server.printer.subscribe_tab(task_id, viewer_tab)
        return state

    def test_find_source_tab_resolves_viewer_to_subagent(self) -> None:
        server, _events = self._make_server()
        self._register_subagent(
            server, task_id="77", viewer_tab="viewer-tab",
        )
        resolved = server._find_source_tab_for_viewer("viewer-tab")
        assert resolved == "task-parent__sub_0", (
            "the viewer tab must resolve to the sub-agent's registry "
            f"entry; got {resolved!r}"
        )

    def test_stop_on_viewer_tab_sets_only_subagent_event(self) -> None:
        server, _events = self._make_server()
        state = self._register_subagent(
            server, task_id="77", viewer_tab="viewer-tab",
        )
        assert state.stop_event is not None
        server._stop_task("viewer-tab")
        assert state.stop_event.is_set(), (
            "Stop on the sub-agent's chat tab must set the "
            "sub-agent's own stop event"
        )
        # The chained parent event must stay unset — only the
        # sub-agent stops.
        assert isinstance(state.stop_event, _SubagentStopEvent)
        parent_ev = state.stop_event._parent_event
        assert parent_ev is not None and not parent_ev.is_set(), (
            "stopping the sub-agent must not stop the parent task"
        )

    def test_append_user_message_routes_to_subagent_queue(self) -> None:
        server, events = self._make_server()
        state = self._register_subagent(
            server, task_id="77", viewer_tab="viewer-tab",
        )
        server._cmd_append_user_message(
            {"tabId": "viewer-tab", "prompt": "add more tests"},
        )
        assert state.pending_user_messages == ["add more tests"], (
            "a prompt typed on the sub-agent's chat tab must land in "
            "the SUB-AGENT's pending_user_messages queue; got "
            f"{state.pending_user_messages!r}"
        )
        echo = [
            e for e in events
            if e.get("type") == "prompt" and e.get("tabId") == "viewer-tab"
        ]
        assert echo and echo[0].get("text") == "add more tests", (
            "the queued prompt must be echoed back on the viewer tab"
        )

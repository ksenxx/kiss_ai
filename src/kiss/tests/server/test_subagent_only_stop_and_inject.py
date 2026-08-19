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

* ``ChatSorcarAgent._run_tasks_parallel`` gives each sub-agent worker
  its OWN per-sub-agent ``_SubagentStopEvent`` (chained to the
  parent's) bound to the printer's thread-local, and stamps
  ``agent._tab_id`` with the deterministic sub-tab id.
* A ``KeyboardInterrupt`` raised inside one worker by the cooperative
  stop is absorbed by ``_run_single`` (reported as that one task's
  failure) unless the PARENT is being stopped.
* The server resolves a frontend viewer tab (subscribed to the
  sub-agent's task stream) to the sub-agent's task-keyed
  ``agent_state`` entry, which makes both ``_stop_task(viewer_tab)``
  and ``_cmd_append_user_message`` reach ONLY that sub-agent.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _SubagentStopEvent
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


def _clear_registry() -> None:
    with agent_state.STATE_LOCK:
        agent_state.agent_states.clear()


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
        """Set the per-sub-agent stop event of ONE worker; that worker
        aborts via the cooperative ``KeyboardInterrupt`` path while its
        sibling completes and the parent's stop event stays unset."""
        release_sibling = threading.Event()
        sub_stop_events: dict[str, threading.Event] = {}
        events_lock = threading.Lock()

        def _stub_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
            printer = kwargs.get("printer")
            assert printer is not None
            self.printer = printer
            task_key = kwargs.get("prompt_template", "")
            stop = getattr(printer._thread_local, "stop_event", None)
            assert stop is not None
            with events_lock:
                sub_stop_events[task_key] = stop
            if task_key == "victim":
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

        def _victim_event() -> threading.Event | None:
            with events_lock:
                return sub_stop_events.get("victim")

        assert _wait_until(lambda: _victim_event() is not None), (
            "the victim sub-agent worker never started"
        )
        victim_stop = _victim_event()
        assert victim_stop is not None
        assert isinstance(victim_stop, _SubagentStopEvent), (
            "each sub-agent worker must get its own per-sub-agent "
            "_SubagentStopEvent chained to the parent's"
        )
        victim_stop.set()

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
        started = threading.Event()
        queued = threading.Event()

        def _stub_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
            printer = kwargs.get("printer")
            assert printer is not None
            self.printer = printer
            captured["tab_id"] = getattr(self, "_tab_id", "")
            captured["agent"] = self
            # Bind a task id on the worker thread (as the real run
            # does) and register the sub-agent's task-keyed state so
            # the server bridge (the only steering channel) can queue
            # into it.
            printer._thread_local.task_id = "sub-task-0"
            state = agent_state.AgentState("sub-task-0")
            agent_state.register(state)
            captured["state"] = state
            started.set()
            assert queued.wait(10)
            model = _RecordingModel()
            SorcarAgent._drain_pending_user_messages(self, model)
            captured["model_calls"] = list(model.calls)
            captured["leftover"] = list(state.pending_user_messages)
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

        assert started.wait(10), "the sub-agent worker never started"
        sub_state = captured["state"]
        sub_state.pending_user_messages.append("focus on tests")
        queued.set()
        runner.join(timeout=15)
        assert not runner.is_alive()

        assert str(captured["tab_id"]).endswith("__sub_0"), (
            "the sub-agent's _tab_id must be the deterministic "
            "'task-<parent>__sub_<idx>' id so per-sub-agent UI "
            f"routing works; got {captured['tab_id']!r}"
        )
        calls = captured["model_calls"]
        assert len(calls) == 1, calls
        role, content = calls[0]
        assert role == "user"
        assert "focus on tests" in content
        assert captured["leftover"] == [], (
            "the drain must empty the queue so the same message is "
            "never injected twice"
        )


class TestViewerTabResolvesToSubagent:
    """A frontend viewer tab subscribed to a sub-agent's task stream
    must resolve to the sub-agent's task-keyed agent state for both
    Stop and prompt injection."""

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
    ) -> agent_state.AgentState:
        """Register a live sub-agent state under its task id and
        subscribe *viewer_tab* to its task stream (as the server does
        when a frontend tab opens the sub-agent's transcript)."""
        agent = ChatSorcarAgent("sub")
        agent._last_task_id = task_id
        state = agent_state.AgentState(
            task_id,
            agent=agent,  # type: ignore[arg-type]
            chat_id="chat-1",
            parent_task_id="parent",
            is_task_active=True,
            stop_event=_SubagentStopEvent(threading.Event()),
        )
        agent_state.register(state)
        server.printer.subscribe_tab(task_id, viewer_tab)
        return state

    def test_stop_on_viewer_tab_sets_only_subagent_event(self) -> None:
        server, _events = self._make_server()
        state = self._register_subagent(
            server, task_id="77", viewer_tab="viewer-tab",
        )
        assert state.is_subagent, (
            "a state with parent_task_id set must report is_subagent"
        )
        assert state.stop_event is not None
        server._stop_task("viewer-tab")
        assert state.stop_event.is_set(), (
            "Stop on the sub-agent's chat tab must set the "
            "sub-agent's own stop event"
        )
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

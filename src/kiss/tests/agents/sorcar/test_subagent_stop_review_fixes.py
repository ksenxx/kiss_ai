# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the review-round fixes to the "interact with a
RUNNING sub-agent" feature:

1. ``VSCodeServer._open_persisted_subagent_tabs`` must SUBSCRIBE the
   reopened deterministic frontend tab (``{parent_tab_id}__sub_{id}``)
   to a STILL-RUNNING sub-agent's live stream — otherwise the input
   textbox shown on that tab is a dead surface (Stop / prompt
   injection cannot resolve the sub-agent, live events never arrive).

2. ``VSCodeServer._stop_task`` must FORCE-STOP a sub-agent wedged in an
   uninterruptible call (never polling its cooperative stop event) by
   injecting ``KeyboardInterrupt`` into the pool worker thread
   published on the sub-agent's registry state.

3. The force-stop watchdog's ownership guard must NEVER interrupt a
   SIBLING task that a reused ``ThreadPoolExecutor`` worker thread
   picked up after the stopped sub-agent finished cooperatively.

4. ``_SubagentStopEvent.wait`` semantics: own set, parent set mid-wait,
   and timeout expiry.

All tests drive the real production code (``_run_tasks_parallel``,
``VSCodeServer._stop_task`` / ``_open_persisted_subagent_tabs``, the
real registry and printer) — no mocks of the code under test.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import (
    ChatSorcarAgent,
    _SubagentStopEvent,
)
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


def _clear_registry() -> None:
    _RunningAgentState.running_agent_states.clear()
    ChatSorcarAgent.running_agents.clear()


def _wait_until(predicate: Any, timeout: float = 10.0) -> bool:
    """Poll *predicate* until truthy or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class _RecordingPrinter(JsonPrinter):
    """``JsonPrinter`` that records broadcasts and subscriptions."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.subscribe_calls: list[tuple[Any, str]] = []
        self._ev_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._ev_lock:
            self.events.append(event)

    def subscribe_tab(self, task_id: Any, tab_id: str) -> None:
        self.subscribe_calls.append((task_id, tab_id))
        super().subscribe_tab(task_id, tab_id)


class TestPersistedReopenSubscribesRunningSubagent:
    """Reopening a persisted parent whose sub-agent is STILL RUNNING
    must wire the deterministic sub tab into the live stream so the
    running-input surface (Stop / inject) actually works."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _setup_rows_and_server(
        self,
    ) -> tuple[VSCodeServer, _RecordingPrinter, str, str, str]:
        chat_id = "chat-persisted-reopen"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        sub_id, _ = th._add_task(
            "sub task body",
            chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_id}},
        )
        printer = _RecordingPrinter()
        server = VSCodeServer(printer=printer)
        return server, printer, chat_id, parent_id, sub_id

    def _register_live_sub(
        self, chat_id: str, parent_id: str, sub_id: str,
    ) -> _RunningAgentState:
        """Register the live sub-agent exactly as ``_run_tasks_parallel``
        and ``ChatSorcarAgent.run`` do mid-flight."""
        agent = ChatSorcarAgent("sub")
        agent._last_task_id = sub_id
        ChatSorcarAgent.running_agents[sub_id] = agent
        backend_tab_id = f"task-{parent_id}__sub_0"
        state = _RunningAgentState(
            backend_tab_id,
            "test-model",
            agent=agent,  # type: ignore[arg-type]
            chat_id=chat_id,
            is_subagent=True,
            parent_task_id=parent_id,
            is_task_active=True,
            stop_event=_SubagentStopEvent(threading.Event()),
        )
        state.task_history_id = sub_id
        _RunningAgentState.register(backend_tab_id, state)
        return state

    def test_running_sub_reopen_subscribes_and_routes_stop_inject(
        self,
    ) -> None:
        server, printer, chat_id, parent_id, sub_id = (
            self._setup_rows_and_server()
        )
        state = self._register_live_sub(chat_id, parent_id, sub_id)
        frontend_sub_tab = f"tab-parent__sub_{sub_id}"

        server._open_persisted_subagent_tabs(
            parent_task_id=parent_id, parent_tab_id="tab-parent",
        )

        # 1. The reopened deterministic tab is subscribed to the LIVE
        #    sub-agent's stream (keyed by the sub-agent's own row id).
        assert (sub_id, frontend_sub_tab) in printer.subscribe_calls, (
            f"reopened running sub tab was not subscribed; got "
            f"{printer.subscribe_calls!r}"
        )
        assert frontend_sub_tab in printer._fanout_targets(sub_id)

        # 2. The broadcast marks the sub-agent as RUNNING so the
        #    frontend shows the input textbox + buttons.
        opens = [
            e for e in printer.events if e.get("type") == "openSubagentTab"
        ]
        assert len(opens) == 1
        assert opens[0]["tab_id"] == frontend_sub_tab
        assert opens[0]["isDone"] is False

        # 3. Stop / prompt injection on the reopened tab resolve to the
        #    sub-agent's backend registry entry — the wiring the input
        #    surface depends on.
        resolved = server._find_source_tab_for_viewer(frontend_sub_tab)
        assert resolved == f"task-{parent_id}__sub_0", resolved

        server._stop_task(frontend_sub_tab)
        assert state.stop_event is not None and state.stop_event.is_set(), (
            "Stop on the reopened running sub tab must set the "
            "sub-agent's own stop event"
        )
        assert isinstance(state.stop_event, _SubagentStopEvent)
        parent_ev = state.stop_event._parent_event
        assert parent_ev is not None and not parent_ev.is_set()

        server._cmd_append_user_message(
            {"tabId": frontend_sub_tab, "prompt": "steer the sub"},
        )
        assert state.pending_user_messages == ["steer the sub"]

    def test_completion_race_during_reopen_emits_subagent_done(
        self,
    ) -> None:
        """The sub-agent finishes at the exact moment the persisted
        parent is reopened: its own ``subagentDone`` fan-out ran before
        the reopened tab subscribed.  ``_open_persisted_subagent_tabs``
        must recheck after broadcasting and emit ``subagentDone`` for
        the reopened tab itself — otherwise the tab pulses "running"
        (with a dead input surface) forever.

        The race is made deterministic through the pluggable printer:
        the moment the ``openSubagentTab`` broadcast goes out, the
        printer emulates the sub-agent's completion exactly as
        production does it (``running_agents`` pop, then state
        unregister) — i.e. AFTER the ``is_done`` snapshot, BEFORE the
        recheck.
        """
        server, printer, chat_id, parent_id, sub_id = (
            self._setup_rows_and_server()
        )
        self._register_live_sub(chat_id, parent_id, sub_id)
        frontend_sub_tab = f"tab-parent__sub_{sub_id}"
        backend_tab_id = f"task-{parent_id}__sub_0"

        original_broadcast = _RecordingPrinter.broadcast

        def _broadcast_with_finish(
            self_p: _RecordingPrinter, event: dict[str, Any],
        ) -> None:
            original_broadcast(self_p, event)
            if event.get("type") == "openSubagentTab":
                # The sub-agent finishes NOW — same order as
                # production: pop running_agents, then unregister.
                ChatSorcarAgent.running_agents.pop(sub_id, None)
                _RunningAgentState.unregister(backend_tab_id)

        printer.broadcast = (  # type: ignore[method-assign]
            _broadcast_with_finish.__get__(printer, _RecordingPrinter)
        )

        server._open_persisted_subagent_tabs(
            parent_task_id=parent_id, parent_tab_id="tab-parent",
        )

        dones = [
            e
            for e in printer.events
            if e.get("type") == "subagentDone"
            and e.get("tab_id") == frontend_sub_tab
        ]
        assert dones, (
            "a sub-agent that finished during the reopen must get a "
            "subagentDone broadcast for the reopened tab, else the tab "
            "shows a running input surface forever; events: "
            f"{[e.get('type') for e in printer.events]}"
        )

    def test_done_sub_reopen_is_not_subscribed(self) -> None:
        """A FINISHED sub-agent row reopens as a plain done tab: no
        live-stream subscription and ``isDone`` is True (the frontend
        then keeps the input hidden)."""
        server, printer, _chat_id, parent_id, sub_id = (
            self._setup_rows_and_server()
        )
        # No running_agents entry and no live registry state → done.

        server._open_persisted_subagent_tabs(
            parent_task_id=parent_id, parent_tab_id="tab-parent",
        )

        frontend_sub_tab = f"tab-parent__sub_{sub_id}"
        assert (sub_id, frontend_sub_tab) not in printer.subscribe_calls
        opens = [
            e for e in printer.events if e.get("type") == "openSubagentTab"
        ]
        assert len(opens) == 1
        assert opens[0]["isDone"] is True


class TestForceStopBlockedSubagent:
    """Stop on a sub-agent that NEVER polls its cooperative stop event
    (wedged in an API call) must still abort it — via the watchdog's
    ``KeyboardInterrupt`` injection into the published worker thread —
    while the sibling and the parent keep running."""

    def setup_method(self) -> None:
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()

    def test_stop_force_interrupts_wedged_subagent_only(
        self, monkeypatch: Any,
    ) -> None:
        release_sibling = threading.Event()
        sibling_interrupted = threading.Event()

        def _stub_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
            printer = kwargs.get("printer")
            assert printer is not None
            self.printer = printer
            task_key = kwargs.get("prompt_template", "")
            if task_key == "victim":
                # Wedged: never polls the stop event.  Sleeps in small
                # slices so the injected KeyboardInterrupt surfaces at
                # the next bytecode boundary.
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    time.sleep(0.02)
                return "success: true\nsummary: never stopped\n"
            try:
                assert release_sibling.wait(20)
            except KeyboardInterrupt:
                sibling_interrupted.set()
                raise
            return "success: true\nsummary: sibling done\n"

        monkeypatch.setattr(ChatSorcarAgent, "run", _stub_run)

        server = VSCodeServer()
        printer = server.printer
        printer._thread_local.task_id = "parent-task"
        parent_stop = threading.Event()
        printer._thread_local.stop_event = parent_stop
        parent = ChatSorcarAgent("parent")
        parent._last_task_id = "ptask"
        parent.printer = printer

        results: list[str] = []
        runner = threading.Thread(
            target=lambda: results.extend(
                parent._run_tasks_parallel(
                    ["victim", "sibling"], max_workers=2,
                ),
            ),
            daemon=True,
        )
        runner.start()

        victim_tab = "task-ptask__sub_0"

        def _victim_armed() -> bool:
            st = _RunningAgentState.running_agent_states.get(victim_tab)
            return (
                st is not None
                and st.stop_event is not None
                and st.task_thread is not None
                and st.task_thread.is_alive()
            )

        assert _wait_until(_victim_armed), (
            "victim sub-agent never published its stop event + worker "
            "thread in the registry"
        )

        # The exact user gesture: Stop clicked on the sub-agent's tab.
        server._stop_task(victim_tab)

        # The watchdog joins for 1 s, then injects KeyboardInterrupt;
        # the victim worker unwinds and unregisters its state.
        assert _wait_until(
            lambda: victim_tab
            not in _RunningAgentState.running_agent_states,
            timeout=15,
        ), "the wedged victim was never force-stopped"
        release_sibling.set()
        runner.join(timeout=20)
        assert not runner.is_alive(), "parallel fan-out never finished"

        assert len(results) == 2, results
        assert "stopped" in results[0].lower(), (
            "the wedged sub-agent must be force-stopped and report a "
            f"stopped-by-user result; got: {results[0]!r}"
        )
        assert "sibling done" in results[1], results[1]
        assert not sibling_interrupted.is_set(), (
            "the sibling sub-agent must never receive the injected "
            "KeyboardInterrupt"
        )
        assert not parent_stop.is_set(), (
            "force-stopping one sub-agent must not stop the parent"
        )

    def test_watchdog_never_interrupts_reused_pool_thread(
        self, monkeypatch: Any,
    ) -> None:
        """max_workers=1: the stopped sub-agent exits cooperatively
        within the watchdog's 1 s grace window and the SAME pool thread
        picks up the sibling.  The ownership guard must observe that
        the victim's registry entry is gone and skip the injection —
        the sibling must complete untouched."""
        sibling_interrupted = threading.Event()

        def _stub_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
            printer = kwargs.get("printer")
            assert printer is not None
            self.printer = printer
            task_key = kwargs.get("prompt_template", "")
            stop = getattr(printer._thread_local, "stop_event", None)
            assert stop is not None
            if task_key == "victim":
                # Cooperative exit: return as soon as the stop fires.
                assert _wait_until(stop.is_set, 10)
                return "success: false\nsummary: victim exited\n"
            # Sibling: run PAST the watchdog's 1 s join + injection
            # window on the same reused pool thread.
            try:
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    time.sleep(0.02)
            except KeyboardInterrupt:
                sibling_interrupted.set()
                raise
            return "success: true\nsummary: sibling done\n"

        monkeypatch.setattr(ChatSorcarAgent, "run", _stub_run)

        server = VSCodeServer()
        printer = server.printer
        printer._thread_local.task_id = "parent-task"
        printer._thread_local.stop_event = threading.Event()
        parent = ChatSorcarAgent("parent")
        parent._last_task_id = "ptask2"
        parent.printer = printer

        results: list[str] = []
        runner = threading.Thread(
            target=lambda: results.extend(
                parent._run_tasks_parallel(
                    ["victim", "sibling"], max_workers=1,
                ),
            ),
            daemon=True,
        )
        runner.start()

        victim_tab = "task-ptask2__sub_0"

        def _victim_armed() -> bool:
            st = _RunningAgentState.running_agent_states.get(victim_tab)
            return (
                st is not None
                and st.stop_event is not None
                and st.task_thread is not None
            )

        assert _wait_until(_victim_armed)
        server._stop_task(victim_tab)

        runner.join(timeout=25)
        assert not runner.is_alive(), "parallel fan-out never finished"
        assert len(results) == 2, results
        assert "victim exited" in results[0], results[0]
        assert "sibling done" in results[1], (
            "the sibling running on the REUSED pool worker thread must "
            f"complete untouched; got: {results[1]!r}"
        )
        assert not sibling_interrupted.is_set(), (
            "the force-stop watchdog interrupted the sibling task that "
            "the reused pool thread picked up after the victim finished"
        )


class TestSubagentStopEventWaitSemantics:
    """``_SubagentStopEvent.wait`` must observe its own flag, the
    parent chain, and timeouts."""

    def test_wait_returns_true_when_own_flag_set(self) -> None:
        ev = _SubagentStopEvent(threading.Event())
        ev.set()
        assert ev.wait(0.0) is True
        assert ev.wait(1.0) is True

    def test_wait_times_out_false_when_nothing_set(self) -> None:
        ev = _SubagentStopEvent(threading.Event())
        start = time.monotonic()
        assert ev.wait(0.15) is False
        assert time.monotonic() - start < 5.0

    def test_wait_wakes_on_parent_set_mid_wait(self) -> None:
        parent = threading.Event()
        ev = _SubagentStopEvent(parent)
        threading.Timer(0.1, parent.set).start()
        assert ev.wait(5.0) is True

    def test_wait_without_parent_and_none_chain(self) -> None:
        ev = _SubagentStopEvent(None)
        assert ev.wait(0.05) is False
        ev.set()
        assert ev.wait(None) is True
        assert ev.is_set() is True

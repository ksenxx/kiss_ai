# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the review-round fixes to the "interact with a
RUNNING sub-agent" feature:

``VSCodeServer._open_persisted_subagent_tabs`` must SUBSCRIBE the
reopened deterministic frontend tab (``{parent_tab_id}__sub_{id}``)
to a STILL-RUNNING sub-agent's live stream — otherwise the input
textbox shown on that tab is a dead surface (Stop / prompt injection
cannot resolve the sub-agent, live events never arrive).  A sub-agent
is a daemon sub-task with its own ``stop_event``, so Stop on its tab
leaves the parent's event untouched.

All tests drive the real production code (``VSCodeServer._stop_task``
/ ``_open_persisted_subagent_tabs``, the real registry and printer) —
no mocks of the code under test.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


def _clear_registry() -> None:
    with agent_state.STATE_LOCK:
        agent_state.agent_states.clear()


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

    def subscribe_tab(
        self, task_id: Any, tab_id: str,
    ) -> list[dict[str, Any]] | None:
        self.subscribe_calls.append((task_id, tab_id))
        return super().subscribe_tab(task_id, tab_id)


class _DbRedirectBase:
    """Per-test ``history.db`` redirection + registry cleanup."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        _clear_registry()

    def teardown_method(self) -> None:
        _clear_registry()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestPersistedReopenSubscribesRunningSubagent(_DbRedirectBase):
    """Reopening a persisted parent whose sub-agent is STILL RUNNING
    must wire the deterministic sub tab into the live stream so the
    running-input surface (Stop / inject) actually works."""

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
    ) -> tuple[agent_state.AgentState, agent_state.AgentState]:
        """Register the live parent and the live sub-agent exactly as
        the printer bridge (``agent_task_allocated``) does mid-flight.
        Returns ``(parent, sub)``."""
        parent = agent_state.AgentState(
            parent_id,
            agent=WorktreeSorcarAgent("parent"),
            chat_id=chat_id,
            tab_id="tab-parent",
            stop_event=threading.Event(),
            is_task_active=True,
        )
        agent_state.register(parent)
        agent = WorktreeSorcarAgent("sub")
        agent._last_task_id = sub_id
        backend_tab_id = f"task-{parent_id}__sub_0"
        state = agent_state.AgentState(
            sub_id,
            agent=agent,
            chat_id=chat_id,
            tab_id=backend_tab_id,
            parent_task_id=parent_id,
            stop_event=threading.Event(),
            is_task_active=True,
        )
        agent_state.register(state)
        return parent, state

    def test_running_sub_reopen_subscribes_and_routes_stop_inject(
        self,
    ) -> None:
        server, printer, chat_id, parent_id, sub_id = (
            self._setup_rows_and_server()
        )
        parent, state = self._register_live_sub(chat_id, parent_id, sub_id)
        frontend_sub_tab = f"tab-parent__sub_{sub_id}"

        server._open_persisted_subagent_tabs(
            parent_task_id=parent_id, parent_tab_id="tab-parent",
        )

        assert (sub_id, frontend_sub_tab) in printer.subscribe_calls, (
            f"reopened running sub tab was not subscribed; got "
            f"{printer.subscribe_calls!r}"
        )
        assert frontend_sub_tab in printer._fanout_targets(sub_id)

        opens = [
            e for e in printer.events if e.get("type") == "openSubagentTab"
        ]
        assert len(opens) == 1
        assert opens[0]["tab_id"] == frontend_sub_tab
        assert opens[0]["isDone"] is False

        server._stop_task(frontend_sub_tab)
        assert state.stop_event is not None and state.stop_event.is_set(), (
            "Stop on the reopened running sub tab must set the "
            "sub-agent's own stop event"
        )
        assert parent.stop_event is not None and not parent.stop_event.is_set(), (
            "stopping the sub-agent must not stop the parent task"
        )

        server._cmd_append_user_message(
            {"tabId": frontend_sub_tab, "prompt": "steer the sub"},
        )
        assert state.pending_user_messages == ["steer the sub"]
        assert parent.pending_user_messages == []

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
        production does it (``agent_task_finished`` unregisters the
        non-server-owned state) — i.e. AFTER the ``is_done`` snapshot,
        BEFORE the recheck.
        """
        server, printer, chat_id, parent_id, sub_id = (
            self._setup_rows_and_server()
        )
        self._register_live_sub(chat_id, parent_id, sub_id)
        frontend_sub_tab = f"tab-parent__sub_{sub_id}"

        original_broadcast = _RecordingPrinter.broadcast

        def _broadcast_with_finish(
            self_p: _RecordingPrinter, event: dict[str, Any],
        ) -> None:
            original_broadcast(self_p, event)
            if event.get("type") == "openSubagentTab":
                agent_state.unregister(sub_id)

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

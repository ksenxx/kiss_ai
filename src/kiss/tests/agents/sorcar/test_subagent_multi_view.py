# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for multi-view of running sub-agents.

Spec
----
A running sub-agent is registered in
:attr:`kiss.agents.sorcar.running_agent_state._RunningAgentState.running_agent_states`
just like a regular task, keyed by the sub-agent's own ``sub_tab_id``,
with two extra flag fields:

* ``is_subagent = True``
* ``parent_task_id = <int>`` (parent's ``task_history.id``)

The state's ``chat_id`` mirrors the parent's chat (sub-agents share
the parent's session) and ``task_history_id`` is mirrored from the
sub-agent's own ``task_history`` row while its
:meth:`ChatSorcarAgent.run` is executing.  This makes the sub-agent
discoverable to :meth:`VSCodeServer._reattach_running_chat` via the
``task_id`` disambiguator, so clicking the sub-agent row in the
history sidebar subscribes the freshly-opened tab to the live event
stream without stealing the parent's tab.

The frontend handles the rest: the ``openSubagentTab`` broadcast
emitted before ``task_events`` flips the new tab into sub-agent
styling and suppresses adjacent-task loading.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class TestSubagentRegistersRunningState:
    """``_run_tasks_parallel`` must register a real ``_RunningAgentState``
    for each sub-agent so multi-view works."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_running_subagent_has_state_with_flags(self) -> None:
        """While the sub-agent is in flight, its ``_RunningAgentState``
        must exist, mirror the parent's ``chat_id``, carry
        ``is_subagent=True`` and ``parent_task_id=<int>``, and (because
        :meth:`ChatSorcarAgent.run` mirrors it) expose
        ``task_history_id`` equal to the sub-agent's own
        ``task_history`` row id.
        """
        parent_chat_id = "chat-parent"
        parent_task_id, _ = th._add_task("parent", chat_id=parent_chat_id)

        parent = ChatSorcarAgent("parent")
        parent._chat_id = parent_chat_id
        parent._last_task_id = parent_task_id
        printer = JsonPrinter()
        printer._thread_local.task_id = "tab-parent"
        parent.printer = printer  # type: ignore[assignment]

        observed: dict[str, Any] = {}
        gate = threading.Event()
        done = threading.Event()

        def fake_run(self: ChatSorcarAgent, **_kw: Any) -> str:
            sub_states = [
                s for s in _RunningAgentState.running_agent_states.values()
                if s.is_subagent
            ]
            assert len(sub_states) == 1, sub_states
            st = sub_states[0]
            observed["tab_id"] = st.tab_id
            observed["state"] = st
            observed["chat_id"] = st.chat_id
            observed["is_subagent"] = st.is_subagent
            observed["parent_task_id"] = st.parent_task_id
            observed["task_history_id"] = st.task_history_id
            observed["is_task_active"] = st.is_task_active
            gate.set()
            done.wait(timeout=2)
            return '{"success": true, "summary": "ok"}'

        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        orig = SorcarAgent.run
        worker_result: dict[str, Any] = {}

        def _worker() -> None:
            printer._thread_local.task_id = "tab-parent"
            worker_result["out"] = parent._run_tasks_parallel(
                ["do something"],
                max_workers=1,
            )

        t = threading.Thread(target=_worker, daemon=True)
        SorcarAgent.run = fake_run  # type: ignore[assignment,method-assign]
        try:
            t.start()
            assert gate.wait(timeout=5), "fake_run never reached"

            assert observed["tab_id"] == f"task-{parent_task_id}__sub_0"
            assert observed["state"] is not None
            assert observed["chat_id"] == parent_chat_id
            assert observed["is_subagent"] is True
            assert observed["parent_task_id"] == parent_task_id
            assert observed["is_task_active"] is True
            assert isinstance(observed["task_history_id"], str)
            assert observed["task_history_id"]
        finally:
            SorcarAgent.run = orig  # type: ignore[method-assign]
            done.set()
            t.join(timeout=5)

        assert f"task-{parent_task_id}__sub_0" not in (
            _RunningAgentState.running_agent_states
        )


class TestReattachRunningChatTaskIdDisambiguation:
    """``_reattach_running_chat`` with ``task_id`` must match the
    sub-agent's state — not the parent's — even though they share
    ``chat_id``."""

    def setup_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def teardown_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_task_id_disambiguates_sub_from_parent(self) -> None:
        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        subs: list[tuple[Any, str]] = []

        def _stub_subscribe(task_id: Any, tab_id: str) -> None:
            subs.append((task_id, tab_id))

        server.printer.subscribe_tab = _stub_subscribe  # type: ignore[assignment]

        parent_state = _RunningAgentState("tab-parent", "test-model")
        parent_state.chat_id = "shared-chat"
        parent_state.task_history_id = "100"
        parent_state.is_task_active = True
        _RunningAgentState.running_agent_states["tab-parent"] = parent_state

        sub_state = _RunningAgentState("tab-parent__sub_0", "test-model")
        sub_state.chat_id = "shared-chat"
        sub_state.task_history_id = "200"
        sub_state.is_subagent = True
        sub_state.parent_task_id = "100"
        sub_state.is_task_active = True
        _RunningAgentState.running_agent_states["tab-parent__sub_0"] = (
            sub_state
        )

        ok = server._reattach_running_chat(
            "shared-chat", "tab-history-click", task_id="200",
        )
        assert ok is True
        assert subs == [("200", "tab-history-click")]

        subs.clear()
        ok2 = server._reattach_running_chat(
            "shared-chat", "tab-fresh-viewer",
        )
        assert ok2 is True
        assert len(subs) == 1

    def test_subagent_row_does_not_fall_back_to_parent(self) -> None:
        """A sub-agent row whose own thread has already ended (no
        ``task_history_id`` match) MUST NOT fall back to the parent's
        live stream — that would land sub-agent-styled tab events
        into the parent's chat.
        """
        server = VSCodeServer()
        subs: list[tuple[Any, str]] = []
        server.printer.subscribe_tab = (  # type: ignore[assignment]
            lambda task_id, tab_id: subs.append((task_id, tab_id))
        )

        parent_state = _RunningAgentState("tab-parent", "test-model")
        parent_state.chat_id = "chat-A"
        parent_state.task_history_id = "100"
        parent_state.is_task_active = True
        _RunningAgentState.running_agent_states["tab-parent"] = parent_state

        ok = server._reattach_running_chat(
            "chat-A",
            "tab-history-click",
            task_id="999",
            is_subagent=True,
        )
        assert ok is False
        assert subs == []

    def test_regular_row_falls_back_to_chat_id(self) -> None:
        """A regular task row whose thread has already finished (no
        ``task_history_id`` match) DOES fall back to a live state in
        the same chat — preserving the existing multi-view behavior
        for chat resumes.
        """
        server = VSCodeServer()
        subs: list[tuple[Any, str]] = []
        server.printer.subscribe_tab = (  # type: ignore[assignment]
            lambda task_id, tab_id: subs.append((task_id, tab_id))
        )

        parent_state = _RunningAgentState("tab-parent", "test-model")
        parent_state.chat_id = "chat-A"
        parent_state.task_history_id = "100"
        parent_state.is_task_active = True
        _RunningAgentState.running_agent_states["tab-parent"] = parent_state

        ok = server._reattach_running_chat(
            "chat-A", "tab-history-click", task_id="999",
        )
        assert ok is True
        assert subs == [("100", "tab-history-click")]


class TestReplaySessionSubscribesRunningSubagent:
    """End-to-end: clicking a still-running sub-agent in the history
    sidebar (``_replay_session(chat_id, tab_id, task_id=sub_id)``)
    subscribes the new tab to the sub-agent's live event stream."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_history_click_subscribes_subagent_stream(self) -> None:
        chat_id = "chat-multi-view"
        parent_id, _ = th._add_task("parent", chat_id=chat_id)
        sub_id, _ = th._add_task("sub-task body", chat_id=chat_id)
        th._append_chat_event(
            {"type": "text_delta", "text": "live"},
            task_id=sub_id,
        )
        th._save_task_extra(
            {
                "model": "m",
                "work_dir": "/tmp",
                "version": "v",
                "tokens": 0,
                "cost": 0.0,
                "is_parallel": False,
                "is_worktree": False,
                "subagent": {"parent_task_id": parent_id},
            },
            task_id=sub_id,
        )

        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        subs: list[tuple[Any, str]] = []
        server.printer.subscribe_tab = (  # type: ignore[assignment]
            lambda task_id, tab_id: subs.append((task_id, tab_id))
        )

        parent_state = _RunningAgentState("tab-parent", "test")
        parent_state.chat_id = chat_id
        parent_state.task_history_id = parent_id
        parent_state.is_task_active = True
        _RunningAgentState.running_agent_states["tab-parent"] = parent_state

        sub_state = _RunningAgentState("tab-parent__sub_0", "test")
        sub_state.chat_id = chat_id
        sub_state.task_history_id = sub_id
        sub_state.is_subagent = True
        sub_state.parent_task_id = parent_id
        sub_state.is_task_active = True
        _RunningAgentState.running_agent_states["tab-parent__sub_0"] = (
            sub_state
        )

        new_tab_id = "tab-history-click"
        server._replay_session(
            chat_id=chat_id, tab_id=new_tab_id, task_id=sub_id,
        )

        assert (sub_id, new_tab_id) in subs
        assert (parent_id, new_tab_id) not in subs

        types = [e.get("type") for e in events]
        assert "openSubagentTab" in types
        assert "task_events" in types
        assert types.index("openSubagentTab") < types.index("task_events")

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1
        assert "isDone" in opens[0]
        assert isinstance(opens[0]["isDone"], bool)


def test_subagent_state_pop_is_idempotent() -> None:
    """Popping the sub-agent's registry entry must be safe even when
    the entry is missing (the printer's ``_persist_agents`` map may
    survive a partial shutdown, but the state pop is unconditional).
    """
    _RunningAgentState.running_agent_states.clear()
    _RunningAgentState.running_agent_states.pop("nonexistent", None)
    time.sleep(0)

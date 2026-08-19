# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for multi-view of running sub-agents.

Spec
----
A running sub-agent is registered in the task-keyed registry
:data:`kiss.server.agent_state.agent_states` just like a regular
task, keyed by the sub-agent's own ``task_history`` row id, with
``parent_task_id`` set (which makes ``is_subagent`` True) and
``chat_id`` mirroring the parent's chat (sub-agents share the
parent's session).  This makes the sub-agent discoverable to
:meth:`VSCodeServer._reattach_running_chat` via the ``task_id``
disambiguator, so clicking the sub-agent row in the history sidebar
subscribes the freshly-opened tab to the live event stream without
stealing the parent's tab.

The frontend handles the rest: the ``openSubagentTab`` broadcast
emitted before ``task_events`` flips the new tab into sub-agent
styling and suppresses adjacent-task loading.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
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


class TestReattachRunningChatTaskIdDisambiguation:
    """``_reattach_running_chat`` with ``task_id`` must match the
    sub-agent's state — not the parent's — even though they share
    ``chat_id``."""

    def setup_method(self) -> None:
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()

    def test_task_id_disambiguates_sub_from_parent(self) -> None:
        server = VSCodeServer()
        subs: list[tuple[Any, str]] = []

        def _stub_subscribe(task_id: Any, tab_id: str) -> None:
            subs.append((task_id, tab_id))

        server.printer.subscribe_tab = _stub_subscribe  # type: ignore[assignment]

        try:
            parent_state = agent_state.AgentState(
                "100",
                chat_id="shared-chat",
                tab_id="tab-parent",
                server_owned=True,
                is_task_active=True,
            )
            agent_state.register(parent_state)

            sub_state = agent_state.AgentState(
                "200",
                chat_id="shared-chat",
                tab_id="tab-parent__sub_0",
                parent_task_id="100",
                is_task_active=True,
            )
            agent_state.register(sub_state)
            assert sub_state.is_subagent is True

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
            assert subs == [("100", "tab-fresh-viewer")]
        finally:
            agent_state.agent_states.clear()

    def test_subagent_row_does_not_fall_back_to_parent(self) -> None:
        """A sub-agent row whose own run has already ended (no
        registry entry for its task id) MUST NOT fall back to the
        parent's live stream — that would land sub-agent-styled tab
        events into the parent's chat.
        """
        server = VSCodeServer()
        subs: list[tuple[Any, str]] = []
        server.printer.subscribe_tab = (  # type: ignore[assignment]
            lambda task_id, tab_id: subs.append((task_id, tab_id))
        )

        try:
            parent_state = agent_state.AgentState(
                "100",
                chat_id="chat-A",
                tab_id="tab-parent",
                server_owned=True,
                is_task_active=True,
            )
            agent_state.register(parent_state)

            ok = server._reattach_running_chat(
                "chat-A",
                "tab-history-click",
                task_id="999",
                is_subagent=True,
            )
            assert ok is False
            assert subs == []
        finally:
            agent_state.agent_states.clear()

    def test_regular_row_falls_back_to_chat_id(self) -> None:
        """A regular task row whose run has already finished (no
        registry entry for its task id) DOES fall back to a live
        state in the same chat — preserving the existing multi-view
        behavior for chat resumes.
        """
        server = VSCodeServer()
        subs: list[tuple[Any, str]] = []
        server.printer.subscribe_tab = (  # type: ignore[assignment]
            lambda task_id, tab_id: subs.append((task_id, tab_id))
        )

        try:
            parent_state = agent_state.AgentState(
                "100",
                chat_id="chat-A",
                tab_id="tab-parent",
                server_owned=True,
                is_task_active=True,
            )
            agent_state.register(parent_state)

            ok = server._reattach_running_chat(
                "chat-A", "tab-history-click", task_id="999",
            )
            assert ok is True
            assert subs == [("100", "tab-history-click")]
        finally:
            agent_state.agent_states.clear()


class TestReplaySessionSubscribesRunningSubagent:
    """End-to-end: clicking a still-running sub-agent in the history
    sidebar (``_replay_session(chat_id, tab_id, task_id=sub_id)``)
    subscribes the new tab to the sub-agent's live event stream."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
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

        parent_state = agent_state.AgentState(
            str(parent_id),
            chat_id=chat_id,
            tab_id="tab-parent",
            server_owned=True,
            is_task_active=True,
        )
        agent_state.register(parent_state)

        sub_state = agent_state.AgentState(
            str(sub_id),
            chat_id=chat_id,
            tab_id="tab-parent__sub_0",
            parent_task_id=str(parent_id),
            is_task_active=True,
        )
        agent_state.register(sub_state)

        new_tab_id = "tab-history-click"
        server._replay_session(
            chat_id=chat_id, tab_id=new_tab_id, task_id=sub_id,
        )

        sub_keys = {str(k) for k, t in subs if t == new_tab_id}
        assert str(sub_id) in sub_keys
        assert str(parent_id) not in sub_keys

        types = [e.get("type") for e in events]
        assert "openSubagentTab" in types
        assert "task_events" in types
        assert types.index("openSubagentTab") < types.index("task_events")

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1
        assert "isDone" in opens[0]
        assert isinstance(opens[0]["isDone"], bool)

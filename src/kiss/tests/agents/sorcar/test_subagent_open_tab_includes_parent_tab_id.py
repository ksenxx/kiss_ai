"""Live sub-agent ``openSubagentTab`` broadcasts must include the parent's
frontend tab id so that closing the parent tab cascade-closes every
sub-agent tab it spawned.

The frontend's ``closeTab`` (in ``media/main.js``) walks the
``parentTabId`` chain it built from incoming ``openSubagentTab``
events.  If the backend omits ``parent_tab_id`` from the broadcast,
the new sub-agent tab is created with no parent linkage and the
recursive close stops there — leaking the sub-agent tab when the user
closes the parent.

Repro: when a parent task spawns a parallel sub-agent, the
``new_tab`` broadcast causes the frontend to allocate a fresh chat
tab and post ``resumeSession`` for the sub-agent's
``task_history.id``.  ``VSCodeServer._replay_session`` then sees
``extra.subagent`` and broadcasts ``openSubagentTab`` to flip the
new tab into sub-agent styling.  This broadcast MUST include
``parent_tab_id`` pointing at the parent's frontend tab so the
frontend can register the parent → child relationship.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer


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


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()
    real = JsonPrinter.broadcast

    def capture(event: dict) -> None:
        ev = server.printer._inject_task_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)

    server.printer.broadcast = capture  # type: ignore[assignment]
    _ = real
    return server, events


def _seed_subagent_row(
    *, parent_task_id: int, chat_id: str, description: str,
) -> int:
    task_id, _ = th._add_task(description, chat_id=chat_id)
    th._save_task_extra(
        {
            "model": "test-model",
            "work_dir": "/tmp",
            "version": "test",
            "tokens": 0,
            "cost": 0.0,
            "is_parallel": False,
            "is_worktree": False,
            "subagent": {"parent_task_id": parent_task_id},
        },
        task_id=task_id,
    )
    return task_id


class TestOpenSubagentTabIncludesParentTabId:
    """``openSubagentTab`` must carry ``parent_tab_id`` so the frontend
    can chain parent → child for recursive tab closes."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _RunningAgentState.running_agent_states.clear()

    def test_replay_subagent_broadcast_carries_parent_tab_id(self) -> None:
        chat_id = "chat-parent-pin"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        sub_task_id = _seed_subagent_row(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Sub-task: research X",
        )
        server, events = _make_server()

        # Simulate the parent task actively running on tab "tab-parent".
        # Its ``_RunningAgentState`` must carry ``task_history_id ==
        # parent_id`` so the broadcaster can resolve parent_tab_id from
        # the registry.
        parent_state = _RunningAgentState("tab-parent", "test-model")
        parent_state.chat_id = chat_id
        parent_state.task_history_id = parent_id
        parent_state.is_subagent = False
        _RunningAgentState.running_agent_states["tab-parent"] = parent_state

        server._replay_session(
            chat_id=chat_id,
            tab_id="tab-newly-created-sub",
            task_id=sub_task_id,
        )

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1, f"events={events}"
        op = opens[0]
        assert op["tab_id"] == "tab-newly-created-sub"
        assert op["parent_tab_id"] == "tab-parent", (
            "openSubagentTab must include parent_tab_id so the frontend "
            "can cascade-close the sub-agent tab when its parent is closed"
        )

    def test_replay_subagent_uses_agent_last_task_id_while_parent_running(
        self,
    ) -> None:
        """During the active spawn window the parent's
        ``_RunningAgentState.task_history_id`` is still ``None`` —
        ``_run_task`` zeroes it at task start and only re-populates
        it in the per-subtask ``finally`` block AFTER ``agent.run``
        returns.  The parent's task row id is exposed on the live
        agent as ``agent._last_task_id``.  The lookup must consult
        ``_last_task_id`` (mirroring :meth:`_get_running_task_ids`)
        so parent_tab_id resolves while the parent is still running
        — this is the path the cascade-close bug travels."""

        class _StubAgent:
            def __init__(self, last_task_id: int) -> None:
                self._last_task_id = last_task_id

        chat_id = "chat-parent-live"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        sub_task_id = _seed_subagent_row(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Sub-task: live",
        )
        server, events = _make_server()

        parent_state = _RunningAgentState("tab-parent-live", "test-model")
        parent_state.chat_id = chat_id
        # Mirror the live state: task_history_id NOT yet populated,
        # but the agent has already allocated the parent's row id.
        parent_state.task_history_id = None
        parent_state.agent = _StubAgent(parent_id)  # type: ignore[assignment]
        parent_state.is_subagent = False
        _RunningAgentState.running_agent_states["tab-parent-live"] = parent_state

        server._replay_session(
            chat_id=chat_id,
            tab_id="tab-sub-live",
            task_id=sub_task_id,
        )

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1, f"events={events}"
        assert opens[0]["parent_tab_id"] == "tab-parent-live", (
            "parent_tab_id must resolve via agent._last_task_id while "
            "the parent task is still running (task_history_id=None)"
        )

    def test_replay_subagent_parent_tab_empty_when_parent_not_running(
        self,
    ) -> None:
        """If the parent task isn't live (no ``_RunningAgentState`` for
        it), the broadcast still includes ``parent_tab_id`` as an empty
        string — the frontend tolerates an empty value and simply skips
        the parent linkage, leaving the cascade-close at this tab."""
        chat_id = "chat-orphan-pin"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        sub_task_id = _seed_subagent_row(
            parent_task_id=parent_id,
            chat_id=chat_id,
            description="Sub-task: orphan",
        )
        server, events = _make_server()

        server._replay_session(
            chat_id=chat_id,
            tab_id="tab-newly-created-sub-orphan",
            task_id=sub_task_id,
        )

        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == 1
        assert opens[0]["parent_tab_id"] == ""

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: sub-agent events are persisted to the sub-agent's
OWN ``task_history`` row (``has_events=1``), not silently dropped.

``JsonPrinter._persist_event`` resolves the agent state registered
under the event's ``taskId`` in the task-keyed ``agent_state``
registry to find the ``task_history`` row id under which to persist —
when the lookup misses, the event is silently dropped.  If a
sub-agent's run never registered itself (via
``printer.agent_task_allocated``), every sub-agent's row had
``has_events=0``, so the history-sidebar click handler took the
no-events branch (``setTaskText`` + leave input populated) instead of
the ``resumeSession`` branch, which from the user's perspective looked
like clicking the sub-task "opened a new chat tab instead of loading
all the events from the sub task".

This test runs the real ``_run_tasks_parallel`` path with a stub
underlying agent (so no LLM is needed), against a real
``JsonPrinter`` and a temp-dir SQLite DB, and asserts that
every sub-agent's row has ``has_events=1`` AND that the persisted
events are retrievable via ``_load_chat_events_by_task_id``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _load_chat_events_by_task_id,
)
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter


def _redirect(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore(saved: tuple) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class _StubAgent(ChatSorcarAgent):
    """ChatSorcarAgent whose ``super().run`` emits one persisted event
    via the shared printer and returns a YAML ``finish`` payload.

    Avoids spinning up a model; we only care that the printer/persist
    plumbing for sub-agents works.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.model_name = "stub"
        self.work_dir = "/tmp"
        self.total_tokens_used = 0
        self.budget_used = 0.0

    def run(self, prompt_template: str = "", **kwargs: Any) -> str:  # type: ignore[override]
        printer = kwargs.get("printer")
        from kiss.agents.sorcar.persistence import _add_task, _save_task_extra, _save_task_result

        task_id, self._chat_id = _add_task(prompt_template, chat_id=self._chat_id)
        self._last_task_id = task_id
        task_key = str(task_id)
        try:
            if printer is not None:
                tl = getattr(printer, "_thread_local", None)
                if tl is not None:
                    tl.task_id = task_key
                printer.agent_task_allocated(self, task_id, self._chat_id)
                printer.broadcast({
                    "type": "text_delta",
                    "text": f"subagent-run-event-{prompt_template[:20]}",
                })
            from kiss.core._version import __version__

            extra_payload: dict[str, object] = {
                "model": self.model_name,
                "work_dir": self.work_dir,
                "version": __version__,
                "tokens": 0,
                "cost": 0.0,
                "is_parallel": False,
                "is_worktree": False,
            }
            if self._subagent_info is not None:
                extra_payload["subagent"] = self._subagent_info
            _save_task_extra(extra_payload, task_id=task_id)
            result: str = yaml.dump(
                {"success": True, "summary": "stub"}, sort_keys=False,
            )
            _save_task_result(task_id=task_id, result="stub")
            return result
        finally:
            if printer is not None:
                printer.agent_task_finished(self, task_key)


class _RecordingPrinter(JsonPrinter):
    """JsonPrinter whose ``broadcast`` runs the SAME side effects
    as a real browser printer (inject tabId, record, persist) so the
    ``_persist_event`` plumbing is exercised end-to-end.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        event = self._inject_task_id(event)
        with self._lock:
            self._record_event(event)
        self.events.append(event)
        self._persist_event(event)


class TestSubagentEventsPersisted:
    """Each sub-agent's events MUST land in its own ``task_history`` row."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def test_subagent_rows_have_persisted_events(self) -> None:
        import kiss.agents.sorcar.chat_sorcar_agent as csa_mod

        real_cls = csa_mod.ChatSorcarAgent
        csa_mod.ChatSorcarAgent = _StubAgent  # type: ignore[misc]
        try:
            printer = _RecordingPrinter()
            printer._thread_local.task_id = "tab-parent"
            parent = real_cls("parent")
            parent.printer = printer
            parent.model_name = "stub"
            parent.work_dir = "/tmp"
            parent._chat_id = "chat-parent-shared"
            parent._last_task_id = "aaaaaaaabbbbccccddddeeeeffff0000"
            parent_state = agent_state.AgentState(
                parent._last_task_id,
                agent=parent,  # type: ignore[arg-type]
                tab_id="tab-parent",
                is_task_active=True,
            )
            agent_state.register(parent_state)

            tasks = ["sub task A", "sub task B", "sub task C"]
            results = parent._run_tasks_parallel(tasks, max_workers=1)
            assert len(results) == 3

            th._flush_chat_events()

            db = th._get_db()
            rows = db.execute(
                "SELECT id, parent_task_id, has_events FROM task_history "
                "WHERE parent_task_id IS NOT NULL AND parent_task_id != '' "
                "ORDER BY rowid ASC"
            ).fetchall()
            sub_rows = [
                {"id": r[0], "parent_task_id": r[1], "has_events": r[2]}
                for r in rows
            ]
            assert len(sub_rows) == 3, f"expected 3 sub-agent rows, got {sub_rows}"
            for h in sub_rows:
                assert h["has_events"] == 1, (
                    f"sub-agent row {h['id']} has_events=0 — events "
                    f"were not persisted: {h}"
                )
                row_id = h["id"]
                assert isinstance(row_id, str)
                loaded = _load_chat_events_by_task_id(row_id)
                assert loaded is not None
                evs = loaded.get("events", [])
                assert isinstance(evs, list)
                assert any(
                    e.get("type") == "text_delta"
                    and "subagent-run-event" in str(e.get("text", ""))
                    for e in evs
                ), f"events table for task {row_id} missing subagent event: {evs}"

            with agent_state.STATE_LOCK:
                leaked = [
                    tid for tid, st in agent_state.agent_states.items()
                    if st is not parent_state
                ]
            assert leaked == [], (
                "agent_task_finished must unregister every sub-agent's "
                f"state after its run; leaked: {leaked}"
            )
        finally:
            csa_mod.ChatSorcarAgent = real_cls  # type: ignore[misc]

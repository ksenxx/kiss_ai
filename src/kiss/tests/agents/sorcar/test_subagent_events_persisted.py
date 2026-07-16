# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: sub-agent events are persisted to the sub-agent's
OWN ``task_history`` row (``has_events=1``), not silently dropped.

Before this fix, ``ChatSorcarAgent._run_tasks_parallel`` set
``printer._thread_local.task_id = sub_tab_id`` so each sub-agent's
broadcast events were tagged with the ``sub_tab_id``, but it never
registered the sub-agent in ``printer._persist_agents[sub_tab_id]``.
``JsonPrinter._persist_event`` looks up the agent in
``_persist_agents`` by the event's ``tabId`` to find the task_id under
which to persist — when the lookup misses, the event is silently
dropped.  Result: every sub-agent's ``task_history`` row had
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
        # Run the real chat-session bookkeeping (which calls _add_task
        # to allocate ``_last_task_id`` for this sub-agent) but skip
        # the actual model invocation.
        printer = kwargs.get("printer")
        from kiss.agents.sorcar.persistence import _add_task, _save_task_extra, _save_task_result

        task_id, self._chat_id = _add_task(prompt_template, chat_id=self._chat_id)
        self._last_task_id = task_id
        # Mirror what the real ``ChatSorcarAgent.run`` does so the
        # printer can route events to this sub-agent's task row:
        # tag this worker thread with the new ``task_id`` and
        # register ``self`` in the printer's ``_persist_agents`` map
        # under the same key.  Without this plumbing
        # ``_inject_task_id`` would emit no ``taskId`` and
        # ``_persist_event`` would drop the event.
        task_key = str(task_id)
        if printer is not None:
            tl = getattr(printer, "_thread_local", None)
            if tl is not None:
                tl.task_id = task_key
            persist_map = getattr(printer, "_persist_agents", None)
            if persist_map is not None:
                persist_map[task_key] = self
        # Emit one display event through the printer's broadcast path.
        # Because we registered ourselves in ``_persist_agents`` keyed
        # by our own ``task_id``, this should land in the events table
        # under THIS sub-agent's ``task_id``.
        if printer is not None:
            printer.broadcast({
                "type": "text_delta",
                "text": f"subagent-run-event-{prompt_template[:20]}",
            })
        from kiss._version import __version__

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

    def test_subagent_rows_have_persisted_events(self) -> None:
        # Patch ChatSorcarAgent to be the stub so _run_tasks_parallel
        # creates stub sub-agents instead of real ones.
        import kiss.agents.sorcar.chat_sorcar_agent as csa_mod

        real_cls = csa_mod.ChatSorcarAgent
        csa_mod.ChatSorcarAgent = _StubAgent  # type: ignore[misc]
        try:
            printer = _RecordingPrinter()
            # Parent has a tab_id so sub-tab ids are deterministic.
            printer._thread_local.task_id = "tab-parent"
            # ``persist_agents`` must contain the parent registration
            # to mirror the production task_runner setup, but it's
            # NOT what we're testing — we're testing sub-agent rows.
            parent = real_cls("parent")
            parent.printer = printer
            parent.model_name = "stub"
            parent.work_dir = "/tmp"
            parent._chat_id = "chat-parent-shared"
            # ``_run_tasks_parallel`` reads ``parent._last_task_id``
            # to stamp the ``parent_task_id`` column on each sub-agent
            # row.  In production the parent's run() sets this very
            # early; here we set it explicitly so the assertion below
            # finds the sub-agent rows by parent_task_id.
            # Must be a 32-char lowercase-hex string to satisfy the
            # persistence ``_coerce_parent_task_id`` contract.
            parent._last_task_id = "aaaaaaaabbbbccccddddeeeeffff0000"
            printer._persist_agents["tab-parent"] = parent

            tasks = ["sub task A", "sub task B", "sub task C"]
            results = parent._run_tasks_parallel(tasks, max_workers=1)
            assert len(results) == 3

            # Persistence goes through the async event-writer thread;
            # under heavy parallel test load the sub-agent rows/events
            # may not have been flushed to SQLite by the time
            # ``_run_tasks_parallel`` returns.  Drain the queue before
            # querying so the assertions below are deterministic.
            th._flush_chat_events()

            # Each sub-agent's row must have has_events=1 AND its
            # events table must contain the emitted text_delta.
            # Note: _load_history filters out sub-agent rows via
            # _HISTORY_NOT_SUBAGENT, so we query the DB directly here
            # to find them.
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

            # Only the parent's entry should remain in
            # ``_persist_agents`` after ``_run_tasks_parallel`` returns;
            # the sub-agents' entries (keyed by their own ``task_id``)
            # are tied to per-task lifecycle and are cleaned by the
            # task runner via ``cleanup_task``.  We assert the keys
            # for the sub-agents — if any — are sub-task ids, NOT the
            # legacy ``tab-parent__sub_N`` sub-tab-id form.
            assert "tab-parent__sub_0" not in printer._persist_agents
            assert "tab-parent__sub_1" not in printer._persist_agents
            assert "tab-parent__sub_2" not in printer._persist_agents
        finally:
            csa_mod.ChatSorcarAgent = real_cls  # type: ignore[misc]

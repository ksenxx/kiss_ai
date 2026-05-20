"""Integration tests for the new sub-agent persistence schema.

Contract
--------
The ``task_history.extra`` JSON blob for a sub-agent row carries
ONLY a ``subagent`` key holding ``{"parent_task_id": <int>}``.  No
``tab_id``, ``parent_tab_id``, ``task_index``, or ``description``
fields are persisted — they are either ephemeral live-event fields
or derivable at render time.

The frontend handler must additionally skip overscroll-triggered
adjacent-task loading on sub-agent tabs so a reopened sub-agent
tab does not pull in the parent's chat siblings (they share the
same ``chat_id`` but the user opened the row explicitly).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.printer import Printer

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media" / "main.js"
)
_MAIN_CSS = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media" / "main.css"
)


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


class _CapturePrinter(BaseBrowserPrinter):
    """Printer that captures broadcasts and exercises persistence."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        event = self._inject_task_id(event)
        with self._lock:
            self._record_event(event)
        self.events.append(event)
        self._persist_event(event)


class _StubChatAgent(ChatSorcarAgent):
    """Sub-agent stub that records one event and writes ``extra``.

    Mirrors what the real :class:`ChatSorcarAgent` does in its
    ``run()`` but skips the LLM call.  The parent agent in
    :meth:`_run_tasks_parallel` already sets
    :attr:`_subagent_info` on the stub before calling ``run()``, so
    the stub just persists the payload verbatim.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.model_name = "stub"
        self.work_dir = "/tmp"
        self.total_tokens_used = 0
        self.budget_used = 0.0

    def run(self, prompt_template: str = "", **kwargs: Any) -> str:  # type: ignore[override]
        from kiss.agents.sorcar.persistence import (
            _add_task,
            _save_task_extra,
            _save_task_result,
        )

        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id,
        )
        self._last_task_id = task_id
        printer = kwargs.get("printer")
        if printer is not None:
            printer.broadcast({"type": "text_delta", "text": "stub-event"})
        extra_payload: dict[str, object] = {
            "model": self.model_name,
            "work_dir": self.work_dir,
            "version": "test",
            "tokens": 0,
            "cost": 0.0,
            "is_parallel": False,
            "is_worktree": False,
        }
        if self._subagent_info is not None:
            extra_payload["subagent"] = self._subagent_info
        _save_task_extra(extra_payload, task_id=task_id)
        _save_task_result(task_id=task_id, result="stub")
        out: str = yaml.dump(
            {"success": True, "summary": "stub"}, sort_keys=False,
        )
        return out


class TestPersistedSubagentBlob:
    """Each sub-agent row's ``extra.subagent`` blob must hold ONLY
    ``parent_task_id`` — no legacy frontend identity fields."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        ChatSorcarAgent.running_agents.clear()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_subagent_extra_contains_only_parent_task_id(self) -> None:
        import kiss.agents.sorcar.chat_sorcar_agent as csa_mod

        real_cls = csa_mod.ChatSorcarAgent
        csa_mod.ChatSorcarAgent = _StubChatAgent  # type: ignore[misc]
        try:
            printer = _CapturePrinter()
            printer._thread_local.task_id = "tab-parent"
            parent = real_cls("parent")
            parent.printer = cast(Printer | None, printer)
            parent.model_name = "stub"
            parent.work_dir = "/tmp"
            parent._chat_id = "chat-shared"
            # Simulate the parent task already being persisted —
            # _run_tasks_parallel reads ``_last_task_id``.
            from kiss.agents.sorcar.persistence import _add_task

            parent_task_id, _ = _add_task(
                "parent task", chat_id=parent._chat_id,
            )
            parent._last_task_id = parent_task_id
            printer._persist_agents["tab-parent"] = parent

            parent._run_tasks_parallel(["sub A", "sub B"], max_workers=1)

            # Sub-agent rows are filtered out of the user-facing
            # ``_load_history`` listing.  To verify the persisted
            # blob shape we query the DB directly.
            with th._rw_lock.read_lock():
                db = th._get_db()
                rows = db.execute(
                    "SELECT extra FROM task_history "
                    "WHERE extra LIKE '%\"subagent\"%' ORDER BY id ASC",
                ).fetchall()
            sub_rows = [dict(r) for r in rows]
            assert len(sub_rows) == 2, sub_rows
            for h in sub_rows:
                extra = json.loads(str(h["extra"]))
                assert "subagent" in extra
                sub = extra["subagent"]
                assert sub == {"parent_task_id": parent_task_id}, (
                    f"subagent payload must hold only parent_task_id, "
                    f"got {sub!r}"
                )
                # And no legacy fields leaked into the parent dict.
                assert "tab_id" not in sub
                assert "parent_tab_id" not in sub
                assert "task_index" not in sub
                assert "description" not in sub
        finally:
            csa_mod.ChatSorcarAgent = real_cls  # type: ignore[misc]




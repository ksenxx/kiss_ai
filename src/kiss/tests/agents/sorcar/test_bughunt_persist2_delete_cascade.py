# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt: ``_delete_task`` must cascade to persisted sub-agent rows.

Sub-agent rows (``extra.subagent.parent_task_id``, written by
``ChatSorcarAgent._run_tasks_parallel``) are reachable ONLY through
their parent task: ``_load_subagent_rows_by_parent_task_id`` is the
single read path, and every history/chat listing filters them out.

When the user deletes the parent task from the history sidebar
(``VSCodeServer._handle_delete_task`` → ``_delete_task``), the
sub-agent rows and their event logs survived as permanently
unreachable zombie rows.  Observable damage beyond the storage leak:
``_chat_has_tasks(chat_id)`` kept returning ``True`` for a chat whose
every visible task was deleted, so the frontend's ``taskDeleted``
broadcast carried ``chatHasMoreTasks: true`` and the now-empty chat
tab was never closed.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _chat_has_tasks,
    _delete_task,
    _load_chat_events_by_task_id,
    _load_subagent_rows_by_parent_task_id,
)


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _add_subagent(self, parent_id: str, chat_id: str, task: str) -> str:
        sub_id, _ = _add_task(
            task,
            chat_id=chat_id,
            extra={
                "subagent": {
                    "parent_task_id": parent_id,
                    "parent_tab_id": f"task-{parent_id}",
                },
            },
        )
        return sub_id


class TestDeleteParentCascadesToSubagents(_TempDbTestBase):
    """Deleting a parent must delete its sub-agent rows and events."""

    def test_subagent_rows_and_events_deleted_with_parent(self) -> None:
        parent_id, chat_id = _add_task("parent task")
        sub1 = self._add_subagent(parent_id, chat_id, "sub task 1")
        sub2 = self._add_subagent(parent_id, chat_id, "sub task 2")
        _append_chat_event({"type": "sub_ev"}, task_id=sub1)
        _append_chat_event({"type": "sub_ev"}, task_id=sub2)

        assert _delete_task(parent_id) is True

        assert _load_subagent_rows_by_parent_task_id(parent_id) == []
        assert _load_chat_events_by_task_id(sub1) is None
        assert _load_chat_events_by_task_id(sub2) is None
        # The chat is now genuinely empty — the frontend's
        # ``chatHasMoreTasks`` flag must agree so the tab gets closed.
        assert _chat_has_tasks(chat_id) is False

    def test_other_parents_subagents_survive(self) -> None:
        parent_a, chat_a = _add_task("parent A")
        parent_b, chat_b = _add_task("parent B")
        sub_a = self._add_subagent(parent_a, chat_a, "sub of A")
        sub_b = self._add_subagent(parent_b, chat_b, "sub of B")

        assert _delete_task(parent_a) is True

        assert _load_subagent_rows_by_parent_task_id(parent_a) == []
        rows_b = _load_subagent_rows_by_parent_task_id(parent_b)
        assert [r["task_id"] for r in rows_b] == [sub_b]
        assert _load_chat_events_by_task_id(sub_a) is None
        assert _chat_has_tasks(chat_b) is True

    def test_lookalike_non_subagent_row_not_deleted(self) -> None:
        parent_id, chat_id = _add_task("parent with lookalike")
        # A regular row whose free-form extra merely EMBEDS the marker
        # substring must NOT be cascade-deleted (same false-positive
        # defense as _load_subagent_rows_by_parent_task_id).
        lookalike_id, _ = _add_task(
            "regular task",
            chat_id=chat_id,
            extra={"note": f'{{"parent_task_id": {parent_id}}} subagent'},
        )

        assert _delete_task(parent_id) is True

        assert _load_chat_events_by_task_id(lookalike_id) is not None
        assert _chat_has_tasks(chat_id) is True

    def test_deleting_subagent_row_does_not_touch_parent(self) -> None:
        parent_id, chat_id = _add_task("parent stays")
        sub_id = self._add_subagent(parent_id, chat_id, "sub goes")

        assert _delete_task(sub_id) is True

        assert _load_chat_events_by_task_id(parent_id) is not None
        assert _load_subagent_rows_by_parent_task_id(parent_id) == []

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: ``_delete_task`` must cascade through NESTED sub-agents.

A sub-agent spawned by ``run_parallel`` is a full ``ChatSorcarAgent``
that itself exposes the ``run_parallel`` tool, so a sub-agent can fan
out its own sub-agents.  Each grandchild row carries
``extra.subagent.parent_task_id == <child id>`` — NOT the top-level
parent's id.  ``_delete_task`` only cascaded one level
(``_subagent_child_ids(db, task_id)``), so deleting the top-level
parent left every grandchild row (and its event log) behind as a
permanently unreachable zombie: the only read path,
``_load_subagent_rows_by_parent_task_id``, walks from a parent id that
no longer exists, and every history/chat listing filters sub-agent
rows out.  Observable damage beyond the storage leak:
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


class TestDeleteCascadesThroughNestedSubagents(_TempDbTestBase):
    """Deleting a parent must delete grandchildren sub-agent rows too."""

    def test_grandchild_rows_and_events_deleted_with_parent(self) -> None:
        parent_id, chat_id = _add_task("top-level parent task")
        child = self._add_subagent(parent_id, chat_id, "child sub task")
        grandchild = self._add_subagent(child, chat_id, "grandchild sub task")
        great = self._add_subagent(grandchild, chat_id, "great-grandchild")
        _append_chat_event({"type": "ev"}, task_id=grandchild)
        _append_chat_event({"type": "ev"}, task_id=great)

        assert _delete_task(parent_id) is True

        assert _load_subagent_rows_by_parent_task_id(parent_id) == []
        assert _load_subagent_rows_by_parent_task_id(child) == []
        assert _load_subagent_rows_by_parent_task_id(grandchild) == []
        assert _load_chat_events_by_task_id(child) is None
        assert _load_chat_events_by_task_id(grandchild) is None
        assert _load_chat_events_by_task_id(great) is None
        # The chat is now genuinely empty — the frontend's
        # ``chatHasMoreTasks`` flag must agree so the tab gets closed.
        assert _chat_has_tasks(chat_id) is False

    def test_other_trees_nested_subagents_survive(self) -> None:
        parent_a, chat_a = _add_task("parent A")
        parent_b, chat_b = _add_task("parent B")
        child_a = self._add_subagent(parent_a, chat_a, "child of A")
        self._add_subagent(child_a, chat_a, "grandchild of A")
        child_b = self._add_subagent(parent_b, chat_b, "child of B")
        grandchild_b = self._add_subagent(child_b, chat_b, "grandchild of B")

        assert _delete_task(parent_a) is True

        rows_b = _load_subagent_rows_by_parent_task_id(child_b)
        assert [r["task_id"] for r in rows_b] == [grandchild_b]
        assert _load_chat_events_by_task_id(child_b) is not None
        assert _chat_has_tasks(chat_b) is True
        assert _chat_has_tasks(chat_a) is False

    def test_self_referencing_subagent_row_terminates(self) -> None:
        # Defensive: a corrupt row whose subagent.parent_task_id points
        # at itself must not send the cascade into an infinite loop.
        parent_id, chat_id = _add_task("parent of weird row")
        weird, _ = _add_task(
            "self-referencing sub row",
            chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_id}},
        )
        # Corrupt it to point at itself: in the new flat-column schema
        # sub-agent parenthood lives in the ``parent_task_id`` column,
        # so the self-cycle is induced by pointing that column at the
        # row's own id.
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET parent_task_id = ? WHERE id = ?",
                (weird, weird),
            )
            db.commit()
        # Deleting the weird row itself must terminate and succeed.
        assert _delete_task(weird) is True
        assert _load_chat_events_by_task_id(weird) is None
        assert _load_chat_events_by_task_id(parent_id) is not None

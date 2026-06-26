# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: non-finite floats in ``extra`` corrupt the JSON column.

Every ``extra`` write (``_add_task``, ``_save_task_extra``,
``_set_task_favorite``) used plain ``json.dumps`` whose default
``allow_nan=True`` serialises ``float("nan")`` / ``float("inf")`` as the
bare tokens ``NaN`` / ``Infinity`` — NOT valid RFC 8259 JSON.  SQLite's
``json_valid`` (used by the ``_HISTORY_NOT_SUBAGENT`` predicate that
hides sub-agent rows from every history reader) rejects those tokens,
while Python's ``json.loads`` (used by ``_is_subagent_row``) accepts
them.  The two sub-agent detectors therefore DISAGREE for any sub-agent
row whose metrics contain a non-finite float (e.g. a NaN ``cost``):

* ``_load_history`` / ``_search_history`` / ``_prefix_match_task``
  (SQL side) treat the row as a REGULAR task and
  surface it in the history sidebar;
* ``_list_recent_chats``'s chat-selection query counts the row's chat
  against ``limit`` but the Python-side ``_is_subagent_row`` filter then
  drops every task — the slot is consumed and an older REAL chat
  silently disappears (resurrecting the iteration-4 limit-slot bug).

Fix: serialise ``extra`` through ``_dumps_extra`` which replaces
non-finite floats with ``None`` so the column always holds valid JSON,
and make ``_is_subagent_row`` (and the other Python-side sub-agent
parsers) reject NaN/Infinity via ``parse_constant`` so legacy corrupt
rows are classified identically by SQL and Python.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _is_subagent_row,
    _list_recent_chats,
    _load_history,
    _save_task_extra,
    _set_task_favorite,
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

    def _set_timestamp(self, task_id: int, ts: float) -> None:
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE id = ?",
                (ts, task_id),
            )
            db.commit()

    def _raw_extra(self, task_id: int) -> str:
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT extra FROM task_history WHERE id = ?", (task_id,),
            ).fetchone()
        return str(row["extra"] or "")

    def _insert_raw_extra(self, task: str, chat_id: str, extra_raw: str) -> int:
        """Insert a row with a verbatim ``extra`` string (legacy-row shim)."""
        db = th._get_db()
        with th._rw_lock.write_lock():
            cursor = db.execute(
                "INSERT INTO task_history "
                "(timestamp, task, chat_id, result, extra) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), task, chat_id, "done", extra_raw),
            )
            db.commit()
        row_id = cursor.lastrowid
        assert row_id is not None
        return int(row_id)


class TestExtraWritesStayValidJson(_TempDbTestBase):
    """Non-finite floats must never corrupt the ``extra`` JSON column."""

    def test_add_task_nan_cost_writes_valid_json(self) -> None:
        task_id, _ = _add_task(
            "task with nan cost",
            extra={"model": "m", "cost": float("nan")},
        )
        raw = self._raw_extra(task_id)
        parsed = json.loads(raw)  # must not need lenient NaN parsing
        assert parsed["model"] == "m"
        assert parsed["cost"] is None or (
            isinstance(parsed["cost"], float) and math.isfinite(parsed["cost"])
        )
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT json_valid(extra) AS ok FROM task_history WHERE id = ?",
                (task_id,),
            ).fetchone()
        assert row["ok"] == 1

    def test_save_task_extra_inf_tokens_writes_valid_json(self) -> None:
        task_id, _ = _add_task("task")
        _save_task_extra(
            {"tokens": float("inf"), "cost": float("-inf"), "steps": 3},
            task_id=task_id,
        )
        raw = self._raw_extra(task_id)
        parsed = json.loads(raw)
        assert parsed["steps"] == 3
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT json_valid(extra) AS ok FROM task_history WHERE id = ?",
                (task_id,),
            ).fetchone()
        assert row["ok"] == 1

    def test_set_favorite_preserves_validity_with_nested_nan(self) -> None:
        task_id, _ = _add_task(
            "task", extra={"metrics": {"cost": float("nan")}},
        )
        assert _set_task_favorite(task_id, True)
        raw = self._raw_extra(task_id)
        parsed = json.loads(raw)
        assert parsed["is_favorite"] is True
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT json_valid(extra) AS ok FROM task_history WHERE id = ?",
                (task_id,),
            ).fetchone()
        assert row["ok"] == 1


class TestSubagentNanExtraConsistency(_TempDbTestBase):
    """SQL and Python sub-agent detection must agree for NaN extras."""

    def test_subagent_row_with_nan_cost_stays_hidden_from_history(self) -> None:
        parent_id, chat = _add_task("parent task")
        _add_task(
            "fanned-out subtask",
            chat_id=chat,
            extra={
                "subagent": {"parent_task_id": parent_id},
                "cost": float("nan"),
            },
        )
        tasks = [e["task"] for e in _load_history()]
        assert "fanned-out subtask" not in tasks
        assert "parent task" in tasks

    def test_subagent_only_chat_with_nan_cost_does_not_eat_limit_slot(
        self,
    ) -> None:
        base = time.time()
        b_id, chat_b = _add_task("real task B")
        self._set_timestamp(b_id, base - 30)
        a_id, chat_a = _add_task("real task A")
        self._set_timestamp(a_id, base - 20)
        x_id, _chat_x = _add_task(
            "orphaned subagent task",
            extra={
                "subagent": {"parent_task_id": 999_999},
                "cost": float("nan"),
            },
        )
        self._set_timestamp(x_id, base - 10)

        chats = _list_recent_chats(limit=2)
        chat_ids = [c["chat_id"] for c in chats]
        assert chat_ids == [chat_a, chat_b]

    def test_legacy_nan_row_classified_identically_by_sql_and_python(
        self,
    ) -> None:
        # A legacy row written by the old code path: bare NaN token makes
        # the stored extra invalid RFC 8259 JSON.  SQLite's json_valid
        # rejects it, so _HISTORY_NOT_SUBAGENT shows the row; the
        # Python-side detector must agree (NOT a sub-agent row) instead
        # of silently dropping the row from _list_recent_chats /
        # _load_chat_context while the SQL side displays it.
        raw = '{"subagent": {"parent_task_id": 1}, "cost": NaN}'
        row_id = self._insert_raw_extra("legacy nan row", "legacychat", raw)

        db = th._get_db()
        with th._rw_lock.read_lock():
            sql_row = db.execute(
                "SELECT "
                + th._HISTORY_NOT_SUBAGENT
                + " AS not_sub FROM task_history WHERE id = ?",
                (row_id,),
            ).fetchone()
        sql_says_subagent = sql_row["not_sub"] == 0
        python_says_subagent = _is_subagent_row(raw)
        assert sql_says_subagent == python_says_subagent

        # And the chat must be visible end-to-end: the SQL side lists
        # the row, so the Python side must keep its chat too.
        chats = _list_recent_chats(limit=10)
        chat_ids = [c["chat_id"] for c in chats]
        assert "legacychat" in chat_ids

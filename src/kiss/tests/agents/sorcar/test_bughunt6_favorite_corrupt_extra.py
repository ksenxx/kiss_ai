# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: starring a legacy corrupt-extra row makes it vanish.

A legacy row written by the pre-iteration-5 code path can carry a bare
``NaN`` token in ``task_history.extra`` — invalid RFC 8259 JSON.  Since
iteration 5, BOTH sub-agent detectors (SQLite ``json_valid`` in
``_HISTORY_NOT_SUBAGENT`` and strict ``_parse_extra_dict`` in
``_is_subagent_row``) reject such extras, so the row is uniformly
classified as a REGULAR task and is visible in the history sidebar —
even when the corrupt extra textually contains a ``"subagent"`` key.

``_set_task_favorite`` however parsed the stored extra with the lenient
default ``json.loads`` (which ACCEPTS NaN), merged ``is_favorite`` into
the recovered dict — including the never-effective ``subagent`` key —
and re-encoded it through ``_dumps_extra`` as VALID JSON.  The rewrite
therefore flipped the row's classification to "sub-agent": clicking the
favourite star made the task permanently disappear from
``_load_history`` / ``_search_history`` / ``_list_recent_chats``.

Fix: when the stored extra is strict-invalid, ``_set_task_favorite``
still preserves the leniently-recovered metadata (the sidebar displays
it) but drops the top-level ``subagent`` key so the merge-rewrite can
never change the row's sub-agent classification.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
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
    _search_history,
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

    def _raw_extra(self, task_id: int) -> str:
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT extra FROM task_history WHERE id = ?", (task_id,),
            ).fetchone()
        return str(row["extra"] or "")


_LEGACY_NAN_SUBAGENT_EXTRA = (
    '{"subagent": {"parent_task_id": 1}, "cost": NaN, "model": "m"}'
)


class TestFavoriteDoesNotFlipClassification(_TempDbTestBase):
    """Starring a row must never change its sub-agent classification."""

    def test_star_legacy_nan_subagent_row_stays_visible(self) -> None:
        row_id = self._insert_raw_extra(
            "legacy nan row", "legacychat", _LEGACY_NAN_SUBAGENT_EXTRA,
        )
        # Pre-condition: the corrupt row is uniformly classified as a
        # regular task and shows up in the history sidebar.
        assert "legacy nan row" in [e["task"] for e in _load_history()]
        assert not _is_subagent_row(_LEGACY_NAN_SUBAGENT_EXTRA)

        assert _set_task_favorite(row_id, True)

        # The starred row must remain visible everywhere.
        assert "legacy nan row" in [e["task"] for e in _load_history()]
        assert "legacy nan row" in [
            e["task"] for e in _search_history("legacy nan")
        ]
        # And SQL/Python classification must still agree post-rewrite.
        assert not _is_subagent_row(self._raw_extra(row_id))

    def test_star_legacy_nan_subagent_row_keeps_chat_in_recent_chats(
        self,
    ) -> None:
        row_id = self._insert_raw_extra(
            "legacy nan row", "legacychat", _LEGACY_NAN_SUBAGENT_EXTRA,
        )
        assert _set_task_favorite(row_id, True)
        chats = _list_recent_chats(limit=10)
        assert "legacychat" in [c["chat_id"] for c in chats]

    def test_unstar_legacy_nan_subagent_row_stays_visible(self) -> None:
        row_id = self._insert_raw_extra(
            "legacy nan row", "legacychat", _LEGACY_NAN_SUBAGENT_EXTRA,
        )
        assert _set_task_favorite(row_id, False)
        entries = [e for e in _load_history() if e["task"] == "legacy nan row"]
        assert len(entries) == 1
        parsed = json.loads(self._raw_extra(row_id))
        assert parsed["is_favorite"] is False

    def test_star_legacy_nan_row_preserves_displayed_metadata(self) -> None:
        # The sidebar displays extra metadata via lenient json.loads, so
        # the favourite rewrite must keep the recovered fields (with the
        # non-finite values sanitised) — only the classification-flipping
        # "subagent" key may be dropped.
        row_id = self._insert_raw_extra(
            "legacy nan row", "legacychat", _LEGACY_NAN_SUBAGENT_EXTRA,
        )
        assert _set_task_favorite(row_id, True)
        raw = self._raw_extra(row_id)
        parsed = json.loads(raw)
        assert parsed["is_favorite"] is True
        assert parsed["model"] == "m"
        assert parsed["cost"] is None
        # The rewritten column must be strict-valid JSON.
        db = th._get_db()
        with th._rw_lock.read_lock():
            row = db.execute(
                "SELECT json_valid(extra) AS ok FROM task_history WHERE id = ?",
                (row_id,),
            ).fetchone()
        assert row["ok"] == 1

    def test_star_valid_subagent_row_stays_hidden(self) -> None:
        # Control: a REAL (valid-JSON) sub-agent row keeps its subagent
        # marker through a favourite toggle and stays hidden.
        parent_id, chat = _add_task("parent task")
        sub_id, _ = _add_task(
            "fanned-out subtask",
            chat_id=chat,
            extra={"subagent": {"parent_task_id": parent_id}},
        )
        assert _set_task_favorite(sub_id, True)
        assert "fanned-out subtask" not in [e["task"] for e in _load_history()]
        assert _is_subagent_row(self._raw_extra(sub_id))

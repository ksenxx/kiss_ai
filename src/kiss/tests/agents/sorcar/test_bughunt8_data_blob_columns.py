# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8: BLOB values in TEXT columns must not crash history readers.

``_row_to_extra_json`` copies the ``model`` / ``work_dir`` /
``version`` / ``parent_task_id`` column values into the synthesized
``extra`` JSON payload verbatim.  SQLite's dynamic typing lets a
hand-edited / 3rd-party-source DB store a BLOB in those TEXT columns;
``json.dumps(bytes)`` raises ``TypeError`` — caught neither by the
local ``(KeyError, IndexError)`` handler nor by ``_dumps_extra``'s
``ValueError`` handler — which propagated out of ``_load_history`` /
``_search_history`` / ``_load_chat_events_by_task_id`` and blanked
the whole history sidebar over one corrupt row (the exact threat
model of the earlier bughunt2 TEXT-in-REAL coercion fixes).

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _load_chat_events_by_task_id,
    _load_history,
    _search_history,
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

    def _corrupt_column(self, task_id: str, column: str, value: object) -> None:
        """Directly store *value* (e.g. a BLOB) into *column* of the row."""
        db = th._get_db()
        db.execute(
            f"UPDATE task_history SET {column} = ? WHERE id = ?",
            (value, task_id),
        )
        db.commit()


class TestBlobInTextColumns(_TempDbTestBase):
    """History readers must survive BLOBs in TEXT columns."""

    def test_load_history_survives_blob_model(self) -> None:
        """A BLOB ``model`` must not blank the whole history list."""
        tid, _ = _add_task("healthy task")
        bad_tid, _ = _add_task("corrupt task")
        self._corrupt_column(bad_tid, "model", b"\x00\x01\xff")

        entries = _load_history()

        assert {e["id"] for e in entries} == {tid, bad_tid}
        for e in entries:
            extra = e["extra"]
            assert isinstance(extra, str)
            if extra:
                parsed = json.loads(extra)
                assert isinstance(parsed.get("model", ""), str)

    def test_search_history_survives_blob_work_dir(self) -> None:
        """A BLOB ``work_dir`` must not abort substring search."""
        tid, _ = _add_task("findme alpha")
        self._corrupt_column(tid, "work_dir", b"\xde\xad\xbe\xef")

        entries = _search_history("findme")

        assert [e["id"] for e in entries] == [tid]
        extra = entries[0]["extra"]
        assert isinstance(extra, str)
        if extra:
            json.loads(extra)

    def test_load_by_task_id_survives_blob_parent_task_id(self) -> None:
        """A BLOB ``parent_task_id`` must not break task-id replay load."""
        tid, _ = _add_task("task with corrupt parent id")
        self._corrupt_column(tid, "parent_task_id", b"\x99\x88")

        session = _load_chat_events_by_task_id(tid)

        assert session is not None
        assert session["task"] == "task with corrupt parent id"
        extra = session["extra"]
        assert isinstance(extra, str)
        if extra:
            json.loads(extra)

    def test_load_history_survives_blob_version(self) -> None:
        """A BLOB ``version`` must not blank the whole history list."""
        tid, _ = _add_task("task with corrupt version")
        self._corrupt_column(tid, "version", b"\x01")

        entries = _load_history()

        assert [e["id"] for e in entries] == [tid]
        extra = entries[0]["extra"]
        assert isinstance(extra, str)
        if extra:
            json.loads(extra)

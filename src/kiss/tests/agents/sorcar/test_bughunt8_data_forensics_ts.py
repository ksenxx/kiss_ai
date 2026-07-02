# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8: orphan-recovery forensics must survive corrupt event rows.

``_log_orphaned_task_forensics`` formats the last few event timestamps
of every orphaned task with ``f"ts={ev_ts:.1f}"``.  The ``ev_ts``
value is read from the ``events.timestamp`` column *inside* the
``try`` block, but the ``:.1f`` format runs *outside* it.  SQLite's
dynamic typing lets a hand-edited / 3rd-party-source DB store TEXT or
BLOB in that REAL column; formatting a str/bytes with ``:.1f`` raises
``ValueError`` ("Unknown format code 'f'"), which propagates out of
``_recover_orphaned_tasks`` — crashing the whole server startup sweep
over one corrupt event row (the same threat model as the bughunt2
``_row_to_extra_json`` coercion fixes).

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import _add_task, _recover_orphaned_tasks


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

    def _insert_event_with_timestamp(self, task_id: str, ts: object) -> None:
        """Insert a raw event row with an arbitrary-typed timestamp."""
        db = th._get_db()
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (?, 0, ?, ?)",
            (task_id, json.dumps({"type": "task_start"}), ts),
        )
        db.commit()


class TestForensicsCorruptEventTimestamp(_TempDbTestBase):
    """_recover_orphaned_tasks must not crash on non-REAL timestamps."""

    def test_text_timestamp_does_not_crash_recovery(self) -> None:
        """A TEXT ``events.timestamp`` must not abort the orphan sweep."""
        tid, _ = _add_task("orphan task killed mid-run")
        self._insert_event_with_timestamp(tid, "garbage-not-a-number")

        recovered = _recover_orphaned_tasks(set())

        assert recovered == 1
        row = th._get_db().execute(
            "SELECT result FROM task_history WHERE id = ?", (tid,)
        ).fetchone()
        assert row["result"] == "Task terminated unexpectedly (process killed)"

    def test_blob_timestamp_does_not_crash_recovery(self) -> None:
        """A BLOB ``events.timestamp`` must not abort the orphan sweep."""
        tid, _ = _add_task("another orphan task")
        self._insert_event_with_timestamp(tid, b"\x00\x01\x02")

        recovered = _recover_orphaned_tasks(set())

        assert recovered == 1
        row = th._get_db().execute(
            "SELECT result FROM task_history WHERE id = ?", (tid,)
        ).fetchone()
        assert row["result"] == "Task terminated unexpectedly (process killed)"

    def test_healthy_rows_still_recovered_alongside_corrupt_one(self) -> None:
        """A corrupt row must not block recovery of other orphans."""
        good_tid, _ = _add_task("healthy orphan")
        self._insert_event_with_timestamp(good_tid, 1234.5)
        bad_tid, _ = _add_task("corrupt orphan")
        self._insert_event_with_timestamp(bad_tid, "NaN-ish text")

        recovered = _recover_orphaned_tasks(set())

        assert recovered == 2
        for tid in (good_tid, bad_tid):
            row = th._get_db().execute(
                "SELECT result FROM task_history WHERE id = ?", (tid,)
            ).fetchone()
            assert (
                row["result"]
                == "Task terminated unexpectedly (process killed)"
            )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: task-text lookups pick the WRONG row on timestamp ties.

``_most_recent_task_id(db, task)`` orders by bare ``timestamp DESC``
with no ``id`` tiebreak.  Its SQL runs through the ``idx_th_task``
index plus a temp B-tree sort, which (empirically, and per SQLite
docs, "unspecified") returns the LOWEST rowid first among equal
timestamps — i.e. the OLDEST run of the task.

Consequences when two runs of the same task text share a timestamp
value (coarse clock ticks, concurrent inserts, imported/restored
databases):

* ``_save_task_result(result, task=...)`` / ``_save_task_extra(...,
  task=...)`` (legacy task-text fallback through ``_resolve_task_id``
  → ``_most_recent_task_id``) UPDATE the older row, clobbering a
  finished task's result while the newest run keeps its
  "Agent Failed Abruptly" sentinel.

This is inconsistent with ``_load_latest_chat_events_by_chat_id`` and
``_get_adjacent_task_by_chat_id`` which already use the total order
``(timestamp, id)``.

Runs against a real SQLite database redirected to a temp dir; the tie
is seeded through the module's own connection (legitimate database
state — ``timestamp`` is plain REAL data).  No mocks, patches, fakes,
or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _most_recent_task_id,
    _save_task_result,
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

    def _force_all_timestamps(self, ts: float) -> None:
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute("UPDATE task_history SET timestamp = ?", (ts,))
            db.commit()
        th._invalidate_chat_context_cache("")


class TestTaskTextLookupTimestampTie(_TempDbTestBase):
    """(timestamp, id) total order for task-text-keyed lookups."""

    def test_most_recent_task_id_returns_latest_run(self) -> None:
        _t1, chat1 = _add_task("deploy app")
        t2, chat2 = _add_task("deploy app")  # new, separate session
        assert chat1 != chat2
        self._force_all_timestamps(1000.0)

        db = th._get_db()
        with th._rw_lock.read_lock():
            assert _most_recent_task_id(db, "deploy app") == t2

    def test_legacy_result_fallback_updates_latest_run(self) -> None:
        t1, _ = _add_task("build project")
        t2, _ = _add_task("build project")
        self._force_all_timestamps(1000.0)

        _save_task_result("done OK", task="build project")

        db = th._get_db()
        with th._rw_lock.read_lock():
            r1 = db.execute(
                "SELECT result FROM task_history WHERE id = ?", (t1,)
            ).fetchone()["result"]
            r2 = db.execute(
                "SELECT result FROM task_history WHERE id = ?", (t2,)
            ).fetchone()["result"]
        # The NEWEST run must receive the result; the older row keeps
        # whatever it had ("Agent Failed Abruptly" sentinel here).
        assert r2 == "done OK"
        assert r1 == "Agent Failed Abruptly"

    def test_distinct_timestamps_still_ordered_by_time(self) -> None:
        # Sanity: the id tiebreak must not override genuine timestamp
        # ordering (older row with a larger id).
        t1, _chat1 = _add_task("run tests")
        _t2, _chat2 = _add_task("run tests")
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET timestamp = 2000.0 WHERE id = ?",
                (t1,),
            )
            db.execute(
                "UPDATE task_history SET timestamp = 1000.0 WHERE id != ?",
                (t1,),
            )
            db.commit()
        th._invalidate_chat_context_cache("")

        with th._rw_lock.read_lock():
            assert _most_recent_task_id(db, "run tests") == t1

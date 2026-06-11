# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt: adjacent-task navigation breaks on timestamp ties.

``_load_latest_chat_events_by_chat_id`` orders rows by
``timestamp DESC, id DESC`` — the row id is the explicit tiebreaker
for equal timestamps.  ``_get_adjacent_task_by_chat_id`` however
compared bare timestamps with strict ``<`` / ``>``, so two tasks in
the same chat carrying the same ``timestamp`` value (concurrent
inserts from multiple viewers of one chat, restored/imported
databases, coarse clock ticks) became mutually unreachable via the
webview's prev/next scrolling — navigation silently returned ``None``
even though an adjacent task exists and the "latest task" loader can
see it.

The fix makes adjacency use the same total order
``(timestamp, id)`` that the latest-task loader already uses.

Runs against a real SQLite database redirected to a temp dir; the
tie is seeded through the module's own connection (a legitimate
database state — ``timestamp`` is plain ``REAL`` data).  No mocks,
patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _get_adjacent_task_by_chat_id,
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

    def _force_equal_timestamps(self, chat_id: str, ts: float) -> None:
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE chat_id = ?",
                (ts, chat_id),
            )
            db.commit()


class TestAdjacentNavigationWithTimestampTies(_TempDbTestBase):
    """(timestamp, id) ordering must make tied rows reachable."""

    def test_next_and_prev_traverse_equal_timestamps_by_id(self) -> None:
        t1, chat_id = _add_task("task one")
        t2, _ = _add_task("task two", chat_id=chat_id)
        t3, _ = _add_task("task three", chat_id=chat_id)
        self._force_equal_timestamps(chat_id, 1000.0)

        nxt = _get_adjacent_task_by_chat_id(chat_id, t1, "next")
        assert nxt is not None
        assert nxt["task_id"] == t2
        assert nxt["task"] == "task two"

        nxt2 = _get_adjacent_task_by_chat_id(chat_id, t2, "next")
        assert nxt2 is not None
        assert nxt2["task_id"] == t3

        prv = _get_adjacent_task_by_chat_id(chat_id, t3, "prev")
        assert prv is not None
        assert prv["task_id"] == t2

        prv2 = _get_adjacent_task_by_chat_id(chat_id, t2, "prev")
        assert prv2 is not None
        assert prv2["task_id"] == t1

        # Endpoints stay endpoints.
        assert _get_adjacent_task_by_chat_id(chat_id, t1, "prev") is None
        assert _get_adjacent_task_by_chat_id(chat_id, t3, "next") is None

    def test_distinct_timestamps_still_ordered_by_time(self) -> None:
        # Sanity: the id tiebreaker must not override genuine
        # timestamp ordering (older row with a larger id).
        t1, chat_id = _add_task("first by time")
        t2, _ = _add_task("second by time", chat_id=chat_id)
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE id = ?",
                (2000.0, t1),
            )
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE id = ?",
                (1000.0, t2),
            )
            db.commit()

        nxt = _get_adjacent_task_by_chat_id(chat_id, t2, "next")
        assert nxt is not None
        assert nxt["task_id"] == t1
        prv = _get_adjacent_task_by_chat_id(chat_id, t1, "prev")
        assert prv is not None
        assert prv["task_id"] == t2

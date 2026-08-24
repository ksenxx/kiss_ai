# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review fix — the seq-dedupe migration is atomic across processes.

``_init_tables_lock`` is process-LOCAL, and the connections run in
autocommit.  The old repair path (``_apply_index_ddl`` catching one
``IntegrityError``, running ``_dedupe_event_seqs`` as bare autocommit
UPDATEs, then retrying ``CREATE UNIQUE INDEX`` exactly once) therefore
had a cross-process hole: another process inserting one more duplicate
``(task_id, seq)`` row between the repair and the retry left the
database with duplicate rows AND without ``idx_ev_task_seq`` — the
migration raised and every later open kept failing the same way.

The fix (``_repair_and_create_index``) runs dedupe + index creation
inside ONE ``BEGIN IMMEDIATE`` transaction — SQLite's cross-process
write lock, which a peer's autocommit INSERT cannot interleave — and
retries the combined step a bounded number of times.

Tests use REAL child processes with their own SQLite connections
racing a REAL migration on a shared database file.  No mocks.

Branch-coverage notes for ``_repair_and_create_index``:

* the retry iterations (``attempt > 0``) and the final ``raise`` need
  the ``CREATE UNIQUE INDEX`` to fail INSIDE ``BEGIN IMMEDIATE`` right
  after ``_dedupe_event_seqs`` repaired every duplicate — the held
  write lock excludes the only actor (a concurrent writer) that could
  cause that, so those branches are unreachable without instrumenting
  the code under test; they are a bounded safety valve replacing the
  old unbounded failure mode, per policy documented here instead of
  mocked.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.tests.agents.sorcar.test_rr_area_e_persistence import (
    _redirect,
    _restore,
)


def _plant_legacy_duplicates(db_path: Path, task_id: str, dups: int) -> None:
    """Drop the unique index and insert *dups* duplicate-seq rows.

    Reproduces the state a pre-index database is in: rows whose
    ``(task_id, seq)`` collide, and no ``idx_ev_task_seq``.
    """
    raw = sqlite3.connect(db_path)
    try:
        raw.execute("DROP INDEX idx_ev_task_seq")
        for i in range(dups):
            raw.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) "
                "VALUES (?, 0, ?, ?)",
                (task_id, f'{{"type": "legacy-dup", "i": {i}}}', 2.0 + i),
            )
        raw.commit()
    finally:
        raw.close()


def _duplicate_hammer(
    db_path: str, task_id: str, stop_file: str, report_file: str,
) -> None:
    """Child process: insert duplicate ``(task_id, seq)`` rows in a loop.

    A real concurrent old-version writer: its own connection, own
    autocommit INSERTs, no knowledge of the migration.  Runs until the
    parent creates *stop_file*.  Once ``idx_ev_task_seq`` exists the
    inserts start failing with ``IntegrityError`` — counted, expected,
    and proof the index actually blocks duplicates cross-process.
    """
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=30000")
    inserted = 0
    refused = 0
    try:
        while not Path(stop_file).exists():
            try:
                conn.execute(
                    "INSERT INTO events "
                    "(task_id, seq, event_json, timestamp) "
                    "VALUES (?, 0, ?, ?)",
                    (task_id, '{"type": "hammer"}', time.time()),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                refused += 1
                if Path(stop_file).exists():  # pragma: no cover — timing
                    break
                time.sleep(0.001)
            except sqlite3.OperationalError:  # pragma: no cover — busy
                time.sleep(0.001)
    finally:
        conn.close()
        Path(report_file).write_text(f"{inserted} {refused}")


def _txn_holding_writer(
    db_path: str, task_id: str, holding_file: str, release_file: str,
) -> None:
    """Child process: hold ``BEGIN IMMEDIATE`` with an uncommitted dup.

    Signals via *holding_file* once the write lock is held, commits
    when *release_file* appears.  The parent's migration must WAIT for
    this commit and then still repair the row it just committed.
    """
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) "
            "VALUES (?, 0, '{\"type\": \"late-dup\"}', 9.0)",
            (task_id,),
        )
        Path(holding_file).write_text("holding")
        deadline = time.monotonic() + 60
        while not Path(release_file).exists():
            if time.monotonic() >= deadline:  # pragma: no cover — hang guard
                break
            time.sleep(0.01)
        conn.execute("COMMIT")
    finally:
        conn.close()


class _MigrationRaceBase(unittest.TestCase):
    """Private database + a legacy duplicate-seq state per test."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_rr_review_mig_"))
        self.saved = _redirect(self.tmp)
        th._close_db()
        self.task_id, _chat = th._add_task("legacy race task")
        for i in range(4):
            th._queue_chat_event({"type": "ev", "i": i}, self.task_id)
        th._flush_chat_events(self.task_id)
        th._close_db()
        _plant_legacy_duplicates(th._DB_PATH, self.task_id, dups=3)

    def tearDown(self) -> None:
        _restore(self.saved)

    def _assert_repaired(self) -> None:
        """Index exists and every task's seqs are duplicate-free."""
        db = sqlite3.connect(th._DB_PATH)
        try:
            index_sql = db.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND name='idx_ev_task_seq'"
            ).fetchone()
            dup_pairs = db.execute(
                "SELECT task_id, seq, COUNT(*) FROM events "
                "GROUP BY task_id, seq HAVING COUNT(*) > 1"
            ).fetchall()
        finally:
            db.close()
        assert index_sql is not None, "idx_ev_task_seq was not created"
        self.assertIn("UNIQUE", index_sql[0].upper())
        self.assertEqual(dup_pairs, [])


class TestMigrationVsConcurrentDuplicateWriter(_MigrationRaceBase):
    """Reviewer's repro: a peer keeps inserting dups mid-migration."""

    def test_index_and_uniqueness_survive_racing_inserts(self) -> None:
        ctx = multiprocessing.get_context()
        stop_file = self.tmp / "stop"
        report_file = self.tmp / "report"
        proc = ctx.Process(
            target=_duplicate_hammer,
            args=(
                str(th._DB_PATH), self.task_id,
                str(stop_file), str(report_file),
            ),
        )
        proc.start()
        try:
            # Let the hammer land some pre-migration duplicates so the
            # migration demonstrably races live inserts.
            time.sleep(0.3)
            th._get_db()  # runs _init_tables → repair + index, atomically
        finally:
            stop_file.write_text("stop")
            proc.join(timeout=60)
        self.assertEqual(proc.exitcode, 0)
        self._assert_repaired()
        inserted, refused = map(
            int, report_file.read_text().split(),
        )
        # The writer really ran concurrently and really got refused by
        # the index afterwards (it keeps trying until told to stop).
        self.assertGreater(inserted, 0)
        self.assertGreater(refused, 0)
        # Every row the hammer landed BEFORE the index went up was
        # preserved by the resequencing repair (dedupe deletes
        # nothing): original 4 + 3 planted dups + inserted.
        db = sqlite3.connect(th._DB_PATH)
        try:
            (count,) = db.execute(
                "SELECT COUNT(*) FROM events WHERE task_id = ?",
                (self.task_id,),
            ).fetchone()
        finally:
            db.close()
        self.assertEqual(count, 7 + inserted)

    def test_repaired_database_replays_and_accepts_new_events(self) -> None:
        # End-to-end: after the racy repair the persistence layer is
        # fully usable — replay sees every preserved event in order
        # and new writes get fresh unique seqs.
        th._get_db()
        self._assert_repaired()
        th._queue_chat_event({"type": "after"}, self.task_id)
        th._flush_chat_events(self.task_id)
        session = th._load_chat_events_by_task_id(self.task_id)
        assert session is not None
        events = session["events"]
        assert isinstance(events, list)
        self.assertEqual(len(events), 8)
        self.assertEqual(events[-1]["type"], "after")


class TestMigrationWaitsForPeerWriteTransaction(_MigrationRaceBase):
    """Deterministic lock hand-off: dup committed just before repair."""

    def test_migration_repairs_duplicate_committed_under_peer_lock(
        self,
    ) -> None:
        ctx = multiprocessing.get_context()
        holding_file = self.tmp / "holding"
        release_file = self.tmp / "release"
        proc = ctx.Process(
            target=_txn_holding_writer,
            args=(
                str(th._DB_PATH), self.task_id,
                str(holding_file), str(release_file),
            ),
        )
        proc.start()
        try:
            deadline = time.monotonic() + 60
            while not holding_file.exists():
                self.assertLess(
                    time.monotonic(), deadline, "writer never took the lock",
                )
                time.sleep(0.01)
            # The peer holds BEGIN IMMEDIATE with an uncommitted
            # duplicate.  Release it just before migrating: its commit
            # lands a fresh duplicate the migration must both wait for
            # and repair inside its own write transaction.
            release_file.write_text("go")
            th._get_db()
        finally:
            release_file.write_text("go")
            proc.join(timeout=60)
        self.assertEqual(proc.exitcode, 0)
        self._assert_repaired()


class TestRepairJoinsCallersTransaction(_MigrationRaceBase):
    """``conn.in_transaction`` branch: the legacy-migration path."""

    def test_repair_inside_existing_begin_immediate(self) -> None:
        # _migrate_old_schema_if_needed calls _apply_index_ddl inside
        # its own BEGIN IMMEDIATE; the repair must join that
        # transaction instead of nesting a second BEGIN (which sqlite
        # would refuse).
        conn = sqlite3.connect(
            th._DB_PATH, timeout=30, isolation_level=None,
        )
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("BEGIN IMMEDIATE")
            th._apply_index_ddl(conn)
            self.assertTrue(conn.in_transaction)
            conn.execute("COMMIT")
        finally:
            conn.close()
        self._assert_repaired()


if __name__ == "__main__":
    unittest.main()

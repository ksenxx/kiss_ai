# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regressions for ``sorcar/persistence.py``.

Covers audit findings:

* **R1** — the cross-process orphan sweep rewrote the result of a task
  that was still RUNNING in another process, and the rewrite was sticky
  so the real outcome could never be recorded afterwards.
* **R2** — multi-statement read-modify-write sequences ran as
  independent autocommit statements guarded only by an in-process lock,
  so the cap of a capped table could be exceeded permanently (shown on
  ``steer_inputs``; the ``frequent_tasks`` writer it was found in is
  gone).
* **R3** — permanently-unwritable events were journalled to a
  write-only sidecar and then acknowledged as flushed, so the real
  transcript was lost.
* **I1** — schema booleans defaulted to 0 while every other layer
  defaults to True, and the migration mapped a MISSING key to 0.
* **I4** — ``_update_task_columns`` interpolates the column names into
  SQL but accepted any string.
* **D2** — one thread's ``_close_thread_db`` invalidated every OTHER
  thread's per-thread connection.
* **D3** — ``_flush_chat_events`` busy-spun at 500 Hz and waited for
  every task's events, not just the caller's.

Every test uses a REAL temporary SQLite database, REAL threads and REAL
child processes.  No mocks, patches or doubles, and no paid LLM calls.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.tests.conftest import posix_only
from kiss.tests.persistence_helpers import close_db

_RACE_DELAY_ENV = "KISS_RACE_DELAY"


def _redirect(tmpdir: Path) -> tuple:
    """Point the persistence module at a private KISS home."""
    saved = (th._DB_PATH, th._KISS_DIR, th._owner_state)
    kiss_dir = tmpdir / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._owner_state = None
    return saved


def _restore(saved: tuple) -> None:
    """Undo :func:`_redirect`."""
    close_db()
    (th._DB_PATH, th._KISS_DIR, th._owner_state) = saved


def _live_task_worker(kiss_dir: str, out_queue) -> None:
    """Child process: insert a task row and stay alive holding it.

    The parent always ends this process with ``kill()``, so the child
    must NOT block on a ``multiprocessing`` primitive: SIGKILL while
    holding one of those shared semaphores wedges the parent too.
    """
    import kiss.agents.sorcar.persistence as child_th

    child_th._KISS_DIR = Path(kiss_dir)
    child_th._DB_PATH = child_th._KISS_DIR / "history.db"
    child_th._owner_state = None
    task_id, _chat = child_th._add_task("live task in another process")
    out_queue.put(task_id)
    time.sleep(300)


def _clean_exit_owner_worker(kiss_dir: str, out_queue) -> None:
    """Child process: publish a liveness marker, then exit normally.

    Models the common case the marker lifecycle has to survive: a
    daemon or CLI run that finishes every task it owns and terminates
    without anyone ever having to sweep its rows.
    """
    import kiss.agents.sorcar.persistence as child_th

    child_th._KISS_DIR = Path(kiss_dir)
    child_th._DB_PATH = child_th._KISS_DIR / "history.db"
    child_th._owner_state = None
    out_queue.put(child_th._process_owner_token())


def _steer_input_worker(kiss_dir: str, text: str, barrier) -> None:
    """Child process: record one brand-new steer input at the barrier."""
    import kiss.agents.sorcar.persistence as child_th

    child_th._KISS_DIR = Path(kiss_dir)
    child_th._DB_PATH = child_th._KISS_DIR / "history.db"
    child_th._owner_state = None
    barrier.wait()
    child_th._record_steer_input(text)


class _PersistenceTestCase(unittest.TestCase):
    """Base fixture giving each test a private database."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_e_db_"))
        self.saved = _redirect(self.tmp)
        close_db()

    def tearDown(self) -> None:
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    @property
    def kiss_dir(self) -> Path:
        """The private KISS home for this test."""
        return self.tmp / ".kiss"

    def _result_of(self, task_id: str) -> str:
        """Return the ``result`` column of *task_id*."""
        row = th._get_db().execute(
            "SELECT result FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        assert row is not None
        return str(row["result"])


class OrphanSweepLivenessTest(_PersistenceTestCase):
    """R1: a task running in ANOTHER process is not an orphan."""

    def setUp(self) -> None:
        super().setUp()
        self.ctx = multiprocessing.get_context("spawn")
        self.queue = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=_live_task_worker,
            args=(str(self.kiss_dir), self.queue),
        )
        self.proc.start()
        self.task_id = self.queue.get(timeout=120)

    def tearDown(self) -> None:
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(timeout=30)
        self.queue.close()
        self.queue.cancel_join_thread()
        super().tearDown()

    def test_live_task_of_another_process_is_not_mislabelled(self) -> None:
        """The sweep must leave a still-running foreign task alone."""
        rewritten = th._recover_orphaned_tasks(set(), created_before=time.time())

        self.assertEqual(rewritten, 0)
        self.assertEqual(self._result_of(self.task_id), "Agent Failed Abruptly")

    def test_the_sentinel_survives_for_a_later_genuine_kill(self) -> None:
        """After a real kill the same row IS recovered — not sticky."""
        th._recover_orphaned_tasks(set(), created_before=time.time())
        self.assertEqual(self._result_of(self.task_id), "Agent Failed Abruptly")

        self.proc.kill()
        self.proc.join(timeout=30)

        rewritten = th._recover_orphaned_tasks(set(), created_before=time.time())
        self.assertEqual(rewritten, 1)
        self.assertEqual(
            self._result_of(self.task_id),
            "Task terminated unexpectedly (process killed)",
        )

    def test_owner_marker_is_removed_once_the_owner_dies(self) -> None:
        """A dead process's liveness marker is not leaked forever."""
        markers = th._owner_dir()
        own_token = th._process_owner_token()
        self.assertEqual(len(list(markers.iterdir())), 2)
        self.proc.kill()
        self.proc.join(timeout=30)

        th._recover_orphaned_tasks(set(), created_before=time.time())

        self.assertEqual(
            [p.name for p in markers.iterdir()], [f"{own_token}.lock"],
        )

    def test_this_process_own_rows_are_never_swept(self) -> None:
        """A row this process created is alive by definition."""
        mine, _chat = th._add_task("local live task")
        th._recover_orphaned_tasks(set(), created_before=time.time() + 1)
        self.assertEqual(self._result_of(mine), "Agent Failed Abruptly")

    def test_legacy_rows_without_an_owner_are_still_recovered(self) -> None:
        """Rows predating owner tracking keep the old behaviour."""
        db = th._get_db()
        legacy = uuid.uuid4().hex
        db.execute(
            "INSERT INTO task_history (id, timestamp, task, result, owner) "
            "VALUES (?, ?, ?, ?, '')",
            (legacy, time.time() - 60, "legacy", "Agent Failed Abruptly"),
        )

        th._recover_orphaned_tasks(set(), created_before=time.time())

        self.assertEqual(
            self._result_of(legacy),
            "Task terminated unexpectedly (process killed)",
        )
        self.assertEqual(self._result_of(self.task_id), "Agent Failed Abruptly")


class SteerInputCapTest(_PersistenceTestCase):
    """R2: the cap of a capped table (``steer_inputs``) must hold across processes."""

    def _seed(self, count: int) -> None:
        """Insert *count* distinct steer-input rows, oldest first.

        The timestamps increase strictly, so eviction (oldest first)
        always targets ``seed-0`` and never a row the test just added.
        """
        now = time.time()
        th._get_db().executemany(
            "INSERT INTO steer_inputs (text, timestamp) VALUES (?, ?)",
            [(f"seed-{i}", now - count + i) for i in range(count)],
        )

    def _texts(self) -> set[str]:
        """Return the texts in ``steer_inputs``."""
        return {r["text"] for r in th._get_db().execute("SELECT text FROM steer_inputs")}

    def test_two_processes_cannot_exceed_the_cap(self) -> None:
        """Concurrent inserts at the cap boundary stay within the cap."""
        self._seed(th._MAX_STEER_INPUTS - 1)
        close_db()
        saved_delay = os.environ.get(_RACE_DELAY_ENV)
        os.environ[_RACE_DELAY_ENV] = "0.05"
        try:
            ctx = multiprocessing.get_context("spawn")
            barrier = ctx.Barrier(2)
            procs = [
                ctx.Process(
                    target=_steer_input_worker,
                    args=(str(self.kiss_dir), f"racer-{i}", barrier),
                )
                for i in range(2)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=120)
                self.assertEqual(p.exitcode, 0)
        finally:
            if saved_delay is None:
                os.environ.pop(_RACE_DELAY_ENV, None)
            else:
                os.environ[_RACE_DELAY_ENV] = saved_delay

        texts = self._texts()
        self.assertLessEqual(len(texts), th._MAX_STEER_INPUTS)
        self.assertLessEqual({"racer-0", "racer-1"}, texts)

    def test_serial_inserts_evict_the_oldest_row(self) -> None:
        """Eviction keeps the table at the cap and drops the oldest row; a repeat evicts none."""
        self._seed(th._MAX_STEER_INPUTS)
        th._record_steer_input("seed-3")
        texts = self._texts()
        self.assertEqual(len(texts), th._MAX_STEER_INPUTS)
        self.assertIn("seed-0", texts)
        th._record_steer_input("brand new")
        th._record_steer_input("  ")
        texts = self._texts()
        self.assertEqual(len(texts), th._MAX_STEER_INPUTS)
        self.assertIn("brand new", texts)
        self.assertNotIn("seed-0", texts)
        self.assertNotIn("  ", texts)


class FailedEventReplayTest(_PersistenceTestCase):
    """R3: journalled events must be recoverable, not just inspectable."""

    def _block_event_writes(self) -> None:
        """Make every INSERT into ``events`` fail at the schema level."""
        th._get_db().execute(
            "CREATE TRIGGER reject_events BEFORE INSERT ON events "
            "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
        )

    def _unblock_event_writes(self) -> None:
        """Allow event inserts again."""
        th._get_db().execute("DROP TRIGGER reject_events")

    def _event_count(self, task_id: str) -> int:
        """Return the number of persisted events for *task_id*."""
        row = th._get_db().execute(
            "SELECT COUNT(*) AS n FROM events WHERE task_id = ?", (task_id,),
        ).fetchone()
        return int(row["n"])

    def _journal_line_count(self) -> int:
        """Total pending journal rows: live sidecar plus claimed snapshots.

        A snapshot whose replay failed stays under its claimed
        ``.consumed-*`` name (restoring it to the live name would let
        later appends mix into it and break replay chronology), so the
        no-loss invariant is counted across both.
        """
        sidecar = Path(th._failed_events_path(th._current_db_path()))
        total = 0
        for path in sidecar.parent.glob(sidecar.name + "*"):
            if path.suffix == ".lock":
                continue
            total += len(path.read_text(encoding="utf-8").splitlines())
        return total

    def test_events_are_replayed_once_the_database_recovers(self) -> None:
        """A transient write outage must not lose the transcript."""
        task_id, _chat = th._add_task("replay target")
        self._block_event_writes()
        for i in range(10):
            th._queue_chat_event({"type": "text", "content": f"e{i}"}, task_id)
        th._flush_chat_events(task_id)

        path = th._failed_events_path(th._current_db_path())
        self.assertTrue(th._journal_has_pending_rows(path))
        self.assertEqual(self._journal_line_count(), 10)
        self.assertEqual(self._event_count(task_id), 0)

        self._unblock_event_writes()
        th._flush_chat_events(task_id)

        self.assertTrue(th._task_has_transcript_events(task_id))
        self.assertEqual(self._event_count(task_id), 10)
        self.assertEqual(self._journal_line_count(), 0)

    def test_journal_stays_when_the_database_is_still_unwritable(self) -> None:
        """A failing replay keeps every journalled row for the next attempt."""
        task_id, _chat = th._add_task("still broken")
        self._block_event_writes()
        th._queue_chat_event({"type": "text", "content": "x"}, task_id)
        th._flush_chat_events(task_id)
        path = th._failed_events_path(th._current_db_path())
        self.assertTrue(th._journal_has_pending_rows(path))

        th._replay_failed_events()

        self.assertTrue(th._journal_has_pending_rows(path))
        self.assertEqual(self._journal_line_count(), 1)
        self._unblock_event_writes()

        th._replay_failed_events()
        self.assertEqual(self._event_count(task_id), 1)
        self.assertEqual(self._journal_line_count(), 0)

    def test_journal_is_written_next_to_its_origin_database(self) -> None:
        """Rows are journalled beside the DB they were produced against."""
        foreign = str(self.kiss_dir / "other.db")
        th._journal_failed_events(
            [("abc", json.dumps({"type": "t"}), 1.0, foreign)], 4,
        )
        self.assertTrue(Path(th._failed_events_path(foreign)).is_file())
        self.assertFalse(
            Path(th._failed_events_path(th._current_db_path())).exists()
        )

    def test_malformed_journal_lines_are_skipped(self) -> None:
        """A corrupt journal never blocks recovery of the good lines.

        Nor does a damaged block that is not even UTF-8, or a line
        nested deeper than the JSON parser recurses: the replay raised
        on them from every flush, and the event writer died on them
        with its batch still queued, so a flush never returned.
        """
        task_id, _chat = th._add_task("partial journal")
        sidecar = Path(th._failed_events_path(th._current_db_path()))
        too_deep = b"[" * 100_000 + b"\n"
        sidecar.write_bytes(
            b"not json\n\xff not text either\n" + too_deep
            + json.dumps({
                "task_id": task_id,
                "event_json": json.dumps({"type": "result"}),
                "timestamp": time.time(),
                "origin_db_path": th._current_db_path(),
            }).encode() + b"\n"
        )

        th._flush_chat_events()

        self.assertEqual(self._event_count(task_id), 1)
        self.assertFalse(sidecar.exists())

        # The event writer replays the journal ahead of its own batch.
        sidecar.write_bytes(too_deep)
        th._queue_chat_event({"type": "text"}, task_id)
        th._flush_chat_events(task_id)

        self.assertEqual(self._event_count(task_id), 2)
        self.assertFalse(sidecar.exists())

    @posix_only("renaming a SQLite file another connection holds open")
    def test_reconnect_failure_does_not_cache_a_closed_connection(self) -> None:
        """A temporarily unopenable database recovers on the next call."""
        th._get_db()
        db_path = Path(th._current_db_path())
        stash = self.kiss_dir / "stashed.db"
        db_path.rename(stash)
        db_path.mkdir()
        with self.assertRaises(sqlite3.Error):
            th._get_db()

        db_path.rmdir()
        stash.rename(db_path)

        db = th._get_db()
        self.assertIsNotNone(
            db.execute("SELECT 1").fetchone(),
        )


class SchemaDefaultsTest(_PersistenceTestCase):
    """I1: absent toggle values must read as the framework default."""

    def _extra_of(self, entry: dict) -> dict:
        """Decode the synthesized ``extra`` JSON of a history entry."""
        decoded: dict = json.loads(str(entry["extra"]))
        return decoded

    def test_a_row_inserted_without_toggles_reads_as_enabled(self) -> None:
        """The DDL default matches the producer's own default."""
        db = th._get_db()
        db.execute(
            "INSERT INTO task_history (id, timestamp, task) VALUES (?, ?, ?)",
            (uuid.uuid4().hex, time.time(), "bare insert"),
        )
        extra = self._extra_of(th._load_history()[0])
        self.assertTrue(extra["is_parallel"])

    def test_migration_keeps_unrecorded_toggles_enabled(self) -> None:
        """A legacy row that never recorded the toggles is not inverted."""
        close_db()
        db_path = self.kiss_dir / "history.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.executescript("""
            CREATE TABLE task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                task TEXT NOT NULL,
                has_events INTEGER DEFAULT 0,
                result TEXT DEFAULT '',
                chat_id CHAR(32) DEFAULT '',
                extra TEXT DEFAULT ''
            );
        """)
        conn.execute(
            "INSERT INTO task_history (timestamp, task, extra) "
            "VALUES (?, ?, ?)",
            (time.time(), "legacy task", json.dumps({"model": "m"})),
        )
        conn.close()

        entries = th._load_history()

        self.assertEqual(len(entries), 1)
        extra = self._extra_of(entries[0])
        self.assertTrue(extra["is_parallel"])
        self.assertEqual(extra["model"], "m")

    def test_migration_preserves_explicitly_disabled_toggles(self) -> None:
        """An explicit ``false`` is still migrated as disabled."""
        close_db()
        db_path = self.kiss_dir / "history.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.executescript("""
            CREATE TABLE task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                task TEXT NOT NULL,
                has_events INTEGER DEFAULT 0,
                result TEXT DEFAULT '',
                chat_id CHAR(32) DEFAULT '',
                extra TEXT DEFAULT ''
            );
        """)
        conn.execute(
            "INSERT INTO task_history (timestamp, task, extra) "
            "VALUES (?, ?, ?)",
            (time.time(), "explicit off", json.dumps({"is_parallel": False})),
        )
        conn.close()

        extra = self._extra_of(th._load_history()[0])

        self.assertFalse(extra["is_parallel"])

    def test_add_task_records_what_the_caller_declared(self) -> None:
        """An explicit ``False`` from the caller is still stored as off."""
        task_id, _chat = th._add_task("explicit", extra={"is_parallel": False})
        row = th._get_db().execute(
            "SELECT is_parallel FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        self.assertEqual(int(row["is_parallel"]), 0)


class UpdateTaskColumnGuardTest(_PersistenceTestCase):
    """I4: only allow-listed columns may be interpolated into SQL."""

    def test_a_non_allowlisted_column_raises(self) -> None:
        """The long-removed ``extra`` column is rejected, not written."""
        task_id, _chat = th._add_task("guard me")
        with self.assertRaises(ValueError):
            th._update_task_columns({"result": "ok", "extra": "{}"}, task_id)
        self.assertEqual(self._result_of(task_id), "Agent Failed Abruptly")

    def test_result_still_round_trips(self) -> None:
        """The allowed columns keep working, with and without a verdict."""
        task_id, _chat = th._add_task("guard me too")
        th._save_task_result("Task completed", task_id=task_id)
        self.assertEqual(self._result_of(task_id), "Task completed")
        self.assertIsNone(self._success_of(task_id))
        th._save_task_result("Could not do it", task_id=task_id, success=False)
        self.assertEqual(self._result_of(task_id), "Could not do it")
        self.assertEqual(self._success_of(task_id), 0)
        th._save_task_result("Done after all", task_id=task_id, success=True)
        self.assertEqual(self._success_of(task_id), 1)

    def _success_of(self, task_id: str) -> object:
        row = th._get_db().execute(
            "SELECT success FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        return row["success"]


class ThreadConnectionCacheTest(_PersistenceTestCase):
    """D2: one thread closing its connection must not disturb others."""

    def test_workers_keep_one_connection_each(self) -> None:
        """No worker is forced to reopen when another thread closes."""
        ready = threading.Barrier(5)
        stop = threading.Event()
        seen: dict[int, list[sqlite3.Connection]] = {}
        lock = threading.Lock()

        def worker(index: int) -> None:
            conns: list[sqlite3.Connection] = [th._get_db()]
            with lock:
                seen[index] = conns
            ready.wait(timeout=30)
            while not stop.is_set():
                conn = th._get_db()
                if conn is not conns[-1]:
                    conns.append(conn)
                time.sleep(0.001)
            th._close_thread_db()

        workers = [
            threading.Thread(target=worker, args=(i,), name=f"w{i}")
            for i in range(4)
        ]
        for w in workers:
            w.start()

        # Connect LAST so this thread's handle is the process's newest
        # one — that is the case ``_close_thread_db`` used to propagate
        # to every other thread.
        ready.wait(timeout=30)
        th._get_db()
        th._close_thread_db()
        time.sleep(0.3)
        stop.set()
        for w in workers:
            w.join(timeout=30)

        for index, conns in seen.items():
            self.assertEqual(
                len(conns), 1,
                f"worker {index} reopened its connection {len(conns)} times",
            )


class FlushScopeTest(_PersistenceTestCase):
    """D3: a flush must not wait for an unrelated task's backlog."""

    def test_flush_of_an_idle_task_is_not_blocked_by_a_backlog(self) -> None:
        """A quiet task's flush is not delayed by another task's queue."""
        busy_id, _c1 = th._add_task("busy task")
        idle_id, _c2 = th._add_task("idle task")
        holding = threading.Event()
        release = threading.Event()

        def hold_writer() -> None:
            """Stall the event writer by owning the DB write lock."""
            with th._rw_lock.write_lock():
                holding.set()
                release.wait(timeout=30)

        holder = threading.Thread(target=hold_writer, name="db-write-holder")
        holder.start()
        self.assertTrue(holding.wait(timeout=10))
        for i in range(100):
            th._queue_chat_event(
                {"type": "text", "content": f"chunk {i}"}, busy_id,
            )
        time.sleep(0.2)
        self.assertGreater(th._pending_count(busy_id), 0)

        started = time.monotonic()
        th._flush_chat_events(idle_id)
        elapsed = time.monotonic() - started

        release.set()
        holder.join(timeout=30)
        self.assertLess(elapsed, 0.3)

        th._flush_chat_events()
        self.assertEqual(th._pending_count(busy_id), 0)

    def test_flush_restarts_a_dead_writer_thread(self) -> None:
        """Pending events are drained even if the writer already exited."""
        task_id, _chat = th._add_task("dead writer")
        th._stop_event_writer()
        th._event_deque.append((
            task_id,
            json.dumps({"type": "result", "content": "late"}),
            time.time(),
            th._current_db_path(),
        ))

        th._flush_chat_events(task_id)

        self.assertEqual(th._pending_count(task_id), 0)
        row = th._get_db().execute(
            "SELECT COUNT(*) AS n FROM events WHERE task_id = ?", (task_id,),
        ).fetchone()
        self.assertEqual(int(row["n"]), 1)

    def test_unscoped_flush_waits_for_everything(self) -> None:
        """The default flush still guarantees a full drain."""
        task_id, _chat = th._add_task("drain me")
        for i in range(500):
            th._queue_chat_event({"type": "text", "content": str(i)}, task_id)
        th._flush_chat_events()
        self.assertEqual(th._pending_count(None), 0)
        row = th._get_db().execute(
            "SELECT COUNT(*) AS n FROM events WHERE task_id = ?", (task_id,),
        ).fetchone()
        self.assertEqual(int(row["n"]), 500)


class ModuleInternalsTest(_PersistenceTestCase):
    """Edge paths of the task-ownership and transaction helpers."""

    def test_race_delay_is_a_no_op_without_the_variable(self) -> None:
        """The test hook costs nothing in production."""
        os.environ.pop(_RACE_DELAY_ENV, None)
        started = time.monotonic()
        th._race_delay()
        self.assertLess(time.monotonic() - started, 0.01)

    def test_race_delay_ignores_a_malformed_value(self) -> None:
        """A bad value must not raise into a database write."""
        saved = os.environ.get(_RACE_DELAY_ENV)
        os.environ[_RACE_DELAY_ENV] = "soon"
        try:
            started = time.monotonic()
            th._race_delay()
            self.assertLess(time.monotonic() - started, 0.01)
        finally:
            if saved is None:
                os.environ.pop(_RACE_DELAY_ENV, None)
            else:
                os.environ[_RACE_DELAY_ENV] = saved

    def test_immediate_txn_rolls_back_on_error(self) -> None:
        """A failure inside the block undoes every statement in it."""
        db = th._get_db()
        task_id = uuid.uuid4().hex
        with self.assertRaises(RuntimeError), th._immediate_txn(db):
            db.execute(
                "INSERT INTO task_history (id, timestamp, task) "
                "VALUES (?, ?, ?)",
                (task_id, time.time(), "rolled back"),
            )
            raise RuntimeError("boom")
        row = db.execute(
            "SELECT 1 FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        self.assertIsNone(row)

    def test_owner_token_is_reminted_when_the_home_moves(self) -> None:
        """A redirected KISS home gets its own liveness marker."""
        first = th._process_owner_token()
        first_marker = th._owner_dir() / f"{first}.lock"
        self.assertTrue(first_marker.is_file())

        moved = self.tmp / "moved"
        moved.mkdir()
        th._KISS_DIR = moved
        second = th._process_owner_token()

        self.assertNotEqual(first, second)
        self.assertTrue((moved / th._OWNER_DIR_NAME / f"{second}.lock").is_file())
        self.assertFalse(
            first_marker.exists(),
            "the abandoned home kept a marker no process will ever "
            "unlink, so it accumulates one file per redirect",
        )

    def test_a_stop_before_the_token_is_kept_leaves_no_marker(self) -> None:
        """A Stop landing once the marker is written, before its token is kept, deletes it.

        No row names such a token, so no sweep would ever find the file.
        """
        def stop_after_flush(frame, event, arg):  # type: ignore[no-untyped-def]
            if (
                event == "c_return" and getattr(arg, "__name__", "") == "flush"
                and frame.f_code.co_name == "_process_owner_token"
            ):
                raise KeyboardInterrupt

        sys.setprofile(stop_after_flush)
        try:
            with self.assertRaises(KeyboardInterrupt):
                th._process_owner_token()
        finally:
            sys.setprofile(None)
        self.assertEqual(list(th._owner_dir().iterdir()), [])
        token = th._process_owner_token()
        self.assertEqual([p.name for p in th._owner_dir().iterdir()], [f"{token}.lock"])

    def test_marker_is_removed_when_its_owner_exits_normally(self) -> None:
        """A process that simply finishes leaves no marker behind.

        The child is a ``spawn`` process: a fresh interpreter that
        finishes through ``sys.exit`` and so runs ``atexit`` hooks, like
        a real daemon or CLI run.  A ``fork`` child would leave through
        ``os._exit`` in ``multiprocessing``'s bootstrap, which skips
        ``atexit`` and cannot exercise :func:`_release_owner_marker`.
        """
        ctx = multiprocessing.get_context("spawn")
        out_queue = ctx.Queue()
        proc = ctx.Process(
            target=_clean_exit_owner_worker,
            args=(str(self.kiss_dir), out_queue),
        )
        proc.start()
        try:
            token = out_queue.get(timeout=60)
        finally:
            proc.join(timeout=60)
        self.assertEqual(proc.exitcode, 0)
        self.assertTrue(token, "the child never published a marker")

        marker = self.kiss_dir / th._OWNER_DIR_NAME / f"{token}.lock"
        self.assertFalse(
            marker.exists(),
            "a process that exited normally left its liveness marker "
            "behind: every daemon and CLI lifecycle leaks one file",
        )
        self.assertFalse(th._owner_is_alive(token))

    @posix_only("chmod 500 directory write denial")
    def test_owner_token_is_empty_when_the_marker_cannot_be_written(
        self,
    ) -> None:
        """An unwritable home degrades gracefully instead of raising."""
        locked = self.tmp / "locked"
        locked.mkdir()
        th._KISS_DIR = locked
        th._owner_state = None
        locked.chmod(0o500)
        try:
            self.assertEqual(th._process_owner_token(), "")
        finally:
            locked.chmod(0o700)

    def test_owner_is_alive_is_false_for_a_missing_marker(self) -> None:
        """An unknown token belongs to a process that is long gone."""
        self.assertFalse(th._owner_is_alive("1234-" + uuid.uuid4().hex))
        self.assertFalse(th._owner_is_alive(""))

    def test_owner_column_is_added_to_an_older_database(self) -> None:
        """A database created before owner tracking is upgraded in place."""
        close_db()
        conn = sqlite3.connect(str(self.kiss_dir / "history.db"))
        conn.execute(
            "CREATE TABLE task_history ("
            "id TEXT PRIMARY KEY, timestamp REAL NOT NULL, task TEXT NOT NULL,"
            " has_events INTEGER DEFAULT 0, result TEXT DEFAULT '',"
            " chat_id CHAR(32) DEFAULT '', model TEXT DEFAULT '',"
            " work_dir TEXT DEFAULT '', version TEXT DEFAULT '',"
            " tokens INTEGER DEFAULT 0, cost REAL DEFAULT 0.0,"
            " steps INTEGER DEFAULT 0, is_parallel INTEGER DEFAULT 1,"
            " is_worktree INTEGER DEFAULT 1,"
            " auto_commit_mode INTEGER DEFAULT 1,"
            " start_ts INTEGER DEFAULT 0, end_ts INTEGER DEFAULT 0,"
            " is_favorite INTEGER DEFAULT 0, parent_task_id TEXT DEFAULT '')"
        )
        conn.commit()
        conn.close()

        db = th._get_db()

        cols = {
            r[1] for r in db.execute(
                "PRAGMA table_info(task_history)"
            ).fetchall()
        }
        self.assertIn("owner", cols)
        task_id, _chat = th._add_task("after upgrade")
        row = db.execute(
            "SELECT owner FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        self.assertEqual(str(row["owner"]), th._process_owner_token())

    def test_sweep_without_a_time_filter(self) -> None:
        """``created_before=None`` applies no timestamp restriction."""
        task_id, _chat = th._add_task("no time filter")
        th._get_db().execute(
            "UPDATE task_history SET owner = '' WHERE id = ?", (task_id,)
        )
        self.assertEqual(th._recover_orphaned_tasks(set()), 1)
        self.assertEqual(
            self._result_of(task_id),
            "Task terminated unexpectedly (process killed)",
        )

    def test_journalling_survives_an_unwritable_sidecar(self) -> None:
        """Losing the journal itself is logged, never raised."""
        sidecar = Path(th._failed_events_path(th._current_db_path()))
        sidecar.mkdir()
        th._journal_failed_events(
            [("abc", json.dumps({"type": "t"}), 1.0, th._current_db_path())], 4,
        )
        self.assertTrue(sidecar.is_dir())
        sidecar.rmdir()

    def test_a_journal_of_only_garbage_is_discarded(self) -> None:
        """An unparsable journal is dropped instead of retried forever.

        So is a snapshot whose name holds digits ``int`` refuses where
        its claim key goes: the replay raised on it from every flush.
        """
        sidecar = Path(th._failed_events_path(th._current_db_path()))
        sidecar.write_text("garbage\nmore garbage\n")
        snapshot = sidecar.with_name(sidecar.name + ".consumed-" + "\u00b2" * 20 + "-1-x")
        snapshot.write_text("garbage\n")

        th._replay_failed_events()

        self.assertFalse(sidecar.exists())
        self.assertFalse(snapshot.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

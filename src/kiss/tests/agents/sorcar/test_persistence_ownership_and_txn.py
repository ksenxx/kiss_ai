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
  so the ``frequent_tasks`` cap could be exceeded permanently.
* **R3** — permanently-unwritable events were journalled to a
  write-only sidecar and then acknowledged as flushed, so the real
  transcript was lost.
* **I1** — schema booleans defaulted to 0 while every other layer
  defaults to True, and the migration mapped a MISSING key to 0.
* **I4** — ``_update_task_column`` interpolates the column name into
  SQL but accepted any string.
* **D2** — the process-global ``_db_conn`` invalidated every OTHER
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
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

import kiss.agents.sorcar.persistence as th

_RACE_DELAY_ENV = "KISS_RACE_DELAY"


def _redirect(tmpdir: Path) -> tuple:
    """Point the persistence module at a private KISS home."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR, th._owner_state)
    kiss_dir = tmpdir / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    th._owner_state = None
    return saved


def _restore(saved: tuple) -> None:
    """Undo :func:`_redirect`."""
    th._close_db()
    (th._DB_PATH, th._db_conn, th._KISS_DIR, th._owner_state) = saved


def _live_task_worker(kiss_dir: str, out_queue) -> None:
    """Child process: insert a task row and stay alive holding it.

    The parent always ends this process with ``kill()``, so the child
    must NOT block on a ``multiprocessing`` primitive: SIGKILL while
    holding one of those shared semaphores wedges the parent too.
    """
    import kiss.agents.sorcar.persistence as child_th

    child_th._KISS_DIR = Path(kiss_dir)
    child_th._DB_PATH = child_th._KISS_DIR / "sorcar.db"
    child_th._db_conn = None
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
    child_th._DB_PATH = child_th._KISS_DIR / "sorcar.db"
    child_th._db_conn = None
    child_th._owner_state = None
    out_queue.put(child_th._process_owner_token())


def _frequent_task_worker(kiss_dir: str, task: str, barrier) -> None:
    """Child process: record one brand-new frequent task at the barrier."""
    import kiss.agents.sorcar.persistence as child_th

    child_th._KISS_DIR = Path(kiss_dir)
    child_th._DB_PATH = child_th._KISS_DIR / "sorcar.db"
    child_th._db_conn = None
    child_th._owner_state = None
    barrier.wait()
    child_th._record_frequent_task(task)


class _PersistenceTestCase(unittest.TestCase):
    """Base fixture giving each test a private database."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss_e_db_"))
        self.saved = _redirect(self.tmp)
        th._close_db()

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


class FrequentTaskCapTest(_PersistenceTestCase):
    """R2: the frequent-task cap must hold across processes."""

    def _seed(self, count: int) -> None:
        """Insert *count* distinct frequent-task rows, oldest first.

        Every row gets ``count = 1`` and a strictly increasing
        timestamp, so eviction (lowest count, then oldest timestamp)
        always targets ``seed-0`` and never a row the test just added.
        """
        db = th._get_db()
        now = time.time()
        for i in range(count):
            db.execute(
                "INSERT INTO frequent_tasks (task, count, timestamp) "
                "VALUES (?, 1, ?)",
                (f"seed-{i}", now - count + i),
            )

    def _count(self) -> int:
        """Return the number of rows in ``frequent_tasks``."""
        row = th._get_db().execute(
            "SELECT COUNT(*) AS n FROM frequent_tasks"
        ).fetchone()
        return int(row["n"])

    def test_two_processes_cannot_exceed_the_cap(self) -> None:
        """Concurrent inserts at the cap boundary stay within the cap."""
        self._seed(th._MAX_FREQUENT_TASKS - 1)
        th._close_db()
        saved_delay = os.environ.get(_RACE_DELAY_ENV)
        os.environ[_RACE_DELAY_ENV] = "0.05"
        try:
            ctx = multiprocessing.get_context("spawn")
            barrier = ctx.Barrier(2)
            procs = [
                ctx.Process(
                    target=_frequent_task_worker,
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

        self.assertLessEqual(self._count(), th._MAX_FREQUENT_TASKS)
        tasks = {r["task"] for r in th._get_db().execute(
            "SELECT task FROM frequent_tasks"
        ).fetchall()}
        self.assertIn("racer-0", tasks)
        self.assertIn("racer-1", tasks)

    def test_serial_inserts_evict_the_least_used_row(self) -> None:
        """Eviction keeps the table at the cap and drops the coldest row."""
        self._seed(th._MAX_FREQUENT_TASKS)
        th._record_frequent_task("brand new")
        self.assertEqual(self._count(), th._MAX_FREQUENT_TASKS)
        tasks = {r["task"] for r in th._get_db().execute(
            "SELECT task FROM frequent_tasks"
        ).fetchall()}
        self.assertIn("brand new", tasks)
        self.assertNotIn("seed-0", tasks)

    def test_repeat_of_an_existing_task_never_evicts(self) -> None:
        """An upsert of a known task does not touch the cap logic."""
        self._seed(th._MAX_FREQUENT_TASKS)
        th._record_frequent_task("seed-3")
        self.assertEqual(self._count(), th._MAX_FREQUENT_TASKS)
        row = th._get_db().execute(
            "SELECT count FROM frequent_tasks WHERE task = 'seed-3'"
        ).fetchone()
        self.assertEqual(int(row["count"]), 2)

    def test_empty_task_is_ignored(self) -> None:
        """An empty task string is not recorded."""
        th._record_frequent_task("")
        self.assertEqual(self._count(), 0)


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

    def test_events_are_replayed_once_the_database_recovers(self) -> None:
        """A transient write outage must not lose the transcript."""
        task_id, _chat = th._add_task("replay target")
        self._block_event_writes()
        for i in range(10):
            th._queue_chat_event({"type": "text", "content": f"e{i}"}, task_id)
        th._flush_chat_events(task_id)

        sidecar = Path(th._failed_events_path(th._current_db_path()))
        self.assertTrue(sidecar.is_file())
        self.assertEqual(len(sidecar.read_text().splitlines()), 10)
        self.assertEqual(self._event_count(task_id), 0)

        self._unblock_event_writes()
        th._flush_chat_events(task_id)

        self.assertTrue(th._task_has_events(task_id))
        self.assertEqual(self._event_count(task_id), 10)
        self.assertFalse(sidecar.exists())

    def test_journal_stays_when_the_database_is_still_unwritable(self) -> None:
        """A failing replay keeps the journal for the next attempt."""
        task_id, _chat = th._add_task("still broken")
        self._block_event_writes()
        th._queue_chat_event({"type": "text", "content": "x"}, task_id)
        th._flush_chat_events(task_id)
        sidecar = Path(th._failed_events_path(th._current_db_path()))
        self.assertTrue(sidecar.is_file())

        th._replay_failed_events()

        self.assertTrue(sidecar.is_file())
        self._unblock_event_writes()

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
        """A corrupt journal never blocks recovery of the good lines."""
        task_id, _chat = th._add_task("partial journal")
        sidecar = Path(th._failed_events_path(th._current_db_path()))
        sidecar.write_text(
            "not json\n"
            + json.dumps({
                "task_id": task_id,
                "event_json": json.dumps({"type": "result"}),
                "timestamp": time.time(),
                "origin_db_path": th._current_db_path(),
            }) + "\n"
        )

        th._flush_chat_events()

        self.assertEqual(self._event_count(task_id), 1)
        self.assertFalse(sidecar.exists())

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
        self.assertTrue(extra["is_worktree"])
        self.assertTrue(extra["is_parallel"])
        self.assertTrue(extra["auto_commit_mode"])

    def test_migration_keeps_unrecorded_toggles_enabled(self) -> None:
        """A legacy row that never recorded the toggles is not inverted."""
        th._close_db()
        db_path = self.kiss_dir / "sorcar.db"
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
        self.assertTrue(extra["is_worktree"])
        self.assertTrue(extra["is_parallel"])
        self.assertTrue(extra["auto_commit_mode"])
        self.assertEqual(extra["model"], "m")

    def test_migration_preserves_explicitly_disabled_toggles(self) -> None:
        """An explicit ``false`` is still migrated as disabled."""
        th._close_db()
        db_path = self.kiss_dir / "sorcar.db"
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
            (time.time(), "explicit off", json.dumps({
                "is_worktree": False,
                "is_parallel": False,
                "auto_commit_mode": "no",
            })),
        )
        conn.close()

        extra = self._extra_of(th._load_history()[0])

        self.assertFalse(extra["is_worktree"])
        self.assertFalse(extra["is_parallel"])
        self.assertFalse(extra["auto_commit_mode"])

    def test_add_task_records_what_the_caller_declared(self) -> None:
        """An explicit ``False`` from the caller is still stored as off."""
        task_id, _chat = th._add_task(
            "explicit", extra={"is_worktree": False, "is_parallel": True},
        )
        row = th._get_db().execute(
            "SELECT is_worktree, is_parallel FROM task_history WHERE id = ?",
            (task_id,),
        ).fetchone()
        self.assertEqual(int(row["is_worktree"]), 0)
        self.assertEqual(int(row["is_parallel"]), 1)


class UpdateTaskColumnGuardTest(_PersistenceTestCase):
    """I4: only allow-listed columns may be interpolated into SQL."""

    def test_a_non_allowlisted_column_raises(self) -> None:
        """The long-removed ``extra`` column is rejected, not written."""
        task_id, _chat = th._add_task("guard me")
        with self.assertRaises(ValueError):
            th._update_task_column("extra", "{}", task_id, None)

    def test_result_still_round_trips(self) -> None:
        """The one allowed column keeps working."""
        task_id, _chat = th._add_task("guard me too")
        th._save_task_result("Task completed", task_id=task_id)
        self.assertEqual(self._result_of(task_id), "Task completed")


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

        # Connect LAST so this thread's handle is the one the global
        # ``_db_conn`` names — that is the case ``_close_thread_db``
        # used to propagate to every other thread.
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
        th._reserve_pending(task_id)
        th._event_queue.put((
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

    def test_marker_is_removed_when_its_owner_exits_normally(self) -> None:
        """A process that simply finishes leaves no marker behind."""
        out_queue: multiprocessing.Queue[str] = multiprocessing.Queue()
        proc = multiprocessing.Process(
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
        th._close_db()
        conn = sqlite3.connect(str(self.kiss_dir / "sorcar.db"))
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
        """An unparsable journal is dropped instead of retried forever."""
        sidecar = Path(th._failed_events_path(th._current_db_path()))
        sidecar.write_text("garbage\nmore garbage\n")

        th._replay_failed_events()

        self.assertFalse(sidecar.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

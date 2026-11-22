# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Read-only and journal paths must not wait on another process's locks.

Two waits in ``persistence`` were unbounded or needless:

* Every new per-thread connection ran the column migration under
  ``BEGIN IMMEDIATE`` even when no column was missing, so the first
  ``_get_db()`` of a thread (a history request, a sub-agent) sat in
  SQLite's 30 s ``busy_timeout`` behind any other process's write
  transaction, holding the ``_rw_lock`` read token for that whole wait.
* ``_journal_file_lock`` took the cross-process ``flock`` with no
  timeout, so a peer wedged under the journal lock stalled every
  ``_flush_chat_events`` (and so every final-result save) in this
  daemon for as long as it held the file.

What a journal write does once that timeout fires is pinned here too.
It must never touch the shared file without the lock: the holder may
be the prune replacing the sidecar's inode or a replayer unlinking the
snapshot it just claimed, and a record written into either inode is
gone.  A final result (task thread) goes to a private spill file that
the loader reads and the next prune folds in; a failed-events batch
(writer thread) waits longer, because its rows should land in the live
sidecar in order, but past ``_JOURNAL_APPEND_TIMEOUT_S`` it is spilled
to a private file that the next locked append or replay claims as a
snapshot AFTER the live sidecar, so a wedged peer can never hang the
task thread's final flush and the transcript order survives.

Also here: ``_record_file_usage`` trims a ``file_usage`` table that is
already past its cap back down to it.

Also here: the metadata backfill honours the ``success`` column the
live finalizer honours, so ``--refresh`` reproduces the daemon's tags.

Also here: the wait a statement does make for another process's write
transaction (30 s at most) is taken in slices, so the Stop of a task
reaches a thread in it; its length, SQLite's error at its end and the
refusals SQLite makes without waiting are those of one unbroken wait.

Everything is real: a real second connection holding SQLite's write
lock, a real second OS process holding the kernel file lock, real
threads and the real module state.  Nothing is mocked or patched.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.history_columns import (
    ADDED_TASK_COLUMNS,
    TASK_TABLE,
    missing_columns,
)
from kiss.agents.sorcar.task_metadata import backfill_task_metadata
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.agents.sorcar import rwlock_probe
from kiss.tests.persistence_helpers import close_db

_HOLD_WRITE_SCRIPT = """
import sqlite3, sys
db = sqlite3.connect(sys.argv[1], isolation_level=None)
db.execute("BEGIN IMMEDIATE")
print("held", flush=True)
sys.stdin.readline()
db.execute("ROLLBACK")
"""

_HOLD_LOCK_SCRIPT = """
import sys, time
from kiss.core.file_lock import lock_exclusive
handle = open(sys.argv[1], "a+")
lock_exclusive(handle)
open(sys.argv[2], "w").close()
time.sleep(float(sys.argv[3]))
"""


@pytest.fixture
def kiss_home(tmp_path: Path) -> Iterator[Path]:
    """Point persistence at a throwaway KISS_HOME and restore it afterwards."""
    saved = (persistence._KISS_DIR, persistence._DB_PATH)
    persistence._KISS_DIR = tmp_path
    persistence._DB_PATH = tmp_path / "history.db"
    try:
        yield tmp_path
    finally:
        close_db()
        persistence._KISS_DIR, persistence._DB_PATH = saved


def get_db_on_new_thread(done: threading.Event, errors: list[BaseException]) -> None:
    """Thread body: open this thread's connection, then signal *done*."""
    try:
        persistence._get_db()
    except BaseException as exc:
        errors.append(exc)
    finally:
        persistence._close_thread_db()
        done.set()


def test_a_new_thread_connects_while_a_peer_holds_the_write_lock(kiss_home: Path) -> None:
    """A complete schema needs no write lock: the connection opens at once."""
    persistence._get_db()  # creates the full schema on the main thread
    peer = sqlite3.connect(str(persistence._DB_PATH), isolation_level=None, timeout=1)
    peer.execute("BEGIN IMMEDIATE")
    try:
        done = threading.Event()
        errors: list[BaseException] = []
        started = time.monotonic()
        threading.Thread(target=get_db_on_new_thread, args=(done, errors), daemon=True).start()
        finished = done.wait(5.0)
        elapsed = time.monotonic() - started
    finally:
        peer.execute("ROLLBACK")
        peer.close()
    assert finished, "the new thread's _get_db() waited for the peer's transaction"
    assert elapsed < 1.0, f"_get_db() took {elapsed:.2f}s behind a peer write transaction"
    assert errors == []


def test_a_missing_column_is_still_added_with_its_backfill(kiss_home: Path) -> None:
    """The write transaction is taken only when a column is absent, and then adds it."""
    conn = sqlite3.connect(str(persistence._DB_PATH), isolation_level=None)
    persistence._init_tables(conn)
    assert missing_columns(conn, TASK_TABLE, ADDED_TASK_COLUMNS) == []
    conn.execute("ALTER TABLE task_history DROP COLUMN sea")
    assert [name for name, _ in missing_columns(conn, TASK_TABLE, ADDED_TASK_COLUMNS)] == ["sea"]
    persistence._add_missing_columns(conn)
    assert missing_columns(conn, TASK_TABLE, ADDED_TASK_COLUMNS) == []
    assert not conn.in_transaction
    conn.close()


def record_a_file_on_new_thread(outcome: list[object]) -> None:
    """Thread body: one write; *outcome* gets how it ended, then whether a transaction is left."""
    try:
        persistence._record_file_usage("waited.py")
        outcome.append("written")
    except BaseException as exc:
        outcome.append(type(exc).__name__)
    finally:
        outcome.append(persistence._get_db().in_transaction)
        persistence._close_thread_db()


def test_a_stop_reaches_a_write_that_waits_for_another_process(kiss_home: Path) -> None:
    """A write behind another process's transaction is stopped at once; left alone, it waits.

    The wait was one C call of up to 30 s (SQLite's busy handler), in
    which no injected exception is raised: the Stop of a task stayed
    pending until the other process let go of the database.
    """
    persistence._get_db()  # the schema: the writers' connections open without a write
    peer = subprocess.Popen(
        [sys.executable, "-c", _HOLD_WRITE_SCRIPT, str(persistence._DB_PATH)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    stopped: list[object] = []
    waited: list[object] = []
    stoppee = threading.Thread(target=record_a_file_on_new_thread, args=(stopped,), daemon=True)
    waiter = threading.Thread(target=record_a_file_on_new_thread, args=(waited,), daemon=True)
    try:
        assert peer.stdout is not None and peer.stdout.readline().strip() == "held"
        stoppee.start()
        deadline = time.monotonic() + 10.0
        while not rwlock_probe.writer(persistence._rw_lock):
            assert time.monotonic() < deadline, "the write never took the history lock"
            time.sleep(0.01)
        time.sleep(0.3)  # its BEGIN IMMEDIATE waits for the peer by now
        assert inject_async_exception(stoppee.ident or 0, KeyboardInterrupt) == 1
        stoppee.join(persistence._BUSY_SLICE_MS / 1000 + 0.9)  # 1 s, unless SQLite is an old one
        assert not stoppee.is_alive(), "the Stop waited for the other process's transaction"
        assert stopped == ["KeyboardInterrupt", False]
        assert not rwlock_probe.writer(persistence._rw_lock)
        waiter.start()
        waiter.join(2.0)
        assert waiter.is_alive(), "the write gave up on a database that was busy for 2 s"
        peer.communicate("release\n", timeout=10)
        waiter.join(10.0)
        assert waited == ["written", False]
    finally:
        peer.kill()
        peer.wait()
        for thread in (stoppee, waiter):
            if thread.is_alive():
                thread.join(10.0)
    counts = persistence._get_db().execute("SELECT path, count FROM file_usage").fetchall()
    assert [tuple(row) for row in counts] == [("waited.py", 1)]


def test_a_busy_database_is_given_up_on_after_the_whole_wait_or_where_sqlite_refuses(
    kiss_home: Path,
) -> None:
    """The wait ends in SQLite's own error after its whole length; none is made in SQLite's place.

    SQLite refuses the write of a connection that holds a read
    transaction without waiting (in rollback-journal mode the two
    would wait for each other); repeating the statement must not wait
    for it either.
    """
    saved = persistence._BUSY_TIMEOUT_MS
    persistence._BUSY_TIMEOUT_MS = 500  # the whole wait, shortened
    try:
        db = persistence._get_db()
        for path in ("a.py", "b.py"):
            persistence._record_file_usage(path)
        peer = sqlite3.connect(str(persistence._DB_PATH), isolation_level=None, timeout=1)
        peer.execute("BEGIN IMMEDIATE")
        try:
            started = time.monotonic()
            with pytest.raises(sqlite3.OperationalError, match="database is locked") as info:
                persistence._record_file_usage("c.py")
            elapsed = time.monotonic() - started
            assert info.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
            assert 0.5 <= elapsed < 3.0, f"gave up after {elapsed:.2f}s of a 0.5 s wait"
            assert not db.in_transaction and not rwlock_probe.writer(persistence._rw_lock)
            persistence._BUSY_TIMEOUT_MS = saved  # 30 s: what a wait in SQLite's place would last
            half_read = db.execute("SELECT path FROM file_usage")
            half_read.fetchone()
            started = time.monotonic()
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                db.execute("BEGIN IMMEDIATE")
            assert time.monotonic() - started < 1.0, "waited where SQLite does not"
            half_read.close()
        finally:
            peer.close()
    finally:
        persistence._BUSY_TIMEOUT_MS = saved


def test_a_new_database_gets_its_tables_once_a_peer_lets_go_of_it(kiss_home: Path) -> None:
    """The schema script waits for a peer's transaction as long as that lasts, not for one slice."""
    peer = sqlite3.connect(str(persistence._DB_PATH), isolation_level=None, timeout=1)
    peer.execute("PRAGMA journal_mode=WAL")
    peer.execute("BEGIN IMMEDIATE")
    done = threading.Event()
    errors: list[BaseException] = []
    try:
        threading.Thread(target=get_db_on_new_thread, args=(done, errors), daemon=True).start()
        assert not done.wait(0.5), f"the schema did not wait for the peer: {errors}"
    finally:
        peer.execute("ROLLBACK")
        peer.close()
    assert done.wait(10.0), "the schema was not created after the peer let go"
    assert errors == []
    assert persistence._get_db().execute("SELECT COUNT(*) FROM task_history").fetchone()[0] == 0


class ReaderInTheWay:
    """A second connection whose half-read SELECT keeps a commit waiting for half a second.

    The database is in rollback-journal mode, where a commit needs
    every reader gone.
    """

    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.inserts = 0

    def read(self) -> None:
        """Start the SELECT; a timer thread ends it 0.5 s later."""
        cursor = self.conn.execute("SELECT x FROM t")
        cursor.fetchone()
        threading.Timer(0.5, cursor.close).start()

    def read_before_the_second_insert(self, sql: str) -> None:
        """Trace callback of the writer: the first INSERT commits freely, the second waits."""
        self.inserts += sql.startswith("INSERT")
        if self.inserts == 2:
            self.inserts += 1
            self.read()


def stop_as_the_slice_is_set_back(frame: FrameType, event: str, arg: object) -> None:
    """Profile hook: a Stop for the calling thread as its connection sets the slice back."""
    if (
        event == "call"
        and frame.f_code is persistence._LockedConnection.execute.__code__
        and frame.f_locals["sql"] == f"PRAGMA busy_timeout={persistence._BUSY_SLICE_MS}"
    ):
        sys.setprofile(None)
        inject_async_exception(threading.get_ident(), KeyboardInterrupt)


def commit_stopped_there(conn: sqlite3.Connection, outcome: list[object]) -> None:
    """Thread body: *outcome* gets how the commit ended and the ``busy_timeout`` it left."""
    sys.setprofile(stop_as_the_slice_is_set_back)
    try:
        conn.commit()
    except BaseException as exc:
        outcome.append(type(exc).__name__)
    finally:
        sys.setprofile(None)
    outcome.append(conn.execute("PRAGMA busy_timeout").fetchone()[0])


def test_the_calls_that_run_their_statements_from_c_wait_as_long(tmp_path: Path) -> None:
    """``commit()``, ``with conn:`` and a later set of an autocommit ``executemany`` wait 0.5 s.

    None can be repeated from the statement that found the database
    busy: SQLite makes their whole wait, which a slice of it would cut
    short here, and no Stop leaves the connection waiting that way.
    So does the last row of a write that returns rows, where it
    commits.  A statement that is repeated is applied once, also when
    SQLite counted a trigger's change in the attempts it undid.
    """
    path = tmp_path / "journal.db"
    conn = persistence._LockedConnection(path, check_same_thread=False, isolation_level=None)
    conn.execute(f"PRAGMA busy_timeout={persistence._BUSY_SLICE_MS}")  # as _get_db() opens one
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
    reader = ReaderInTheWay(path)
    waits = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t VALUES (3)")
        reader.read()
        started = time.monotonic()
        conn.commit()
        waits.append(time.monotonic() - started)
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO t VALUES (4)")
            reader.read()
            started = time.monotonic()
        waits.append(time.monotonic() - started)
        conn.set_trace_callback(reader.read_before_the_second_insert)
        started = time.monotonic()
        conn.executemany("INSERT INTO t VALUES (?)", [(5,), (6,)])
        waits.append(time.monotonic() - started)
        conn.set_trace_callback(None)
        returned = conn.execute("WITH v(x) AS (SELECT 7) INSERT INTO t SELECT x FROM v RETURNING x")
        reader.read()
        started = time.monotonic()
        assert [tuple(row) for row in returned.fetchall()] == [(7,)]
        waits.append(time.monotonic() - started)
        conn.execute("CREATE TABLE log (x INTEGER)")
        conn.execute(
            "CREATE TRIGGER logged AFTER INSERT ON t BEGIN INSERT INTO log VALUES (new.x); END",
        )
        reader.read()
        started = time.monotonic()
        conn.execute("INSERT INTO t VALUES (8)")
        waits.append(time.monotonic() - started)
        assert [wait >= 0.4 for wait in waits] == [True] * 5, waits
        assert [row[0] for row in conn.execute("SELECT x FROM t")] == list(range(1, 9))
        assert [row[0] for row in conn.execute("SELECT x FROM log")] == [8]
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == persistence._BUSY_SLICE_MS
        stopped: list[object] = []
        worker = threading.Thread(target=commit_stopped_there, args=(conn, stopped), daemon=True)
        worker.start()
        worker.join(10.0)
        assert stopped == ["KeyboardInterrupt", persistence._BUSY_SLICE_MS]
        conn._unsliced(conn.close)  # closed by the call itself: nothing is left to set back
    finally:
        conn.close()
        reader.conn.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        conn.commit()


def hold_journal_lock_in_subprocess(lock_file: Path, hold_s: float) -> subprocess.Popen[bytes]:
    """Start a child process that holds the kernel lock on *lock_file* for *hold_s* seconds."""
    locked = lock_file.with_suffix(".locked")
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK_SCRIPT, str(lock_file), str(locked), str(hold_s)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    deadline = time.monotonic() + 20.0
    while not locked.exists():
        if child.poll() is not None or time.monotonic() > deadline:
            child.kill()
            raise AssertionError("the lock-holding child never took the journal lock")
        time.sleep(0.02)
    return child


def test_flush_returns_while_a_peer_process_holds_the_journal_lock(kiss_home: Path) -> None:
    """A wedged peer costs ``_flush_chat_events`` at most the lock timeout."""
    persistence._get_db()
    journal = Path(persistence._failed_events_path(str(persistence._DB_PATH)))
    journal.write_text("", encoding="utf-8")  # makes the replay take the lock
    child = hold_journal_lock_in_subprocess(Path(str(journal) + ".lock"), 60.0)
    try:
        started = time.monotonic()
        persistence._flush_chat_events()
        elapsed = time.monotonic() - started
    finally:
        child.kill()
        child.wait()
    assert persistence._JOURNAL_LOCK_TIMEOUT_S <= elapsed < persistence._JOURNAL_LOCK_TIMEOUT_S + 3
    # The replay was skipped, not run unserialised: the live journal is untouched.
    assert journal.exists()


def test_journal_appends_go_through_when_the_lock_holder_is_slow(kiss_home: Path) -> None:
    """An append waits out a slow peer and, past the timeout, still lands.

    The timed-out record goes to a private spill file, never into the
    shared sidecar the lock holder may be rewriting; the next
    uncontended prune folds it into the sidecar.
    """
    persistence._get_db()
    sidecar = Path(persistence._final_results_path(str(persistence._DB_PATH)))
    child = hold_journal_lock_in_subprocess(Path(str(sidecar) + ".lock"), 60.0)
    try:
        persistence._journal_final_result("t1", "done", 1)
        assert not sidecar.exists()
        assert len(persistence._journal_spills(str(sidecar))) == 1
        assert persistence._load_final_results(str(sidecar)) == {
            "t1": {"result": "done", "success": 1},
        }
    finally:
        child.kill()
        child.wait()
    persistence._prune_final_results_journal(str(sidecar))
    assert persistence._journal_spills(str(sidecar)) == []
    assert '"task_id": "t1"' in sidecar.read_text(encoding="utf-8")
    assert persistence._load_final_results(str(sidecar)) == {
        "t1": {"result": "done", "success": 1},
    }


_PAUSED_PRUNE_SCRIPT = """
import os, sys, time
from kiss.agents.sorcar import persistence
sidecar, paused, go = sys.argv[1], sys.argv[2], sys.argv[3]
prune = persistence._prune_final_results_journal.__code__
def hook(frame, event, arg):
    if event == "c_call" and arg is os.replace and frame.f_code is prune:
        open(paused, "w").close()
        while not os.path.exists(go):
            time.sleep(0.02)
sys.setprofile(hook)
persistence._prune_final_results_journal(sidecar)
"""

_PAUSED_REPLAY_SCRIPT = """
import os, sys, time
from pathlib import Path
from kiss.agents.sorcar import persistence
from kiss.tests.persistence_helpers import close_db
home, paused, go = sys.argv[1], sys.argv[2], sys.argv[3]
persistence._KISS_DIR = Path(home)
persistence._DB_PATH = Path(home) / "history.db"
claim = persistence._claim_journal_snapshots.__code__
def hook(frame, event, arg):
    if event == "call" and frame.f_code is claim:
        open(paused, "w").close()
        while not os.path.exists(go):
            time.sleep(0.02)
sys.setprofile(hook)
persistence._replay_failed_events()
close_db()
"""


def start_paused_child(script: str, *args: str, paused: Path) -> subprocess.Popen[bytes]:
    """Run *script* in a child process and wait until it has created *paused*."""
    child = subprocess.Popen(
        [sys.executable, "-c", script, *args],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    deadline = time.monotonic() + 20.0
    while not paused.exists():
        if child.poll() is not None or time.monotonic() > deadline:
            child.kill()
            raise AssertionError("the child never reached its pause point")
        time.sleep(0.02)
    return child


def test_a_timed_out_final_result_survives_a_concurrent_prune(kiss_home: Path) -> None:
    """A record journalled past the lock timeout is not erased by the prune
    that held the lock: it never shares the inode the prune replaces."""
    persistence._get_db()
    sidecar = Path(persistence._final_results_path(str(persistence._DB_PATH)))
    sidecar.write_text(
        '{"task_id": "stale", "result": "x", "success": 1, "ts": 1.0}\n', encoding="utf-8",
    )
    paused, go = kiss_home / "prune-paused", kiss_home / "prune-go"
    child = start_paused_child(
        _PAUSED_PRUNE_SCRIPT, str(sidecar), str(paused), str(go), paused=paused,
    )
    try:
        started = time.monotonic()
        persistence._journal_final_result("new", "new result", 1)
        assert time.monotonic() - started >= persistence._JOURNAL_LOCK_TIMEOUT_S
        go.touch()
        assert child.wait(timeout=20) == 0
    finally:
        child.kill()
    assert persistence._load_final_results(str(sidecar)) == {
        "new": {"result": "new result", "success": 1},
    }
    persistence._prune_final_results_journal(str(sidecar))
    assert persistence._journal_spills(str(sidecar)) == []
    assert persistence._load_final_results(str(sidecar)) == {
        "new": {"result": "new result", "success": 1},
    }


def test_a_later_record_in_the_sidecar_beats_an_earlier_spill(kiss_home: Path) -> None:
    """Last write wins by journal time, whichever file the record sits in."""
    persistence._get_db()
    sidecar = Path(persistence._final_results_path(str(persistence._DB_PATH)))
    child = hold_journal_lock_in_subprocess(Path(str(sidecar) + ".lock"), 60.0)
    try:
        persistence._journal_final_result("t1", "first", None)
    finally:
        child.kill()
        child.wait()
    persistence._journal_final_result("t1", "second", 1)
    assert sidecar.exists() and len(persistence._journal_spills(str(sidecar))) == 1
    assert persistence._load_final_results(str(sidecar)) == {
        "t1": {"result": "second", "success": 1},
    }
    persistence._prune_final_results_journal(str(sidecar))
    assert persistence._load_final_results(str(sidecar)) == {
        "t1": {"result": "second", "success": 1},
    }


def test_a_batch_journalled_during_a_slow_replay_is_still_replayed(kiss_home: Path) -> None:
    """The failed-events append waits out a replayer that lets go within
    the append timeout and lands in the live sidecar: a batch written
    into the sidecar a replayer is consuming would be unlinked with it."""
    persistence._get_db()
    task_id, _chat = persistence._add_task("slow replay task")
    sidecar = Path(persistence._failed_events_path(str(persistence._DB_PATH)))
    sidecar.write_text("", encoding="utf-8")  # makes the replay take the lock
    paused, go = kiss_home / "replay-paused", kiss_home / "replay-go"
    child = start_paused_child(
        _PAUSED_REPLAY_SCRIPT, str(kiss_home), str(paused), str(go), paused=paused,
    )
    opened, resume = threading.Event(), threading.Event()
    journal_code = persistence._journal_failed_events.__code__

    def pause_after_opening_the_sidecar(frame: object, event: str, arg: object) -> None:
        if event == "c_return" and arg is open and getattr(frame, "f_code") is journal_code:
            opened.set()
            resume.wait(60)

    origin = str(persistence._DB_PATH)
    batch = [(task_id, json.dumps({"type": "text", "text": "late"}), 1.5, origin)]
    threading.setprofile(pause_after_opening_the_sidecar)
    appender = threading.Thread(target=persistence._journal_failed_events, args=(batch, 4))
    appender.start()
    try:
        started = time.monotonic()
        opened_before_release = opened.wait(persistence._JOURNAL_LOCK_TIMEOUT_S + 3)
        go.touch()
        assert child.wait(timeout=20) == 0
        resume.set()
        appender.join(30)
        assert not appender.is_alive()
    finally:
        threading.setprofile(None)
        resume.set()
        child.kill()
    elapsed = time.monotonic() - started
    assert sidecar.exists()
    persistence._flush_chat_events()
    rows = persistence._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ?", (task_id,),
    ).fetchall()
    assert [json.loads(r[0])["text"] for r in rows] == ["late"]
    assert not opened_before_release, "the sidecar was opened while the replayer held the lock"
    assert elapsed >= persistence._JOURNAL_LOCK_TIMEOUT_S


def failed_event_batch(
    task_id: str, text: str, timestamp: float,
) -> list[tuple[str, str, float, str]]:
    """Return a one-event failed-events batch for *task_id* against the active database."""
    origin = str(persistence._DB_PATH)
    return [(task_id, json.dumps({"type": "text", "text": text}), timestamp, origin)]


def test_a_failed_batch_spills_when_a_peer_keeps_the_journal_lock(kiss_home: Path) -> None:
    """A peer wedged under the journal lock costs the writer thread at most
    the append timeout; the batch lands in a spill file the next locked
    append or replay folds in AFTER the rows already in the live sidecar
    and BEFORE the rows appended later, so the flush that waits for the
    writer returns and the transcript keeps its order."""
    persistence._get_db()
    task_id, _chat = persistence._add_task("wedged peer task")
    sidecar = Path(persistence._failed_events_path(str(persistence._DB_PATH)))
    persistence._journal_failed_events(failed_event_batch(task_id, "first", 1.0), 4)
    live_before = sidecar.read_text(encoding="utf-8")
    saved = persistence._JOURNAL_APPEND_TIMEOUT_S
    persistence._JOURNAL_APPEND_TIMEOUT_S = 2.0
    child = hold_journal_lock_in_subprocess(Path(str(sidecar) + ".lock"), 120.0)
    appender = threading.Thread(
        target=persistence._journal_failed_events,
        args=(failed_event_batch(task_id, "second", 2.0), 4),
    )
    try:
        started = time.monotonic()
        appender.start()
        appender.join(persistence._JOURNAL_APPEND_TIMEOUT_S + 5)
        elapsed = time.monotonic() - started
        stalled = appender.is_alive()
    finally:
        child.kill()
        child.wait()
        appender.join(30)
        persistence._JOURNAL_APPEND_TIMEOUT_S = saved
    assert not stalled, "the append waited on the wedged peer past the append timeout"
    assert elapsed >= 2.0
    assert sidecar.read_text(encoding="utf-8") == live_before, "sidecar written without the lock"
    spills = persistence._journal_spills(str(sidecar))
    assert len(spills) == 1
    assert not [p for p in kiss_home.iterdir() if ".consumed-" in p.name]
    # The peer is gone: this append takes the lock and must land after the spill.
    persistence._journal_failed_events(failed_event_batch(task_id, "third", 3.0), 4)
    persistence._flush_chat_events()
    rows = persistence._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq", (task_id,),
    ).fetchall()
    assert [json.loads(r[0])["text"] for r in rows] == ["first", "second", "third"]
    assert not persistence._journal_has_pending_rows(str(sidecar))
    assert not [p for p in kiss_home.iterdir() if ".spill-" in p.name or ".consumed-" in p.name]


def test_a_spill_alone_is_found_and_replayed(kiss_home: Path) -> None:
    """A spill left with no live sidecar (the peer replayed it) is still pending and replayed."""
    persistence._get_db()
    task_id, _chat = persistence._add_task("lone spill task")
    sidecar = Path(persistence._failed_events_path(str(persistence._DB_PATH)))
    saved = persistence._JOURNAL_APPEND_TIMEOUT_S
    persistence._JOURNAL_APPEND_TIMEOUT_S = 0.2
    child = hold_journal_lock_in_subprocess(Path(str(sidecar) + ".lock"), 120.0)
    try:
        persistence._journal_failed_events(failed_event_batch(task_id, "lone", 1.0), 4)
    finally:
        child.kill()
        child.wait()
        persistence._JOURNAL_APPEND_TIMEOUT_S = saved
    assert not sidecar.exists()
    assert persistence._journal_has_pending_rows(str(sidecar))
    persistence._flush_chat_events()
    rows = persistence._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ?", (task_id,),
    ).fetchall()
    assert [json.loads(r[0])["text"] for r in rows] == ["lone"]
    assert persistence._journal_spills(str(sidecar)) == []
    assert not persistence._journal_has_pending_rows(str(sidecar))


def test_an_oversized_file_usage_table_is_trimmed_to_the_cap(kiss_home: Path) -> None:
    """A table already past the cap (an older daemon's database) shrinks
    back to the cap on the first new path, evicting the least recently used."""
    db = persistence._get_db()
    cap = persistence._MAX_FILE_USAGE_ENTRIES
    with persistence._immediate_txn(db):
        db.executemany(
            "INSERT INTO file_usage (path, count, last_used) VALUES (?, 1, ?)",
            [(f"file-{i}", float(i)) for i in range(cap + 2)],
        )
    persistence._record_file_usage("file-0")  # an existing path: refreshed, nothing evicted
    assert db.execute("SELECT COUNT(*) FROM file_usage").fetchone()[0] == cap + 2
    persistence._record_file_usage("new-path")
    assert db.execute("SELECT COUNT(*) FROM file_usage").fetchone()[0] == cap
    kept = {row[0] for row in db.execute("SELECT path FROM file_usage").fetchall()}
    assert "new-path" in kept and "file-0" in kept
    assert {"file-1", "file-2", "file-3"}.isdisjoint(kept)


def test_the_prune_is_skipped_while_a_peer_holds_the_journal_lock(kiss_home: Path) -> None:
    """A rewrite without the lock could drop a peer's append: the prune waits for the next sweep."""
    sidecar = Path(persistence._final_results_path(str(persistence._DB_PATH)))
    stale = '{"task_id": "old", "result": "x", "success": 1, "ts": 1.0}\n'
    sidecar.write_text(stale, encoding="utf-8")
    child = hold_journal_lock_in_subprocess(Path(str(sidecar) + ".lock"), 60.0)
    try:
        persistence._prune_final_results_journal(str(sidecar))
    finally:
        child.kill()
        child.wait()
    assert sidecar.read_text(encoding="utf-8") == stale
    persistence._prune_final_results_journal(str(sidecar))
    assert sidecar.read_text(encoding="utf-8") == ""


def test_backfill_flags_a_row_the_agent_declared_failed(kiss_home: Path) -> None:
    """``success == 0`` with an ordinary summary tags ``failed``, as the live path does."""
    conn = sqlite3.connect(str(persistence._DB_PATH), isolation_level=None)
    persistence._init_tables(conn)
    conn.execute(
        "INSERT INTO task_history (id, timestamp, task, result, success, tags) "
        "VALUES ('t1', 1.0, 'Write the quarterly report', 'done', 0, '')"
    )
    counts = backfill_task_metadata(conn)
    tags = conn.execute("SELECT tags FROM task_history WHERE id = 't1'").fetchone()[0]
    conn.close()
    assert counts["tags"] == 1
    assert "failed" in tags.split(",")

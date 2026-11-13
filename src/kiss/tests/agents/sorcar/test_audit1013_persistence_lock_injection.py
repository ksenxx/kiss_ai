# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Persistence's locks survive the runner's Stop injection on a task thread.

Every task thread writes its history row, its events and its result
through ``persistence``, so its module-level locks are daemon-wide
and contended: ``_init_tables_lock`` (every first connection of a
thread, every retired handle), the pending-event condition, the
seq caches, the event-writer and journal locks, the chat-context
cache and the per-connection statement locks.  The runner's Stop
delivers ``KeyboardInterrupt`` with ``PyThreadState_SetAsyncExc``; an
interrupt raised at the check point right after a plain ``with
lock:``'s ``__enter__`` returned skips ``__exit__`` (CPython gh-74174)
and leaks the lock for the daemon's life.  A thread blocked in a
contended ``acquire()`` receives the pending interrupt exactly there.
A leaked connection lock is as bad: ``_prune_dead_thread_conns``
closes the dead thread's handle under ``_init_tables_lock`` and
``close()`` waits for that lock forever, so every later ``_get_db()``
in the process hangs.  Every such site therefore goes through
:func:`~kiss.core.locked.locked_call`.

Each test hammers one real persistence path from a worker thread with
hundreds of injections while two peer threads loop the same path and
a contender holds the path's condition lock in bursts, so every
shared lock the path takes is contended, in a child process (a leaked
lock would wedge the rest of the pytest session).  The writes are
real: the driver counts the rounds it completed, the only exception
the workers may raise is the injected interrupt, and the rows and
events the path wrote are read back from the database afterwards, so
a hammer that only failed on SQLite would not pass.  Every lock is
asserted free after, and the process can still connect and close.
"""

from __future__ import annotations

import gc
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence
from kiss.tests.lock_injection import (
    INJECTIONS,
    Contender,
    Hammered,
    ignore_thread_interrupt,
    inject,
    lock_is_free,
)
from kiss.tests.persistence_helpers import close_db


class _Peer(threading.Thread):
    """A thread repeating *call* until stopped, keeping the shared locks contended."""

    def __init__(self, call: object) -> None:
        super().__init__(daemon=True)
        self.call = call
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            self.call()  # type: ignore[operator]


class _OwnRounds:
    """Calls *driver*, counting the rounds completed through this object alone."""

    def __init__(self, driver: _TaskRow) -> None:
        self.driver = driver
        self.completed = 0

    def __call__(self) -> None:
        self.driver()
        self.completed += 1


class _TaskRow:
    """``_add_task`` then ``_save_task_result``: the row, the journal and the caches."""

    contended = persistence._chat_context_cache_lock

    def __init__(self) -> None:
        self.completed = 0
        self.last_task_id = ""

    def __call__(self) -> None:
        task_id, _chat_id = persistence._add_task("audit1013 persistence", chat_id="")
        persistence._save_task_result("done", task_id, success=True)
        persistence._load_chat_context_text("chat-of-" + task_id)
        self.last_task_id = task_id
        self.completed += 1

    def verify(self) -> None:
        """The last completed round's row and result are in the database."""
        row = persistence._get_db().execute(
            "SELECT task, result, success FROM task_history WHERE id = ?", (self.last_task_id,),
        ).fetchone()
        assert row is not None and tuple(row) == ("audit1013 persistence", "done", 1), row


class _Events:
    """``_queue_chat_event`` then ``_flush_chat_events``: reservations, writer, caches."""

    contended = persistence._pending_cond

    def __init__(self) -> None:
        self.completed = 0
        self.task_id, _chat_id = persistence._add_task("audit1013 events", chat_id="")

    def __call__(self) -> None:
        persistence._queue_chat_event({"type": "status", "n": 1}, self.task_id)
        persistence._flush_chat_events(self.task_id)
        self.completed += 1

    def verify(self) -> None:
        """At least every completed round's event reached the events table."""
        persistence._flush_chat_events(self.task_id)
        row = persistence._load_chat_events_by_task_id(self.task_id)
        assert row is not None
        events = row["events"]
        assert isinstance(events, list)
        stored = sum(1 for e in events if e.get("type") == "status")
        assert stored >= self.completed, (stored, self.completed)


class _Reconnect:
    """``_close_thread_db`` then ``_get_db``: ``_init_tables_lock`` and the handle locks."""

    contended = persistence._init_tables_lock

    def __init__(self) -> None:
        self.completed = 0

    def __call__(self) -> None:
        persistence._close_thread_db()
        persistence._get_db().execute("SELECT 1").fetchall()
        self.completed += 1

    def verify(self) -> None:
        """A fresh handle still answers."""
        assert persistence._get_db().execute("SELECT 1").fetchone()[0] == 1


_DRIVERS: dict[str, type[_TaskRow | _Events | _Reconnect]] = {
    "task_row": _TaskRow,
    "events": _Events,
    "reconnect": _Reconnect,
}


def _persistence_locks() -> dict[str, object]:
    return {
        "_init_tables_lock": persistence._init_tables_lock,
        "_chat_context_cache_lock": persistence._chat_context_cache_lock,
        "_owner_state_lock": persistence._owner_state_lock,
        "_caches_lock": persistence._caches_lock,
        "_event_writer_lock": persistence._event_writer_lock,
        "_journal_lock": persistence._journal_lock,
        "_pending_cond": persistence._pending_cond,
        "TASK_USAGE_LOCK": persistence.TASK_USAGE_LOCK,
    }


def _take_write_lock() -> None:
    with persistence._rw_lock.write_lock():
        pass


def _check_connections_closable() -> None:
    """The write lock is free, and every handle a dead worker left behind gets closed."""
    leaked = [
        key for key, (conn, owner, _path) in list(persistence._open_conns.items())
        if not owner.is_alive() and not lock_is_free(conn.lock)
    ]
    assert not leaked, f"a dead thread left its connection lock held: {leaked}"
    for name, target in (("_rw_lock", _take_write_lock), ("close_db", close_db)):
        probe = threading.Thread(target=target, daemon=True)
        probe.start()
        probe.join(20.0)
        assert not probe.is_alive(), f"{name} hung on a leaked lock"


class _UnexpectedErrors:
    """``threading.excepthook`` keeping every worker error that is not the injected interrupt."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def __call__(self, args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not KeyboardInterrupt:
            self.errors.append(f"{args.thread.name if args.thread else '?'}: {args.exc_value!r}")


def _hammer_contended(driver: _TaskRow | _Events | _Reconnect) -> None:
    """Inject ``INJECTIONS`` interrupts into a thread looping *driver*; two peers loop it too.

    The peers and a contender on the driver's condition lock make every
    shared lock contended, so the worker's ``acquire()`` blocks and
    receives the interrupt on return, which is the gh-74174 window.
    They are stopped before the locks are checked.
    """
    previous_hook = threading.excepthook
    unexpected = _UnexpectedErrors()
    threading.excepthook = unexpected
    gc.collect()
    gc.disable()
    delivered = 0
    peers: list[threading.Thread] = [_Peer(driver) for _ in range(2)]
    peers.append(Contender(driver.contended))
    for peer in peers:
        peer.start()
    worker = Hammered(driver)
    worker.start()
    try:
        for _ in range(INJECTIONS):
            time.sleep(0.0005)
            if not worker.is_alive():
                worker = Hammered(driver)
                worker.start()
            delivered += inject(worker)
        worker.stop.set()
        worker.join(20.0)
        for peer in peers:
            peer.stop.set()  # type: ignore[attr-defined]
        for peer in peers:
            peer.join(20.0)
    finally:
        gc.enable()
        threading.excepthook = previous_hook
    assert not worker.is_alive(), "the worker hung on a lock"
    assert not any(peer.is_alive() for peer in peers), "a peer hung on a leaked lock"
    assert delivered > INJECTIONS // 2, "the interrupts did not reach the worker"
    assert not unexpected.errors, unexpected.errors
    assert driver.completed > 0, "the path never completed a round"
    leaked = [name for name, lock in _persistence_locks().items() if not lock_is_free(lock)]
    assert not leaked, f"an interrupt leaked {leaked}"


def _redirect_home() -> None:
    """Point persistence at a fresh database under a temporary home."""
    home = Path(tempfile.mkdtemp(prefix="audit1013-persist-"))
    os.environ["KISS_HOME"] = str(home)
    persistence._KISS_DIR = home
    persistence._DB_PATH = home / "history.db"


def run_hammer(name: str) -> None:
    """Child-process body: hammer the path *name*, check its writes landed and no lock leaked."""
    _redirect_home()
    driver = _DRIVERS[name]()
    _hammer_contended(driver)
    driver.verify()
    _check_connections_closable()


def _run_in_child(call: str) -> None:
    """Run the module-level *call* (``"f(args)"``) in a child process; it must exit 0."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.{call}"],
        capture_output=True, text=True, timeout=240, check=False,
    )
    assert proc.returncode == 0, f"{call}: {proc.stderr[-3000:]}"


@pytest.mark.parametrize("path", sorted(_DRIVERS))
def test_persistence_lock_paths(path: str) -> None:
    """No persistence site a task thread runs leaks a lock under injection."""
    _run_in_child(f"run_hammer({path!r})")


def run_two_stops(second: str) -> None:
    """Child-process body: a stop in a fetch, then a second one while the first unwinds.

    The first stop lands between a SELECT's execute and its fetch, so
    the interrupted frames hold a half-read cursor.  The second lands
    at *second*: ``"teardown"`` is inside ``_RWLock._teardown``'s
    repair loop, where the two stops and the interrupted frames end up
    in a reference cycle that keeps the cursor and its read snapshot
    alive (garbage collection is off, as on any thread between
    collections); ``"rollback"`` is on the way to ``_immediate_txn``'s
    ``ROLLBACK``, which the stop would skip.  Either way the thread's
    next ``BEGIN IMMEDIATE`` must go through, instead of "database is
    locked" (``SQLITE_BUSY_SNAPSHOT``) or "cannot start a transaction
    within a transaction".
    """
    _redirect_home()
    driver = _TaskRow()
    rounds = _OwnRounds(driver)  # the worker's rounds alone: the peer shares the driver
    worker = Hammered(rounds)
    fired = [False, False]
    rounds_at_stop = [0, 0]

    def stop_now(index: int) -> bool:
        ready = rounds.completed > 2 if index == 0 else fired[0]
        if ready and threading.current_thread() is worker and not fired[index]:
            rounds_at_stop[index] = rounds.completed
            fired[index] = True
            return True
        return False

    def recovered() -> bool:
        """The worker itself completed 100 rounds after the second stop."""
        return fired[1] and rounds.completed >= rounds_at_stop[1] + 100

    original_fetchone = persistence._LockedCursor.fetchone
    original_drop = persistence._RWLock._drop_writer_ref
    original_execute = persistence._LockedConnection.execute

    def fetchone_with_first_stop(self: persistence._LockedCursor) -> Any:
        if stop_now(0):
            raise KeyboardInterrupt
        return original_fetchone(self)

    def drop_with_second_stop(self: persistence._RWLock, ref: Any) -> None:
        if second == "teardown" and stop_now(1):
            raise KeyboardInterrupt
        original_drop(self, ref)

    def execute_with_second_stop(
        conn: persistence._LockedConnection, sql: str, parameters: Any = (), /,
    ) -> Any:
        if second == "rollback" and sql == "ROLLBACK" and stop_now(1):
            raise KeyboardInterrupt
        return original_execute(conn, sql, parameters)

    persistence._LockedCursor.fetchone = fetchone_with_first_stop  # type: ignore[method-assign]
    persistence._RWLock._drop_writer_ref = drop_with_second_stop  # type: ignore[method-assign]
    persistence._LockedConnection.execute = execute_with_second_stop  # type: ignore[method-assign]
    unexpected = _UnexpectedErrors()
    threading.excepthook = unexpected
    gc.collect()
    gc.disable()
    peer = _Peer(driver)
    peer.start()
    worker.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and worker.is_alive() and not recovered():
        time.sleep(0.01)
    worker.stop.set()
    peer.stop.set()
    worker.join(20.0)
    peer.join(20.0)
    gc.enable()
    assert fired == [True, True], fired
    assert not unexpected.errors, unexpected.errors
    assert not worker.is_alive() and not peer.is_alive(), "a thread hung"
    assert recovered(), (rounds.completed, rounds_at_stop)


@pytest.mark.parametrize("second", ["teardown", "rollback"])
def test_a_second_stop_while_the_first_unwinds_keeps_the_connection_writable(second: str) -> None:
    """The thread writes on after two stops, the second in the lock's repair or the rollback."""
    _run_in_child(f"run_two_stops({second!r})")


def run_stop_inside_token_callback() -> None:
    """Child-process body: a stop landing in a token callback blocked on the mutex.

    ``_drop_writer_ref`` runs as a token's weakref callback on whatever
    thread dropped the token.  Blocked on the lock's mutex, that thread
    receives a pending stop the moment its ``acquire`` returns; the
    mutex must not stay its own afterwards (every later reader and
    writer would wait on it for good).
    """
    threading.excepthook = ignore_thread_interrupt
    lock = persistence._rw_lock
    token = persistence._Token()  # alive throughout: a dead plain ref cannot be hashed
    with lock._cond:
        callback = threading.Thread(
            target=lock._drop_writer_ref, args=(weakref.ref(token),), daemon=True,
        )
        callback.start()
        time.sleep(0.2)  # the callback thread is parked in ``acquire``
        assert inject(callback) == 1
    callback.join(5.0)
    assert not callback.is_alive(), "the callback thread hung"
    assert lock_is_free(lock._mutex), "the stopped callback kept the mutex"


def test_a_stop_in_a_token_callback_releases_the_mutex() -> None:
    """A token callback that takes a stop on its acquire leaves the mutex free."""
    _run_in_child("run_stop_inside_token_callback()")


def _connect(path: Path) -> persistence._LockedConnection:
    conn = persistence._LockedConnection(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _orphan_half_read_cursor(conn: sqlite3.Connection) -> None:
    """Leave a half-read cursor of *conn* reachable only through a reference cycle.

    The table must hold two rows at least: reading the last row steps
    the statement to its end, which resets it and ends the snapshot.
    """
    cursor = conn.execute("SELECT x FROM t")
    cursor.fetchone()
    holder: list[object] = [cursor]
    holder.append(holder)


def test_a_stale_snapshot_retry_replays_every_executemany_row(tmp_path: Path) -> None:
    """After an orphaned half-read cursor, ``executemany`` inserts every parameter set, once.

    SQLite takes the first set before the write that finds the snapshot
    stale, so a retry over the same iterator would skip it.
    """
    a = _connect(tmp_path / "snapshot.db")
    b = _connect(tmp_path / "snapshot.db")
    gc.disable()  # the orphan lives until the retry collects it
    try:
        a.execute("CREATE TABLE t (x INTEGER)")
        a.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
        _orphan_half_read_cursor(a)
        b.execute("INSERT INTO t VALUES (3)")  # a's snapshot is stale from here on
        a.executemany("INSERT INTO t VALUES (?)", iter([(10,), (11,)]))
        rows = a.execute("SELECT x FROM t ORDER BY x").fetchall()
        assert rows == [(1,), (2,), (3,), (10,), (11,)], rows
    finally:
        gc.enable()
        a.close()
        b.close()


def test_a_snapshot_gone_stale_after_a_written_row_is_not_replayed(tmp_path: Path) -> None:
    """An autocommit ``executemany`` failing after its first row raises; no row is written twice.

    Each parameter set commits on its own, so a commit elsewhere in
    between (here from the trace callback, right before the second
    set) makes the second one fail with ``SQLITE_BUSY_SNAPSHOT`` while
    an orphaned half-read cursor keeps the read snapshot.  A retry
    that replayed the sets once the orphan is collected would write
    the first one twice.
    """
    a = _connect(tmp_path / "snapshot.db")
    b = _connect(tmp_path / "snapshot.db")
    inserts = [0]

    def commit_on_b_before_the_second_set(sql: str) -> None:
        if sql.startswith("INSERT"):
            inserts[0] += 1
            if inserts[0] == 2:
                b.execute("INSERT INTO t VALUES (3)")

    gc.disable()
    try:
        a.execute("CREATE TABLE t (x INTEGER)")
        a.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
        _orphan_half_read_cursor(a)
        a.set_trace_callback(commit_on_b_before_the_second_set)
        with pytest.raises(sqlite3.OperationalError) as info:
            a.executemany("INSERT INTO t VALUES (?)", [(10,), (11,)])
        assert info.value.sqlite_errorcode == sqlite3.SQLITE_BUSY_SNAPSHOT
        a.set_trace_callback(None)
        gc.collect()  # frees the orphan: the read below sees b's commit
        rows = a.execute("SELECT x FROM t ORDER BY x").fetchall()
        assert rows == [(1,), (2,), (3,), (10,)], rows
    finally:
        gc.enable()
        a.close()
        b.close()


if __name__ == "__main__":
    pytest.main([__file__])

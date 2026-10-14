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
:func:`~kiss.agents.sorcar._concurrency.locked_call`.

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
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence
from kiss.tests.lock_injection import (
    INJECTIONS,
    Contender,
    Hammered,
    inject,
    lock_is_free,
)


class _Peer(threading.Thread):
    """A thread repeating *call* until stopped, keeping the shared locks contended."""

    def __init__(self, call: object) -> None:
        super().__init__(daemon=True)
        self.call = call
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            self.call()  # type: ignore[operator]


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
    for name, target in (("_rw_lock", _take_write_lock), ("_close_db", persistence._close_db)):
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


def run_hammer(name: str) -> None:
    """Child-process body: hammer the path *name*, check its writes landed and no lock leaked."""
    home = Path(tempfile.mkdtemp(prefix="audit1013-persist-"))
    os.environ["KISS_HOME"] = str(home)
    persistence._KISS_DIR = home
    persistence._DB_PATH = home / "history.db"
    persistence._db_conn = None
    driver = _DRIVERS[name]()
    _hammer_contended(driver)
    driver.verify()
    _check_connections_closable()


@pytest.mark.parametrize("path", sorted(_DRIVERS))
def test_persistence_lock_paths(path: str) -> None:
    """No persistence site a task thread runs leaks a lock under injection."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({path!r})"],
        capture_output=True, text=True, timeout=240, check=False,
    )
    assert proc.returncode == 0, f"{path}: {proc.stderr[-3000:]}"


if __name__ == "__main__":
    pytest.main([__file__])

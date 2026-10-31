# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Orphaned-sidecar recovery on the ``_get_db()`` path survives the runner's Stop.

``_recover_orphaned_sidecars_locked`` closes every open connection of
the process once it can take that connection's lock, with
``conn.lock.acquire(timeout=...)``.  Those locks belong to other
threads' handles.  The acquire used to sit outside its ``try/finally``:
a ``KeyboardInterrupt`` injected with ``PyThreadState_SetAsyncExc`` at
the check point right after the call returned (CPython gh-74174) left
the lock held by the interrupted thread for good, every later
``_get_db()`` in the process then waited the full
``_RECOVERY_IDLE_WAIT_S`` on that lock and raised, and the handle's
owner could never retire it.  The acquire now sits inside the ``try``
and the ``finally`` releases unconditionally.

The test runs the real path: a worker unlinks the ``-wal``/``-shm``
sidecars and calls ``_get_db()`` (which detects the orphaned mapping
and recovers) while hundreds of interrupts are injected into it, two
peers loop ``_get_db()`` + a statement on their own handles, and a
contender holds the open connections' locks in 1 ms bursts so the
worker's acquire blocks and receives the interrupt on return.  It runs
in a child process (a leaked lock would wedge the pytest session) and
asserts afterwards that every connection lock is free and that a
fresh ``_get_db()`` answers at once instead of stalling.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence
from kiss.tests.conftest import posix_only
from kiss.tests.lock_injection import INJECTIONS, Hammered, inject, lock_is_free


class _Recover:
    """Orphan the sidecars, then ``_get_db()``: the recovery closes every handle."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = str(db_path)
        self.completed = 0

    def __call__(self) -> None:
        for suffix in ("-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.db_path + suffix)
        try:
            persistence._get_db().execute("SELECT 1").fetchall()
        except sqlite3.Error:  # the documented fail-once of a handle closed by a peer's recovery
            return
        self.completed += 1


class _Peer(threading.Thread):
    """A thread looping ``_get_db()`` and a statement on it until stopped."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                persistence._get_db().execute("SELECT 1").fetchall()
            except sqlite3.Error:  # closed or interrupted by a recovery
                pass


class _ConnContender(threading.Thread):
    """A thread holding each registered connection's lock for 1 ms at a time."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            for conn, _owner, _path in list(persistence._open_conns.values()):
                with conn.lock:
                    time.sleep(0.001)
            time.sleep(0.0002)


class _UnexpectedErrors:
    """``threading.excepthook`` keeping every worker error that is not the injected interrupt."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def __call__(self, args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not KeyboardInterrupt:
            self.errors.append(f"{args.thread.name if args.thread else '?'}: {args.exc_value!r}")


def _probe(name: str, target: object) -> None:
    """Run *target* on a thread; it must return well within ``_RECOVERY_IDLE_WAIT_S``."""
    thread = threading.Thread(target=target, daemon=True)  # type: ignore[arg-type]
    thread.start()
    thread.join(20.0)
    assert not thread.is_alive(), f"{name} stalled on a leaked connection lock"


def _select_one() -> None:
    assert persistence._get_db().execute("SELECT 1").fetchone()[0] == 1


def run_hammer() -> None:
    """Child-process body: hammer the recovery, then check no connection lock leaked."""
    home = Path(tempfile.mkdtemp(prefix="audit1017-get-db-"))
    os.environ["KISS_HOME"] = str(home)
    persistence._KISS_DIR = home
    persistence._DB_PATH = home / "history.db"
    persistence._db_conn = None
    persistence._get_db().execute("SELECT 1").fetchall()
    driver = _Recover(persistence._DB_PATH)
    unexpected = _UnexpectedErrors()
    previous_hook = threading.excepthook
    threading.excepthook = unexpected
    gc.collect()
    gc.disable()
    helpers: list[_Peer | _ConnContender] = [_Peer(), _Peer(), _ConnContender()]
    for helper in helpers:
        helper.start()
    worker = Hammered(driver)
    worker.start()
    generation_before = persistence._db_generation
    delivered = attempts = 0
    try:
        while delivered < INJECTIONS and attempts < 10 * INJECTIONS:
            attempts += 1
            time.sleep(0.0005)
            if not worker.is_alive():
                worker = Hammered(driver)
                worker.start()
            if worker.in_call:
                delivered += inject(worker)
        for helper in helpers:
            helper.stop.set()
        for helper in helpers:
            helper.join(20.0)
        # A round (interrupt, checkpoint, close, reopen) outlasts the
        # 0.5 ms between injections; the worker completes one on its
        # own once they stop, unless a lock leaked.  The last interrupt
        # can land in the worker's loop instead of its call and end it,
        # so a fresh worker is started, as in the loop above.
        deadline = time.monotonic() + 20.0
        while driver.completed == 0 and time.monotonic() < deadline:
            if not worker.is_alive():
                worker = Hammered(driver)
                worker.start()
            time.sleep(0.01)
        worker.stop.set()
        worker.join(20.0)
    finally:
        gc.enable()
        threading.excepthook = previous_hook
    assert not worker.is_alive(), "the worker hung on a lock"
    assert not any(helper.is_alive() for helper in helpers), "a helper hung on a leaked lock"
    assert delivered > INJECTIONS // 2, "the interrupts did not reach the worker"
    assert not unexpected.errors, unexpected.errors
    assert persistence._db_generation > generation_before, "no recovery ran under the interrupts"
    assert driver.completed > 0, "the recovery never completed a round"
    assert lock_is_free(persistence._init_tables_lock), "_init_tables_lock leaked"
    leaked = [
        key for key, (conn, _owner, _path) in list(persistence._open_conns.items())
        if not lock_is_free(conn.lock)
    ]
    assert not leaked, f"a connection lock stayed held after the interrupts: {leaked}"
    _probe("_get_db", _select_one)
    _probe("_close_db", persistence._close_db)


@posix_only("unlinking WAL sidecars another connection holds open")
def test_recovery_acquire_survives_stop_injection() -> None:
    """A Stop landing on the recovery's ``acquire`` never leaks another thread's lock."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer()"],
        capture_output=True, text=True, timeout=240, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


if __name__ == "__main__":
    pytest.main([__file__])

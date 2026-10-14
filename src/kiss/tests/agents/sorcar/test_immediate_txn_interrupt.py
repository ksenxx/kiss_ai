# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_immediate_txn`` leaves no transaction open when a Stop interrupt lands on it.

The runner's Stop raises ``KeyboardInterrupt`` on a task thread at the
next bytecode boundary — for a thread inside ``BEGIN IMMEDIATE`` that
is the boundary right after the statement returned.  With the BEGIN
outside the ``try`` the interrupt escaped before the rollback handler
existed, the thread's connection kept SQLite's write lock, and every
other thread's next write sat in the 30 s ``busy_timeout`` (seen as
"a peer hung on a leaked lock" in the persistence hammer on macOS).

A peer thread keeps the database write-locked in 1 ms bursts so the
worker's BEGIN blocks and the injection is delivered exactly on its
return.  The database is in WAL mode like the daemon's, so a COMMIT
never waits on readers and the only contention is the write lock.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.persistence import _immediate_txn
from kiss.tests.lock_injection import INJECTIONS, Hammered, ignore_thread_interrupt, inject


def _connect(path: Path, busy_timeout_ms: int) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(path), check_same_thread=False, isolation_level=None, timeout=10,
    )
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    return conn


class _Peer(threading.Thread):
    """Holds the write lock for 1 ms at a time until stopped, like ``Contender``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(daemon=True)
        self.conn = conn
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            with _immediate_txn(self.conn):
                self.conn.execute("INSERT INTO t VALUES (1)")
                time.sleep(0.001)
            time.sleep(0.001)


def test_interrupted_transaction_is_rolled_back(tmp_path: Path) -> None:
    """Hundreds of injections into a transacting thread never leave its BEGIN open."""
    path = tmp_path / "txn.db"
    conn = _connect(path, busy_timeout_ms=10000)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (n INTEGER)")
    peer = _Peer(_connect(path, busy_timeout_ms=10000))

    def transact() -> None:
        with _immediate_txn(conn):
            conn.execute("INSERT INTO t VALUES (0)")

    previous_hook = threading.excepthook
    threading.excepthook = ignore_thread_interrupt
    peer.start()
    worker = Hammered(transact)
    worker.start()
    try:
        for _ in range(INJECTIONS):
            time.sleep(0.0005)
            if not worker.is_alive():
                assert not conn.in_transaction, "a dead worker left its transaction open"
                worker = Hammered(transact)
                worker.start()
            inject(worker)
        worker.stop.set()
        worker.join(30.0)
    finally:
        peer.stop.set()
        peer.join(30.0)
        threading.excepthook = previous_hook
    assert not worker.is_alive(), "the worker hung"
    assert not peer.is_alive(), "the peer hung on the worker's open transaction"
    assert not conn.in_transaction, "the interrupted transaction was left open"
    # The write lock is free again for anyone.
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("ROLLBACK")
    conn.close()
    peer.conn.close()


def test_interrupt_on_a_closed_connection_keeps_the_original_exception(tmp_path: Path) -> None:
    """The rollback probe of a connection closed inside the block cannot mask the stop."""
    conn = _connect(tmp_path / "closed.db", busy_timeout_ms=1000)
    with pytest.raises(KeyboardInterrupt), _immediate_txn(conn):
        conn.close()  # what the cross-thread sidecar recovery does to idle handles
        raise KeyboardInterrupt

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing inside a journal replay must not leak SQLite's write lock.

``_flush_chat_events`` runs on the task thread and, when a
``.failed_events.jsonl`` journal exists, replays it through
``_write_event_batch`` right there.  The replay's transaction used to be
a hand-written ``BEGIN IMMEDIATE`` / ``COMMIT`` whose only rollback
handler caught ``Exception``: the runner's Stop (``KeyboardInterrupt``
injected on the task thread) escaping between the two left the
transaction open on that thread's cached connection, so SQLite's write
lock stayed held for the connection's life and every other writer — in
this process and in every other Sorcar process on the database — sat in
the 30 s ``busy_timeout`` and then failed with ``database is locked``.

The interrupt is delivered deterministically with a per-thread trace
function that raises on entry to the connection's ``executemany`` —
after the ``BEGIN IMMEDIATE`` and the marker insert, before the
``COMMIT`` — the same check point ``PyThreadState_SetAsyncExc`` can
hit.  The journal, the replay, the connection and the second writer are
all real.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th

_EXECUTEMANY_CODE = th._LockedConnection.executemany.__code__


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the persistence module at a fresh database under *tmp_path*."""
    home = tmp_path / ".kiss"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    th._KISS_DIR = home
    th._DB_PATH = home / "history.db"
    th._db_conn = None
    try:
        yield th._DB_PATH
    finally:
        th._close_db()
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _interrupt_entering_executemany(frame: Any, event: str, arg: Any) -> None:
    """Trace function: raise ``KeyboardInterrupt`` entering ``executemany`` once.

    A trace function that raises is unset by the interpreter, so it
    fires for the first ``executemany`` only — the replay's event
    insert, inside its open ``BEGIN IMMEDIATE``.
    """
    if event == "call" and frame.f_code is _EXECUTEMANY_CODE:
        raise KeyboardInterrupt("stop injected inside the replay transaction")
    return None


class _InterruptedReplayer(threading.Thread):
    """The task thread: flushes (and so replays the journal) under the injected Stop."""

    def __init__(self, task_id: str) -> None:
        super().__init__(daemon=True)
        self.task_id = task_id
        self.interrupted = False
        self.in_transaction: bool | None = None
        self.result_saved = False
        self.replay_done = threading.Event()
        self.release = threading.Event()

    def run(self) -> None:
        sys.settrace(_interrupt_entering_executemany)
        try:
            th._flush_chat_events(self.task_id)
        except KeyboardInterrupt:
            self.interrupted = True
        finally:
            sys.settrace(None)
        self.in_transaction = th._get_db().in_transaction
        self.replay_done.set()
        # Stay alive like a real task thread (its connection stays cached
        # and registered) while the other writer is timed.
        self.release.wait(60)
        # The runner's cleanup saves the stopped task's result on this
        # same connection; with a transaction left open it fails with
        # "cannot start a transaction within a transaction".
        th._save_task_result("stopped", self.task_id)
        self.result_saved = True


def test_stop_inside_journal_replay_leaves_no_open_transaction(db_path: Path) -> None:
    task_id, chat_id = th._add_task("replay under stop")
    batch = [(task_id, json.dumps({"type": "line", "text": "journalled"}),
              time.time(), str(db_path))]
    th._journal_failed_events(batch, 1)
    assert th._journal_has_pending_rows(th._failed_events_path(str(db_path)))

    replayer = _InterruptedReplayer(task_id)
    replayer.start()
    assert replayer.replay_done.wait(30)
    assert replayer.interrupted, "the trace function did not reach the replay's insert"
    assert replayer.in_transaction is False, (
        "the interrupted replay left its transaction open on the task thread"
    )

    # Another writer (any thread, any process) must not wait on the
    # task thread's leaked write lock: 30 s busy timeout, then failure.
    started = time.monotonic()
    assert th._set_chat_summary(chat_id, "title")
    assert time.monotonic() - started < 5.0

    # The task thread's own cleanup write succeeds on the same connection.
    replayer.release.set()
    replayer.join(30)
    assert not replayer.is_alive()
    assert replayer.result_saved

    # The kept snapshot was replayed by the first flush after the stop
    # (the result save's), exactly once, and nothing of it is left behind.
    th._flush_chat_events(task_id)
    rows = th._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ?", (task_id,),
    ).fetchall()
    assert [json.loads(r[0])["text"] for r in rows] == ["journalled"]
    assert not th._journal_has_pending_rows(th._failed_events_path(str(db_path)))
    assert th._get_db().execute("SELECT COUNT(*) FROM replayed_journals").fetchone()[0] == 0

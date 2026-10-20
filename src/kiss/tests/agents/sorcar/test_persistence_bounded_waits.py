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
(writer thread) keeps waiting, because its rows must land in the live
sidecar in order.

Also here: the metadata backfill honours the ``success`` column the
live finalizer honours, so ``--refresh`` reproduces the daemon's tags.

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

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.history_columns import (
    ADDED_TASK_COLUMNS,
    TASK_TABLE,
    missing_columns,
)
from kiss.agents.sorcar.task_metadata import backfill_task_metadata

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
        persistence._close_db()
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
        assert len(persistence._final_results_spills(str(sidecar))) == 1
        assert persistence._load_final_results(str(sidecar)) == {
            "t1": {"result": "done", "success": 1},
        }
    finally:
        child.kill()
        child.wait()
    persistence._prune_final_results_journal(str(sidecar))
    assert persistence._final_results_spills(str(sidecar)) == []
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
persistence._close_db()
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
    assert persistence._final_results_spills(str(sidecar)) == []
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
    assert sidecar.exists() and len(persistence._final_results_spills(str(sidecar))) == 1
    assert persistence._load_final_results(str(sidecar)) == {
        "t1": {"result": "second", "success": 1},
    }
    persistence._prune_final_results_journal(str(sidecar))
    assert persistence._load_final_results(str(sidecar)) == {
        "t1": {"result": "second", "success": 1},
    }


def test_a_batch_journalled_during_a_slow_replay_is_still_replayed(kiss_home: Path) -> None:
    """The failed-events append waits for the replayer's lock however long
    it takes: a batch written into the sidecar a replayer is consuming
    would be unlinked with it."""
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

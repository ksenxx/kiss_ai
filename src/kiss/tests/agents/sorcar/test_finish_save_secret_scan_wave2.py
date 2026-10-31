# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The finish save scans a task's trajectory for secrets before it locks the database.

``_save_task_extra`` with an ``endTs`` derives the finished task's tags,
and the ``secret`` tag needs a regex scan of every persisted event of the
task (seconds for a long task).  The scan ran inside the save's
``_rw_lock.write_lock()`` and ``BEGIN IMMEDIATE`` transaction, so for its
whole duration every reader and writer of the daemon, and every other
process writing ``history.db``, waited.  The tests probe both locks at
the moment the scan's ``SELECT`` starts, and check that the verdict
still reaches the ``tags`` column.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from functools import partial

import kiss.agents.sorcar.persistence as th

_SCAN_PREFIX = "SELECT event_json FROM events"
_SCAN_SUFFIX = "ORDER BY seq, id"
_TOKEN = "ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


def _insert_events(task_id: str, contents: list[str]) -> None:
    """Persist one ``tool_result`` event per entry of *contents* for *task_id*."""
    with closing(sqlite3.connect(th._DB_PATH)) as conn, conn:
        for seq, content in enumerate(contents):
            conn.execute(
                "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
                (task_id, seq, json.dumps({"type": "tool_result", "content": content}), 1.0),
            )


def _read_lock_once(done: threading.Event) -> None:
    """Take and release the daemon's read lock, then set *done*."""
    with th._rw_lock.read_lock():
        done.set()


def _probe_locks(other: sqlite3.Connection, seen: list[str], statement: str) -> None:
    """Trace callback: when the secret scan starts, record which locks are free.

    Records ``writer free``/``writer locked`` for another connection's
    ``BEGIN IMMEDIATE`` (another process) and ``reader free``/``reader
    blocked`` for a thread taking ``_rw_lock.read_lock()`` (the daemon's
    history panel, transcript loads).
    """
    if not (statement.startswith(_SCAN_PREFIX) and statement.endswith(_SCAN_SUFFIX)):
        return
    try:
        other.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        seen.append("writer locked")
    else:
        other.execute("ROLLBACK")
        seen.append("writer free")
    done = threading.Event()
    reader = threading.Thread(target=_read_lock_once, args=(done,), daemon=True)
    reader.start()
    seen.append("reader free" if done.wait(2.0) else "reader blocked")


def _tags(task_id: str) -> list[str]:
    """Return the stored ``tags`` of *task_id*."""
    with closing(sqlite3.connect(th._DB_PATH)) as conn:
        row = conn.execute("SELECT tags FROM task_history WHERE id = ?", (task_id,)).fetchone()
    return str(row[0] or "").split(",")


def _finish_traced(task_id: str | None, extra: dict[str, object]) -> list[str]:
    """Run ``_save_task_extra(extra, task_id)`` with the lock probe; return what it saw."""
    seen: list[str] = []
    db = th._get_db()
    with closing(sqlite3.connect(th._DB_PATH, timeout=0, isolation_level=None)) as other:
        db.set_trace_callback(partial(_probe_locks, other, seen))
        try:
            th._save_task_extra(extra, task_id)
        finally:
            db.set_trace_callback(None)
    return seen


def test_secret_scan_runs_with_both_locks_free_and_tags_the_task() -> None:
    task_id, _ = th._add_task("list the files")
    _insert_events(task_id, ["x" * 200] * 20 + [f"export GITHUB_TOKEN={_TOKEN}"])
    th._save_task_result("done", task_id, success=True)

    seen = _finish_traced(task_id, {"endTs": 1_900_000_000_000, "steps": 3})

    assert seen == ["writer free", "reader free"]
    assert "secret" in _tags(task_id)


def test_a_clean_trajectory_is_scanned_once_and_not_tagged_secret() -> None:
    task_id, _ = th._add_task("list the files")
    _insert_events(task_id, ["plain output with no credential"] * 5)
    th._save_task_result("done", task_id, success=True)

    seen = _finish_traced(task_id, {"endTs": 1_900_000_000_000})

    assert seen == ["writer free", "reader free"]
    assert _tags(task_id) == ["work"]


def test_saves_without_end_ts_or_without_a_row_scan_nothing() -> None:
    task_id, _ = th._add_task("list the files")
    _insert_events(task_id, [f"token {_TOKEN}"])

    assert _finish_traced(task_id, {"steps": 2}) == []
    assert _finish_traced(None, {"endTs": 1_900_000_000_000}) == []
    assert _finish_traced("0" * 32, {"endTs": 1_900_000_000_000}) == []
    with closing(sqlite3.connect(th._DB_PATH)) as conn:
        row = conn.execute(
            "SELECT steps, end_ts, tags FROM task_history WHERE id = ?", (task_id,)
        ).fetchone()
    assert row == (2, 0, "")

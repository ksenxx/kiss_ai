# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S8 F1/F4: the metadata backfill against a live daemon's writes.

* The chat-summary batch of ``backfill_task_metadata`` ran under
  ``with db:``, which opens no transaction before ``upsert_chat_summary``'s
  SELECTs.  A task finishing on another connection between those reads
  and the UPSERT had its fresh ``last_finished`` replaced by the stale
  0, and a chat deleted between the two SELECTs crashed the backfill
  with ``TypeError``.  Each batch now runs in one ``BEGIN IMMEDIATE``
  transaction, so the daemon's write lands before or after the whole
  batch.
* The SEA backfill wrote every inferred value unconditionally, over a
  ``sea`` a live ``_save_task_extra`` had just supplied.  The UPDATE now
  fills an empty column only.

Two real SQLite connections each time (the daemon's write on its own
thread or through the real persistence API); an SQLite trace callback
on the backfill's connection schedules the write at the statement the
race needs.  Nothing is mocked.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from functools import partial
from pathlib import Path

from kiss.agents.sorcar.chat_summary import upsert_chat_summary
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _flush_chat_events,
    _get_db,
    _immediate_txn,
    _save_task_extra,
)
from kiss.agents.sorcar.task_metadata import backfill_task_metadata
from kiss.tests.agents.sorcar.test_task_metadata_sweep1030 import _insert_task, _new_db
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

FINISHED_AT = 200_000


def _finish_task(db_path: Path) -> None:
    """The daemon's finalisation of the running task ``task``, on its own connection."""
    with closing(sqlite3.connect(db_path, timeout=30, isolation_level=None)) as other:
        with _immediate_txn(other):
            other.execute("UPDATE task_history SET end_ts = ? WHERE id = 'task'", (FINISHED_AT,))
            upsert_chat_summary(other, "chat")


def _delete_chat(db_path: Path) -> None:
    """The daemon's deletion of chat ``chat``, on its own connection."""
    with closing(sqlite3.connect(db_path, timeout=30, isolation_level=None)) as other:
        with _immediate_txn(other):
            other.execute("DELETE FROM task_history WHERE chat_id = 'chat'")


def _write_at(
    prefix: str, write: Callable[[], None], writers: list[threading.Thread], statement: str,
) -> None:
    """Trace callback: at the backfill's first *prefix* statement, start *write* and wait a second.

    A backfill holding the write lock keeps the writer waiting past
    that second; it then lands once the batch has committed.
    """
    if writers or not statement.startswith(prefix):
        return
    writer = threading.Thread(target=write, daemon=True)
    writers.append(writer)
    writer.start()
    writer.join(1.0)


def _backfill_racing(
    db_path: Path, prefix: str, write: Callable[[], None],
) -> tuple[dict[str, int], sqlite3.Connection]:
    """Backfill a chat whose one task is running while *write* lands at the *prefix* statement."""
    conn = _new_db(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    _insert_task(conn, "task", 100.0, "fix the issue in the task parser", "chat")
    conn.execute("UPDATE task_history SET start_ts = 100000, end_ts = 0 WHERE id = 'task'")
    conn.commit()
    writers: list[threading.Thread] = []
    conn.set_trace_callback(partial(_write_at, prefix, write, writers))
    try:
        counts = backfill_task_metadata(conn)
    finally:
        conn.set_trace_callback(None)
        for writer in writers:
            writer.join(30)
    assert writers and not writers[0].is_alive(), "the daemon's write never ran"
    return counts, conn


def test_a_task_finishing_during_the_summary_batch_keeps_its_completion(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    counts, conn = _backfill_racing(
        db_path, "INSERT INTO chat_summaries", partial(_finish_task, db_path),
    )
    with closing(conn):
        (end_ts,) = conn.execute("SELECT end_ts FROM task_history WHERE id = 'task'").fetchone()
        (last_finished,) = conn.execute(
            "SELECT last_finished FROM chat_summaries WHERE chat_id = 'chat'"
        ).fetchone()
    assert counts["chats"] == 1
    assert end_ts == FINISHED_AT
    assert last_finished == FINISHED_AT


def test_a_chat_deleted_during_the_summary_batch_does_not_crash_the_backfill(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "history.db"
    counts, conn = _backfill_racing(
        db_path, "SELECT start_ts, timestamp, end_ts", partial(_delete_chat, db_path),
    )
    with closing(conn):
        (tasks,) = conn.execute("SELECT COUNT(*) FROM task_history").fetchone()
    assert counts["chats"] == 1
    assert tasks == 0


def _live_writer_supplies_sea(child: str, seen: list[str], statement: str) -> None:
    """Trace callback: at the backfill's SEA UPDATE, the finishing child saves its own SEA."""
    if seen or not statement.startswith("UPDATE task_history SET sea ="):
        return
    _save_task_extra({"sea": "authoritative_live_sea"}, task_id=child)
    (sea,) = _get_db().execute("SELECT sea FROM task_history WHERE id = ?", (child,)).fetchone()
    seen.append(sea)


def test_sea_backfill_keeps_the_sea_a_live_writer_supplied() -> None:
    home = IsolatedKissHome("audit1101-s8-sea-")
    try:
        parent, _ = _add_task("parent task")
        # The call is stamped before the child it launches, as in a real run.
        _append_chat_event(
            {
                "type": "tool_call",
                "name": "run_agent",
                "extras": {"agent": "/tmp/dispatched_alias.py", "task": "do exact work"},
            },
            task_id=parent,
        )
        child, _ = _add_task("do exact work", extra={"subagent": {"parent_task_id": parent}})
        _flush_chat_events()
        seen: list[str] = []
        with closing(sqlite3.connect(home.kiss_home / "history.db")) as db:
            db.set_trace_callback(partial(_live_writer_supplies_sea, child, seen))
            try:
                counts = backfill_task_metadata(db)
            finally:
                db.set_trace_callback(None)
            (sea,) = db.execute("SELECT sea FROM task_history WHERE id = ?", (child,)).fetchone()
        assert seen == ["authoritative_live_sea"]
        assert sea == "authoritative_live_sea"
        assert counts["sea"] == 0
    finally:
        home.cleanup()

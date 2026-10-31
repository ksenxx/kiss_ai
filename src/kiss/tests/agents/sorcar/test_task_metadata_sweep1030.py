# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the derived task metadata writers on hostile databases.

* A BLOB in the TEXT ``task`` column (a hand-edited or third-party
  database, the case ``persistence._safe_str`` defends against) made
  ``upsert_chat_summary`` and ``infer_subagent_seas`` run ``str`` code on
  ``bytes``: the ``TypeError`` rolled back the finished task's
  ``end_ts`` / ``steps`` / ``cost`` in the daemon's finish save and
  aborted ``backfill_task_metadata``.
* ``backfill_task_metadata`` scanned a batch's trajectories for secrets
  after the batch's first ``UPDATE`` had opened the write transaction,
  so a live daemon sharing the database could not write during the
  scans of the rest of the batch.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from functools import partial
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_summary import summarize_chat
from kiss.agents.sorcar.task_metadata import backfill_task_metadata


def _new_db(path: Path) -> sqlite3.Connection:
    """Open a fresh ``history.db`` at *path* with the current schema."""
    conn = sqlite3.connect(path)
    th._init_tables(conn)
    conn.commit()
    return conn


def _insert_task(
    conn: sqlite3.Connection, task_id: str, ts: float, task: object, chat: str, parent: str = "",
) -> None:
    """Insert a ``task_history`` row whose ``task`` column holds *task* as given."""
    conn.execute(
        "INSERT INTO task_history (id, timestamp, task, chat_id, parent_task_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, ts, task, chat, parent),
    )


def _insert_event(conn: sqlite3.Connection, task_id: str, seq: int, event: dict) -> None:
    """Persist one event of *task_id*."""
    conn.execute(
        "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
        (task_id, seq, json.dumps(event), 100.0 + seq),
    )


def _set_task_text(task_id: str, task: object) -> None:
    """Store *task* as given in the ``task`` column of *task_id* of the live database."""
    with closing(sqlite3.connect(th._DB_PATH)) as conn, conn:
        conn.execute("UPDATE task_history SET task = ? WHERE id = ?", (task, task_id))


def test_finish_save_in_a_chat_with_a_blob_task_stores_the_usage_and_summary() -> None:
    first_id, chat_id = th._add_task("placeholder")
    second_id, _ = th._add_task("add a logout button", chat_id=chat_id)
    _set_task_text(first_id, b"fix the login bug in the portal")
    try:
        th._save_task_extra({"endTs": 1_900_000_000_000, "steps": 4, "cost": 0.25}, second_id)

        with closing(sqlite3.connect(th._DB_PATH)) as conn:
            row = conn.execute(
                "SELECT end_ts, steps, cost FROM task_history WHERE id = ?", (second_id,)
            ).fetchone()
            summary = conn.execute(
                "SELECT summary, last_finished FROM chat_summaries WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
    finally:
        # The database outlives this test: a BLOB left in it would be
        # the later tests' problem (their ``getInputHistory`` replies
        # carried it), not the hostile-row case this test is about.
        _set_task_text(first_id, "fix the login bug in the portal")
    assert row == (1_900_000_000_000, 4, 0.25)
    expected = summarize_chat(["fix the login bug in the portal", "add a logout button"])
    assert summary == (expected, 1_900_000_000_000)
    assert expected.startswith("fix the login bug")


def test_backfill_of_blob_task_texts_tags_infers_and_summarises(tmp_path: Path) -> None:
    with closing(_new_db(tmp_path / "history.db")) as conn:
        _insert_task(conn, "p1", 100.0, b"write the paper on routing", "chatA")
        _insert_task(
            conn, "c1", 101.0, b"PREAMBLE: draft section 2 of the paper", "chatA", "p1",
        )
        _insert_task(conn, "c2", 102.0, b"\xff\xfe not utf-8 at all", "chatA", "p1")
        dispatch = {
            "type": "tool_call",
            "name": "run_agent",
            "callId": 1,
            "extras": {"agent": "/x/draft_sea.py", "task": "draft section 2 of the paper"},
        }
        _insert_event(conn, "p1", 0, dispatch)
        conn.commit()

        counts = backfill_task_metadata(conn)

        rows = {r[0]: r[1:] for r in conn.execute("SELECT id, sea, tags FROM task_history")}
        summary = conn.execute(
            "SELECT summary FROM chat_summaries WHERE chat_id = 'chatA'"
        ).fetchone()
    assert counts == {"tags": 3, "sea": 1, "chats": 1}
    assert rows["c1"][0] == "draft_sea" and rows["c2"][0] == ""
    assert "paper" in rows["p1"][1].split(",")
    assert summary == (summarize_chat(["write the paper on routing"]),)


def _probe_other_writer(other: sqlite3.Connection, seen: list[str], statement: str) -> None:
    """Trace callback: at each trajectory scan, record whether *other* could write now."""
    if not (statement.startswith("SELECT event_json FROM events")
            and statement.endswith("ORDER BY seq, id")):
        return
    try:
        other.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        seen.append("locked")
        return
    other.execute("ROLLBACK")
    seen.append("free")


def test_backfill_scans_trajectories_outside_its_write_transaction(tmp_path: Path) -> None:
    """While the backfill reads a task's events, another process can still write."""
    db_path = tmp_path / "history.db"
    with closing(_new_db(db_path)) as conn:
        for i in range(3):
            _insert_task(conn, f"t{i}", 100.0 + i, f"task {i}", f"chat{i}")
            for seq in range(20):
                _insert_event(conn, f"t{i}", seq, {"type": "tool_result", "content": "x" * 200})
        conn.commit()
        seen: list[str] = []
        with closing(sqlite3.connect(db_path, timeout=0, isolation_level=None)) as other:
            conn.set_trace_callback(partial(_probe_other_writer, other, seen))
            counts = backfill_task_metadata(conn)
            conn.set_trace_callback(None)
        tags = [r[0] for r in conn.execute("SELECT tags FROM task_history ORDER BY id")]
    assert counts["tags"] == 3
    assert seen == ["free", "free", "free"]
    assert all(tags)

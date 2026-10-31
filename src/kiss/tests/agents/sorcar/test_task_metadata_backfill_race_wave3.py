# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 1, R4 finding 1: a backfill never installs a stale verdict over a live writer's.

``backfill_task_metadata`` scans a batch's trajectories before its write
transaction, then wrote every row's tags from those scans.  A task that
finished meanwhile (the daemon wrote a secret-bearing event and the
``secret`` tag) lost the tag to the backfill's earlier verdict, and a
running task's new secret event was never looked at again.  A row is
now written only while it is still untagged and has no event newer
than its scan; a skipped untagged row is left to the next backfill.

The daemon's write is a second real connection, committed while the
backfill reads the next task's events (an SQLite trace callback on the
backfill's connection, the hook the scan-outside-the-lock test uses).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from functools import partial
from pathlib import Path

from kiss.agents.sorcar.task_metadata import backfill_task_metadata
from kiss.tests.agents.sorcar.test_task_metadata_sweep1030 import (
    _insert_event,
    _insert_task,
    _new_db,
)

# A fake token in the shape ``has_secret_value`` matches.
SECRET_EVENT = {
    "type": "tool_result",
    "content": "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789",
}


def _daemon_writes(
    other: sqlite3.Connection, writes: list[str], done: list[bool], statement: str,
) -> None:
    """Trace callback: once the backfill reads ``t3``'s events, commit *writes* for ``t2``."""
    if done or not statement.startswith("SELECT event_json FROM events") or "'t3'" not in statement:
        return
    done.append(True)
    other.execute("BEGIN IMMEDIATE")
    _insert_event(other, "t2", 1, SECRET_EVENT)
    for write in writes:
        other.execute(write)
    other.execute("COMMIT")


def _backfill_racing(db_path: Path, writes: list[str]) -> tuple[dict[str, int], dict[str, str]]:
    """Backfill three untagged tasks while the daemon commits *writes* for ``t2``."""
    with closing(_new_db(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        for i in (1, 2, 3):
            _insert_task(conn, f"t{i}", 100.0 + i, f"fix bug {i} in the parser", f"chat{i}")
            _insert_event(conn, f"t{i}", 0, {"type": "tool_result", "content": "all tests pass"})
        conn.commit()
        done: list[bool] = []
        with closing(sqlite3.connect(db_path, isolation_level=None)) as other:
            conn.set_trace_callback(partial(_daemon_writes, other, writes, done))
            counts = backfill_task_metadata(conn)
            conn.set_trace_callback(None)
        assert done, "the daemon never wrote"
        tags = dict(conn.execute("SELECT id, COALESCE(tags, '') FROM task_history").fetchall())
    return counts, tags


def test_a_task_finishing_during_the_backfill_keeps_its_secret_tag(tmp_path: Path) -> None:
    counts, tags = _backfill_racing(
        tmp_path / "history.db",
        ["UPDATE task_history SET tags = 'work,secret,coding' WHERE id = 't2'"],
    )
    assert tags["t2"] == "work,secret,coding"
    assert counts["tags"] == 2
    assert tags["t1"] and tags["t3"] and "secret" not in tags["t1"] + tags["t3"]


def test_a_task_whose_events_changed_during_the_backfill_is_scanned_again(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    counts, tags = _backfill_racing(db_path, [])
    assert tags["t2"] == ""
    assert counts["tags"] == 2
    with closing(sqlite3.connect(db_path)) as conn:
        assert backfill_task_metadata(conn)["tags"] == 1
        (again,) = conn.execute("SELECT tags FROM task_history WHERE id = 't2'").fetchone()
    assert "secret" in again.split(","), again

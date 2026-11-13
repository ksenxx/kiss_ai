# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.scripts.sync_db``.

Every test drives the real command-line entry point against real SQLite
database files that carry the production ``history.db`` schema, then reads
the resulting files back with plain SQL.  Nothing is mocked.
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.scripts import sync_db
from kiss.tests.conftest import posix_only

REMOTE_HOST = os.environ.get("KISS_SYNC_TEST_HOST", "ksen@34.42.88.157")

TASK_DDL = """
CREATE TABLE task_history (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    task TEXT NOT NULL,
    has_events INTEGER DEFAULT 0,
    result TEXT DEFAULT '',
    chat_id CHAR(32) DEFAULT '',
    model TEXT DEFAULT '',
    work_dir TEXT DEFAULT '',
    version TEXT DEFAULT '',
    tokens INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    steps INTEGER DEFAULT 0,
    is_parallel INTEGER DEFAULT 0,
    is_worktree INTEGER DEFAULT 0,
    auto_commit_mode INTEGER DEFAULT 0,
    start_ts INTEGER DEFAULT 0,
    end_ts INTEGER DEFAULT 0,
    is_favorite INTEGER DEFAULT 0,
    parent_task_id TEXT DEFAULT ''
);
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES task_history(id),
    seq INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    timestamp REAL NOT NULL
);
CREATE TABLE model_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL UNIQUE,
    count INTEGER DEFAULT 0,
    is_last INTEGER DEFAULT 0
);
CREATE TABLE frequent_tasks (
    task TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0,
    timestamp REAL NOT NULL DEFAULT 0
);
CREATE INDEX idx_ev_task_id ON events(task_id);
"""

UNIQUE_EVENT_INDEX = "CREATE UNIQUE INDEX idx_ev_task_seq ON events(task_id, seq)"

REMOTE_SCHEMA_SCRIPT = (
    "import sqlite3,sys\n"
    "c=sqlite3.connect(sys.argv[1])\n"
    "c.executescript(sys.stdin.read())\n"
    "c.commit()\n"
)


def make_db(path: Path, unique_event_index: bool = True) -> sqlite3.Connection:
    """Create a database with the production sorcar schema.

    Args:
        path: File to create.
        unique_event_index: Also create the unique ``(task_id, seq)``
            index that some deployed databases have.

    Returns:
        An open autocommit connection to the new database.
    """
    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript(TASK_DDL)
    if unique_event_index:
        conn.execute(UNIQUE_EVENT_INDEX)
    return conn


def add_task(
    conn: sqlite3.Connection,
    task_id: str | None,
    steps: int = 1,
    events: int = 0,
    result: str = "ok",
    end_ts: int = 0,
    chat_id: str = "",
    parent_task_id: str = "",
    task: str | None = None,
    timestamp: float = 1000.0,
    start_ts: int = 0,
) -> None:
    """Insert one task row plus a run of events for it.

    Args:
        conn: Open connection.
        task_id: Value for ``task_history.id`` (``None`` is allowed and
            exercises the NULL-primary-key edge case).
        steps: Value for ``task_history.steps``.
        events: Number of event rows to add, numbered from ``seq`` 1.
        result: Value for ``task_history.result``.
        end_ts: Value for ``task_history.end_ts``.
        chat_id: The chat the task belongs to.
        parent_task_id: Set for a sub-agent task.
        task: The task text (defaults to ``task <id>``).
        timestamp: Insertion time, epoch seconds.
        start_ts: Launch time, epoch milliseconds (0 when unknown).
    """
    values = {
        "id": task_id,
        "timestamp": timestamp,
        "task": f"task {task_id}" if task is None else task,
        "steps": steps,
        "result": result,
        "end_ts": end_ts,
        "chat_id": chat_id,
        "parent_task_id": parent_task_id,
        "start_ts": start_ts,
    }
    # Some tests build a sparse task table: write only the columns it has.
    present = {row[1] for row in conn.execute("PRAGMA table_info(task_history)")}
    columns = [name for name in values if name in present]
    conn.execute(
        f"INSERT INTO task_history ({', '.join(columns)})"
        f" VALUES ({', '.join('?' for _ in columns)})",
        [values[name] for name in columns],
    )
    append_events(conn, task_id, 1, events)


def append_events(
    conn: sqlite3.Connection, task_id: str | None, first_seq: int, count: int
) -> None:
    """Append a contiguous run of event rows for one task.

    Args:
        conn: Open connection.
        task_id: Task the events belong to.
        first_seq: ``seq`` of the first event.
        count: Number of events to append.
    """
    conn.executemany(
        "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
        [
            (task_id, seq, f'{{"n": {seq}}}', 2000.0 + seq)
            for seq in range(first_seq, first_seq + count)
        ],
    )


def sync(
    *args: str, env: dict[str, str] | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the sync_db command line.

    Args:
        *args: Command-line arguments after the module name.
        env: Environment of the process; the current one when ``None``.
        timeout: Seconds after which the run counts as hung (``TimeoutExpired``).

    Returns:
        The completed process, with text stdout and stderr.
    """
    return subprocess.run(
        [sys.executable, "-m", "kiss.scripts.sync_db", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=timeout,
    )


def rows(path: Path, sql: str) -> list[tuple]:
    """Run a query against a database file.

    Args:
        path: Database file.
        sql: Query to run.

    Returns:
        All result rows.
    """
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def task_ids(path: Path) -> list[str]:
    """List the task ids of a database in order.

    Args:
        path: Database file.

    Returns:
        Sorted ``task_history.id`` values.
    """
    return [r[0] for r in rows(path, "SELECT id FROM task_history ORDER BY id")]


def event_keys(path: Path) -> list[tuple]:
    """List the natural keys of every event row.

    Args:
        path: Database file.

    Returns:
        Sorted ``(task_id, seq)`` pairs.
    """
    return rows(path, "SELECT task_id, seq FROM events ORDER BY task_id, seq")


def test_sync_copies_tasks_and_events_only(tmp_path: Path) -> None:
    """Task and event rows are copied; other tables are left alone."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=3, events=2)
    add_task(src, "b", steps=1, events=1)
    src.execute("INSERT INTO model_usage (model, count) VALUES ('src-model', 7)")
    src.close()
    dst = make_db(target)
    dst.execute("INSERT INTO model_usage (model, count) VALUES ('dst-model', 1)")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert task_ids(target) == ["a", "b"]
    assert event_keys(target) == [("a", 1), ("a", 2), ("b", 1)]
    assert rows(target, "SELECT model FROM model_usage") == [("dst-model",)]
    assert "2 task row(s) added" in done.stdout
    assert "3 event row(s) added" in done.stdout


def test_source_is_never_modified(tmp_path: Path) -> None:
    """The source database is untouched, including its event rowids."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=2)
    src.close()
    make_db(target).close()
    before = rows(source, "SELECT * FROM events ORDER BY id")

    assert sync(str(source), str(target)).returncode == 0

    assert rows(source, "SELECT * FROM events ORDER BY id") == before
    assert task_ids(source) == ["a"]


def test_repeated_sync_is_a_no_op(tmp_path: Path) -> None:
    """A second sync of unchanged databases copies nothing."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=3)
    src.close()
    make_db(target).close()

    assert sync(str(source), str(target)).returncode == 0
    second = sync(str(source), str(target))

    assert second.returncode == 0
    assert "0 task row(s) added, 0 event row(s) added" in second.stdout
    assert event_keys(target) == [("a", 1), ("a", 2), ("a", 3)]


def test_repeated_sync_without_unique_index_does_not_duplicate(tmp_path: Path) -> None:
    """De-duplication does not depend on a unique (task_id, seq) index."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source, unique_event_index=False)
    add_task(src, "a", events=2)
    src.close()
    make_db(target, unique_event_index=False).close()

    assert sync(str(source), str(target), "--full").returncode == 0
    assert sync(str(source), str(target), "--full").returncode == 0

    assert event_keys(target) == [("a", 1), ("a", 2)]


def test_incremental_sync_ships_only_new_rows(tmp_path: Path) -> None:
    """After the first sync only new tasks and later events travel."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=2, events=2)
    add_task(src, "b", steps=5, events=4)
    src.close()
    make_db(target).close()
    assert sync(str(source), str(target)).returncode == 0

    src = sqlite3.connect(source, isolation_level=None)
    append_events(src, "a", 3, 2)
    add_task(src, "c", steps=1, events=1)
    src.close()
    second = sync(str(source), str(target))

    assert second.returncode == 0
    assert "1 task row(s) added, 3 event row(s) added" in second.stdout
    assert task_ids(target) == ["a", "b", "c"]
    assert event_keys(target) == [
        ("a", 1),
        ("a", 2),
        ("a", 3),
        ("a", 4),
        ("b", 1),
        ("b", 2),
        ("b", 3),
        ("b", 4),
        ("c", 1),
    ]


def test_an_existing_task_row_is_never_modified(tmp_path: Path) -> None:
    """A task id the target already has leaves the target's row intact.

    Task ids are unique and a task row is immutable once created, so two
    copies of one id are the same row; whatever the target recorded stays.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=80, result="differs")
    add_task(src, "b", steps=1)
    src.close()
    dst = make_db(target)
    add_task(dst, "a", steps=2, result="kept", end_ts=99)
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0
    assert rows(
        target, "SELECT id, steps, result FROM task_history ORDER BY id"
    ) == [("a", 2, "kept"), ("b", 1, "ok")]
    assert "1 task row(s) added, 0 event row(s) added" in done.stdout


def test_a_differing_copy_never_travels_again(tmp_path: Path) -> None:
    """Row contents are never compared: a known id ships nothing.

    Even ``--full``, which ships every source row, cannot overwrite a row
    the target already has.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=80, result="differs")
    src.close()
    dst = make_db(target)
    add_task(dst, "a", steps=2, result="kept")
    dst.close()

    plain = sync(str(source), str(target))
    full = sync(str(source), str(target), "--full")

    assert plain.returncode == 0 and full.returncode == 0
    assert "0 task row(s) added" in plain.stdout
    assert "0 task row(s) added" in full.stdout
    assert rows(target, "SELECT steps, result FROM task_history") == [(2, "kept")]


def test_removed_mode_flags_are_rejected(tmp_path: Path) -> None:
    """The retired ``--force`` and ``--insert-only`` flags are errors."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    make_db(source).close()
    make_db(target).close()

    for flag in ("--force", "--insert-only"):
        done = sync(str(source), str(target), flag)
        assert done.returncode == 2, done.stderr
        assert "unrecognized arguments" in done.stderr


def test_event_rowids_never_collide(tmp_path: Path) -> None:
    """Overlapping ``events.id`` values on both sides are not a conflict."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=3)
    src.close()
    dst = make_db(target)
    add_task(dst, "z", events=3)
    dst.close()
    assert rows(source, "SELECT id FROM events") == rows(target, "SELECT id FROM events")

    done = sync(str(source), str(target))

    assert done.returncode == 0
    assert event_keys(target) == [
        ("a", 1),
        ("a", 2),
        ("a", 3),
        ("z", 1),
        ("z", 2),
        ("z", 3),
    ]
    ids = [r[0] for r in rows(target, "SELECT id FROM events")]
    assert len(set(ids)) == 6


def test_null_task_ids_are_skipped(tmp_path: Path) -> None:
    """Rows with a NULL primary key are never copied or duplicated."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, None)
    add_task(src, "a", events=1)
    src.close()
    make_db(target).close()

    assert sync(str(source), str(target)).returncode == 0
    assert sync(str(source), str(target)).returncode == 0

    assert task_ids(target) == ["a"]
    assert event_keys(target) == [("a", 1)]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """``--dry-run`` reports the changes and leaves the target unchanged."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=2)
    src.close()
    make_db(target).close()

    done = sync(str(source), str(target), "--dry-run")

    assert done.returncode == 0
    assert "would add 1 task row(s) and 2 event row(s)" in done.stdout
    assert task_ids(target) == []
    assert event_keys(target) == []


def test_events_of_partially_synced_task_catch_up(tmp_path: Path) -> None:
    """Events added to an already-synced task are picked up later."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=1, events=1)
    src.close()
    make_db(target).close()
    assert sync(str(source), str(target)).returncode == 0

    src = sqlite3.connect(source, isolation_level=None)
    append_events(src, "a", 2, 2)
    src.close()
    done = sync(str(source), str(target))

    assert done.returncode == 0
    assert event_keys(target) == [("a", 1), ("a", 2), ("a", 3)]


def test_missing_databases_are_reported(tmp_path: Path) -> None:
    """A missing source or target is an error, not a fresh empty file."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    make_db(source).close()

    missing_target = sync(str(source), str(target))
    missing_source = sync(str(tmp_path / "nope.db"), str(source))

    assert missing_target.returncode == 1
    assert "database not found" in missing_target.stderr
    assert not target.exists()
    assert missing_source.returncode == 1
    assert "database not found" in missing_source.stderr


def test_same_database_is_rejected(tmp_path: Path) -> None:
    """Syncing a database onto itself is refused."""
    source = tmp_path / "src.db"
    make_db(source).close()

    done = sync(str(source), str(source))

    assert done.returncode == 1
    assert "same database" in done.stderr


def test_target_without_the_schema_is_reported(tmp_path: Path) -> None:
    """A target that lacks the expected tables produces a clear error."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    make_db(source).close()
    other = sqlite3.connect(target)
    try:
        other.execute("CREATE TABLE other (x INTEGER)")
    finally:
        other.close()

    done = sync(str(source), str(target))

    assert done.returncode == 1
    assert "task_history" in done.stderr


def test_only_events_above_the_targets_highest_seq_travel(tmp_path: Path) -> None:
    """Events are append-only, so a routine sync never looks below the highest seq.

    A hole the writer never produces is not searched for -- that is what
    keeps the sync from reading the events table -- so it stays until a
    ``--full`` sync ships everything and the merge's de-duplication fills
    it.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=4)
    src.close()
    dst = make_db(target)
    add_task(dst, "a", events=0)
    append_events(dst, "a", 1, 1)
    append_events(dst, "a", 3, 1)
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert event_keys(target) == [("a", 1), ("a", 3), ("a", 4)]
    assert "1 event row(s) added" in done.stdout

    done = sync("--full", str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert event_keys(target) == [("a", 1), ("a", 2), ("a", 3), ("a", 4)]
    assert "1 event row(s) added" in done.stdout


def test_a_task_the_target_has_without_events_gets_them_all(tmp_path: Path) -> None:
    """A task row that arrived ahead of its events reports no highest seq."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=3)
    add_task(src, "b", events=2)
    src.close()
    dst = make_db(target)
    add_task(dst, "a", events=0)
    add_task(dst, "b", events=2)
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert event_keys(target) == [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2)]
    assert "3 event row(s) added" in done.stdout


def test_events_of_a_task_without_a_task_row_do_not_travel(tmp_path: Path) -> None:
    """``events.task_id`` references ``task_history``; orphans are never looked at.

    Neither for a target without the task nor, above the events it has,
    for a target that holds a task of that id.
    """
    source, empty, holding = (tmp_path / name for name in ("src.db", "empty.db", "holding.db"))
    src = make_db(source)
    add_task(src, "a", events=1)
    append_events(src, "orphan", 1, 2)
    src.close()
    make_db(empty).close()
    dst = make_db(holding)
    add_task(dst, "orphan", events=1)
    dst.close()

    for target, expected in ((empty, [("a", 1)]), (holding, [("a", 1), ("orphan", 1)])):
        done = sync(str(source), str(target))

        assert done.returncode == 0, done.stderr
        assert event_keys(target) == expected


CRON_MARKER = "You are running as an unattended scheduled automation (cron job)."
"""First sentence of the cron preamble (``history_columns.CRON_TASK_MARKER``)."""


def column_info(path: Path, table: str) -> dict[str, tuple[str, int, str | None]]:
    """Map each column of a table to its ``(type, notnull, default)`` declaration."""
    return {
        row[1]: (row[2], row[3], row[4])
        for row in rows(path, f"PRAGMA table_info({table})")
    }


def upgrade(conn: sqlite3.Connection) -> None:
    """Give a task table the columns a later release added (as the daemon does)."""
    conn.execute("ALTER TABLE task_history ADD COLUMN success INTEGER")
    conn.execute("ALTER TABLE task_history ADD COLUMN is_cron INTEGER DEFAULT 0")
    conn.execute("ALTER TABLE task_history ADD COLUMN machine TEXT DEFAULT ''")


def test_a_target_from_an_older_release_gains_the_source_columns(tmp_path: Path) -> None:
    """The deploy syncs before it upgrades the remote, so the target's table
    is one release behind the source's: it gets the source's columns, with
    the source's declarations, and the incoming rows keep their values."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    upgrade(src)
    add_task(src, "a", events=1)
    src.execute("UPDATE task_history SET success = 1, is_cron = 1 WHERE id = 'a'")
    add_task(src, "b", task="plain new task")
    src.execute("UPDATE task_history SET success = 0 WHERE id = 'b'")
    src.close()
    dst = make_db(target)
    # Rows the target wrote before the column existed: a cron run and a
    # sub-task of it, recognisable only by their traces.
    add_task(dst, "old-cron", task=CRON_MARKER + " Check the mail.")
    add_task(dst, "old-child", task="look at the inbox", parent_task_id="old-cron")
    add_task(dst, "old-plain", task="an ordinary old task")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert "column(s) success, is_cron, machine added to task_history" in done.stdout
    assert "is_cron worked out for the rows written without" in done.stdout
    info = column_info(target, "task_history")
    assert info["success"] == ("INTEGER", 0, None)
    assert info["is_cron"] == ("INTEGER", 0, "0")
    assert column_info(target, "task_history") == column_info(source, "task_history")
    assert rows(
        target, "SELECT id, success, is_cron FROM task_history ORDER BY id"
    ) == [
        ("a", 1, 1),
        ("b", 0, 0),
        ("old-child", None, 1),
        ("old-cron", None, 1),
        ("old-plain", None, 0),
    ]
    assert event_keys(target) == [("a", 1)]
    # The databases now share a schema: the next sync has nothing to do.
    again = sync(str(source), str(target))
    assert again.returncode == 0, again.stderr
    assert "0 task row(s) added" in again.stdout and "column(s)" not in again.stdout


def test_rows_from_an_older_release_get_the_columns_worked_out(tmp_path: Path) -> None:
    """Pulling from a remote still on the older release: its rows lack the
    target's newer columns, which take their defaults -- except that the
    daemon's backfill decides ``is_cron`` for them, as it would have when
    adding the column, so a cron run from the remote is not filed as a
    plain task for good (task rows are never updated once present)."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "remote-cron", task=CRON_MARKER + " Check the mail.", events=2)
    add_task(src, "remote-child", task="a sub-task", parent_task_id="remote-cron")
    add_task(src, "remote-rundir", task="/good_morning")
    src.execute(
        "UPDATE task_history SET work_dir = '/home/u/.kiss/cron/runs/good-abc'"
        " WHERE id = 'remote-rundir'"
    )
    add_task(src, "remote-plain", task="an ordinary task")
    src.close()
    dst = make_db(target)
    upgrade(dst)
    add_task(dst, "local", task=CRON_MARKER + " Already stamped?")
    dst.execute("UPDATE task_history SET success = 1 WHERE id = 'local'")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert "4 task row(s) added" in done.stdout
    assert "column(s)" not in done.stdout
    assert "is_cron worked out for the rows written without" in done.stdout
    assert rows(
        target, "SELECT id, success, is_cron FROM task_history ORDER BY id"
    ) == [
        ("local", 1, 1),
        ("remote-child", None, 1),
        ("remote-cron", None, 1),
        ("remote-plain", None, 0),
        ("remote-rundir", None, 1),
    ]
    assert event_keys(target) == [("remote-cron", 1), ("remote-cron", 2)]
    assert column_info(target, "task_history") != column_info(source, "task_history")


def test_each_side_may_have_a_column_the_other_lacks(tmp_path: Path) -> None:
    """Two machines on two different newer releases: the target gains the
    source's column, the source's rows take the default of the target's."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    src.execute("ALTER TABLE task_history ADD COLUMN only_source TEXT DEFAULT 's'")
    add_task(src, "a")
    src.close()
    dst = make_db(target)
    dst.execute("ALTER TABLE task_history ADD COLUMN only_target TEXT DEFAULT 't'")
    add_task(dst, "b")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert rows(
        target, "SELECT id, only_source, only_target FROM task_history ORDER BY id"
    ) == [("a", "s", "t"), ("b", "s", "t")]


def test_a_synced_row_keeps_the_machine_it_ran_on(tmp_path: Path) -> None:
    """A row that ran on a third machine keeps its name; this machine's own rows read ''.

    The extract stamps the source's own rows with the source's host
    name and the merge clears the stamp of rows that ran on the
    target's host; source and target share a host here, so the stamp
    is cleared -- ``test_remote_round_trip`` sees it survive.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    src.execute("ALTER TABLE task_history ADD COLUMN machine TEXT DEFAULT ''")
    add_task(src, "here")
    add_task(src, "lab-task")
    src.execute("UPDATE task_history SET machine = 'lab' WHERE id = 'lab-task'")
    src.close()
    make_db(target).close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert rows(target, "SELECT id, machine FROM task_history ORDER BY id") == [
        ("here", ""), ("lab-task", "lab"),
    ]
    assert rows(source, "SELECT id, machine FROM task_history ORDER BY id") == [
        ("here", ""), ("lab-task", "lab"),
    ]


def test_the_event_table_gains_the_source_columns_too(tmp_path: Path) -> None:
    """A column added to ``events`` travels the same way, rowid alias aside."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    src.execute("ALTER TABLE events ADD COLUMN origin TEXT NOT NULL DEFAULT 'local'")
    add_task(src, "a", events=2)
    src.execute("UPDATE events SET origin = 'replayed' WHERE seq = 2")
    src.close()
    make_db(target).close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert "column(s) origin added to events" in done.stdout
    assert column_info(target, "events")["origin"] == ("TEXT", 1, "'local'")
    assert rows(target, "SELECT seq, origin FROM events ORDER BY seq") == [
        (1, "local"),
        (2, "replayed"),
    ]


def test_a_dry_run_rolls_the_added_columns_back(tmp_path: Path) -> None:
    """A dry run reports the columns it would add and leaves the schema alone."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    upgrade(src)
    add_task(src, "a")
    src.close()
    dst = make_db(target)
    add_task(dst, "old-cron", task=CRON_MARKER + " Check the mail.")
    dst.close()
    before = column_info(target, "task_history")

    done = sync(str(source), str(target), "--dry-run")

    assert done.returncode == 0, done.stderr
    assert "column(s) success, is_cron, machine would be added to task_history" in done.stdout
    assert "is_cron would be worked out" in done.stdout
    assert column_info(target, "task_history") == before
    assert task_ids(target) == ["old-cron"]


def test_a_column_add_column_cannot_express_is_refused(tmp_path: Path) -> None:
    """A ``NOT NULL`` column without a default cannot be added to a table
    that has rows -- the daemon could not have added it either -- so the
    sync stops before touching anything rather than guess a value."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    src.execute(
        "CREATE TABLE task_history_new AS SELECT *, 'x' AS strict FROM task_history"
    )
    src.executescript(
        """
        DROP TABLE task_history;
        CREATE TABLE task_history (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            task TEXT NOT NULL,
            strict TEXT NOT NULL,
            parent_task_id TEXT DEFAULT ''
        );
        DROP TABLE task_history_new;
        """
    )
    src.execute(
        "INSERT INTO task_history (id, timestamp, task, strict) VALUES ('a', 1, 't', 'x')"
    )
    src.close()
    dst = make_db(target)
    add_task(dst, "b")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 1
    assert "NOT NULL column 'strict' without a default" in done.stderr
    assert task_ids(target) == ["b"]
    assert "strict" not in column_info(target, "task_history")


def keep_migrating(source: Path, begun: threading.Event, done: threading.Event) -> None:
    """Rename a task column of *source* every millisecond, each a new schema, until *done*."""
    conn = sqlite3.connect(source, isolation_level=None, timeout=60.0)
    try:
        number = 0
        while not done.wait(0.001):
            conn.execute(f"ALTER TABLE task_history RENAME COLUMN m{number} TO m{number + 1}")
            number += 1
            begun.set()
    finally:
        conn.close()


def test_a_source_migrated_during_the_extract_is_read_at_one_instant(tmp_path: Path) -> None:
    """The extract takes the schema and the rows of a live source from one snapshot.

    Another connection keeps changing the schema of the WAL-mode source, and a large
    manifest holds the extract between its copy of the schema and its copy of the rows:
    read apart, the rows named a column the delta's table did not have and the sync died.
    """
    source, target, manifest = tmp_path / "src.db", tmp_path / "dst.db", tmp_path / "have.gz"
    src = make_db(source)
    src.execute("PRAGMA journal_mode=WAL")
    src.execute("ALTER TABLE task_history ADD COLUMN m0 TEXT DEFAULT ''")
    add_task(src, "a", events=1)
    src.close()
    make_db(target).close()
    with open(manifest, "wb") as handle:
        sync_db.write_json_gz({"tasks": {f"t{n}": None for n in range(100_000)}}, handle)
    runner = sync_db.Runner("python3", None, [])
    ends = sync_db.Location(str(source)), sync_db.Location(str(target))
    begun, done = threading.Event(), threading.Event()
    migration = threading.Thread(target=keep_migrating, args=(source, begun, done))
    migration.start()
    try:
        assert begun.wait(60)
        sync_db.synchronize(*ends, runner, target_manifest=str(manifest))
        assert migration.is_alive()  # the schema kept changing until the sync was over
    finally:
        done.set()
        migration.join(60)
    assert not migration.is_alive()
    assert task_ids(target) == ["a"] and event_keys(target) == [("a", 1)]


def test_dry_run_reports_what_a_real_run_would_do(tmp_path: Path) -> None:
    """A dry run applies the merge rules instead of counting the delta."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=2, result="early")
    add_task(src, "b", steps=1, events=2)
    src.close()
    dst = make_db(target)
    add_task(dst, "a", steps=40, result="late", end_ts=99)
    dst.close()

    done = sync(str(source), str(target), "--dry-run")

    assert done.returncode == 0, done.stderr
    assert "would add 1 task row(s) and 2 event row(s)" in done.stdout
    assert task_ids(target) == ["a"]
    assert rows(target, "SELECT steps FROM task_history") == [(40,)]
    assert event_keys(target) == []


@posix_only("a colon cannot appear in a Windows file name (it denotes a stream)")
def test_relative_path_with_a_colon_stays_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``./a:b.db`` is a file name, not a host name."""
    monkeypatch.chdir(tmp_path)
    src = make_db(tmp_path / "s:1.db")
    add_task(src, "a", events=1)
    src.close()
    make_db(tmp_path / "t:2.db").close()

    done = sync("./s:1.db", "./t:2.db")

    assert done.returncode == 0, done.stderr
    assert task_ids(tmp_path / "t:2.db") == ["a"]


def test_duplicate_natural_keys_are_reported(tmp_path: Path) -> None:
    """Two source events sharing (task_id, seq) produce a warning."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source, unique_event_index=False)
    add_task(src, "a", events=1)
    append_events(src, "a", 1, 1)
    src.close()
    make_db(target, unique_event_index=False).close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert "shared by more than one event row" in done.stderr


def test_unexpected_constraint_failure_aborts_the_merge(tmp_path: Path) -> None:
    """A row the target rejects fails the sync instead of vanishing."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=1)
    src.close()
    dst = make_db(target)
    dst.execute(
        "CREATE TRIGGER no_events BEFORE INSERT ON events"
        " BEGIN SELECT RAISE(ABORT, 'events are frozen'); END"
    )
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 1
    assert "events are frozen" in done.stderr
    # ``synchronize``, which the peer sync calls, reports SQLite's failure as its own error.
    ends = sync_db.Location(str(source)), sync_db.Location(str(target))
    with pytest.raises(sync_db.SyncError, match="events are frozen"):
        sync_db.synchronize(*ends, sync_db.Runner("python3", None, []))
    assert task_ids(target) == []
    assert event_keys(target) == []


def test_a_hole_in_a_target_without_the_unique_index_is_filled_by_full(
    tmp_path: Path,
) -> None:
    """``--full`` fills a hole through the anti-join path too, without duplicating."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source, unique_event_index=False)
    add_task(src, "a", events=3)
    src.close()
    dst = make_db(target, unique_event_index=False)
    add_task(dst, "a")
    append_events(dst, "a", 1, 1)
    append_events(dst, "a", 1, 1)
    append_events(dst, "a", 3, 1)
    dst.close()

    done = sync("--full", str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert event_keys(target) == [("a", 1), ("a", 1), ("a", 2), ("a", 3)]


def test_partial_unique_index_on_the_target(tmp_path: Path) -> None:
    """A partial unique index cannot be an ON CONFLICT target."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source, unique_event_index=False)
    add_task(src, "a", events=2)
    src.close()
    dst = make_db(target, unique_event_index=False)
    dst.execute(
        "CREATE UNIQUE INDEX idx_part ON events(task_id, seq) WHERE seq >= 0"
    )
    dst.close()

    assert sync(str(source), str(target)).returncode == 0
    assert sync(str(source), str(target), "--full").returncode == 0

    assert event_keys(target) == [("a", 1), ("a", 2)]


def test_reversed_unique_index_on_the_target(tmp_path: Path) -> None:
    """A unique index declared (seq, task_id) is still the natural key."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source, unique_event_index=False)
    add_task(src, "a", events=1)
    append_events(src, "a", 1, 1)
    src.close()
    dst = make_db(target, unique_event_index=False)
    dst.execute("CREATE UNIQUE INDEX idx_rev ON events(seq, task_id)")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert event_keys(target) == [("a", 1)]


def test_a_minimal_schema_still_syncs(tmp_path: Path) -> None:
    """A schema with only the required columns syncs insert-only too."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    ddl = (
        "CREATE TABLE task_history (id TEXT PRIMARY KEY, task TEXT NOT NULL,"
        " result TEXT DEFAULT '');"
        " CREATE TABLE events (task_id TEXT NOT NULL, seq INTEGER NOT NULL,"
        " event_json TEXT NOT NULL, timestamp REAL NOT NULL);"
    )
    for path, note in ((source, "from the source"), (target, "from the target")):
        conn = sqlite3.connect(path, isolation_level=None)
        conn.executescript(ddl)
        conn.execute(
            "INSERT INTO task_history (id, task, result) VALUES ('a', 't', ?)", (note,)
        )
        conn.execute(
            "INSERT INTO task_history (id, task, result) VALUES (?, 't', '')",
            (f"only {note}",),
        )
        conn.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert "1 task row(s) added" in done.stdout
    assert rows(target, "SELECT result FROM task_history WHERE id = 'a'") == [
        ("from the target",)
    ]
    assert sorted(task_ids(target)) == [
        "a",
        "only from the source",
        "only from the target",
    ]


def test_without_rowid_event_table(tmp_path: Path) -> None:
    """Tables that have no rowid are copied through the direct path."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    ddl = (
        "CREATE TABLE task_history (id TEXT PRIMARY KEY, timestamp REAL NOT NULL,"
        " task TEXT NOT NULL, steps INTEGER DEFAULT 0, result TEXT DEFAULT '',"
        " end_ts INTEGER DEFAULT 0);"
        " CREATE TABLE events (task_id TEXT NOT NULL, seq INTEGER NOT NULL,"
        " event_json TEXT NOT NULL, timestamp REAL NOT NULL,"
        " PRIMARY KEY (task_id, seq)) WITHOUT ROWID;"
    )
    for path in (source, target):
        conn = sqlite3.connect(path, isolation_level=None)
        conn.executescript(ddl)
        conn.close()
    src = sqlite3.connect(source, isolation_level=None)
    add_task(src, "a", events=2)
    src.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert event_keys(target) == [("a", 1), ("a", 2)]
    assert rows(target, "SELECT event_json FROM events ORDER BY seq") == [
        ('{"n": 1}',),
        ('{"n": 2}',),
    ]


def test_column_named_rowid_is_not_confused(tmp_path: Path) -> None:
    """A real column called ``rowid`` does not break the fast path."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    for path in (source, target):
        conn = make_db(path)
        conn.execute("ALTER TABLE events ADD COLUMN rowid_ TEXT DEFAULT ''")
        conn.execute("ALTER TABLE task_history ADD COLUMN rowid_ TEXT DEFAULT ''")
        conn.close()
    src = sqlite3.connect(source, isolation_level=None)
    add_task(src, "a", events=2)
    src.execute("UPDATE events SET rowid_ = 'kept'")
    src.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert rows(target, "SELECT DISTINCT rowid_ FROM events") == [("kept",)]


def test_dry_run_needs_a_writable_target(tmp_path: Path) -> None:
    """A read-only target is reported instead of failing obscurely."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=1)
    src.close()
    make_db(target).close()
    target.chmod(0o444)
    try:
        done = sync(str(source), str(target), "--dry-run")
    finally:
        target.chmod(0o644)

    assert done.returncode == 1
    assert "cannot write to" in done.stderr
    assert "dry run needs" in done.stderr


@posix_only("limits the file size with bash's ulimit")
def test_a_target_that_cannot_grow_fails_with_the_cause(tmp_path: Path) -> None:
    """A merge the disk refuses reports the I/O error, and the target stays as it was.

    SQLite rolls the whole transaction back by itself after an I/O error, so the
    ``ROLLBACK`` issued next failed in turn (``cannot rollback - no transaction is
    active``) and that was the only thing reported.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    for number in range(12):
        add_task(src, f"s{number}", events=10, task="x" * 2000)
    src.close()
    dst = make_db(target)
    for number in range(40):
        add_task(dst, f"d{number}", events=30, task="y" * 2000)
    dst.close()
    kibibytes = target.stat().st_size // 1024 + 8  # room for 8 KiB more; the merge needs 72
    done = subprocess.run(
        ["bash", "-c", f'ulimit -f {kibibytes}; exec "$@"', "bash",
         sys.executable, "-m", "kiss.scripts.sync_db", str(source), str(target)],
        capture_output=True, text=True, check=False, timeout=120,
    )
    assert done.returncode == 1
    assert done.stderr.rstrip().endswith("sync_db: disk I/O error"), done.stderr
    assert task_ids(target) == sorted(f"d{number}" for number in range(40))


@posix_only("needs the OpenSSH client")
def test_a_host_that_never_answers_the_ssh_handshake_fails_the_sync(tmp_path: Path) -> None:
    """The real ``ssh`` gives up on a host that takes the connection and then says nothing.

    Nothing accepts on the listener, so ssh connects, sends its banner and waits for the
    host's: for ever before it had a connect deadline.  The caller's own ``-o`` still wins
    over that default, one second instead of ten.  (The deadline for a connection that dies
    later, ServerAliveInterval, needs a real sshd to test.)
    """
    source = tmp_path / "src.db"
    make_db(source).close()
    with socket.create_server(("127.0.0.1", 0)) as silent:
        args = (str(source), "127.0.0.1:/x.db", "--port", str(silent.getsockname()[1]))
        started = time.monotonic()
        default = sync(*args, timeout=60)
        between = time.monotonic()
        explicit = sync(*args, "-o", "ConnectTimeout=1", timeout=60)
        explicit_seconds = time.monotonic() - between
    for done in (default, explicit):
        assert done.returncode == 1 and "timed out" in done.stderr, done.stderr
        assert "_manifest failed on 127.0.0.1 (exit 255)" in done.stderr
    assert explicit_seconds < between - started - 5


def ssh_available() -> bool:
    """Report whether the test's remote host answers a batch-mode ssh.

    Returns:
        True when ssh can run ``python3`` on the remote host.
    """
    try:
        done = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                REMOTE_HOST,
                "python3 -c 'print(1)'",
            ],
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and done.stdout.strip() == "1"


def remote_run(script: str, argument: str, stdin: str = "") -> str:
    """Run a small python snippet on the remote test host.

    Args:
        script: Python source executed with ``python3 -c``.
        argument: Single argument passed to the snippet as ``sys.argv[1]``.
        stdin: Text piped to the snippet.

    Returns:
        The snippet's stdout.
    """
    command = f"python3 -c {shlex.quote(script)} {shlex.quote(argument)}"
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE_HOST, command],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_remote_round_trip(tmp_path: Path) -> None:
    """Push rows to a remote database, then pull them back down."""
    if not ssh_available():
        pytest.skip(f"no ssh access to {REMOTE_HOST}")
    remote_db = f"/tmp/kiss-sync-test-{os.getpid()}.db"
    local = tmp_path / "local.db"
    back = tmp_path / "back.db"
    src = make_db(local)
    add_task(src, "r1", steps=4, events=3)
    add_task(src, "r2", steps=1, events=1)
    src.close()
    make_db(back).close()
    remote_run(REMOTE_SCHEMA_SCRIPT, remote_db, TASK_DDL + UNIQUE_EVENT_INDEX + ";")
    try:
        push = sync(str(local), f"{REMOTE_HOST}:{remote_db}")
        assert push.returncode == 0, push.stderr
        assert "2 task row(s) added, 4 event row(s) added" in push.stdout

        again = sync(str(local), f"{REMOTE_HOST}:{remote_db}")
        assert again.returncode == 0, again.stderr
        assert "0 task row(s) added, 0 event row(s) added" in again.stdout

        # The remote copy names this machine as the one the tasks ran on.
        stamped = remote_run(
            "import socket,sqlite3,sys\n"
            "c=sqlite3.connect(sys.argv[1])\n"
            "print(sorted({r[0] for r in c.execute('SELECT machine FROM task_history')}))\n"
            "c.execute(\"INSERT INTO task_history (id, timestamp, task) VALUES ('r3', 1, 'r3')\")\n"
            "c.commit()\n"
            "print(socket.gethostname().split('.')[0])\n",
            remote_db,
        ).split()

        pull = sync(f"{REMOTE_HOST}:{remote_db}", str(back))
        assert pull.returncode == 0, pull.stderr
    finally:
        remote_run("import os,sys\nos.remove(sys.argv[1])\n", remote_db)

    assert stamped[0] == repr([sync_db.host_name()])
    # Back home r1 and r2 are this machine's own rows again; r3 ran there.
    assert rows(back, "SELECT id, machine FROM task_history ORDER BY id") == [
        ("r1", ""), ("r2", ""), ("r3", stamped[1]),
    ]
    assert event_keys(back) == [("r1", 1), ("r1", 2), ("r1", 3), ("r2", 1)]


FAKE_SSH = """#!/bin/bash
while [ "$1" = -o ] || [ "$1" = -p ]; do shift 2; done
shift                       # drop the host argument
exec bash -c "$1"
"""


@posix_only("drives the script with a bash ssh stand-in")
def test_a_script_too_long_for_one_raw_argument_still_runs_on_the_remote(tmp_path: Path) -> None:
    """A remote phase travels in one command-line argument, which Linux caps at 128 KiB.

    The checkout here is the real one with the script grown by some 10 KB of comments:
    base64 of the plain sources was 7 KB short of the cap and no longer fitted (``OSError:
    Argument list too long: 'ssh'``), so the sources travel compressed.  Both directions
    run, and the merge on the remote uses the chat-summary module shipped with the script.
    """
    package = tmp_path / "kiss"
    script = package / "scripts" / "sync_db.py"
    script.parent.mkdir(parents=True)
    (package / "agents" / "sorcar").mkdir(parents=True)
    real = Path(sync_db.__file__).resolve()
    for module in sync_db.SHIPPED_MODULES.values():
        shutil.copy(real.parent.parent / module, package / module)
    columns_source = (package / sync_db.SHIPPED_MODULES["history_columns"]).read_text()
    script.write_text(
        real.read_text() + "".join(f"# {line}\n" for line in columns_source.splitlines())
    )
    ssh = tmp_path / "bin" / "ssh"
    ssh.parent.mkdir()
    ssh.write_text(FAKE_SSH)
    ssh.chmod(0o755)
    env = {**os.environ, "PATH": f"{ssh.parent}{os.pathsep}{os.environ['PATH']}"}
    local, remote, back = (tmp_path / name for name in ("local.db", "remote.db", "back.db"))
    src = make_db(local)
    add_task(src, "r1", events=2, chat_id="c1", task="Plot the results")
    src.close()
    dst = make_db(remote)
    add_chats(dst)
    dst.close()
    make_db(back).close()
    for source, target in (
        (str(local), f"me@fakehost:{remote}"),
        (f"me@fakehost:{remote}", str(back)),
    ):
        done = subprocess.run(
            [sys.executable, str(script), source, target],
            capture_output=True, text=True, check=False, env=env, timeout=120,
        )
        assert done.returncode == 0, done.stderr
        assert "1 task row(s) added, 2 event row(s) added" in done.stdout
    assert chats(remote) == [("c1", "Plot the results", 1000000)]
    assert event_keys(back) == [("r1", 1), ("r1", 2)]


def test_a_machine_without_an_ssh_client_fails_like_any_other_sync_error(tmp_path: Path) -> None:
    """An ssh that cannot be started is reported in one line with exit status 1, not a traceback."""
    local = tmp_path / "local.db"
    make_db(local).close()
    done = sync(str(local), "me@fakehost:/remote.db", env={**os.environ, "PATH": str(tmp_path)})
    assert done.returncode == 1
    assert done.stderr.splitlines()[-1].startswith("sync_db: cannot run ssh: "), done.stderr
    assert "Traceback" not in done.stderr


@posix_only("drives the script with a bash ssh stand-in")
def test_a_remote_shell_that_prints_fails_like_any_other_sync_error(tmp_path: Path) -> None:
    """What is not the gzip stream a phase writes is reported in one line, not a traceback.

    A remote start-up file that prints puts its text in front of the
    manifest (a push) or of the delta (a pull), and a connection that
    drops cuts them short; neither database changes.  A remote disk
    that refuses the delta is not such a stream's fault and keeps its
    own words.
    """
    ssh = tmp_path / "bin" / "ssh"
    ssh.parent.mkdir()
    ssh.touch(mode=0o755)
    env = {**os.environ, "PATH": f"{ssh.parent}{os.pathsep}{os.environ['PATH']}"}
    local, remote = tmp_path / "local.db", tmp_path / "remote.db"
    for path in (local, remote):
        conn = make_db(path)
        add_task(conn, path.stem, events=1)
        conn.close()
    ways = ((str(local), f"me@fakehost:{remote}"), (f"me@fakehost:{remote}", str(local)))

    for stand_in, complaint in (
        (FAKE_SSH.replace("exec ", "echo Welcome\nexec "), "Not a gzipped file (b'We')"),
        (FAKE_SSH.replace('exec bash -c "$1"', 'bash -c "$1" | head -c 12'), "ended before"),
    ):
        ssh.write_text(stand_in)
        for source, target in ways:
            done = sync(source, target, env=env)
            assert done.returncode == 1
            assert done.stderr.splitlines()[-1].startswith("sync_db: a phase's output is not")
            assert complaint in done.stderr
            assert "Traceback" not in done.stderr
    ssh.write_text(FAKE_SSH.replace("exec ", "ulimit -f 4\nexec "))
    done = sync(*ways[0], env=env)
    assert done.returncode == 1
    assert "File too large" in done.stderr and "is not the gzip stream" not in done.stderr
    assert task_ids(local) == ["local"]
    assert task_ids(remote) == ["remote"]


CHAT_DDL = """
CREATE TABLE chat_summaries (
    chat_id TEXT PRIMARY KEY,
    summary TEXT DEFAULT '',
    last_launched INTEGER DEFAULT 0
);
"""


def add_chats(conn: sqlite3.Connection, *chats: tuple[str | None, str, object]) -> None:
    """Create the ``chat_summaries`` table and insert ``(chat_id, summary, last_launched)`` rows."""
    conn.executescript(CHAT_DDL)
    conn.executemany("INSERT INTO chat_summaries VALUES (?, ?, ?)", chats)


def chats(path: Path) -> list[tuple]:
    """List every chat summary row of a database, ordered by chat id."""
    return rows(path, "SELECT chat_id, summary, last_launched FROM chat_summaries ORDER BY chat_id")


def test_chat_summaries_are_recomputed_from_the_merged_tasks(tmp_path: Path) -> None:
    """The target's row of every chat that received a task is rebuilt from its merged tasks.

    Neither side's existing row is copied or compared: the target's stale
    row and the source's (deliberately wrong) row are both ignored, the
    rebuilt summary covers the target's own older task as well as the
    arriving ones, ``last_launched`` is the newest task's launch instant,
    a chat the target never saw gets a row, chats that received nothing
    keep theirs, and sub-agent rows and tasks without a chat count for
    nothing.  A repeated sync recomputes nothing; a dry run reports and
    writes nothing.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a2", chat_id="c1", task="Then write the related work section.", timestamp=1200.0)
    add_task(src, "a3", chat_id="c1", task="run the tests", timestamp=1300.0, start_ts=1300500)
    add_task(src, "b1", chat_id="c2", task="Please plot the benchmark results as a bar chart")
    add_task(src, "s1", chat_id="c3", parent_task_id="b1", task="Sub-agent: gather numbers")
    add_task(src, "n1", chat_id="", task="No chat at all")
    add_chats(src, ("c1", "wrong on the source", 99999999), ("c2", "also wrong", 99999999))
    src.close()
    dst = make_db(target)
    add_task(dst, "a1", chat_id="c1", task="Write the introduction of the paper", timestamp=1100.0)
    add_chats(dst, ("c1", "stale", 1), ("c9", "kept", 9))
    dst.close()

    preview = sync(str(source), str(target), "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert "would add 5 task row(s) and 0 event row(s) and refresh 2 chat summary row(s)" in (
        preview.stdout
    )
    assert chats(target) == [("c1", "stale", 1), ("c9", "kept", 9)]

    done = sync(str(source), str(target))
    assert done.returncode == 0, done.stderr
    assert "5 task row(s) added, 0 event row(s) added, 2 chat summary row(s) refreshed" in (
        done.stdout
    )
    assert chats(target) == [
        ("c1", "Write the introduction of the paper", 1300500),
        ("c2", "plot the benchmark results as a bar chart", 1000000),
        ("c9", "kept", 9),
    ]
    assert chats(source) == [
        ("c1", "wrong on the source", 99999999),
        ("c2", "also wrong", 99999999),
    ]

    again = sync(str(source), str(target))
    assert again.returncode == 0, again.stderr
    assert "0 task row(s) added, 0 event row(s) added, 0 chat summary row(s) refreshed" in (
        again.stdout
    )

    # A full sync names every chat again and rebuilds the same rows.
    full = sync(str(source), str(target), "--full")
    assert full.returncode == 0 and "2 chat summary row(s) refreshed" in full.stdout
    assert chats(target)[:2] == [
        ("c1", "Write the introduction of the paper", 1300500),
        ("c2", "plot the benchmark results as a bar chart", 1000000),
    ]


def test_chat_summaries_need_the_table_on_the_target_only(tmp_path: Path) -> None:
    """A target from before the table existed syncs its tasks and gets no chat rows.

    Whether the source has the table is irrelevant: the target's rows are
    rebuilt from the arriving tasks.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=1, chat_id="c1")
    add_chats(src, ("c1", "src one", 100))
    src.close()
    make_db(target).close()
    for extra in ((), ("--full",)):
        done = sync(str(source), str(target), *extra)
        assert done.returncode == 0, done.stderr
        assert "0 chat summary row(s) refreshed" in done.stdout
    assert rows(target, "SELECT name FROM sqlite_master WHERE name = 'chat_summaries'") == []
    assert task_ids(target) == ["a"]

    # The other way round: the target's rows are rebuilt from the arriving
    # tasks whether or not the source ever had the table -- or the column
    # that tells a sub-agent's row from a chat's own, which the target fills in.
    older, newer = tmp_path / "older.db", tmp_path / "newer.db"
    src = make_db(older)
    src.execute("ALTER TABLE task_history DROP COLUMN parent_task_id")
    add_task(src, "b", events=1, chat_id="c2", task="Fix the login bug on the settings page")
    src.close()
    dst = make_db(newer)
    add_chats(dst, ("c9", "kept", 9))
    dst.close()
    done = sync(str(older), str(newer))
    assert done.returncode == 0, done.stderr
    assert "1 task row(s) added" in done.stdout and "1 chat summary row(s)" in done.stdout
    assert chats(newer) == [
        ("c2", "Fix the login bug on the settings page", 1000000),
        ("c9", "kept", 9),
    ]


def test_chat_summaries_without_the_expected_columns_are_refused(tmp_path: Path) -> None:
    """A target chat table the summary code cannot write stops the sync before any row lands."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=1, chat_id="c1")
    src.close()
    dst = make_db(target)
    dst.executescript(
        "CREATE TABLE chat_summaries (chat_id TEXT PRIMARY KEY, last_launched INTEGER)"
    )
    dst.close()
    done = sync(str(source), str(target))
    assert done.returncode == 1
    assert "chat_summaries in the target database lacks required column(s): summary" in done.stderr
    assert task_ids(target) == []


def test_chat_summaries_need_the_summary_code_next_to_the_script(tmp_path: Path) -> None:
    """A copy of the script without the checkout's shipped modules stops rather than guesses.

    The merge reads the carry-over script first (what a deletion takes
    with it); the chat-summary phase then reads the column module (the
    target's ``chat_summaries`` columns) before the summary module.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", chat_id="c1")
    src.close()
    dst = make_db(target)
    add_chats(dst)
    dst.close()
    stray = tmp_path / "scripts" / "sync_db.py"
    stray.parent.mkdir()
    shutil.copy(sync_db.__file__, stray)
    for missing in ("carry_over_tables.py", "history_columns.py"):
        done = subprocess.run(
            [sys.executable, str(stray), str(source), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 1
        assert "cannot read" in done.stderr and missing in done.stderr
        assert task_ids(target) == []
        shutil.copy(Path(sync_db.__file__).with_name("carry_over_tables.py"), stray.parent)


EDIT_SCRIPT = (
    "import sqlite3,sys\n"
    "c=sqlite3.connect(sys.argv[1])\n"
    "count='SELECT count(*) FROM task_history'\n"
    "before=c.execute(count).fetchone()[0]\n"
    "c.execute(sys.argv[2])\n"
    "c.commit()\n"
    "print('tasks', before, '->', c.execute(count).fetchone()[0])\n"
)


def edit_command(tmp_path: Path, sql: str) -> str:
    """Build an ``--edit-delta`` command that runs one SQL statement on the delta.

    Args:
        tmp_path: Directory the helper script is written to.
        sql: Statement to run against the delta.

    Returns:
        A shell command naming the delta as ``{}``.
    """
    script = tmp_path / "edit.py"
    script.write_text(EDIT_SCRIPT)
    quote = sync_db.local_shell_quote
    return f"{quote(sys.executable)} {quote(str(script))} {{}} {quote(sql)}"


def test_edit_delta_rewrites_and_drops_rows_before_the_merge(tmp_path: Path) -> None:
    """The commands run in order, see only the rows about to travel, and shape what arrives."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=1, result="/laptop/kiss")
    add_task(src, "b", events=2, result="/laptop/kiss")
    add_task(src, "live", events=5)
    src.close()
    dst = make_db(target)
    add_task(dst, "a", events=1, result="kept")
    dst.close()
    drop_live = edit_command(tmp_path, "DELETE FROM task_history WHERE id = 'live'")
    drop_live_events = edit_command(tmp_path, "DELETE FROM events WHERE task_id = 'live'")
    relocate = edit_command(
        tmp_path, "UPDATE task_history SET result = replace(result, '/laptop/', '/server/')"
    )

    done = sync(
        str(source),
        str(target),
        "--edit-delta",
        drop_live,
        "--edit-delta",
        drop_live_events,
        "--edit-delta",
        relocate,
    )

    assert done.returncode == 0, done.stderr
    assert "1 task row(s) added, 2 event row(s) added" in done.stdout
    # The first command saw the two task rows the target lacked, the later ones one.
    assert done.stdout.startswith("tasks 2 -> 1\ntasks 1 -> 1\ntasks 1 -> 1\n")
    assert rows(target, "SELECT id, result FROM task_history ORDER BY id") == [
        ("a", "kept"),
        ("b", "/server/kiss"),
    ]
    assert event_keys(target) == [("a", 1), ("b", 1), ("b", 2)]


def test_a_failing_edit_delta_stops_the_sync_before_the_merge(tmp_path: Path) -> None:
    """Nothing is merged when a command exits non-zero, and the sync says which."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=2)
    src.close()
    make_db(target).close()

    done = sync(str(source), str(target), "--edit-delta", "exit 3")

    assert done.returncode == 1
    assert "delta edit exited with 3: exit 3" in done.stderr
    assert task_ids(target) == []
    assert event_keys(target) == []


def test_edit_delta_sees_the_source_columns_and_a_quoted_path(tmp_path: Path) -> None:
    """The delta carries the source's tables, at a path quoted for the shell."""
    source, target = tmp_path / "with space", tmp_path / "dst.db"
    source.mkdir()
    source = source / "src.db"
    src = make_db(source)
    add_task(src, "a", events=1)
    src.close()
    make_db(target).close()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sqlite3,sys\n"
        "print(sys.argv[1])\n"
        "c=sqlite3.connect(sys.argv[1])\n"
        "print(sorted(r[0] for r in c.execute("
        "\"SELECT name FROM sqlite_master WHERE type='table'\")))\n"
    )
    # The delta is a temporary file: put the temp dir where its name needs
    # quoting in every shell (a space, and `&`, a command separator for
    # cmd.exe even in a word without spaces).
    temp = tmp_path / "a&b c"
    temp.mkdir()
    env = {**os.environ, "TMPDIR": str(temp), "TEMP": str(temp), "TMP": str(temp)}

    done = sync(
        str(source),
        str(target),
        "--edit-delta",
        f"{sync_db.local_shell_quote(sys.executable)} {sync_db.local_shell_quote(str(probe))} {{}}",
        env=env,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.count(str(temp)) == 1, done.stdout
    assert (
        "['deleted_chats', 'deleted_tasks', 'events', 'sqlite_sequence', 'task_history']"
        in done.stdout
    )
    assert event_keys(target) == [("a", 1)]


def _tombstone(conn: sqlite3.Connection, table: str, key: str, deleted_ts: float) -> None:
    """Record one deletion as the web app does, creating the table if needed.

    Args:
        conn: Open connection.
        table: ``deleted_tasks`` or ``deleted_chats``.
        key: The task id or chat id deleted.
        deleted_ts: When, as an epoch timestamp.
    """
    column = "id" if table == "deleted_tasks" else "chat_id"
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {table} ({column} TEXT PRIMARY KEY,"
        " deleted_ts REAL NOT NULL)"
    )
    conn.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?)", (key, deleted_ts))


def test_a_deletion_travels_and_the_deleted_task_never_comes_back(tmp_path: Path) -> None:
    """The target's deletion keeps the task out; the source's deletion is carried out."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "kept", events=1)
    add_task(src, "gone-there", events=2)  # the target deleted it
    add_task(src, "gone-here-too", events=1)  # the source deleted it
    _tombstone(src, "deleted_tasks", "gone-here-too", 2000.0)
    src.execute("DELETE FROM task_history WHERE id = 'gone-here-too'")
    src.execute("DELETE FROM events WHERE task_id = 'gone-here-too'")
    src.close()
    dst = make_db(target)
    add_task(dst, "gone-here-too", events=1)
    _tombstone(dst, "deleted_tasks", "gone-there", 2000.0)
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert "1 task row(s) deleted" in done.stdout
    assert task_ids(target) == ["kept"]
    assert event_keys(target) == [("kept", 1)]
    assert rows(target, "SELECT id FROM deleted_tasks ORDER BY id") == [
        ("gone-here-too",), ("gone-there",)
    ]
    # A second run has nothing left to do.
    again = sync(str(source), str(target))
    assert "0 task row(s) added, 0 event row(s) added, 0 chat" in again.stdout
    assert task_ids(target) == ["kept"]


def test_a_full_sync_does_not_resurrect_a_deleted_task(tmp_path: Path) -> None:
    """``--full`` ships every row; the target's deletions still keep them out."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "stale", events=2, result="Agent Failed Abruptly", timestamp=1.0)
    add_task(src, "fine", events=1)
    src.close()
    dst = make_db(target)
    _tombstone(dst, "deleted_tasks", "stale", 2.0)
    dst.close()

    done = sync(str(source), str(target), "--full")

    assert done.returncode == 0, done.stderr
    assert task_ids(target) == ["fine"]
    assert event_keys(target) == [("fine", 1)]


def test_a_deletion_spares_a_task_still_running_on_the_target(tmp_path: Path) -> None:
    """A row the web app will rewrite is not deleted from under the agent.

    Both a row still carrying the placeholder result and a row without
    it whose task wrote an event moments ago are running tasks; the
    deletion is recorded and waits for a sync after they end.  A row
    younger than its deletion is a new task with a reused id.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    for task_id in ("started", "busy", "reused", "old"):
        _tombstone(src, "deleted_tasks", task_id, 5000.0)
    src.close()
    dst = make_db(target)
    add_task(dst, "started", result="Agent Failed Abruptly", events=1)
    add_task(dst, "busy", result="", end_ts=0, events=1)
    dst.execute("UPDATE events SET timestamp = ? WHERE task_id = 'busy'", (time.time(),))
    add_task(dst, "reused", timestamp=6000.0, events=1)
    add_task(dst, "old", end_ts=1500, events=1)
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert task_ids(target) == ["busy", "reused", "started"]
    assert ("old", 1) not in event_keys(target)
    assert len(rows(target, "SELECT id FROM deleted_tasks")) == 4

    # Once they have ended, the recorded deletions are carried out.
    dst = sqlite3.connect(target, isolation_level=None)
    dst.execute("UPDATE task_history SET result = 'done', end_ts = 7000")
    dst.execute("UPDATE events SET timestamp = 1.0")
    dst.close()
    assert sync(str(source), str(target)).returncode == 0
    assert task_ids(target) == ["reused"]

    # Deleted again on the source after its new life began: the newer
    # tombstone travels, although the target knows the id, and the reused
    # task goes with it.
    src = sqlite3.connect(source, isolation_level=None)
    _tombstone(src, "deleted_tasks", "reused", 9000.0)
    src.close()
    assert sync(str(source), str(target)).returncode == 0
    assert task_ids(target) == []
    assert rows(target, "SELECT deleted_ts FROM deleted_tasks WHERE id = 'reused'") == [(9000.0,)]


def test_a_task_recreated_after_its_deletion_travels(tmp_path: Path) -> None:
    """A tombstone older than the task is another task's: the row and its events travel.

    The extract must leave the recreated task in, as the merge would
    keep it, so that a plain sync and a ``--full`` one agree; a task the
    tombstone does date stays out of both.  Each one's sub-task shares
    its fate.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "reused", timestamp=6000.0, events=2)
    add_task(src, "reused-sub", parent_task_id="reused", timestamp=6001.0, events=1)
    add_task(src, "older", timestamp=4000.0, events=1)
    add_task(src, "older-sub", parent_task_id="older", timestamp=4001.0, events=1)
    src.close()
    dst = make_db(target)
    _tombstone(dst, "deleted_tasks", "reused", 5000.0)
    _tombstone(dst, "deleted_tasks", "older", 5000.0)
    dst.close()
    full = tmp_path / "dst-full.db"
    shutil.copy(target, full)

    for flags, path in (((), target), (("--full",), full)):
        done = sync(str(source), str(path), *flags)
        assert done.returncode == 0, done.stderr
        assert task_ids(path) == ["reused", "reused-sub"], flags
        assert event_keys(path) == [("reused", 1), ("reused", 2), ("reused-sub", 1)], flags


def test_a_deleted_chat_takes_the_tasks_its_deleting_machine_never_saw(tmp_path: Path) -> None:
    """A chat deleted on one machine does not come back with a task only the other one had.

    The deleting machine recorded the chat and the tasks it held.  A task
    of the chat it had never seen, and that task's sub-task, are older
    than the deletion and go with the chat on both machines; a task
    started after the deletion is the chat's new life.
    """
    here, there = tmp_path / "here.db", tmp_path / "there.db"
    dst = make_db(here)
    _tombstone(dst, "deleted_tasks", "known", 2000.0)
    _tombstone(dst, "deleted_chats", "shared", 2000.0)
    dst.close()
    src = make_db(there)
    add_task(src, "known", chat_id="shared", events=1)
    add_task(src, "unseen", chat_id="shared", events=2)
    add_task(src, "sub", chat_id="sub-chat", parent_task_id="unseen", events=1)
    add_task(src, "later", chat_id="shared", timestamp=3000.0, events=1)
    src.close()
    full, pristine = tmp_path / "here-full.db", tmp_path / "here-pristine.db"
    shutil.copy(here, full)
    shutil.copy(here, pristine)
    gone = [("known", 2000.0), ("sub", 2000.0), ("unseen", 2000.0)]

    for flags, path in (((), here), (("--full",), full)):
        done = sync(str(there), str(path), *flags)
        assert done.returncode == 0, done.stderr
        assert task_ids(path) == ["later"], flags
        assert event_keys(path) == [("later", 1)], flags
        assert rows(path, "SELECT id, deleted_ts FROM deleted_tasks ORDER BY id") == gone, flags

    # The other way, the machine that holds them carries the deletion out.
    done = sync(str(pristine), str(there))
    assert done.returncode == 0, done.stderr
    assert "3 task row(s) deleted" in done.stdout
    assert task_ids(there) == ["later"]
    assert event_keys(there) == [("later", 1)]
    assert rows(there, "SELECT id, deleted_ts FROM deleted_tasks ORDER BY id") == gone


def test_a_deleted_task_takes_its_sub_tasks_wherever_they_are(tmp_path: Path) -> None:
    """A task deleted on a machine that held none of its sub-tasks still takes them all.

    The sub-tasks the target holds go with the parent's row, whatever
    their age -- a running one once it has ended -- and one that arrives
    later from a third machine stays out, with its events and its own
    sub-task.  One that began after the parent's deletion is recorded as
    deleted a millisecond after it began, whatever the clocks say (one
    machine's runs an hour ahead): a task of its id that began later
    than that, on another machine, is another task and lives.
    """
    source, target, third = (tmp_path / name for name in ("src.db", "dst.db", "third.db"))
    ahead = time.time() + 3600
    src = make_db(source)
    _tombstone(src, "deleted_tasks", "parent", 2000.0)
    src.close()
    dst = make_db(target)
    add_task(dst, "parent", events=1)
    add_task(dst, "child", parent_task_id="parent", events=2)
    add_task(dst, "grandchild", parent_task_id="child", events=1)
    add_task(dst, "late", parent_task_id="parent", timestamp=3000.0, events=1)
    add_task(dst, "ahead", parent_task_id="parent", timestamp=ahead, events=1)
    add_task(dst, "ahead-sub", parent_task_id="ahead", timestamp=ahead + 1, events=1)
    add_task(dst, "busy", parent_task_id="child", result="Agent Failed Abruptly", events=1)
    add_task(dst, "kept", events=1)
    dst.close()
    other = make_db(third)
    add_task(other, "straggler", parent_task_id="grandchild", events=1)
    add_task(other, "straggler-sub", parent_task_id="straggler", events=1)
    add_task(other, "late", timestamp=4000.0, events=1)
    other.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert "6 task row(s) deleted" in done.stdout
    assert task_ids(target) == ["busy", "kept"]
    assert event_keys(target) == [("busy", 1), ("kept", 1)]
    assert dict(rows(target, "SELECT id, deleted_ts FROM deleted_tasks")) == {
        "parent": 2000.0, "child": 2000.0, "grandchild": 2000.0, "busy": 2000.0,
        "late": 3000.0 + 0.001, "ahead": ahead + 0.001, "ahead-sub": ahead + 1 + 0.001,
    }  # fmt: skip

    dst = sqlite3.connect(target, isolation_level=None)
    dst.execute("UPDATE task_history SET result = 'done' WHERE id = 'busy'")
    dst.close()
    # The third machine's own "late" began after the deletion recorded for that id.
    for origin, destination in ((third, target), (target, third)):
        done = sync(str(origin), str(destination))

        assert done.returncode == 0, done.stderr
        assert task_ids(destination) == ["kept", "late"]
        assert event_keys(destination) == [("kept", 1), ("late", 1)]


def test_a_deleted_chat_loses_its_summary_row_but_a_deleted_task_does_not(
    tmp_path: Path,
) -> None:
    """Only a chat deletion drops the chat's row: a task deletion keeps its title.

    A chat deleted long ago and started again is a deleted chat once
    more when its last task goes: its row is dropped by the merge that
    removes the task, not by the next one.  Losing a sub-agent's row
    changes nothing a chat's row says, so such a chat keeps all of it.
    """
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    for task_id in ("t1", "c1", "a1", "s1"):
        _tombstone(src, "deleted_tasks", task_id, 5000.0)
    _tombstone(src, "deleted_chats", "chat-gone", 5000.0)
    for chat_id in ("chat-again", "chat-sub"):
        _tombstone(src, "deleted_chats", chat_id, 1.0)
    src.close()
    dst = make_db(target)
    add_task(dst, "t1", chat_id="chat-kept", events=1)
    add_task(dst, "c1", chat_id="chat-gone", events=1)
    add_task(dst, "a1", chat_id="chat-again", events=1)
    add_task(dst, "s1", chat_id="chat-sub", parent_task_id="elsewhere", events=1)
    dst.execute(
        "CREATE TABLE chat_summaries (chat_id TEXT PRIMARY KEY, summary TEXT,"
        " last_launched INTEGER, last_finished INTEGER, user_summary TEXT DEFAULT '')"
    )
    dst.execute("INSERT INTO chat_summaries VALUES ('chat-kept', 's', 1, 1, 'My title')")
    dst.execute("INSERT INTO chat_summaries VALUES ('chat-gone', 's', 1, 1, '')")
    dst.execute("INSERT INTO chat_summaries VALUES ('chat-again', 's', 2000, 1, 'Again')")
    dst.execute("INSERT INTO chat_summaries VALUES ('chat-sub', 's', 2000, 1, 'Sub')")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert task_ids(target) == []
    # What was worked out from the deleted task goes with it, as in the web app.
    assert rows(target, "SELECT * FROM chat_summaries ORDER BY chat_id") == [
        ("chat-kept", "", 0, 0, "My title"),
        ("chat-sub", "s", 2000, 1, "Sub"),
    ]
    assert "3 chat summary row(s) refreshed" in done.stdout
    assert "0 chat summary row(s) refreshed" in sync(str(source), str(target)).stdout


def test_a_chat_started_again_after_its_deletion_keeps_its_row(tmp_path: Path) -> None:
    """An old chat tombstone does not reach a chat that lives again under the same id."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    _tombstone(src, "deleted_chats", "again", 1000.0)
    _tombstone(src, "deleted_chats", "stale", 1000.0)
    src.close()
    dst = make_db(target)
    dst.execute(
        "CREATE TABLE chat_summaries (chat_id TEXT PRIMARY KEY, summary TEXT,"
        " last_launched INTEGER, last_finished INTEGER, user_summary TEXT DEFAULT '')"
    )
    # Launched after the deletion (milliseconds), its tasks since deleted one by one.
    dst.execute("INSERT INTO chat_summaries VALUES ('again', 's', 5000000, 0, 'New life')")
    dst.execute("INSERT INTO chat_summaries VALUES ('stale', 's', 500000, 0, '')")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert rows(target, "SELECT chat_id, user_summary FROM chat_summaries") == [
        ("again", "New life")
    ]

    # Deleted again on the source, after its new life began: the newer
    # deletion reaches the target, with or without --full, and the chat goes.
    src = sqlite3.connect(source, isolation_level=None)
    _tombstone(src, "deleted_chats", "again", 9000.0)
    src.close()
    full = tmp_path / "dst-full.db"
    shutil.copy(target, full)
    for flags, path in (((), target), (("--full",), full)):
        done = sync(str(source), str(path), *flags)
        assert done.returncode == 0, done.stderr
        assert rows(path, "SELECT deleted_ts FROM deleted_chats WHERE chat_id = 'again'") == [
            (9000.0,)
        ], flags
        assert rows(path, "SELECT chat_id FROM chat_summaries") == [], flags
    # Synced the other way, the older deletion cannot lower the newer one.
    stale = tmp_path / "stale.db"
    old = make_db(stale)
    _tombstone(old, "deleted_chats", "again", 1000.0)
    old.close()
    for flags, path in (((), target), (("--full",), full)):
        assert sync(str(stale), str(path), *flags).returncode == 0
        assert rows(path, "SELECT deleted_ts FROM deleted_chats WHERE chat_id = 'again'") == [
            (9000.0,)
        ], flags

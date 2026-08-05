# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.scripts.sync_db``.

Every test drives the real command-line entry point against real SQLite
database files that carry the production ``sorcar.db`` schema, then reads
the resulting files back with plain SQL.  Nothing is mocked.
"""

from __future__ import annotations

import os
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

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
    """
    conn.execute(
        "INSERT INTO task_history (id, timestamp, task, steps, result, end_ts)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, 1000.0, f"task {task_id}", steps, result, end_ts),
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


def sync(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the sync_db command line.

    Args:
        *args: Command-line arguments after the module name.

    Returns:
        The completed process, with text stdout and stderr.
    """
    return subprocess.run(
        [sys.executable, "-m", "kiss.scripts.sync_db", *args],
        capture_output=True,
        text=True,
        check=False,
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
    assert "0 task row(s) added, 0 updated, 0 event row(s) added" in second.stdout
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
    """After the first sync only later events and changed tasks travel."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=2, events=2)
    add_task(src, "b", steps=5, events=4)
    src.close()
    make_db(target).close()
    assert sync(str(source), str(target)).returncode == 0

    src = sqlite3.connect(source, isolation_level=None)
    append_events(src, "a", 3, 2)
    src.execute("UPDATE task_history SET steps = 9, result = 'done' WHERE id = 'a'")
    add_task(src, "c", steps=1, events=1)
    src.close()
    second = sync(str(source), str(target))

    assert second.returncode == 0
    assert "1 task row(s) added, 1 updated, 3 event row(s) added" in second.stdout
    assert task_ids(target) == ["a", "b", "c"]
    assert rows(target, "SELECT steps, result FROM task_history WHERE id = 'a'") == [
        (9, "done")
    ]
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


def test_stale_source_task_does_not_clobber_target(tmp_path: Path) -> None:
    """A less advanced source task row leaves the target's row intact."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=2, result="early")
    src.close()
    dst = make_db(target)
    add_task(dst, "a", steps=40, result="late", end_ts=99)
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0
    assert rows(target, "SELECT steps, result FROM task_history") == [(40, "late")]
    assert "0 task row(s) added, 0 updated" in done.stdout


def test_force_overwrites_the_target_task_row(tmp_path: Path) -> None:
    """``--force`` makes the source authoritative on conflict."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=2, result="early")
    src.close()
    dst = make_db(target)
    add_task(dst, "a", steps=40, result="late", end_ts=99)
    dst.close()

    done = sync(str(source), str(target), "--force")

    assert done.returncode == 0
    assert rows(target, "SELECT steps, result FROM task_history") == [(2, "early")]
    assert "0 task row(s) added, 1 updated" in done.stdout


def test_insert_only_keeps_existing_rows(tmp_path: Path) -> None:
    """``--insert-only`` adds new rows but never updates existing ones."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", steps=80, result="newer")
    add_task(src, "b", steps=1)
    src.close()
    dst = make_db(target)
    add_task(dst, "a", steps=2, result="older")
    dst.close()

    done = sync(str(source), str(target), "--insert-only")

    assert done.returncode == 0
    assert rows(
        target, "SELECT id, steps, result FROM task_history ORDER BY id"
    ) == [("a", 2, "older"), ("b", 1, "ok")]
    assert "1 task row(s) added, 0 updated" in done.stdout


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
    assert "would add 1 task row(s), update 0, add 2 event row(s)" in done.stdout
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


def test_missing_event_below_the_highest_one_is_healed(tmp_path: Path) -> None:
    """A hole in the target's event sequence is filled on the next sync."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=3)
    src.close()
    dst = make_db(target)
    add_task(dst, "a", events=0)
    append_events(dst, "a", 1, 1)
    append_events(dst, "a", 3, 1)
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert event_keys(target) == [("a", 1), ("a", 2), ("a", 3)]
    assert "1 event row(s) added" in done.stdout


def test_neighbouring_column_values_are_not_confused(tmp_path: Path) -> None:
    """Shifting a byte between two columns still counts as a change."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a")
    src.execute("UPDATE task_history SET model = 'a', parent_task_id = char(0) || 'sb'")
    src.close()
    dst = make_db(target)
    add_task(dst, "a")
    dst.execute("UPDATE task_history SET model = 'a' || char(0) || 's', parent_task_id = 'b'")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert "1 updated" in done.stdout
    assert rows(target, "SELECT model, parent_task_id FROM task_history") == [
        ("a", "\x00sb")
    ]


def test_schema_mismatch_is_refused(tmp_path: Path) -> None:
    """Databases whose columns differ are not synced at all."""
    source, target = tmp_path / "src.db", tmp_path / "dst.db"
    src = make_db(source)
    add_task(src, "a", events=1)
    src.close()
    dst = make_db(target)
    dst.execute("ALTER TABLE task_history ADD COLUMN extra TEXT DEFAULT 'x'")
    dst.close()

    done = sync(str(source), str(target))

    assert done.returncode == 1
    assert "schemas for 'task_history' differ" in done.stderr
    assert "extra" in done.stderr
    assert task_ids(target) == []


def test_dry_run_reports_what_the_mode_would_really_do(tmp_path: Path) -> None:
    """A dry run applies the conflict rules instead of counting the delta."""
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
    assert "would add 1 task row(s), update 0, add 2 event row(s)" in done.stdout
    assert task_ids(target) == ["a"]
    assert rows(target, "SELECT steps FROM task_history") == [(40,)]
    assert event_keys(target) == []


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
    assert task_ids(target) == []
    assert event_keys(target) == []


def test_repeated_sequence_in_the_target_still_heals(tmp_path: Path) -> None:
    """A duplicated event does not disguise a hole as a complete run."""
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

    done = sync(str(source), str(target))

    assert done.returncode == 0, done.stderr
    assert ("a", 2) in event_keys(target)


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
        assert "2 task row(s) added, 0 updated, 4 event row(s) added" in push.stdout

        again = sync(str(local), f"{REMOTE_HOST}:{remote_db}")
        assert again.returncode == 0, again.stderr
        assert "0 task row(s) added, 0 updated, 0 event row(s) added" in again.stdout

        pull = sync(f"{REMOTE_HOST}:{remote_db}", str(back))
        assert pull.returncode == 0, pull.stderr
    finally:
        remote_run("import os,sys\nos.remove(sys.argv[1])\n", remote_db)

    assert task_ids(back) == ["r1", "r2"]
    assert event_keys(back) == [("r1", 1), ("r1", 2), ("r1", 3), ("r2", 1)]

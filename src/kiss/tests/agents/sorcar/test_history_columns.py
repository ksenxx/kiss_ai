# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for :mod:`kiss.agents.sorcar.history_columns`.

The module is the one piece of schema code two programs share: the
daemon migrating a database it opens, and ``sync_db`` merging a database
from another release.  These tests pin what both rely on: the column
list matches the daemon's ``CREATE TABLE``, the cron marker is what the
scheduler really writes, and the backfill is idempotent.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent, history_columns, persistence
from kiss.agents.sorcar.history_columns import TASK_TABLE

LEGACY_DDL = """
CREATE TABLE task_history (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    task TEXT NOT NULL,
    work_dir TEXT DEFAULT '',
    parent_task_id TEXT DEFAULT ''
);
"""


def legacy_db() -> sqlite3.Connection:
    """Return an in-memory database with a first-release task table."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.executescript(LEGACY_DDL)
    return conn


def add(
    conn: sqlite3.Connection, task_id: str, task: str, work_dir: str = "/p", parent: str = ""
) -> None:
    """Insert one task row with the columns a first-release table has."""
    conn.execute(
        "INSERT INTO task_history (id, timestamp, task, work_dir, parent_task_id)"
        " VALUES (?, 1.0, ?, ?, ?)",
        (task_id, task, work_dir, parent),
    )


def add_task_columns(conn: sqlite3.Connection) -> list[str]:
    """Add the task columns a first-release table lacks, as the daemon does on open."""
    return history_columns.add_missing_columns(conn, TASK_TABLE, history_columns.ADDED_TASK_COLUMNS)


def flags(conn: sqlite3.Connection) -> dict[str, int]:
    """Map task id to ``is_cron``."""
    return dict(conn.execute("SELECT id, is_cron FROM task_history").fetchall())


def test_module_is_stdlib_only() -> None:
    """The file is shipped to remote hosts by sync_db and run with a bare python3."""
    tree = ast.parse(Path(history_columns.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "logging", "socket", "sqlite3", "collections"}, imported


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        (TASK_TABLE, history_columns.ADDED_TASK_COLUMNS),
        (history_columns.CHAT_TABLE, history_columns.ADDED_SUMMARY_COLUMNS),
    ],
)
def test_added_columns_are_the_daemons(table: str, columns: tuple[tuple[str, str], ...]) -> None:
    """Every added column is in the daemon's CREATE TABLE, with the same declaration."""
    conn = sqlite3.connect(":memory:")
    persistence._init_tables(conn)
    declared = {
        row[1]: (row[2], row[3], row[4]) for row in conn.execute(f"PRAGMA table_info({table})")
    }
    for name, declaration in columns:
        assert name in declared, name
        probe = sqlite3.connect(":memory:")
        probe.execute(f"CREATE TABLE t (x {declaration})")
        (_, _, kind, notnull, default, _) = probe.execute("PRAGMA table_info(t)").fetchone()
        assert (kind, notnull, default) == declared[name], name


def test_the_preamble_starts_with_the_marker() -> None:
    """The scheduler writes the marker first, so a prompt job's row is recognisable."""
    assert cron_agent.PROMPT_PREAMBLE.startswith(history_columns.CRON_TASK_MARKER + " ")
    assert cron_agent.UNATTENDED_CHILD_PREAMBLE.startswith("You are a sub-task")
    assert not cron_agent.UNATTENDED_CHILD_PREAMBLE.startswith(history_columns.CRON_TASK_MARKER)


def test_add_missing_columns_reports_what_it_added() -> None:
    """A first-release table gets every added column once; a current one none."""
    conn = legacy_db()
    assert add_task_columns(conn) == [
        name for name, _ in history_columns.ADDED_TASK_COLUMNS
    ]
    assert add_task_columns(conn) == []
    present = {row[1] for row in conn.execute("PRAGMA table_info(task_history)")}
    assert {name for name, _ in history_columns.ADDED_TASK_COLUMNS} <= present


def test_backfill_recognises_the_marker_not_the_whole_preamble() -> None:
    """A row from a release whose preamble was worded differently after the
    first sentence is still a cron run; a row that merely quotes the
    sentence is not."""
    conn = legacy_db()
    add(conn, "current", cron_agent.PROMPT_PREAMBLE + "check the mail")
    add(conn, "older", history_columns.CRON_TASK_MARKER + " Do not ask questions. Check the mail")
    add(conn, "rundir", "/good_morning", work_dir="/home/u/.kiss/cron/runs/good-1")
    add(conn, "quoted", "Explain why '" + history_columns.CRON_TASK_MARKER + "' is used")
    add(conn, "plain", "an ordinary task")
    add(conn, "child", "sub-task", parent="older")
    add(conn, "grandchild", "sub-sub-task", parent="child")
    add(conn, "plainchild", "sub-task", parent="plain")
    add_task_columns(conn)

    assert history_columns.backfill_is_cron(conn) == 5
    assert flags(conn) == {
        "current": 1,
        "older": 1,
        "rundir": 1,
        "quoted": 0,
        "plain": 0,
        "child": 1,
        "grandchild": 1,
        "plainchild": 0,
    }


def test_backfill_only_ever_sets_the_flag() -> None:
    """Running it again, or over rows written with the flag, changes nothing."""
    conn = legacy_db()
    add(conn, "cron", history_columns.CRON_TASK_MARKER + " Check the mail.")
    add(conn, "plain", "an ordinary task")
    add_task_columns(conn)
    conn.execute("UPDATE task_history SET is_cron = 1 WHERE id = 'plain'")  # explicit flag

    assert history_columns.backfill_is_cron(conn) == 1
    assert history_columns.backfill_is_cron(conn) == 0
    assert flags(conn) == {"cron": 1, "plain": 1}


def test_complete_rows_runs_the_backfills_it_is_asked_for() -> None:
    """Only the named columns with a backfill run; the rest are ignored."""
    conn = legacy_db()
    add(conn, "cron", history_columns.CRON_TASK_MARKER + " Check the mail.")
    add_task_columns(conn)

    assert history_columns.complete_rows(conn, ["success", "no_such_column"]) == []
    assert flags(conn) == {"cron": 0}
    assert history_columns.complete_rows(conn, {"is_cron", "is_cron", "owner"}) == ["is_cron"]
    assert flags(conn) == {"cron": 1}


def test_synced_rows_learn_the_machine_they_ran_on_from_their_first_events(
    tmp_path: Path,
) -> None:
    """The one-time repair stamps the rows whose first events name another host, once."""
    from kiss.tests.server.conftest import redirect_persistence, restore_persistence

    saved = redirect_persistence(tmp_path)
    try:
        here = history_columns.host_name()
        ids = {
            name: persistence._add_task(name, chat_id="c-" + name)[0]
            for name in ("settings", "prompt", "own", "silent", "stamped", "blob")
        }
        db = persistence._get_db()
        events = [
            ("settings", 0, '{"type": "task_settings", "settings": {"host": "gpu-box.lan"}}'),
            ("prompt", 0, '{"type": "system_prompt", "text": "- Machine info: lab (x86_64)\\n"}'),
            ("own", 0, '{"type": "task_settings", "settings": {"host": "' + here + '"}}'),
            ("own", 1, '{"type": "system_prompt", "text": "Machine info: ' + here + ' (arm64)"}'),
            ("stamped", 0, '{"type": "system_prompt", "text": "Machine info: lab (x86_64)"}'),
            # A foreign writer's BLOB reads as its text: characters, not bytes, are
            # counted up to the host (the two differ after the non-ASCII one).
            ("blob", 0, '{"note": "caf\u00e9", "settings": {"host": "blob-box.lan"}}'.encode()),
        ]
        db.executemany(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, 1.0)",
            [(ids[name], seq, text) for name, seq, text in events],
        )
        db.execute("UPDATE task_history SET machine = 'box' WHERE id = ?", (ids["stamped"],))
        db.commit()

        assert sorted(history_columns.task_origins(db)) == sorted(
            [("gpu-box", ids["settings"]), ("lab", ids["prompt"]), ("blob-box", ids["blob"])]
        )
        assert persistence._backfill_task_machines() == 3
        machines = dict(db.execute("SELECT id, machine FROM task_history").fetchall())
        assert {name: machines[task_id] for name, task_id in ids.items()} == {
            "settings": "gpu-box", "prompt": "lab", "own": "", "silent": "", "stamped": "box",
            "blob": "blob-box",
        }
        # Once per database: the marker makes a second start a no-op.
        db.execute("UPDATE task_history SET machine = '' WHERE id = ?", (ids["prompt"],))
        db.commit()
        assert persistence._backfill_task_machines() == 0
        # A database it cannot read logs and is tried again on the next start.
        (persistence._KISS_DIR / persistence._MACHINES_BACKFILLED).unlink()
        db.execute("ALTER TABLE task_history RENAME COLUMN machine TO gone")
        db.commit()
        assert persistence._backfill_task_machines() == 0
        assert not (persistence._KISS_DIR / persistence._MACHINES_BACKFILLED).exists()
    finally:
        restore_persistence(saved)

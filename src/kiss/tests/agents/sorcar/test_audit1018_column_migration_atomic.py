# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A column and its backfill land together, or not at all (audit 2026-10-18).

``persistence._add_missing_columns`` runs on whichever thread opens the
database first, which may be a task thread: a Stop injected there, or
a crash, between the ``ALTER TABLE ... ADD COLUMN is_cron`` and the
``UPDATE`` that flags the older cron rows left the column in place
with every row at 0.  The next open found the column present, so the
backfill never ran again and the history's cron runs stayed unflagged
for good.  Both statements now run inside one ``BEGIN IMMEDIATE``, so
an interruption rolls the column back and the next open redoes both.

The interruption is SQLite's own: a progress handler on the opening
connection aborts the backfill's ``UPDATE`` (``OperationalError:
interrupted``), exactly where the lost-work window was.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.history_columns import CRON_TASK_MARKER


class AbortBackfill:
    """Interrupt the ``is_cron`` backfill ``UPDATE`` of *conn*, and only it."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.aborting = False
        conn.set_trace_callback(self.statement_started)
        conn.set_progress_handler(self.progress, 1)

    def statement_started(self, sql: str) -> None:
        """Arm the abort for the backfill; every other statement (the ROLLBACK too) runs."""
        self.aborting = sql.startswith("UPDATE task_history SET is_cron = 1")

    def progress(self) -> int:
        """SQLite's progress callback: a non-zero return aborts the running statement."""
        return 1 if self.aborting else 0


def pre_is_cron_db(path: Path) -> None:
    """Create a database the current daemon wrote, minus ``is_cron``, holding one cron row.

    The table is recreated from its own ``CREATE TABLE`` text with the
    ``is_cron`` definition (and the comment lines above it) cut out:
    SQLite's ``DROP COLUMN`` of the last column looks back for the
    preceding comma and trips over the comma inside that comment
    ("incomplete input").
    """
    conn = sqlite3.connect(path, isolation_level=None)
    persistence._init_tables(conn)
    (sql,) = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'task_history'").fetchone()
    without_is_cron = re.sub(r",\s*(?:--[^\n]*\n\s*)*is_cron INTEGER DEFAULT 0", "", sql)
    assert "is_cron" not in without_is_cron
    conn.execute("DROP TABLE task_history")
    conn.execute(without_is_cron)
    conn.execute(
        "INSERT INTO task_history (id, timestamp, task, work_dir) VALUES ('cron', 1.0, ?, '/p')",
        (CRON_TASK_MARKER + " Check the mail.",),
    )
    conn.close()


def columns(path: Path, table: str) -> set[str]:
    """The column names of *table* in the database at *path*."""
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_an_interrupted_backfill_takes_its_column_with_it(tmp_path: Path) -> None:
    """An open cut off between the ALTER and the UPDATE leaves no trace; the next flags the row."""
    path = tmp_path / "history.db"
    pre_is_cron_db(path)
    assert "is_cron" not in columns(path, "task_history")

    conn = sqlite3.connect(path, isolation_level=None)  # autocommit, as the daemon opens it
    AbortBackfill(conn)
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        persistence._init_tables(conn)
    assert not conn.in_transaction
    conn.close()
    # The column did not outlive its backfill.
    assert "is_cron" not in columns(path, "task_history")

    conn = sqlite3.connect(path, isolation_level=None)
    persistence._init_tables(conn)
    assert conn.execute("SELECT is_cron FROM task_history WHERE id = 'cron'").fetchone() == (1,)
    conn.close()


def test_a_caller_owning_a_transaction_keeps_it(tmp_path: Path) -> None:
    """The migration joins a transaction its caller already opened instead of nesting one."""
    path = tmp_path / "history.db"
    pre_is_cron_db(path)
    conn = sqlite3.connect(path)
    conn.execute("BEGIN")
    persistence._add_missing_columns(conn)
    assert conn.in_transaction
    conn.rollback()
    assert "is_cron" not in columns(path, "task_history")
    with conn:  # ``backfill_task_metadata``'s style, on a non-autocommit connection
        persistence._init_tables(conn)
    conn.close()
    assert {"is_cron"} <= columns(path, "task_history")
    assert {"user_summary", "last_finished"} <= columns(path, "chat_summaries")


if __name__ == "__main__":
    pytest.main([__file__])

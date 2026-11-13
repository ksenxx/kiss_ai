# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""How a history table grows a column, and how older rows catch up.

``CREATE TABLE IF NOT EXISTS`` cannot extend a table that already exists,
so every column added to ``task_history`` or ``chat_summaries`` after
the first release is listed in :data:`ADDED_TASK_COLUMNS` or
:data:`ADDED_SUMMARY_COLUMNS` and ``ALTER TABLE``-d in by
:func:`add_missing_columns` when the daemon opens a database from
before it.  A few of those columns mean something for the rows written
before they existed, and :func:`complete_rows` works that meaning out
from the traces the rows carry -- today only ``is_cron``, which
:func:`backfill_is_cron` recovers from the cron scheduler's prompt
preamble and run directory.

Two programs run this code, which is why it has no dependency outside
the standard library:

* the daemon, in ``persistence._add_missing_columns``, once per database
  it opens;
* ``kiss.scripts.sync_db``, which ships this file to the machine it
  synchronizes a database with and runs it there, so that a database
  synced across two releases ends up with the same columns and the same
  flags as one the newer daemon migrated itself.

Both call it inside a ``BEGIN IMMEDIATE`` transaction, so the ``PRAGMA``
and the ``ALTER`` cannot be interleaved by another process and the
columns land together with their backfill, or not at all.
"""

from __future__ import annotations

import socket
import sqlite3
from collections.abc import Callable, Iterable

TASK_TABLE = "task_history"
CHAT_TABLE = "chat_summaries"

ADDED_TASK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("owner", "TEXT DEFAULT ''"),
    ("max_budget", "REAL DEFAULT 0.0"),
    ("is_side_channel", "INTEGER DEFAULT 0"),
    ("tags", "TEXT DEFAULT ''"),
    ("sea", "TEXT DEFAULT ''"),
    ("success", "INTEGER"),
    ("is_cron", "INTEGER DEFAULT 0"),
    ("model_costs", "TEXT DEFAULT ''"),
    ("machine", "TEXT DEFAULT ''"),
)
"""Columns of ``task_history`` added after the first release, oldest first.

Each entry is ``(name, declaration)``; the declaration is what follows
the name in ``ALTER TABLE task_history ADD COLUMN``, so it must be one
SQLite accepts there: no ``PRIMARY KEY``, no ``UNIQUE``, and ``NOT NULL``
only with a default.  Keep it in step with the ``CREATE TABLE`` in
``persistence._init_tables``.
"""

ADDED_SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("user_summary", "TEXT DEFAULT ''"),
    ("last_finished", "INTEGER DEFAULT 0"),
)
"""Columns of ``chat_summaries`` added after the first release, oldest first.

Same shape and rules as :data:`ADDED_TASK_COLUMNS`.  The rows a column
extends read its default until their chat's next recompute
(``last_finished`` 0: unknown) or until the user types one
(``user_summary``).
"""

CRON_TASK_MARKER = "You are running as an unattended scheduled automation (cron job)."
"""The first sentence of the cron scheduler's prompt-job preamble.

Every prompt job the scheduler ever launched has a task text that starts
with this sentence (``cron_agent.PROMPT_PREAMBLE`` builds on it), while
the rest of the preamble has been reworded over time -- so this, and not
the whole preamble, is what a row from an older release is recognised
by.
"""

CRON_RUN_DIR_PATTERN = "%/cron/runs/%"
"""``LIKE`` pattern of the run directory a job without a ``work_dir`` of its own ran in."""

SETTINGS_HOST_MARKER = '"host": "'
"""What precedes the host name in a ``task_settings`` event's JSON."""

MACHINE_INFO_MARKER = "Machine info: "
"""What precedes the host name in the Task Settings of a system prompt."""


def missing_columns(
    conn: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Return the listed columns the table lacks.

    A read-only check (``PRAGMA table_info`` needs no lock), so a
    caller can find out whether a migration is due before taking
    SQLite's write lock for it.

    Args:
        conn: Connection to the database.
        table: The table to inspect.
        columns: ``(name, declaration)`` pairs, as in
            :data:`ADDED_TASK_COLUMNS`.

    Returns:
        The absent ``(name, declaration)`` pairs, in declaration order.
    """
    present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return [(name, declaration) for name, declaration in columns if name not in present]


def add_missing_columns(
    conn: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]
) -> list[str]:
    """Add every listed column the table lacks.

    Args:
        conn: Connection holding a write transaction on the database.
        table: The table to extend.
        columns: ``(name, declaration)`` pairs, as in
            :data:`ADDED_TASK_COLUMNS`.

    Returns:
        The names of the columns this call added, in declaration order.
    """
    added: list[str] = []
    for name, declaration in missing_columns(conn, table, columns):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
        added.append(name)
    return added


def backfill_is_cron(conn: sqlite3.Connection) -> int:
    """Stamp ``is_cron`` on the rows a cron launch left recognisable traces on.

    Rows written by a release without the column carry no explicit
    marker, so the two traces a cron launch always left are used
    instead: the task text of a prompt job starts with
    :data:`CRON_TASK_MARKER` (slash-command jobs included), and a job
    without a ``work_dir`` of its own ran inside ``$KISS_HOME/cron/runs/``.
    The sub-tasks of such a run (``parent_task_id`` chains) are then
    stamped too, generation by generation, as the daemon marks them at
    launch.  Only ever sets the flag, never clears it, so running it
    again -- over rows that already carry the flag, or that were written
    with it -- changes nothing.

    Args:
        conn: Connection whose ``task_history`` rows to update; the
            table must have the ``is_cron`` column.

    Returns:
        The number of rows stamped.
    """
    marker = CRON_TASK_MARKER.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    stamped = conn.execute(
        f"UPDATE {TASK_TABLE} SET is_cron = 1 WHERE is_cron = 0"
        " AND (task LIKE ? ESCAPE '\\' OR work_dir LIKE ?)",
        (marker + "%", CRON_RUN_DIR_PATTERN),
    ).rowcount
    while True:
        children = conn.execute(
            f"UPDATE {TASK_TABLE} SET is_cron = 1 WHERE is_cron = 0 AND parent_task_id IN"
            f" (SELECT id FROM {TASK_TABLE} WHERE is_cron = 1)"
        ).rowcount
        if children <= 0:
            break
        stamped += children
    return stamped


COLUMN_BACKFILLS: dict[str, Callable[[sqlite3.Connection], int]] = {
    "is_cron": backfill_is_cron,
}
"""The columns whose value for older rows can be worked out, and how."""


def host_name() -> str:
    """Return this machine's short host name, what ``task_history.machine`` calls a machine."""
    return socket.gethostname().strip().split(".")[0] or "localhost"


def task_origins(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return ``(host, task id)`` for the unstamped rows whose first events name another host.

    A row a history sync copied in before ``machine`` existed reads ''
    like one of this machine's own, but the first events of every task
    since mid-2026 say where it ran: the system prompt's Task Settings
    line ``Machine info: <host> (<arch>)`` and, later, the
    ``task_settings`` event's ``host``.  Both are read straight from the
    ``(task_id, seq)`` index (``seq`` 0 and 1), but the system prompts
    are large, so on a history of several GB this takes about a minute:
    it is not a :data:`COLUMN_BACKFILLS` entry, which runs inside the
    migration's write transaction, and the caller runs it on a thread of
    its own.  A row whose events name no host, or this host, is left out.
    An event a foreign writer stored as a BLOB is read as its text.

    Args:
        conn: Connection to the database; only read.

    Returns:
        ``(short host name, task id)`` pairs, one per task.
    """
    here = host_name().lower()
    origins: dict[str, str] = {}
    event = "CAST(e.event_json AS TEXT)"
    after_marker = f"substr({event}, nullif(instr({event}, ?), 0) + ?, 120)"
    for task_id, settings_host, info_host in conn.execute(
        f"SELECT t.id, {after_marker}, {after_marker}"
        f" FROM {TASK_TABLE} t JOIN events e ON e.task_id = t.id AND e.seq <= 1"
        " WHERE COALESCE(t.machine, '') = ''",
        (SETTINGS_HOST_MARKER, len(SETTINGS_HOST_MARKER),
         MACHINE_INFO_MARKER, len(MACHINE_INFO_MARKER)),
    ):
        if settings_host:
            text = settings_host.split('"', 1)[0]
        else:
            text = (info_host or "").split(" ", 1)[0]
        host = text.split(".")[0]
        if host and host.lower() not in {here, "unknown"}:
            origins.setdefault(str(task_id), host)
    return [(host, task_id) for task_id, host in origins.items()]


def complete_rows(conn: sqlite3.Connection, columns: Iterable[str]) -> list[str]:
    """Work out, for the rows that predate them, the columns that can be.

    Called with the columns that some rows of the table were written
    without: the ones :func:`add_missing_columns` just added, or the
    ones the rows a sync merged in did not carry.  Each named column that
    has a backfill in :data:`COLUMN_BACKFILLS` gets it run; the others
    keep their declared default, which is their meaning for an older row
    (``success`` NULL: not recorded).

    Args:
        conn: Connection whose ``task_history`` rows to update.
        columns: Column names; unknown names and duplicates are ignored.

    Returns:
        The names of the columns that were backfilled, in the order of
        :data:`COLUMN_BACKFILLS`.
    """
    wanted = set(columns)
    done: list[str] = []
    for name, backfill in COLUMN_BACKFILLS.items():
        if name in wanted:
            backfill(conn)
            done.append(name)
    return done

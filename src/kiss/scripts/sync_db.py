#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""One-way synchronization of ``history.db``-shaped SQLite databases.

Copies rows of the task table (``task_history``) and the ``events``
table from SOURCE into TARGET, then recomputes TARGET's per-chat
summaries (``chat_summaries``) for the chats that received tasks.
Nothing is ever written to SOURCE and no other table of TARGET is
touched.

Both SOURCE and TARGET are unix-style paths to a SQLite database file,
optionally prefixed with ``user@host:`` when the database lives on a
machine reachable over ssh::

    uv run python -m kiss.scripts.sync_db ~/.kiss/history.db /tmp/backup.db
    uv run python -m kiss.scripts.sync_db ksen@1.2.3.4:~/.kiss/history.db \\
        ~/.kiss/history.db

How it works (three phases, each run on the machine that owns the
database, so a database file is never copied across machines):

1. ``_manifest`` runs on TARGET and emits a small gzipped JSON summary
   of what TARGET already has: every ``task_history.id`` it holds, each
   with the highest ``events.seq`` TARGET has for that task.
2. ``_extract`` runs on SOURCE, reads that summary, and builds a
   throw-away *delta* database holding only the rows TARGET is missing.
   The delta is streamed back gzipped.
3. ``_merge`` runs on TARGET, ``ATTACH``-es the delta and inserts the
   new rows in a single transaction.

Only the delta crosses the network, which keeps a repeated sync of a
multi-gigabyte ``history.db`` down to seconds.  Rows are matched by their
stable natural keys -- ``task_history.id`` for tasks and
``(task_id, seq)`` for events -- and ``events.id`` (an ``AUTOINCREMENT``
rowid that means different things in different databases) is never
copied, so syncing the same pair of databases twice is a no-op.

The history tables are append-only, and the sync leans on that: a task
row is never modified or deleted once written, and a task's events are
only ever appended, each with the next ``seq``, so the highest ``seq``
TARGET holds says exactly which of SOURCE's events it lacks -- the ones
above it, and nothing below.  Neither phase therefore reads the events
table itself: the manifest finds each task's highest ``seq`` with one
index seek per task, and the extract seeks straight to the events above
it, so a routine sync costs the same whether the databases hold a
thousand events or ten million.  A hole below a task's highest ``seq``
(which the writer never produces) is not looked for; ``--full`` ships
everything and fills one.  Every event belongs to a task row -- the
``events.task_id`` column references ``task_history`` -- so an event of
a task SOURCE has no row for never travels.

Both databases must have the same columns for the synced tables; a
mismatch is refused rather than half-applied.

Task ids are unique and a task row, once created, is never modified, so
the sync never has to compare row contents: a task travels exactly when
TARGET does not have its id, and a ``task_history`` row that exists on
both sides is always left as TARGET recorded it.

``chat_summaries`` is not history and is not synced as rows.  It is a
cache with one row per chat -- a few-word summary of the chat's tasks and
``last_launched``, the launch instant of its newest task -- that is
neither append-only nor maintained incrementally: the web app rewrites a
chat's row from scratch when one of its tasks finishes, a metadata
backfill may drop and rebuild the whole table, and a row can change
without any stamp on it moving.  No column of such a row can say which
of two copies is right, so none is compared: once the task rows are
merged, the merge recomputes the row of every chat that received a task
from TARGET's own, now complete, ``task_history`` -- the same code
(``kiss/agents/sorcar/chat_summary.py``) the web app runs when a task
finishes.  TARGET's rows for chats that received nothing are left as
they are; a TARGET from before the table existed gets no chat rows.

The remote side runs this very file through ``ssh <host> python3 -c
...``: the script and the chat-summary module it ships along with
itself are stdlib-only and self-contained, so nothing has to be
installed on the remote machine beyond ``python3``.

Usage:
    uv run python -m kiss.scripts.sync_db SOURCE TARGET [OPTIONS]

Options:
    --full          Ignore TARGET's manifest and ship every source row
                    (slow but assumes nothing about how rows were added)
    --dry-run       Perform the merge and roll it back, reporting exactly
                    what a real run would change (TARGET must be writable)
    --edit-delta COMMAND
                    Run a shell command on this machine against the
                    uncompressed delta before it is merged, ``{}``
                    standing for the delta's path (repeatable, in
                    order).  The delta holds the tables ``task_history``
                    and ``events`` with SOURCE's columns and only the
                    rows about to travel, so a
                    command can rewrite or drop rows cheaply -- re-point
                    recorded work directories, hold back a task that is
                    still running -- without a copy of either database.
                    A command that exits non-zero stops the sync before
                    anything is merged.
    --python PATH   Remote python interpreter (default: python3)
    --port PORT     ssh port
    -o OPT          Extra ``ssh -o`` option (repeatable)
    --quiet         Only print the final one-line summary
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gzip
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, BinaryIO

TASK_TABLE = "task_history"
EVENT_TABLE = "events"
CHAT_TABLE = "chat_summaries"
# Where each table's column list lives in the manifest.
MANIFEST_COLUMN_KEYS = {
    TASK_TABLE: "task_columns",
    EVENT_TABLE: "event_columns",
}
# The chat-summary code, stdlib-only, shipped to a remote host with this
# script (see ``_remote_shell_command``) and read from the checkout here.
CHAT_SUMMARY_MODULE = os.path.join("agents", "sorcar", "chat_summary.py")
COPY_BUFFER = 1 << 20

PHASE_MANIFEST = "_manifest"
PHASE_EXTRACT = "_extract"
PHASE_MERGE = "_merge"

# ``user@host`` or ``host``: everything ssh accepts before the colon.
_HOST_PATTERN = re.compile(r"^[A-Za-z0-9._-]+(@[A-Za-z0-9._-]+)?$")


class SyncError(Exception):
    """A synchronization step failed; the message is user-facing."""


# --------------------------------------------------------------------------
# locations: "[user@host:]/path/to/db"
# --------------------------------------------------------------------------


class Location:
    """A database file, either local or on an ssh-reachable machine."""

    def __init__(self, spec: str) -> None:
        """Parse a ``[user@host:]path`` database location.

        Args:
            spec: Unix-style path to a SQLite database, optionally
                prefixed with ``user@host:`` (or ``host:``).  A local
                relative path that itself contains a colon must be
                written with a ``./`` prefix.

        Raises:
            SyncError: If the specification is empty or has an empty
                host or path component.
        """
        if not spec or not spec.strip():
            raise SyncError("empty database location")
        if any(ch in spec for ch in "\r\n\0"):
            raise SyncError(f"illegal character in location {spec!r}")
        self.spec = spec
        head, sep, tail = spec.partition(":")
        if sep and not tail:
            raise SyncError(f"missing database path in {spec!r}")
        if sep and _is_remote_prefix(head, tail):
            self.host: str | None = head
            self.path = tail
        else:
            if sep and not head:
                raise SyncError(f"missing host in {spec!r}")
            self.host = None
            self.path = os.path.abspath(os.path.expanduser(spec))

    @property
    def is_remote(self) -> bool:
        """True when the database is reached over ssh."""
        return self.host is not None

    def __str__(self) -> str:
        """Render the location the way the user wrote it."""
        return f"{self.host}:{self.path}" if self.host else self.path


def _is_remote_prefix(head: str, tail: str) -> bool:
    """Decide whether the part before a colon names an ssh host.

    A bare relative file name that happens to contain a colon, such as
    ``a:b.db``, stays local; ``host:/path``, ``host:~/path`` and any
    ``user@host:path`` are remote.

    Args:
        head: Text before the first colon.
        tail: Text after the first colon.

    Returns:
        True when the location is on another machine.
    """
    if not head or not _HOST_PATTERN.match(head):
        return False
    return "@" in head or tail.startswith("/") or tail.startswith("~")


# --------------------------------------------------------------------------
# small sqlite / stream helpers
# --------------------------------------------------------------------------


def quote_name(name: str) -> str:
    """Quote an SQLite identifier for safe interpolation into SQL.

    Args:
        name: Table or column name.

    Returns:
        The name wrapped in double quotes with inner quotes doubled.
    """
    return '"' + name.replace('"', '""') + '"'


def open_db(path: str, must_exist: bool = True) -> sqlite3.Connection:
    """Open a SQLite database in autocommit mode.

    Args:
        path: Filesystem path of the database (``~`` is expanded).
        must_exist: When True, refuse to create a new empty database.

    Returns:
        An open connection with a generous busy timeout.

    Raises:
        SyncError: If ``must_exist`` and the file does not exist.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if must_exist and not os.path.isfile(path):
        raise SyncError(f"database not found: {path}")
    conn = sqlite3.connect(path, isolation_level=None, timeout=60.0)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def table_columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    """List the column names of a table in declaration order.

    Args:
        conn: Open connection.
        schema: Database name, e.g. ``main`` or an attached alias.
        table: Table name.

    Returns:
        Column names in declaration order.

    Raises:
        SyncError: If the table does not exist.
    """
    rows = conn.execute(
        f"PRAGMA {quote_name(schema)}.table_info({quote_name(table)})"
    ).fetchall()
    if not rows:
        raise SyncError(f"table {table!r} is missing from the {schema} database")
    return [r[1] for r in rows]


def rowid_alias_column(conn: sqlite3.Connection, schema: str, table: str) -> str | None:
    """Return the ``INTEGER PRIMARY KEY`` column of a table, if any.

    Such a column is an alias for the table's rowid: its values are local
    to one database file, so they must never be copied between databases,
    and it is the cheapest handle for looking a row up again.  A
    ``WITHOUT ROWID`` table has no such alias -- there the primary key is
    ordinary data that has to be copied like any other column.

    Args:
        conn: Open connection.
        schema: Database name, e.g. ``main`` or an attached alias.
        table: Table name.

    Returns:
        The column name, or None when the table has no rowid alias.
    """
    listing = conn.execute(
        f"PRAGMA {quote_name(schema)}.table_list({quote_name(table)})"
    ).fetchall()
    if listing and listing[0][4]:
        return None
    rows = conn.execute(
        f"PRAGMA {quote_name(schema)}.table_info({quote_name(table)})"
    ).fetchall()
    keys = [r for r in rows if r[5]]
    if len(keys) != 1:
        return None
    name, decl_type = keys[0][1], (keys[0][2] or "").strip().upper()
    return name if decl_type == "INTEGER" else None


def compare_columns(source: list[str], target: list[str], table: str) -> None:
    """Fail unless a source and a target table have the same columns.

    Column order may differ -- every statement names its columns -- but a
    differing column *set* means the two databases do not share a schema,
    and syncing them would corrupt row content or loop forever re-copying
    rows that can never come to match.

    Args:
        source: Column names on the source side.
        target: Column names on the target side.
        table: Table name, for the error message.

    Raises:
        SyncError: If the two column sets differ.
    """
    only_source = sorted(set(source) - set(target))
    only_target = sorted(set(target) - set(source))
    if not only_source and not only_target:
        return
    details = []
    if only_source:
        details.append(f"missing from the target: {', '.join(only_source)}")
    if only_target:
        details.append(f"missing from the source: {', '.join(only_target)}")
    raise SyncError(
        f"the source and target schemas for {table!r} differ ({'; '.join(details)});"
        " both databases must have the same schema"
    )


def require_columns(columns: list[str], needed: tuple[str, ...], where: str) -> None:
    """Fail unless every required column is present.

    Args:
        columns: Columns that exist.
        needed: Columns the sync depends on.
        where: Human-readable description used in the error message.

    Raises:
        SyncError: If any required column is missing.
    """
    absent = [c for c in needed if c not in columns]
    if absent:
        raise SyncError(f"{where} lacks required column(s): {', '.join(absent)}")


def temp_path(suffix: str) -> str:
    """Create an empty temporary file and return its path.

    Args:
        suffix: File name suffix, e.g. ``".db"``.

    Returns:
        Path of a fresh, empty, user-only readable file.
    """
    fd, path = tempfile.mkstemp(prefix="kiss-sync-", suffix=suffix)
    os.close(fd)
    return path


def write_json_gz(payload: dict[str, Any], out: BinaryIO) -> None:
    """Write a JSON payload gzipped to a binary stream.

    Args:
        payload: JSON-serializable object.
        out: Destination binary stream.
    """
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=6, mtime=0) as gz:
        gz.write(json.dumps(payload).encode("utf-8"))
    out.flush()


def read_json_gz(inp: BinaryIO) -> dict[str, Any]:
    """Read a gzipped JSON payload from a binary stream.

    Args:
        inp: Source binary stream.

    Returns:
        The decoded object, or an empty dict when the stream is empty.
    """
    with gzip.GzipFile(fileobj=inp, mode="rb") as gz:
        raw = gz.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def write_file_gz(path: str, out: BinaryIO) -> None:
    """Stream a file gzipped to a binary stream.

    Args:
        path: File to send.
        out: Destination binary stream.
    """
    with open(path, "rb") as src:
        with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=6, mtime=0) as gz:
            shutil.copyfileobj(src, gz, COPY_BUFFER)
    out.flush()


def read_file_gz(inp: BinaryIO, path: str) -> None:
    """Read a gzipped stream into a file.

    Args:
        inp: Source binary stream.
        path: File to (over)write.
    """
    with gzip.GzipFile(fileobj=inp, mode="rb") as gz:
        with open(path, "wb") as dst:
            shutil.copyfileobj(gz, dst, COPY_BUFFER)


# --------------------------------------------------------------------------
# phase 1: manifest of what the target already has
# --------------------------------------------------------------------------


def phase_manifest(path: str, args: list[str], inp: BinaryIO, out: BinaryIO) -> None:
    """Emit the target's sync manifest as gzipped JSON.

    The manifest maps every task id the target holds to the highest
    ``events.seq`` it has for that task (``null`` when it has none).
    That is all the source needs under the append-only rule the history
    follows: a task row is immutable once created, so an id the target
    holds never has to travel again, and a task's events are only ever
    appended with increasing ``seq``, so the source's events above that
    number are exactly the ones the target lacks.  Each highest ``seq``
    is one seek in the ``(task_id, seq)`` index, so the events table is
    never scanned however many rows it holds.

    Args:
        path: Target database path.
        args: Unused; present for a uniform phase signature.
        inp: Unused; present for a uniform phase signature.
        out: Destination binary stream for the gzipped JSON.
    """
    del args, inp
    conn = open_db(path)
    try:
        task_cols = table_columns(conn, "main", TASK_TABLE)
        event_cols = table_columns(conn, "main", EVENT_TABLE)
        require_columns(task_cols, ("id",), f"{TASK_TABLE} in the target database")
        require_columns(
            event_cols, ("task_id", "seq"), f"{EVENT_TABLE} in the target database"
        )
        tasks = {
            str(task_id): high
            for task_id, high in conn.execute(
                f'SELECT t."id", (SELECT MAX(e."seq") FROM main.{quote_name(EVENT_TABLE)} e'
                ' WHERE e."task_id" = t."id")'
                f' FROM main.{quote_name(TASK_TABLE)} t WHERE t."id" IS NOT NULL'
            )
        }
    finally:
        conn.close()
    manifest: dict[str, Any] = {
        MANIFEST_COLUMN_KEYS[TASK_TABLE]: sorted(task_cols),
        MANIFEST_COLUMN_KEYS[EVENT_TABLE]: sorted(event_cols),
        "tasks": tasks,
    }
    write_json_gz(manifest, out)


def has_table(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    """Report whether *table* exists in the *schema* database of *conn*."""
    row = conn.execute(
        f"SELECT 1 FROM {quote_name(schema)}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


# --------------------------------------------------------------------------
# phase 2: delta database built on the source
# --------------------------------------------------------------------------


def phase_extract(path: str, args: list[str], inp: BinaryIO, out: BinaryIO) -> None:
    """Build a delta database of rows the target lacks and stream it out.

    The whole extract is one transaction: one read transaction on the
    source, so a live database is seen at a single instant, and one write
    transaction on the delta.  The delta is a throw-away file that is
    rebuilt from scratch on any failure, so it is written without a
    journal and without fsync -- row by row in autocommit mode, every
    insert used to wait for the disk, and loading the manifest alone took
    tens of seconds for a few tens of thousands of tasks.

    Args:
        path: Source database path.
        args: Unused; present for a uniform phase signature.
        inp: Gzipped JSON manifest produced by :func:`phase_manifest`.
        out: Destination binary stream for the gzipped delta database.
    """
    del args
    manifest = read_json_gz(inp)
    source = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(source):
        raise SyncError(f"database not found: {source}")
    delta_path = temp_path(".db")
    try:
        conn = sqlite3.connect(delta_path, isolation_level=None, timeout=60.0)
        try:
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA main.journal_mode=OFF")
            conn.execute("PRAGMA main.synchronous=OFF")
            conn.execute("ATTACH DATABASE ? AS src", (source,))
            _create_delta_tables(conn)
            for table, key in MANIFEST_COLUMN_KEYS.items():
                if manifest.get(key) and has_table(conn, "src", table):
                    compare_columns(
                        table_columns(conn, "src", table), list(manifest[key]), table
                    )
            conn.execute("BEGIN")
            _load_target_tasks(conn, manifest.get("tasks") or {})
            _extract_tasks(conn)
            _extract_events(conn)
            conn.execute("COMMIT")
            conn.execute("DETACH DATABASE src")
        finally:
            conn.close()
        write_file_gz(delta_path, out)
    finally:
        _unlink(delta_path)


def _create_delta_tables(conn: sqlite3.Connection) -> None:
    """Recreate the source's task and event tables in the delta database.

    Args:
        conn: Connection to the delta database with the source attached
            as ``src``.

    Raises:
        SyncError: If the source lacks the task or the event table.
    """
    for table in (TASK_TABLE, EVENT_TABLE):
        row = conn.execute(
            "SELECT sql FROM src.sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not row or not row[0]:
            raise SyncError(f"table {table!r} is missing from the source database")
        conn.execute(row[0])


def _load_target_tasks(conn: sqlite3.Connection, tasks: dict[str, Any]) -> None:
    """Load the manifest's tasks into a temporary table the extract joins against.

    Args:
        conn: Connection to the delta database, inside its transaction.
        tasks: Manifest entries ``{task_id: highest seq or None}``; empty
            for a ``--full`` sync.
    """
    conn.execute(
        'CREATE TEMP TABLE "_sync_have" (task_id TEXT PRIMARY KEY, high_seq INTEGER)'
    )
    conn.executemany(
        'INSERT OR REPLACE INTO temp."_sync_have" VALUES (?, ?)', list(tasks.items())
    )


def _extract_tasks(conn: sqlite3.Connection) -> None:
    """Copy the task rows whose id the target does not have into the delta.

    Task ids are unique and a task row is never modified after it is
    created, so an id the target already holds identifies a row that is
    already there in full and never has to travel.  A row with a NULL id
    cannot be matched across databases and would be duplicated on every
    run, so such rows are skipped.

    Args:
        conn: Connection to the delta database with the source attached
            and the target's tasks loaded by :func:`_load_target_tasks`.
    """
    columns = table_columns(conn, "src", TASK_TABLE)
    require_columns(columns, ("id",), f"{TASK_TABLE} in the source database")
    table = quote_name(TASK_TABLE)
    names = ", ".join(quote_name(c) for c in columns)
    conn.execute(
        f"INSERT INTO main.{table} ({names})"
        f" SELECT {', '.join('s.' + quote_name(c) for c in columns)} FROM src.{table} s"
        ' LEFT JOIN temp."_sync_have" w ON w.task_id = s."id"'
        ' WHERE s."id" IS NOT NULL AND w.task_id IS NULL'
    )


def _extract_events(conn: sqlite3.Connection) -> None:
    """Copy the events the target is missing into the delta database.

    Events are append-only with increasing ``seq``, so the target lacks
    exactly the events above the highest ``seq`` it reported for a task,
    and every event of a task it reported no events for.  Both are found
    by seeking the source's ``(task_id, seq)`` index once per task; no
    event the target already has is ever read, nor is the events table
    scanned.  Events whose ``task_id`` has no ``task_history`` row are
    never looked at: the column references that table.

    Args:
        conn: Connection to the delta database with the source attached
            and the target's tasks loaded by :func:`_load_target_tasks`.
    """
    columns = table_columns(conn, "src", EVENT_TABLE)
    require_columns(columns, ("task_id", "seq"), f"{EVENT_TABLE} in the source database")
    rowid_alias = rowid_alias_column(conn, "src", EVENT_TABLE)
    copied = [c for c in columns if c != rowid_alias]
    table = quote_name(EVENT_TABLE)
    insert = (
        f"INSERT INTO main.{table} ({', '.join(quote_name(c) for c in copied)})"
        f" SELECT {', '.join('s.' + quote_name(c) for c in copied)}"
    )
    # CROSS JOIN pins the loop order: tasks on the outside, one index seek
    # into the events on the inside.
    conn.execute(
        f"{insert} FROM src.{quote_name(TASK_TABLE)} t"
        ' LEFT JOIN temp."_sync_have" w ON w.task_id = t."id"'
        f' CROSS JOIN src.{table} s ON s."task_id" = t."id"'
        ' WHERE w.high_seq IS NULL AND s."seq" IS NOT NULL'
    )
    conn.execute(
        f'{insert} FROM temp."_sync_have" w'
        f' CROSS JOIN src.{table} s ON s."task_id" = w.task_id AND s."seq" > w.high_seq'
        " WHERE w.high_seq IS NOT NULL"
    )
    _warn_on_duplicate_event_keys(conn)


def _warn_on_duplicate_event_keys(conn: sqlite3.Connection) -> None:
    """Warn when the extracted events repeat a ``(task_id, seq)`` key.

    Such rows cannot be told apart by a sync, so the target keeps
    whichever of them its own constraints allow.

    Args:
        conn: Connection to the delta database.
    """
    duplicates = conn.execute(
        f"SELECT COUNT(*) FROM (SELECT 1 FROM main.{quote_name(EVENT_TABLE)}"
        ' GROUP BY "task_id", "seq" HAVING COUNT(*) > 1)'
    ).fetchone()[0]
    if duplicates:
        print(
            f"warning: the source has {duplicates} (task_id, seq) pair(s) shared by"
            " more than one event row",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------
# phase 3: merge the delta into the target
# --------------------------------------------------------------------------


def phase_merge(path: str, args: list[str], inp: BinaryIO, out: BinaryIO) -> None:
    """Merge a gzipped delta database into the target and report counts.

    Args:
        path: Target database path.
        args: ``commit`` or ``rollback``; committing is the default.
        inp: Gzipped delta database from :func:`phase_extract`.
        out: Destination binary stream for the JSON statistics.
    """
    commit = not args or args[0] == "commit"
    delta_path = temp_path(".db")
    try:
        read_file_gz(inp, delta_path)
        stats = _apply_delta(path, delta_path, commit)
    finally:
        _unlink(delta_path)
    out.write(json.dumps(stats).encode("utf-8"))
    out.flush()


def _apply_delta(target_path: str, delta_path: str, commit: bool) -> dict[str, int]:
    """Insert every row of a delta database into the target database.

    Args:
        target_path: Target database path.
        delta_path: Path of the plain (unzipped) delta database.
        commit: When False the merge is rolled back after counting the
            rows it would have changed, which is how ``--dry-run``
            reports exactly what a real run would do.

    Returns:
        Counts of the task rows and event rows inserted and of the chat
        summary rows recomputed.
    """
    conn = open_db(target_path)
    try:
        conn.execute("ATTACH DATABASE ? AS delta", (delta_path,))
        for table in (TASK_TABLE, EVENT_TABLE):
            compare_columns(
                table_columns(conn, "delta", table),
                table_columns(conn, "main", table),
                table,
            )
        conn.execute("BEGIN IMMEDIATE")
        try:
            tasks_inserted = _merge_tasks(conn)
            events_inserted = _merge_events(conn)
            chats_refreshed = _merge_chats(conn)
            conn.execute("COMMIT" if commit else "ROLLBACK")
        except BaseException as exc:
            conn.execute("ROLLBACK")
            if isinstance(exc, sqlite3.OperationalError) and "readonly" in str(exc):
                raise SyncError(
                    f"cannot write to {target_path}: {exc}. Even a dry run needs"
                    " write access: it performs the merge and rolls it back"
                ) from exc
            raise
        conn.execute("DETACH DATABASE delta")
    finally:
        conn.close()
    return {
        "tasks_inserted": tasks_inserted,
        "events_inserted": events_inserted,
        "chats_refreshed": chats_refreshed,
    }


def _merge_tasks(conn: sqlite3.Connection) -> int:
    """Insert the delta's task rows the target does not already have.

    Task ids are unique and a task row is immutable once created, so an
    id that is already in the target denotes the very same row and the
    incoming copy is simply skipped; the target's rows are never updated.

    Args:
        conn: Target connection with the delta attached as ``delta``.

    Returns:
        The number of task rows inserted.
    """
    columns = table_columns(conn, "main", TASK_TABLE)
    require_columns(columns, ("id",), f"{TASK_TABLE} in the target database")
    table = quote_name(TASK_TABLE)
    names = ", ".join(quote_name(c) for c in columns)
    sql = (
        f"INSERT INTO main.{table} ({names})"
        f" SELECT {names} FROM delta.{table} WHERE \"id\" IS NOT NULL"
        ' ON CONFLICT("id") DO NOTHING'
    )
    return conn.execute(sql).rowcount


def _merge_events(conn: sqlite3.Connection) -> int:
    """Insert delta events the target does not already have.

    Rows are matched on ``(task_id, seq)`` rather than on ``events.id``,
    which is a per-database rowid.  When the target has a unique index on
    that pair, the duplicates are skipped by an ``ON CONFLICT`` clause
    naming it, which is a single pass over the delta; otherwise an
    anti-join filters them out.  Either way a repeated merge is a no-op
    and a genuine constraint violation still aborts the transaction
    instead of quietly dropping rows.

    Args:
        conn: Target connection with the delta attached as ``delta``.

    Returns:
        The number of event rows inserted.
    """
    columns = table_columns(conn, "main", EVENT_TABLE)
    require_columns(
        columns, ("task_id", "seq"), f"{EVENT_TABLE} in the target database"
    )
    copied = [c for c in columns if c != rowid_alias_column(conn, "main", EVENT_TABLE)]
    table = quote_name(EVENT_TABLE)
    sql = (
        f"INSERT INTO main.{table}"
        f" ({', '.join(quote_name(c) for c in copied)})"
        f" SELECT {', '.join('d.' + quote_name(c) for c in copied)}"
        f" FROM delta.{table} d"
    )
    if _has_unique_event_key(conn):
        sql += ' WHERE true ON CONFLICT("task_id", "seq") DO NOTHING'
    else:
        sql += (
            f" WHERE NOT EXISTS (SELECT 1 FROM main.{table} t"
            ' WHERE t."task_id" = d."task_id" AND t."seq" = d."seq")'
        )
    return conn.execute(sql).rowcount


def _merge_chats(conn: sqlite3.Connection) -> int:
    """Recompute the target's summary row of every chat the delta brought tasks for.

    The delta's task rows name the chats whose ``task_history`` just
    changed on the target (the few rows the target already had, skipped
    by :func:`_merge_tasks`, cost one harmless recomputation).  Each such
    chat's ``chat_summaries`` row is rebuilt from the target's merged
    tasks by the web app's own code, so it is exactly what the app would
    have written had those tasks finished here; chats that received
    nothing keep their rows.  Sub-agent rows do not count: the summary is
    computed from the chat's listable tasks only.  A target without the
    table (or from before tasks carried a chat) gets no rows.

    Args:
        conn: Target connection with the delta attached as ``delta``.

    Returns:
        The number of chat summary rows recomputed.
    """
    if not has_table(conn, "main", CHAT_TABLE):
        return 0
    require_columns(
        table_columns(conn, "main", CHAT_TABLE),
        ("chat_id", "summary", "last_launched"),
        f"{CHAT_TABLE} in the target database",
    )
    if not {"chat_id", "parent_task_id", "start_ts"} <= set(
        table_columns(conn, "main", TASK_TABLE)
    ):
        return 0
    chat_ids = [
        row[0]
        for row in conn.execute(
            f'SELECT DISTINCT "chat_id" FROM delta.{quote_name(TASK_TABLE)}'
            ' WHERE "chat_id" IS NOT NULL AND "chat_id" != \'\''
            ' AND ("parent_task_id" IS NULL OR "parent_task_id" = \'\')'
        )
    ]
    summaries = _chat_summary_module()
    for chat_id in chat_ids:
        summaries.upsert_chat_summary(conn, chat_id)
    return len(chat_ids)


def _chat_summary_module() -> Any:
    """Load the chat-summary code as a module.

    On a remote host the source arrives with this script, bound to the
    ``CHAT_SUMMARY_SOURCE`` global by the ssh bootstrap; here it is read
    from the checkout this script is part of.  Either way the module is
    built from the text rather than imported, so it needs neither the
    package on ``sys.path`` nor anything installed.

    Returns:
        The module, with ``upsert_chat_summary``.
    """
    source = globals().get("CHAT_SUMMARY_SOURCE") or _read_source(_chat_summary_path())
    module = types.ModuleType("chat_summary")
    exec(compile(source, "chat_summary.py", "exec"), module.__dict__)
    return module


def _has_unique_event_key(conn: sqlite3.Connection) -> bool:
    """Report whether the target enforces unique ``(task_id, seq)`` pairs.

    Args:
        conn: Target connection.

    Returns:
        True when a unique index covers exactly ``(task_id, seq)``.
    """
    table = quote_name(EVENT_TABLE)
    for index in conn.execute(f"PRAGMA main.index_list({table})").fetchall():
        is_unique, is_partial = index[2], index[4] if len(index) > 4 else 0
        if not is_unique or is_partial:
            continue
        indexed = sorted(
            str(row[2])
            for row in conn.execute(
                f"PRAGMA main.index_info({quote_name(index[1])})"
            ).fetchall()
        )
        if indexed == ["seq", "task_id"]:
            return True
    return False


def _unlink(path: str) -> None:
    """Delete a file, ignoring a missing file.

    Args:
        path: File to remove.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


PHASES = {
    PHASE_MANIFEST: phase_manifest,
    PHASE_EXTRACT: phase_extract,
    PHASE_MERGE: phase_merge,
}


# --------------------------------------------------------------------------
# driving the phases locally or over ssh
# --------------------------------------------------------------------------


class Runner:
    """Runs sync phases on the machine that owns each database."""

    def __init__(self, python: str, port: str | None, ssh_options: list[str]) -> None:
        """Configure how remote phases are launched.

        Args:
            python: Remote python interpreter command.
            port: ssh port, or None for the default.
            ssh_options: Extra ``-o`` options passed to ssh.
        """
        self.python = python
        self.port = port
        self.ssh_options = ssh_options

    def run(
        self, location: Location, phase: str, args: list[str], inp: str | None, out: str
    ) -> None:
        """Run one phase against a database, writing its output to a file.

        Args:
            location: Database the phase operates on.
            phase: One of the ``PHASE_*`` names.
            args: Extra string arguments for the phase.
            inp: Path of the phase's input stream, or None for no input.
            out: Path the phase's output stream is written to.

        Raises:
            SyncError: If a remote phase exits non-zero.
        """
        if location.is_remote:
            self._run_remote(location, phase, args, inp, out)
            return
        with open(out, "wb") as out_file:
            if inp is None:
                PHASES[phase](location.path, args, _EMPTY_STREAM, out_file)
            else:
                with open(inp, "rb") as in_file:
                    PHASES[phase](location.path, args, in_file, out_file)

    def _run_remote(
        self, location: Location, phase: str, args: list[str], inp: str | None, out: str
    ) -> None:
        """Run one phase on a remote host through ssh.

        Args:
            location: Remote database the phase operates on.
            phase: One of the ``PHASE_*`` names.
            args: Extra string arguments for the phase.
            inp: Path of the phase's input stream, or None for no input.
            out: Path the phase's output stream is written to.

        Raises:
            SyncError: If ssh or the remote phase fails.
        """
        command = ["ssh", "-o", "BatchMode=yes"]
        for option in self.ssh_options:
            command += ["-o", option]
        if self.port:
            command += ["-p", self.port]
        command += [str(location.host), self._remote_shell_command(location, phase, args)]
        with contextlib.ExitStack() as streams:
            out_file = streams.enter_context(open(out, "wb"))
            in_file: Any = subprocess.DEVNULL
            if inp:
                in_file = streams.enter_context(open(inp, "rb"))
            done = subprocess.run(
                command,
                stdin=in_file,
                stdout=out_file,
                stderr=subprocess.PIPE,
                check=False,
            )
        stderr = done.stderr.decode("utf-8", "replace").strip()
        if done.returncode != 0:
            raise SyncError(
                f"{phase} failed on {location.host} (exit {done.returncode})"
                + (f": {stderr}" if stderr else "")
            )
        if stderr:
            print(f"[{location.host}] {stderr}", file=sys.stderr)

    def _remote_shell_command(
        self, location: Location, phase: str, args: list[str]
    ) -> str:
        """Build the shell command that runs this script on a remote host.

        The script's own source and the chat-summary module's are shipped
        inline, base64 encoded, so the remote machine needs nothing but a
        python interpreter.

        Args:
            location: Remote database the phase operates on.
            phase: One of the ``PHASE_*`` names.
            args: Extra string arguments for the phase.

        Returns:
            A single shell command string for ``ssh``.
        """
        script = base64.b64encode(_read_source(_script_path())).decode("ascii")
        summaries = base64.b64encode(_read_source(_chat_summary_path())).decode("ascii")
        bootstrap = (
            f"import base64;CHAT_SUMMARY_SOURCE=base64.b64decode('{summaries}');"
            f"exec(base64.b64decode('{script}'))"
        )
        parts = [self.python, "-c", bootstrap, phase, location.path, *args]
        return " ".join(shlex.quote(p) for p in parts)


def _script_path() -> str:
    """Return the path of this script's source file.

    Raises:
        SyncError: When running from exec'd text with no file behind it.
    """
    path = globals().get("__file__")
    if not path:
        raise SyncError("cannot locate this script's source for remote execution")
    return str(path)


def _chat_summary_path() -> str:
    """Return the path of the chat-summary module in the checkout this script is part of."""
    return os.path.join(
        os.path.dirname(os.path.abspath(_script_path())), os.pardir, CHAT_SUMMARY_MODULE
    )


def _read_source(path: str) -> bytes:
    """Read a source file of this checkout, for shipping to or running on a host.

    Args:
        path: The file to read.

    Returns:
        The bytes of the file.

    Raises:
        SyncError: If the file cannot be read.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise SyncError(f"cannot read {path}: {exc}") from exc


class _EmptyStream:
    """A readable binary stream that is always at end of file."""

    def read(self, size: int = -1) -> bytes:
        """Return no data.

        Args:
            size: Ignored.

        Returns:
            An empty bytes object.
        """
        del size
        return b""


_EMPTY_STREAM: Any = _EmptyStream()


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def synchronize(
    source: Location,
    target: Location,
    runner: Runner,
    full: bool = False,
    dry_run: bool = False,
    delta_edits: list[str] | None = None,
) -> dict[str, Any]:
    """Copy task and event rows from a source database into a target and refresh its chat summaries.

    Args:
        source: Database rows are read from; never modified.
        target: Database rows are written to.
        runner: Launches each phase locally or over ssh.
        full: Ship every source row instead of only the target's gaps.
        dry_run: Roll the merge back instead of committing it, so the
            target is left untouched but the reported counts are the ones
            a real run would apply.
        delta_edits: Shell commands run on this machine, in order,
            against the uncompressed delta before it is merged, ``{}``
            standing for the delta's path (see :func:`edit_delta`).

    Returns:
        Statistics with the compressed delta size, the rows inserted and
        the elapsed wall-clock seconds.

    Raises:
        SyncError: If source and target are the same database, or if any
            phase or delta edit fails.
    """
    if (source.host, source.path) == (target.host, target.path):
        raise SyncError("source and target are the same database")
    started = time.monotonic()
    temporary: list[str] = []
    try:
        manifest_path = _track(temporary, ".json.gz")
        delta_path = _track(temporary, ".db.gz")
        stats_path = _track(temporary, ".json")
        if full:
            with open(manifest_path, "wb") as handle:
                write_json_gz({}, handle)
        else:
            runner.run(target, PHASE_MANIFEST, [], None, manifest_path)
        runner.run(source, PHASE_EXTRACT, [], manifest_path, delta_path)
        if delta_edits:
            edit_delta(delta_path, delta_edits)
        commit = "rollback" if dry_run else "commit"
        runner.run(target, PHASE_MERGE, [commit], delta_path, stats_path)
        with open(stats_path, "rb") as handle:
            raw = handle.read()
        if not raw:
            raise SyncError("the merge phase produced no result")
        stats: dict[str, Any] = {"delta_bytes": os.path.getsize(delta_path)}
        stats.update(json.loads(raw.decode("utf-8")))
    finally:
        for path in temporary:
            _unlink(path)
    stats["seconds"] = round(time.monotonic() - started, 2)
    stats["dry_run"] = dry_run
    return stats


def local_shell_quote(arg: str) -> str:
    """Quote *arg* for the shell ``subprocess.run(..., shell=True)`` uses on this machine.

    That is ``/bin/sh`` everywhere but Windows, where it is ``cmd.exe``:
    it knows nothing of POSIX single quotes, and ``&``, ``|``, ``<``,
    ``>`` and ``^`` are literal only inside double quotes.  A file name
    cannot contain a double quote on Windows, so a word that does (not a
    path) gets the C runtime's quoting instead.

    Args:
        arg: The word to quote.

    Returns:
        *arg* quoted as one shell word.
    """
    if os.name != "nt":
        return shlex.quote(arg)
    if '"' in arg:
        return subprocess.list2cmdline([arg])
    return f'"{arg}"'


def edit_delta(delta_gz: str, commands: list[str]) -> None:
    """Run shell commands against the delta between the extract and the merge.

    The delta is unpacked to a temporary file, every command is run in
    order with ``{}`` replaced by that file's (shell-quoted) path, and the
    result is packed again in place of the original.  The commands
    inherit this process's standard streams.  A command that exits
    non-zero stops the sync before anything is merged.

    Args:
        delta_gz: Path of the gzipped delta database; rewritten.
        commands: Shell commands to run, each naming the delta as ``{}``.

    Raises:
        SyncError: If a command exits non-zero.
    """
    plain = temp_path(".db")
    try:
        with open(delta_gz, "rb") as inp:
            read_file_gz(inp, plain)
        for command in commands:
            shell_command = command.replace("{}", local_shell_quote(plain))
            done = subprocess.run(shell_command, shell=True, check=False)
            if done.returncode != 0:
                raise SyncError(f"delta edit exited with {done.returncode}: {command}")
        with open(delta_gz, "wb") as out:
            write_file_gz(plain, out)
    finally:
        _unlink(plain)


def _track(paths: list[str], suffix: str) -> str:
    """Create a temporary file and remember it for later cleanup.

    Args:
        paths: List every created path is appended to.
        suffix: File name suffix.

    Returns:
        Path of the new empty file.
    """
    path = temp_path(suffix)
    paths.append(path)
    return path


def format_stats(source: Location, target: Location, stats: dict[str, Any]) -> str:
    """Render sync statistics as one human-readable line.

    Args:
        source: Source location.
        target: Target location.
        stats: Result of :func:`synchronize`.

    Returns:
        A single summary line.
    """
    chats = stats.get("chats_refreshed", 0)
    if stats["dry_run"]:
        applied = (
            f"would add {stats['tasks_inserted']} task row(s)"
            f" and {stats['events_inserted']} event row(s)"
            f" and refresh {chats} chat summary row(s)"
        )
    else:
        applied = (
            f"{stats['tasks_inserted']} task row(s) added,"
            f" {stats['events_inserted']} event row(s) added,"
            f" {chats} chat summary row(s) refreshed"
        )
    return (
        f"{source} -> {target}: {applied}"
        f" [delta {stats['delta_bytes'] / 1024:.1f} KiB,"
        f" {stats['seconds']}s]"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="sync_db",
        description=(
            "One-way sync of the task_history and events tables of history.db-shaped "
            "SQLite databases, recomputing the target's chat_summaries rows of the "
            "chats that received tasks. SOURCE and TARGET are "
            "unix paths, optionally prefixed with user@host: for a "
            "database reachable over ssh."
        ),
    )
    parser.add_argument("source", help="[user@host:]/path/to/source.db (read only)")
    parser.add_argument("target", help="[user@host:]/path/to/target.db (updated)")
    parser.add_argument(
        "--full",
        action="store_true",
        help="ship every source row instead of only the target's gaps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "report exactly what a real run would change and roll it back"
            " (the target must still be writable)"
        ),
    )
    parser.add_argument(
        "--edit-delta",
        dest="delta_edits",
        action="append",
        default=[],
        metavar="COMMAND",
        help=(
            "shell command run here against the uncompressed delta before it is"
            " merged, {} standing for the delta's path (repeatable, run in order;"
            " a non-zero exit stops the sync)"
        ),
    )
    parser.add_argument(
        "--python", default="python3", help="remote python interpreter (default: python3)"
    )
    parser.add_argument("--port", help="ssh port")
    parser.add_argument(
        "-o",
        dest="ssh_options",
        action="append",
        default=[],
        metavar="OPTION",
        help="extra ssh -o option (repeatable)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only print the final summary line"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the synchronization or an internal remote phase.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 on a synchronization error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in PHASES:
        return _run_phase_from_argv(args)
    options = build_parser().parse_args(args)
    try:
        source = Location(options.source)
        target = Location(options.target)
        if not options.quiet:
            print(f"syncing {source} -> {target} ...", file=sys.stderr)
        stats = synchronize(
            source,
            target,
            Runner(options.python, options.port, options.ssh_options),
            full=options.full,
            dry_run=options.dry_run,
            delta_edits=options.delta_edits,
        )
    except (SyncError, sqlite3.Error) as exc:
        print(f"sync_db: {exc}", file=sys.stderr)
        return 1
    print(format_stats(source, target, stats))
    return 0


def _run_phase_from_argv(args: list[str]) -> int:
    """Execute one phase named on the command line, on stdin/stdout.

    This entry point is what the remote side of an ssh sync runs.

    Args:
        args: ``[phase, db_path, *phase_args]``.

    Returns:
        Process exit status: 0 on success, 1 on a synchronization error.
    """
    if len(args) < 2:
        print(f"sync_db: {args[0]} needs a database path", file=sys.stderr)
        return 1
    try:
        PHASES[args[0]](args[1], args[2:], sys.stdin.buffer, sys.stdout.buffer)
    except (SyncError, sqlite3.Error) as exc:
        print(f"sync_db: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

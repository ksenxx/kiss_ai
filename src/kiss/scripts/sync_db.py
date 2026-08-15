#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""One-way synchronization of ``sorcar.db``-shaped SQLite databases.

Copies rows of the task table (``task_history``) and the ``events``
table from SOURCE into TARGET.  Nothing is ever written to SOURCE and no
other table of TARGET is touched.

Both SOURCE and TARGET are unix-style paths to a SQLite database file,
optionally prefixed with ``user@host:`` when the database lives on a
machine reachable over ssh::

    uv run python -m kiss.scripts.sync_db ~/.kiss/sorcar.db /tmp/backup.db
    uv run python -m kiss.scripts.sync_db ksen@1.2.3.4:~/.kiss/sorcar.db \\
        ~/.kiss/sorcar.db

How it works (three phases, each run on the machine that owns the
database, so a database file is never copied across machines):

1. ``_manifest`` runs on TARGET and emits a small gzipped JSON summary
   of what TARGET already has: a content digest per ``task_history.id``,
   and the lowest and highest ``events.seq`` plus the number of distinct
   sequence numbers per task.
2. ``_extract`` runs on SOURCE, reads that summary, and builds a
   throw-away *delta* database holding only the rows TARGET is missing
   or that differ.  The delta is streamed back gzipped.
3. ``_merge`` runs on TARGET, ``ATTACH``-es the delta and inserts the
   new rows in a single transaction.

Only the delta crosses the network, which keeps a repeated sync of a
multi-gigabyte ``sorcar.db`` down to seconds.  Rows are matched by their
stable natural keys -- ``task_history.id`` for tasks and
``(task_id, seq)`` for events -- and ``events.id`` (an ``AUTOINCREMENT``
rowid that means different things in different databases) is never
copied, so syncing the same pair of databases twice is a no-op.  When
TARGET holds an unbroken run of a task's events, only events outside
that run travel; when the run has a gap -- left by an interrupted
earlier sync, say -- the source ships that task's events in full and the
gap heals.

Both databases must have the same columns for the two tables; a mismatch
is refused rather than half-applied.

A ``task_history`` row that exists on both sides is updated from SOURCE
only when SOURCE's copy actually carried the task further: no smaller
``steps``, ``tokens``, ``cost`` or ``end_ts``, and at least one of them
larger.  That stops a stale SOURCE from clobbering a run that progressed
on TARGET, and -- because the two databases sync in both directions --
it also stops a row that merely *differs* from overwriting whatever the
receiving machine has recorded about the same task since.  The two
columns that a user or a later event sets on their own, ``is_favorite``
and ``has_events``, are merged rather than copied: once either side has
them set, both do.  ``--force`` always takes SOURCE's row as it stands,
``--insert-only`` never updates an existing row.

The remote side runs this very file through ``ssh <host> python3 -c
...``: the script is stdlib-only and self-contained, so nothing has to
be installed on the remote machine beyond ``python3``.

Usage:
    uv run python -m kiss.scripts.sync_db SOURCE TARGET [OPTIONS]

Options:
    --insert-only   Only add missing rows; never update existing ones
    --force         On conflict always overwrite TARGET's task row
    --full          Ignore TARGET's manifest and ship every source row
                    (slow but assumes nothing about how rows were added)
    --dry-run       Perform the merge and roll it back, reporting exactly
                    what a real run would change (TARGET must be writable)
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
import hashlib
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
from typing import Any, BinaryIO

TASK_TABLE = "task_history"
EVENT_TABLE = "events"
# Columns of the task table that only ever grow while a task runs; used
# to decide whether SOURCE's copy of a task supersedes TARGET's copy.
MONOTONE_COLUMNS = ("steps", "tokens", "cost", "end_ts")
# Columns that are set independently of a task's progress -- by the user
# marking a task a favourite, and by the first event a task emits -- and
# that are therefore merged into the target rather than copied over it.
STICKY_COLUMNS = ("is_favorite", "has_events")
# Where each table's column list lives in the manifest.
MANIFEST_COLUMN_KEYS = {TASK_TABLE: "task_columns", EVENT_TABLE: "event_columns"}
COPY_BUFFER = 1 << 20
INSERT_BATCH = 1000

PHASE_MANIFEST = "_manifest"
PHASE_EXTRACT = "_extract"
PHASE_MERGE = "_merge"

MODE_DEFAULT = "default"
MODE_FORCE = "force"
MODE_INSERT_ONLY = "insert-only"

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


def row_digest(values: list[Any]) -> str:
    """Hash one row's values into a short, cross-machine stable digest.

    Each value is hashed as a type tag, its byte length and its bytes, so
    that no two different rows can be framed into the same byte stream.

    Args:
        values: Column values in a fixed column order.

    Returns:
        A hex SHA-1 digest of the framed values.
    """
    h = hashlib.sha1(usedforsecurity=False)
    for value in values:
        if value is None:
            tag, payload = b"n", b""
        elif isinstance(value, bytes):
            tag, payload = b"b", value
        elif isinstance(value, str):
            tag, payload = b"s", value.encode("utf-8", "surrogatepass")
        elif isinstance(value, bool):
            tag, payload = b"i", b"1" if value else b"0"
        elif isinstance(value, int):
            tag, payload = b"i", str(value).encode()
        else:
            tag, payload = b"f", repr(float(value)).encode()
        h.update(tag)
        h.update(str(len(payload)).encode())
        h.update(b":")
        h.update(payload)
    return h.hexdigest()


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

    The manifest holds a content digest per task id and, per task, the
    lowest and highest ``events.seq`` together with the number of events
    the target has.  That is all the source needs to compute a minimal
    delta: when the count matches the span, the target owns an unbroken
    run of events and only rows outside it have to travel, and otherwise
    the source ships every event of that task so earlier gaps heal.

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
        digest_cols = sorted(task_cols)
        selection = ", ".join(quote_name(c) for c in digest_cols)
        tasks: dict[str, str] = {}
        for row in conn.execute(
            f"SELECT {selection} FROM main.{quote_name(TASK_TABLE)}"
            ' WHERE "id" IS NOT NULL'
        ):
            values = list(row)
            task_id = values[digest_cols.index("id")]
            tasks[str(task_id)] = row_digest(values)
        events = {
            str(task_id): [low, high, count]
            for task_id, low, high, count in conn.execute(
                'SELECT "task_id", MIN("seq"), MAX("seq"), COUNT(DISTINCT "seq")'
                f" FROM main.{quote_name(EVENT_TABLE)} GROUP BY \"task_id\""
            )
            if task_id is not None and low is not None
        }
    finally:
        conn.close()
    write_json_gz(
        {
            MANIFEST_COLUMN_KEYS[TASK_TABLE]: digest_cols,
            MANIFEST_COLUMN_KEYS[EVENT_TABLE]: sorted(event_cols),
            "tasks": tasks,
            "events": events,
        },
        out,
    )


# --------------------------------------------------------------------------
# phase 2: delta database built on the source
# --------------------------------------------------------------------------


def phase_extract(path: str, args: list[str], inp: BinaryIO, out: BinaryIO) -> None:
    """Build a delta database of rows the target lacks and stream it out.

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
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("ATTACH DATABASE ? AS src", (source,))
            _create_delta_tables(conn)
            for table, key in MANIFEST_COLUMN_KEYS.items():
                if manifest.get(key):
                    compare_columns(
                        table_columns(conn, "src", table), list(manifest[key]), table
                    )
            _extract_tasks(conn, manifest)
            _extract_events(conn, manifest)
            conn.execute("DETACH DATABASE src")
        finally:
            conn.close()
        write_file_gz(delta_path, out)
    finally:
        _unlink(delta_path)


def _create_delta_tables(conn: sqlite3.Connection) -> None:
    """Recreate the source's task and event tables inside the delta db.

    Args:
        conn: Connection to the delta database with the source attached
            as ``src``.

    Raises:
        SyncError: If the source lacks one of the two tables.
    """
    for table in (TASK_TABLE, EVENT_TABLE):
        row = conn.execute(
            "SELECT sql FROM src.sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not row or not row[0]:
            raise SyncError(f"table {table!r} is missing from the source database")
        conn.execute(row[0])


def _extract_tasks(conn: sqlite3.Connection, manifest: dict[str, Any]) -> None:
    """Copy task rows the target is missing or that differ into the delta.

    Args:
        conn: Connection to the delta database with the source attached.
        manifest: Target manifest; empty for a ``--full`` sync.
    """
    columns = table_columns(conn, "src", TASK_TABLE)
    require_columns(columns, ("id",), f"{TASK_TABLE} in the source database")
    have: dict[str, str] = manifest.get("tasks") or {}
    digest_cols = manifest.get(MANIFEST_COLUMN_KEYS[TASK_TABLE]) or sorted(columns)
    selection = ", ".join(quote_name(c) for c in columns)
    insert = (
        f"INSERT INTO main.{quote_name(TASK_TABLE)} ({selection})"
        f" VALUES ({', '.join('?' for _ in columns)})"
    )
    index = {name: pos for pos, name in enumerate(columns)}
    batch: list[tuple[Any, ...]] = []
    cursor = conn.execute(f"SELECT {selection} FROM src.{quote_name(TASK_TABLE)}")
    for row in cursor:
        task_id = row[index["id"]]
        if task_id is None:
            # A NULL primary key cannot be matched across databases and
            # would be duplicated on every run, so such rows are skipped.
            continue
        digest = row_digest([row[index[c]] if c in index else None for c in digest_cols])
        if have.get(str(task_id)) == digest:
            continue
        batch.append(tuple(row))
        if len(batch) >= INSERT_BATCH:
            conn.executemany(insert, batch)
            batch.clear()
    if batch:
        conn.executemany(insert, batch)


def _extract_events(conn: sqlite3.Connection, manifest: dict[str, Any]) -> None:
    """Copy the events the target is missing into the delta database.

    An event travels when its task is unknown to the target, when its
    ``seq`` falls outside the run of sequence numbers the target already
    holds, or when that run has a gap -- in which case every event of the
    task is shipped and the merge's de-duplication sorts it out.

    Args:
        conn: Connection to the delta database with the source attached.
        manifest: Target manifest; empty for a ``--full`` sync.
    """
    columns = table_columns(conn, "src", EVENT_TABLE)
    require_columns(columns, ("task_id", "seq"), f"{EVENT_TABLE} in the source database")
    rowid_alias = rowid_alias_column(conn, "src", EVENT_TABLE)
    copied = [c for c in columns if c != rowid_alias]
    _create_watermark_table(conn, manifest.get("events") or {})
    table = quote_name(EVENT_TABLE)
    names = ", ".join(quote_name(c) for c in copied)
    selection = ", ".join(f"s.{quote_name(c)}" for c in copied)
    missing = (
        f" FROM src.{table} s"
        ' LEFT JOIN main."_sync_watermark" w ON w.task_id = s."task_id"'
        ' WHERE s."task_id" IS NOT NULL AND s."seq" IS NOT NULL'
        " AND (w.task_id IS NULL OR w.complete = 0"
        ' OR s."seq" > w.high_seq OR s."seq" < w.low_seq)'
    )
    if rowid_alias:
        # Picking the rows by their rowid first keeps the scan inside a
        # ``(task_id, seq)`` index when the source has one, so a routine
        # sync never reads the payload of an event it does not copy.
        key = quote_name(rowid_alias)
        conn.execute('CREATE TABLE main."_sync_todo" (rid INTEGER PRIMARY KEY)')
        conn.execute(f'INSERT INTO main."_sync_todo" SELECT s.{key}' + missing)
        conn.execute(
            f"INSERT INTO main.{table} ({names}) SELECT {selection}"
            f' FROM main."_sync_todo" t JOIN src.{table} s ON s.{key} = t.rid'
        )
        conn.execute('DROP TABLE main."_sync_todo"')
    else:
        conn.execute(f"INSERT INTO main.{table} ({names}) SELECT {selection}{missing}")
    conn.execute('DROP TABLE main."_sync_watermark"')
    _warn_on_duplicate_event_keys(conn)


def _create_watermark_table(
    conn: sqlite3.Connection, watermarks: dict[str, Any]
) -> None:
    """Load the target's per-task event coverage into the delta database.

    Args:
        conn: Connection to the delta database.
        watermarks: Manifest entries of the form
            ``{task_id: [low_seq, high_seq, count]}``.
    """
    conn.execute(
        'CREATE TABLE main."_sync_watermark" (task_id TEXT PRIMARY KEY,'
        " low_seq INTEGER, high_seq INTEGER, complete INTEGER NOT NULL)"
    )
    rows = [
        (task_id, low, high, int(count == high - low + 1))
        for task_id, (low, high, count) in watermarks.items()
    ]
    conn.executemany(
        'INSERT OR REPLACE INTO main."_sync_watermark" VALUES (?, ?, ?, ?)', rows
    )


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
        args: The conflict mode (``default``, ``force`` or
            ``insert-only``) followed by ``commit`` or ``rollback``.
        inp: Gzipped delta database from :func:`phase_extract`.
        out: Destination binary stream for the JSON statistics.
    """
    mode = args[0] if args else MODE_DEFAULT
    commit = len(args) < 2 or args[1] == "commit"
    delta_path = temp_path(".db")
    try:
        read_file_gz(inp, delta_path)
        stats = _apply_delta(path, delta_path, mode, commit)
    finally:
        _unlink(delta_path)
    out.write(json.dumps(stats).encode("utf-8"))
    out.flush()


def _apply_delta(
    target_path: str, delta_path: str, mode: str, commit: bool
) -> dict[str, int]:
    """Insert every row of a delta database into the target database.

    Args:
        target_path: Target database path.
        delta_path: Path of the plain (unzipped) delta database.
        mode: ``default``, ``force`` or ``insert-only``.
        commit: When False the merge is rolled back after counting the
            rows it would have changed, which is how ``--dry-run``
            reports exactly what a real run would do.

    Returns:
        Counts of the task rows inserted and updated and of the event
        rows inserted.
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
            tasks_inserted, tasks_updated = _merge_tasks(conn, mode)
            events_inserted = _merge_events(conn)
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
        "tasks_updated": tasks_updated,
        "events_inserted": events_inserted,
    }


def _merge_tasks(conn: sqlite3.Connection, mode: str) -> tuple[int, int]:
    """Insert new task rows and update existing ones per the sync mode.

    Args:
        conn: Target connection with the delta attached as ``delta``.
        mode: ``default``, ``force`` or ``insert-only``.

    Returns:
        The number of rows inserted and the number of rows updated.
    """
    columns = table_columns(conn, "main", TASK_TABLE)
    require_columns(columns, ("id",), f"{TASK_TABLE} in the target database")
    table = quote_name(TASK_TABLE)
    names = ", ".join(quote_name(c) for c in columns)
    sql = (
        f"INSERT INTO main.{table} ({names})"
        f" SELECT {names} FROM delta.{table} WHERE \"id\" IS NOT NULL"
    )
    updatable = [c for c in columns if c != "id"]
    if mode == MODE_INSERT_ONLY or not updatable:
        sql += ' ON CONFLICT("id") DO NOTHING'
    else:
        assignments = ", ".join(_assignment(c, mode) for c in updatable)
        sql += f' ON CONFLICT("id") DO UPDATE SET {assignments}'
        if mode != MODE_FORCE:
            sql += " WHERE " + _supersedes_clause(columns)
    before = _count(conn, "main", TASK_TABLE)
    changed = conn.execute(sql).rowcount
    inserted = _count(conn, "main", TASK_TABLE) - before
    updated = max(changed - inserted, 0)
    if mode == MODE_DEFAULT:
        updated += _raise_sticky_columns(conn, columns)
    return inserted, updated


def _raise_sticky_columns(conn: sqlite3.Connection, columns: list[str]) -> int:
    """Carry a set ``is_favorite`` or ``has_events`` flag over to the target.

    These two columns are not progress: the user marks a favourite
    whenever they like, and the flag saying a task has events is set by
    the first event it emits.  A row that is otherwise not further along
    -- the usual case for two copies of a finished task -- does not get
    to overwrite the target's row, so the flags are raised here instead.
    Only raised, never cleared: once either machine has one set, both do.

    All the flags are raised by a single statement, so a row counts once
    however many of them rose, and a column the incoming row does not
    raise is not written at all.

    Args:
        conn: Target connection with the delta attached as ``delta``.
        columns: Columns of the target's task table.

    Returns:
        The number of rows whose flags were raised.
    """
    table = quote_name(TASK_TABLE)
    present = [c for c in STICKY_COLUMNS if c in columns]
    if not present:
        return 0
    assignments, rose = [], []
    for column in present:
        name = quote_name(column)
        incoming = (
            f"(SELECT d.{name} FROM delta.{table} d"
            f' WHERE d."id" = main.{table}."id")'
        )
        larger = f"COALESCE({incoming}, 0) > COALESCE({name}, 0)"
        assignments.append(f"{name} = CASE WHEN {larger} THEN {incoming} ELSE {name} END")
        rose.append(larger)
    return int(
        conn.execute(
            f"UPDATE main.{table} SET {', '.join(assignments)}"
            f" WHERE {' OR '.join(rose)}"
        ).rowcount
    )


def _assignment(column: str, mode: str) -> str:
    """Build one ``SET`` assignment of the conflict update.

    Args:
        column: Column being written.
        mode: ``default``, ``force`` or ``insert-only``.

    Returns:
        An SQL assignment: the source's value, or the larger of the two
        values for a sticky column, which is how a favourite marked on
        one machine survives a row arriving from the other.
    """
    name = quote_name(column)
    if mode == MODE_FORCE or column not in STICKY_COLUMNS:
        return f"{name} = excluded.{name}"
    table = quote_name(TASK_TABLE)
    return (
        f"{name} = MAX(COALESCE(excluded.{name}, 0),"
        f" COALESCE({table}.{name}, 0))"
    )


def _supersedes_clause(columns: list[str]) -> str:
    """Build the guard that keeps a stale source row from winning.

    A row is only allowed to overwrite the target's row when it carried
    the task *further*: not behind on any of the columns that grow as a
    task runs, and ahead on at least one of them.  A row that merely
    differs -- same progress, some other column edited on the receiving
    machine since -- leaves the target's row alone, which is what makes
    syncing the same two databases in both directions non-destructive.

    A table without any of those columns says nothing about which of two
    copies of a task is the later one, so no copy is allowed to overwrite
    the other; ``--force`` is how such a table is overwritten on purpose.

    Args:
        columns: Columns available on both sides of the merge.

    Returns:
        An SQL boolean expression that is true when the incoming row
        supersedes the row already in the target.
    """
    present = [c for c in MONOTONE_COLUMNS if c in columns]
    if not present:
        return "0"
    not_behind = " AND ".join(_compare_monotone(c, ">=") for c in present)
    ahead = " OR ".join(_compare_monotone(c, ">") for c in present)
    return f"({not_behind}) AND ({ahead})"


def _compare_monotone(column: str, operator: str) -> str:
    """Compare one column of the incoming task row with the target's.

    Args:
        column: Name of a column that grows as a task runs.
        operator: SQL comparison operator, ``">="`` or ``">"``.

    Returns:
        An SQL boolean expression comparing the two values, treating a
        missing value as zero on either side.
    """
    name = quote_name(column)
    return (
        f"COALESCE(excluded.{name}, 0) {operator}"
        f" COALESCE({quote_name(TASK_TABLE)}.{name}, 0)"
    )


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


def _count(conn: sqlite3.Connection, schema: str, table: str) -> int:
    """Count the rows of a table.

    Args:
        conn: Open connection.
        schema: Database name, e.g. ``main`` or ``delta``.
        table: Table name.

    Returns:
        The number of rows.
    """
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM {quote_name(schema)}.{quote_name(table)}"
        ).fetchone()[0]
    )


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

        The script's own source is shipped inline, base64 encoded, so the
        remote machine needs nothing but a python interpreter.

        Args:
            location: Remote database the phase operates on.
            phase: One of the ``PHASE_*`` names.
            args: Extra string arguments for the phase.

        Returns:
            A single shell command string for ``ssh``.
        """
        payload = base64.b64encode(_script_source()).decode("ascii")
        bootstrap = f"import base64;exec(base64.b64decode('{payload}'))"
        parts = [self.python, "-c", bootstrap, phase, location.path, *args]
        return " ".join(shlex.quote(p) for p in parts)


def _script_source() -> bytes:
    """Read this script's own source, for shipping to a remote host.

    Returns:
        The bytes of this file.

    Raises:
        SyncError: If the source file cannot be read.
    """
    path = globals().get("__file__")
    if not path:
        raise SyncError("cannot locate this script's source for remote execution")
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
    mode: str = MODE_DEFAULT,
    full: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy task and event rows from a source database into a target.

    Args:
        source: Database rows are read from; never modified.
        target: Database rows are written to.
        runner: Launches each phase locally or over ssh.
        mode: Conflict handling for existing task rows: ``default``,
            ``force`` or ``insert-only``.
        full: Ship every source row instead of only the target's gaps.
        dry_run: Roll the merge back instead of committing it, so the
            target is left untouched but the reported counts are the ones
            a real run would apply.

    Returns:
        Statistics with the compressed delta size, the rows inserted and
        updated and the elapsed wall-clock seconds.

    Raises:
        SyncError: If source and target are the same database, or if any
            phase fails.
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
        commit = "rollback" if dry_run else "commit"
        runner.run(target, PHASE_MERGE, [mode, commit], delta_path, stats_path)
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
    if stats["dry_run"]:
        applied = (
            f"would add {stats['tasks_inserted']} task row(s),"
            f" update {stats['tasks_updated']},"
            f" add {stats['events_inserted']} event row(s)"
        )
    else:
        applied = (
            f"{stats['tasks_inserted']} task row(s) added,"
            f" {stats['tasks_updated']} updated,"
            f" {stats['events_inserted']} event row(s) added"
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
            "One-way sync of the task_history and events tables of "
            "sorcar.db-shaped SQLite databases. SOURCE and TARGET are "
            "unix paths, optionally prefixed with user@host: for a "
            "database reachable over ssh."
        ),
    )
    parser.add_argument("source", help="[user@host:]/path/to/source.db (read only)")
    parser.add_argument("target", help="[user@host:]/path/to/target.db (updated)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--insert-only",
        action="store_true",
        help="only add missing rows; never update an existing task row",
    )
    group.add_argument(
        "--force",
        action="store_true",
        help="on conflict always overwrite the target's task row",
    )
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
    mode = MODE_DEFAULT
    if options.insert_only:
        mode = MODE_INSERT_ONLY
    elif options.force:
        mode = MODE_FORCE
    try:
        source = Location(options.source)
        target = Location(options.target)
        if not options.quiet:
            print(f"syncing {source} -> {target} ...", file=sys.stderr)
        stats = synchronize(
            source,
            target,
            Runner(options.python, options.port, options.ssh_options),
            mode=mode,
            full=options.full,
            dry_run=options.dry_run,
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

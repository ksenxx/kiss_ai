#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Carry the counters of a replaced ``sorcar.db`` into its replacement.

``sync_db.py`` synchronizes the two tables that hold a machine's history
-- ``task_history`` and ``events``.  A ``sorcar.db`` holds three more
tables that no sync moves, each a tally the web app keeps so that its
menus offer what you actually use:

* ``model_usage``    -- how often each model was chosen;
* ``file_usage``     -- how often each file was opened, and when last;
* ``frequent_tasks`` -- how often each task text was run, and when last.

When a deploy has to replace a database wholesale instead of merging
into it, those three tables would be replaced along with it -- the
deployment's own tallies gone, in a step that is supposed to lose
nothing.  This script puts them back from the file the replacement kept:
rows only the old database had are inserted, and where both have the
same row every numeric column is raised to the larger of the two values.
Nothing is ever lowered or deleted, so running it twice changes nothing
the second time.

Usage:
    python3 carry_over_tables.py OLD_DATABASE NEW_DATABASE

The number of rows inserted and raised is printed.  An old database that
cannot be opened is not an error -- that is exactly why it was replaced
-- and leaves the new one untouched.  It is stdlib-only and
self-contained so that it can be piped straight into a remote
``python3`` over ssh.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import sys

# The tables no sync moves, each with the column that identifies a row.
CARRIED_TABLES = {
    "model_usage": "model",
    "file_usage": "path",
    "frequent_tasks": "task",
}
NUMERIC_TYPES = ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")


def quote(name: str) -> str:
    """Quote an identifier for use in a statement.

    Args:
        name: Table or column name.

    Returns:
        The name in double quotes, with inner quotes doubled.
    """
    return '"' + name.replace('"', '""') + '"'


def _columns(
    con: sqlite3.Connection, schema: str, table: str
) -> list[tuple[str, str, bool]]:
    """Describe a table's columns.

    Args:
        con: Open connection.
        schema: Attached database name, ``main`` or ``old``.
        table: Table name.

    Returns:
        One ``(name, declared_type, is_integer_primary_key)`` triple per
        column, in declaration order; empty when the table is absent.
    """
    described = []
    for row in con.execute(
        f"PRAGMA {quote(schema)}.table_info({quote(table)})"
    ).fetchall():
        name, declared, primary_key = str(row[1]), str(row[2] or "").upper(), bool(row[5])
        described.append((name, declared, primary_key and "INT" in declared))
    return described


def _is_numeric(declared_type: str) -> bool:
    """Report whether a declared column type holds numbers.

    Args:
        declared_type: Type as declared in the schema, upper-cased.

    Returns:
        True when the column is a counter or a timestamp rather than text.
    """
    return any(marker in declared_type for marker in NUMERIC_TYPES)


def carry_over_table(con: sqlite3.Connection, table: str, key: str) -> tuple[int, int]:
    """Insert the rows one table lacks and raise its numeric columns.

    Args:
        con: Connection to the new database with the old one attached as
            ``old``.
        table: Table to carry over.
        key: Column that identifies a row in it.

    Returns:
        The number of rows inserted and the number of rows raised.
    """
    new_columns = _columns(con, "main", table)
    old_columns = _columns(con, "old", table)
    if not new_columns or [c[0] for c in new_columns] != [c[0] for c in old_columns]:
        return 0, 0
    names = [name for name, _, is_rowid in new_columns if not is_rowid]
    if key not in names:
        return 0, 0
    selection = ", ".join(quote(name) for name in names)
    # A row whose key is NULL cannot be matched against anything -- SQL equality
    # never holds for NULL -- so it would be inserted again on every run, and
    # would be refused outright by a schema that has since made the column NOT
    # NULL.  It is a tally of nothing in particular; it stays where it is.
    inserted = con.execute(
        f"INSERT INTO main.{quote(table)} ({selection})"
        f" SELECT {', '.join('o.' + quote(name) for name in names)}"
        f" FROM old.{quote(table)} o WHERE o.{quote(key)} IS NOT NULL"
        f" AND NOT EXISTS (SELECT 1 FROM main.{quote(table)} m"
        f" WHERE m.{quote(key)} = o.{quote(key)})"
    ).rowcount
    raisable = [
        name
        for name, declared, _ in new_columns
        if name != key and name in names and _is_numeric(declared)
    ]
    if not raisable:
        return max(inserted, 0), 0
    assignments, conditions = [], []
    for name in raisable:
        incoming = (
            f"(SELECT o.{quote(name)} FROM old.{quote(table)} o"
            f" WHERE o.{quote(key)} = main.{quote(table)}.{quote(key)})"
        )
        larger = f"COALESCE({incoming}, 0) > COALESCE({quote(name)}, 0)"
        assignments.append(
            f"{quote(name)} = CASE WHEN {larger} THEN {incoming} ELSE {quote(name)} END"
        )
        conditions.append(larger)
    raised = con.execute(
        f"UPDATE main.{quote(table)} SET {', '.join(assignments)}"
        f" WHERE {' OR '.join(conditions)}"
    ).rowcount
    return max(inserted, 0), max(raised, 0)


def carry_over(old_database: str, new_database: str) -> tuple[int, int]:
    """Carry every unsynced table of one database into another.

    Args:
        old_database: Database that was replaced; only read.
        new_database: Database that replaced it; updated in place.

    Returns:
        The total number of rows inserted and raised.  ``(0, 0)`` when
        there is no old database, or none that can be opened -- which is
        usually why it was replaced.

    Raises:
        sqlite3.Error: If the old database opened but the carry-over then
            failed.  Reporting that as success would leave the counters
            only in a backup file nobody looks at.
    """
    old_path = os.path.expanduser(old_database)
    if not os.path.isfile(old_path):
        return 0, 0
    # Reading a database that was left in WAL mode makes SQLite create the
    # shared-memory index beside it, even read-only.  It is machine-local and
    # regenerated on demand, and the file it appears beside is a backup nobody
    # is meant to open again, so it is cleaned up below rather than left as
    # debris in ~/.kiss.
    had_shm = os.path.exists(old_path + "-shm")
    # uri=True is what makes the read-only ATTACH below a URI rather than the
    # name of a file called "file:...": the flag belongs to the connection.
    con = sqlite3.connect(os.path.expanduser(new_database), timeout=60.0, uri=True)
    inserted = raised = 0
    try:
        try:
            con.execute("ATTACH DATABASE ? AS old", (f"file:{old_path}?mode=ro",))
        except sqlite3.Error as exc:
            # Not a database, or not one this SQLite can read: that is why it
            # was replaced, and there is nothing in it to carry over.
            print(f"carry_over_tables: {old_database}: {exc}", file=sys.stderr)
            return 0, 0
        try:
            with con:
                for table, key in CARRIED_TABLES.items():
                    added, lifted = carry_over_table(con, table, key)
                    inserted += added
                    raised += lifted
        finally:
            con.execute("DETACH DATABASE old")
    finally:
        con.close()
        if not had_shm:
            with contextlib.suppress(OSError):
                os.unlink(old_path + "-shm")
    return inserted, raised


def main(argv: list[str] | None = None) -> int:
    """Carry the unsynced tables over from the command line.

    Args:
        argv: Arguments ``[old_database, new_database]``; defaults to
            ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 when the carry-over failed,
        2 on a usage error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: carry_over_tables.py OLD_DATABASE NEW_DATABASE", file=sys.stderr
        )
        return 2
    try:
        inserted, raised = carry_over(args[0], args[1])
    except sqlite3.Error as exc:
        print(f"carry_over_tables: {args[1]}: {exc}", file=sys.stderr)
        return 1
    print(f"{inserted} row(s) restored, {raised} counter(s) raised")
    return 0


if __name__ == "__main__":
    sys.exit(main())

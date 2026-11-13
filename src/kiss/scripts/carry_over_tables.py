#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Carry the counters of a replaced ``history.db`` into its replacement.

``sync_db.py`` synchronizes the two tables that hold a machine's history
-- ``task_history`` and ``events``.  A ``history.db`` holds four more
tables that no sync moves, each a tally the web app keeps so that its
menus offer what you actually use:

* ``model_usage``    -- how often each model was chosen;
* ``file_usage``     -- how often each file was opened, and when last;
* ``frequent_tasks`` -- how often each task text was run, and when last
  (earlier releases wrote it; the rows they left are still carried);
* ``steer_inputs``   -- each message typed into a running task, and when
  last (autocomplete history).

and two more that record what the user deleted (``deleted_tasks`` and
``deleted_chats``, see ``sync_db.py``), which a sync moves but a
replacement that skipped the sync -- the old database could not be
merged from -- would lose.

When a deploy has to replace a database wholesale instead of merging
into it, those tables would be replaced along with it -- the
deployment's own tallies and deletions gone, in a step that is supposed
to lose nothing.  This script puts them back from the file the
replacement kept: rows only the old database had are inserted, and
where both have the same row every numeric column is raised to the
larger of the two values.  The titles the user typed over chats there
(``chat_summaries.user_summary``, which no sync moves either) go back
over the same chats, in place of a title that came with the
replacement.  The deletions carried over are then carried
out on the new database, as a sync would: a task the old database had
deleted does not come back in the new one, nor does what the web app
deletes with it -- the tasks of a deleted chat, the sub-tasks of a
deleted task.  Nothing else is ever lowered or deleted, so running it
twice changes nothing the second time.

Usage:
    python3 carry_over_tables.py OLD_DATABASE NEW_DATABASE

The number of rows inserted and raised is printed.  An old database that
cannot be opened is not an error -- that is exactly why it was replaced
-- and leaves the new one untouched.  It is stdlib-only and
self-contained so that it can be piped straight into a remote
``python3`` over ssh.  ``sync_db.py`` runs :func:`spread_deletions` of
this file too, so that a merge and a replacement take the same tasks
with a deletion, and ships the file with itself for that.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import sys
from pathlib import Path

# The tables no sync moves, each with the column that identifies a row.
CARRIED_TABLES = {
    "model_usage": "model",
    "file_usage": "path",
    "frequent_tasks": "task",
    "steer_inputs": "text",
    "deleted_tasks": "id",
    "deleted_chats": "chat_id",
}
# The deletion records, created in a new database from before they existed
# (``persistence._init_tables`` declares them the same way).
DELETED_DDL = (
    'CREATE TABLE IF NOT EXISTS main."deleted_tasks"'
    ' ("id" TEXT PRIMARY KEY, "deleted_ts" REAL NOT NULL)',
    'CREATE TABLE IF NOT EXISTS main."deleted_chats"'
    ' ("chat_id" TEXT PRIMARY KEY, "deleted_ts" REAL NOT NULL)',
)
NUMERIC_TYPES = ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")
# How long after it began a sub-task younger than its parent's deletion is
# recorded as deleted (``spread_deletions``): late enough to be after its
# start in a REAL of epoch seconds, early enough that a task of the same id
# begun later is another task.
SPREAD_MARGIN = 0.001


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


def carry_over_titles(con: sqlite3.Connection) -> int:
    """Put the titles the user typed over the old database's chats back.

    ``chat_summaries.user_summary`` is the one column of that table
    nothing can recompute.  A title the old database holds replaces the
    one in the new database's row for the chat, whose generated columns
    stay as they are; a chat the new database has a task of but no row
    for yet (a row is written when a task finishes) gets a row holding
    the title alone, as typing it there would have made.  A chat the new
    database has neither a row nor a task of gets nothing, so a deleted
    chat does not come back.

    Args:
        con: Connection to the new database with the old one attached as
            ``old``.

    Returns:
        The number of rows that got a title.
    """
    # A database from before these columns has no titles to give, or no
    # chats to put them on.
    titled = {"chat_id", "user_summary"}
    if not (
        titled <= {c[0] for c in _columns(con, "old", "chat_summaries")}
        and titled <= {c[0] for c in _columns(con, "main", "chat_summaries")}
        and "chat_id" in {c[0] for c in _columns(con, "main", "task_history")}
    ):
        return 0
    title = (
        '(SELECT o."user_summary" FROM old."chat_summaries" o'
        ' WHERE o."chat_id" = main."chat_summaries"."chat_id" AND o."user_summary" != \'\')'
    )
    replaced = con.execute(
        f'UPDATE main."chat_summaries" SET "user_summary" = {title}'
        f" WHERE {title} != COALESCE(\"user_summary\", '')"
    ).rowcount
    added = con.execute(
        'INSERT INTO main."chat_summaries" ("chat_id", "user_summary")'
        ' SELECT o."chat_id", o."user_summary" FROM old."chat_summaries" o'
        " WHERE o.\"user_summary\" != ''"
        ' AND NOT EXISTS (SELECT 1 FROM main."chat_summaries" m'
        ' WHERE m."chat_id" = o."chat_id")'
        ' AND EXISTS (SELECT 1 FROM main."task_history" t WHERE t."chat_id" = o."chat_id")'
    ).rowcount
    return replaced + added


def _started(alias: str, columns: set[str]) -> str:
    """Return the SQL for when a task row began: 0 in a table without a ``timestamp``.

    Args:
        alias: Alias of the task row in the surrounding statement.
        columns: The columns of ``task_history``.
    """
    return f'COALESCE({alias}."timestamp", 0)' if "timestamp" in columns else "0"


def _unrecorded(columns: set[str]) -> str:
    """Return the SQL condition, led by ``AND``, that no deletion is recorded for task row ``t``.

    A record older than the row does not count: a reused id is a new task.

    Args:
        columns: The columns of the ``task_history`` that ``t`` is a row of.
    """
    return (
        ' AND NOT EXISTS (SELECT 1 FROM main."deleted_tasks" x'
        f' WHERE x."id" = t."id" AND x."deleted_ts" > {_started("t", columns)})'
    )


def spread_deletions(con: sqlite3.Connection, columns: dict[str, set[str]]) -> None:
    """Record as deleted every task that a recorded deletion takes with it.

    The web app deletes a chat with all its tasks and a task with all its
    sub-tasks, and records each of them -- each of those the machine
    holds.  A task of the chat, or a sub-task, that the deleting machine
    had never seen has no record of its own and would come back from, or
    stay behind on, the machine that holds it.  So a task row is recorded
    as deleted too when it is older than the deletion of its chat, or
    when its parent task is deleted (no row of the parent's id younger
    than that deletion is in any of the schemas: a reused id is a new
    task), sub-tasks of sub-tasks included.  It is recorded at the
    moment of the deletion that takes it; a sub-task that began after
    that moment, :data:`SPREAD_MARGIN` seconds after it began -- the
    same record on every machine, whatever their clocks say, and one
    that leaves a later task of its id alone.

    Args:
        con: Connection whose ``main`` database holds the deletion
            records.
        columns: By schema, the columns of the ``task_history`` whose
            rows the deletions are spread over: ``main`` alone here, a
            sync's attached ``delta`` as well in ``sync_db.py``.
    """
    insert = 'INSERT INTO main."deleted_tasks" ("id", "deleted_ts") '
    newer_wins = (
        ' ON CONFLICT("id") DO UPDATE SET "deleted_ts" = excluded."deleted_ts"'
        ' WHERE excluded."deleted_ts" > "deleted_ts"'
    )
    chats = 'main."deleted_chats" d'
    for schema, present in columns.items():
        if "chat_id" not in present:
            continue
        rows = f'{schema}."task_history" t'
        # CROSS JOIN pins the loop order.  The rows of ``main`` are found by
        # one index seek per deletion; the rows of another schema, which has
        # no index on the chat, each seek their deletion.
        first, second = (chats, rows) if schema == "main" else (rows, chats)
        con.execute(
            f'{insert}SELECT t."id", d."deleted_ts" FROM {first} CROSS JOIN {second}'
            ' ON d."chat_id" = t."chat_id"'
            " WHERE t.\"id\" IS NOT NULL AND t.\"chat_id\" != ''"
            f' AND d."deleted_ts" > {_started("t", present)}{_unrecorded(present)}{newer_wins}'
        )
    # A row of the parent's id that its deletion does not date is a new
    # task (a reused id) and keeps its sub-tasks.
    orphaned = "".join(
        f' AND NOT EXISTS (SELECT 1 FROM {schema}."task_history" p WHERE p."id" = d."id"'
        f' AND {_started("p", present)} >= d."deleted_ts")'
        for schema, present in columns.items()
    )
    sub_tasks = [
        f'SELECT t."id", MAX(d."deleted_ts", {_started("t", present)} + {SPREAD_MARGIN})'
        ' FROM temp."_spread_frontier" f CROSS JOIN main."deleted_tasks" d ON d."id" = f."id"'
        f' CROSS JOIN {schema}."task_history" t ON t."parent_task_id" = d."id"'
        " WHERE t.\"id\" IS NOT NULL AND t.\"parent_task_id\" != ''"
        f"{orphaned}{_unrecorded(present)}"
        for schema, present in columns.items()
        if "parent_task_id" in present
    ]
    # Level by level.  The frontier holds the deletions whose sub-tasks are
    # still to be looked for: all of them at first, then the ones the level
    # before recorded.  It ends: a row its record covers is not taken again,
    # and one no record can cover (a start that is no number) is never a
    # deleted parent.
    con.execute('CREATE TEMP TABLE "_spread_frontier" AS SELECT "id" FROM main."deleted_tasks"')
    while True:
        taken = [row for select in sub_tasks for row in con.execute(select)]
        if not taken:
            break
        con.executemany(f"{insert}VALUES (?, ?){newer_wins}", taken)
        con.execute('DELETE FROM temp."_spread_frontier"')
        con.executemany(
            'INSERT INTO temp."_spread_frontier" VALUES (?)', {(task_id,) for task_id, _ in taken}
        )
    con.execute('DROP TABLE temp."_spread_frontier"')


def apply_deletions(con: sqlite3.Connection) -> int:
    """Carry out the deletions a database records on the rows it holds.

    A task in ``deleted_tasks`` loses its events and its row unless the
    row is younger than the deletion (a reused id is a new task), and
    so does every task the deletions take with them
    (:func:`spread_deletions`).  A chat that lost a task keeps its
    ``chat_summaries`` row and the title the user typed, without what
    the row said of its tasks: the web app works that out again when the
    chat's next task finishes, and ``backfill_task_metadata.py`` does
    for every such chat.  A chat in ``deleted_chats`` loses the row once
    none of its tasks is left.  Nothing runs out of a database being
    replaced, so no row is a running task's.

    Args:
        con: Connection to the new database, with its deletion records
            carried over.

    Returns:
        The number of task rows deleted.
    """
    tables = {
        str(row[0])
        for row in con.execute("SELECT name FROM main.sqlite_master WHERE type = 'table'")
    }
    columns = {str(row[1]) for row in con.execute('PRAGMA main.table_info("task_history")')}
    if "id" not in columns:
        return 0
    spread_deletions(con, {"main": columns})
    doomed = (
        'FROM main."task_history" t JOIN main."deleted_tasks" d'
        f' ON d."id" = t."id" WHERE d."deleted_ts" > {_started("t", columns)}'
    )
    summarised = "chat_id" in columns and "chat_summaries" in tables
    # Empty when the database has no such table.
    summary_columns = {
        str(row[1]) for row in con.execute('PRAGMA main.table_info("chat_summaries")')
    }
    if summarised:
        cleared = [
            f"{quote(name)} = {blank}"
            for name, blank in (("summary", "''"), ("last_launched", "0"), ("last_finished", "0"))
            if name in summary_columns
        ]
        # Sub-agent rows do not count, as in the web app: the row is
        # worked out from the chat's listable tasks.
        listable = (
            ' AND (t."parent_task_id" IS NULL OR t."parent_task_id" = \'\')'
            if "parent_task_id" in columns
            else ""
        )
        if cleared:
            con.execute(
                f'UPDATE main."chat_summaries" SET {", ".join(cleared)}'
                f' WHERE "chat_id" IN (SELECT t."chat_id" {doomed}{listable})'
            )
    if "events" in tables:
        con.execute(f'DELETE FROM main."events" WHERE "task_id" IN (SELECT t."id" {doomed})')
    deleted = con.execute(
        f'DELETE FROM main."task_history" WHERE "id" IN (SELECT t."id" {doomed})'
    ).rowcount
    if summarised:
        # A chat started again after its deletion (a launch younger than
        # the deletion) is a new chat with the old id and keeps its row.
        launched = (
            'COALESCE("chat_summaries"."last_launched", 0)'
            if "last_launched" in summary_columns
            else "0"
        )
        con.execute(
            'DELETE FROM main."chat_summaries" WHERE EXISTS'
            ' (SELECT 1 FROM main."deleted_chats" d'
            ' WHERE d."chat_id" = "chat_summaries"."chat_id"'
            f' AND d."deleted_ts" * 1000 > {launched})'
            ' AND NOT EXISTS (SELECT 1 FROM main."task_history" t'
            ' WHERE t."chat_id" = "chat_summaries"."chat_id")'
        )
    return max(deleted, 0)


def _open(database: str) -> sqlite3.Connection:
    """Open a database that is already there, to update it in place.

    Args:
        database: Path of the database; ``~`` is expanded.

    Returns:
        The connection, its header read: a connection alone opens nothing,
        and a file that is no database would fail the first statement run
        on it, whatever that statement names.  It is opened by URI: that
        is what keeps a file that is not there from being created, and
        what lets an ``ATTACH`` on it name a read-only URI rather than a
        file called "file:...".

    Raises:
        sqlite3.Error: If there is no such file, or it is no database.
    """
    uri = f"{Path(os.path.expanduser(database)).absolute().as_uri()}?mode=rw"
    con = sqlite3.connect(uri, timeout=60.0, uri=True)
    try:
        con.execute("PRAGMA schema_version").fetchone()
    except sqlite3.Error:
        con.close()
        raise
    return con


def carry_over(old_database: str, new_database: str) -> tuple[int, int, int]:
    """Carry every unsynced table of one database into another.

    Args:
        old_database: Database that was replaced; only read.
        new_database: Database that replaced it; updated in place.

    Returns:
        The total number of rows inserted (or given their title back)
        and raised, and the number of task rows the carried-over
        deletions removed.  ``(0, 0, 0)`` when there is no old database,
        or none that can be opened -- which is usually why it was
        replaced.

    Raises:
        sqlite3.Error: If the old database opened but the carry-over then
            failed.  Reporting that as success would leave the counters
            only in a backup file nobody looks at.
    """
    old_path = os.path.expanduser(old_database)
    if not os.path.isfile(old_path):
        return 0, 0, 0
    # Reading a database that was left in WAL mode makes SQLite create the
    # shared-memory index and an empty write-ahead log beside it, even
    # read-only.  The index is machine-local and regenerated on demand, the
    # log holds nothing, and the file they appear beside is a backup nobody
    # is meant to open again, so both are cleaned up below rather than left
    # as debris in ~/.kiss.
    had_shm = os.path.exists(old_path + "-shm")
    had_wal = os.path.exists(old_path + "-wal")
    con = _open(new_database)
    inserted = raised = deleted = 0
    try:
        try:
            old_uri = f"{Path(old_path).absolute().as_uri()}?mode=ro"
            con.execute("ATTACH DATABASE ? AS old", (old_uri,))
        except sqlite3.Error as exc:
            # Not a database, or not one this SQLite can read: that is why it
            # was replaced, and there is nothing in it to carry over.
            print(f"carry_over_tables: {old_database}: {exc}", file=sys.stderr)
            return 0, 0, 0
        try:
            with con:
                for ddl in DELETED_DDL:
                    con.execute(ddl)
                for table, key in CARRIED_TABLES.items():
                    added, lifted = carry_over_table(con, table, key)
                    inserted += added
                    raised += lifted
                inserted += carry_over_titles(con)
                deleted = apply_deletions(con)
        finally:
            con.execute("DETACH DATABASE old")
    finally:
        con.close()
        if not had_shm:
            with contextlib.suppress(OSError):
                os.unlink(old_path + "-shm")
        if not had_wal:
            # Only while it is empty: frames in it would be rows of the backup.
            with contextlib.suppress(OSError):
                if not os.path.getsize(old_path + "-wal"):
                    os.unlink(old_path + "-wal")
    return inserted, raised, deleted


def main(argv: list[str] | None = None) -> int:
    """Carry the unsynced tables over from the command line.

    Args:
        argv: Arguments ``[old_database, new_database]``, or
            ``[--apply-deletions, database]`` to carry out the deletions
            one database records on itself (prints how many task rows
            went); defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 when the carry-over failed,
        2 on a usage error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "--apply-deletions":
        # Only the last step, on a database that already holds its deletion
        # records: a snapshot downloaded whole carries deletions the machine
        # it came from had recorded but not carried out yet.
        try:
            con = _open(args[1])
            try:
                with con:
                    for ddl in DELETED_DDL:
                        con.execute(ddl)
                    deleted = apply_deletions(con)
            finally:
                con.close()
        except sqlite3.Error as exc:
            print(f"carry_over_tables: {args[1]}: {exc}", file=sys.stderr)
            return 1
        print(deleted)
        return 0
    if len(args) != 2:
        print(
            "usage: carry_over_tables.py OLD_DATABASE NEW_DATABASE"
            " | --apply-deletions DATABASE",
            file=sys.stderr,
        )
        return 2
    old, new = (os.path.expanduser(arg) for arg in args)
    if os.path.isfile(old) and os.path.isfile(new) and os.path.samefile(old, new):
        # Its read-only attachment of itself would hold the lock its writes wait for.
        print(
            "carry_over_tables: OLD_DATABASE and NEW_DATABASE are the same file", file=sys.stderr
        )
        return 2
    try:
        inserted, raised, deleted = carry_over(args[0], args[1])
    except sqlite3.Error as exc:
        print(f"carry_over_tables: {args[1]}: {exc}", file=sys.stderr)
        return 1
    print(
        f"{inserted} row(s) restored, {raised} counter(s) raised"
        + (f", {deleted} deleted task(s) removed" if deleted else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

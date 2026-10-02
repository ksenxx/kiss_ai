#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Summarise the synced content of a ``sorcar.db`` in one line.

A deploy that has to replace a machine's task database wholesale must
know that the rows it brought back from that machine are still all it
has.  Counting tasks is not enough: a task that ran there can gain
events, gain steps and finish without the number of tasks changing at
all, and the replacement would then put an older copy of it in place.

So both sides are compared on everything that a sync moves: the number
of tasks, the work they have done, the number of events and the newest
one -- and then, because a task can also be marked a favourite, cost more
or simply finish without any of those counts moving, a digest of the task
rows themselves.  It is printed as a single line so that a shell can
compare two of them with ``=``:

    <tasks> <steps> <tokens> <events> <newest_rowid> <newest_timestamp> <digest>

The events are covered by their count and their newest row alone: an
event is written once and never edited, so a change to that table always
shows up in one of the two.  The newest row is the one with the highest
rowid (``events.id`` is an ``AUTOINCREMENT`` key, so a new row always
gets a higher one), which SQLite finds without reading the table; a
maximum over ``seq`` or ``timestamp`` would scan every event, and on a
database with millions of them that took longer than the sync itself.

Usage:
    python3 db_fingerprint.py DATABASE

Nothing is printed and the exit status is 1 when the database cannot be
read or does not have the columns a sync needs, so that a caller cannot
mistake "I could not look" for "nothing changed".  It is stdlib-only and
self-contained so that it can be piped straight into a remote ``python3``
over ssh.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys

TASK_QUERY = (
    'SELECT count(*), COALESCE(SUM("steps"), 0), COALESCE(SUM("tokens"), 0)'
    " FROM task_history"
)
# Three scalar subqueries rather than one aggregate query: SQLite answers a
# lone count(*) from the index's page counts and a lone MAX(rowid) with one
# seek, but a SELECT that aggregates both loops over every row.
EVENT_QUERY = (
    "SELECT (SELECT count(*) FROM events),"
    " COALESCE((SELECT MAX(rowid) FROM events), 0),"
    ' COALESCE((SELECT "timestamp" FROM events ORDER BY rowid DESC LIMIT 1), 0)'
)
# Ordered by the primary key, so that the digest describes what the rows say
# and not the order sqlite happens to return them in.
DIGEST_QUERY = 'SELECT * FROM task_history ORDER BY "id"'
DIGEST_LENGTH = 16


def task_digest(con: sqlite3.Connection) -> str:
    """Return a digest of every task row, whatever the columns are.

    Args:
        con: Open connection to the database to read.

    Returns:
        A short hexadecimal digest that changes when any task row does.
    """
    digest = hashlib.sha256()
    for row in con.execute(DIGEST_QUERY):
        digest.update(repr(row).encode("utf-8", "surrogatepass"))
        digest.update(b"\n")
    return digest.hexdigest()[:DIGEST_LENGTH]


def fingerprint(database: str) -> str:
    """Return the one-line summary of a database's synced content.

    Args:
        database: Path of the ``sorcar.db`` to summarise.

    Returns:
        The seven fields separated by spaces.

    Raises:
        sqlite3.Error: If the database cannot be read, or lacks the
            tables and columns a sync moves.
    """
    path = os.path.expanduser(database)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60.0)
    try:
        tasks = con.execute(TASK_QUERY).fetchone()
        events = con.execute(EVENT_QUERY).fetchone()
        digest = task_digest(con)
    finally:
        con.close()
    return " ".join(str(value) for value in (*tasks, *events, digest))


def main(argv: list[str] | None = None) -> int:
    """Print a database's fingerprint from the command line.

    Args:
        argv: Arguments ``[database]``; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 when the database cannot be
        read, 2 on a usage error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: db_fingerprint.py DATABASE", file=sys.stderr)
        return 2
    try:
        print(fingerprint(args[0]))
    except sqlite3.Error as exc:
        print(f"db_fingerprint: {args[0]}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

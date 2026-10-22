#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Report the tasks of a ``history.db`` that are still running.

A deploy rebuilds the Python environment under the web app and restarts
it, and replacing the task database wholesale stops it outright.  Either
one kills a task that is running at that moment: the agent stops
mid-step and the steps it had left are gone.  This script is what lets a
deploy see that coming and refuse.

A task counts as running when it has no end time *and* its newest event
is younger than the window -- a heartbeat.  The end time alone says
nothing: a database accumulates thousands of rows with no end time,
left by tasks that were interrupted or crashed months ago, and refusing
to deploy because of those would mean never deploying again.

Usage:
    python3 running_tasks.py DATABASE [WINDOW_SECONDS] [--unfinished]

The number of running tasks is printed on the first line, followed by
one ``<task_id> <seconds_since_last_event>`` line each.

``--unfinished`` answers a different question, the one the task-database
sync asks: which rows are not final yet?  A task's row is written when
the task starts, with ``"Agent Failed Abruptly"`` for a result, and
rewritten when the task ends -- or, for a task whose process died, by
the next start of the web app.  The sync only ever adds rows to the
receiving database, so a row copied before that rewrite would stay
unfinished there for good.  An unfinished task is one that still
carries that placeholder, or that is running by the heartbeat above
(a row written by a version without the placeholder); the age printed
after its id is that of its newest event, or of the row itself when it
has none.

A machine with no database, or whose ``history.db`` is not an SQLite file
at all, has no running task and prints ``0`` -- a first deploy has
nothing to lose.  A database that exists but cannot be *read* is a
different answer: it prints ``unknown`` and exits 1, because it might be
holding a running task and this script cannot tell.  Reading the answer
as "nothing is running" is how a deploy kills a task it was told to look
for.

One thing this does not see: a task waiting for an answer from the user
writes no events while it waits, so after WINDOW_SECONDS of waiting it
looks idle.  Raise the window when that matters.

It is stdlib-only and self-contained so that it can be piped straight
into a remote ``python3`` over ssh.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

DEFAULT_WINDOW = 300.0
TASK_TABLE = "task_history"
EVENT_TABLE = "events"
SQLITE_MAGIC = b"SQLite format 3\x00"
# The result the web app writes into a task's row when the task starts
# (``persistence._add_task``) and replaces when it ends.
UNFINISHED_RESULT = "Agent Failed Abruptly"


def is_sqlite_file(path: str) -> bool:
    """Report whether a file starts with SQLite's format marker.

    Args:
        path: File to look at.

    Returns:
        True when the file could be a database.  A file that is not one
        cannot be holding a running task, whatever else is wrong with it.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


def running_tasks(
    database: str, window: float = DEFAULT_WINDOW, now: float | None = None
) -> list[tuple[str, float]] | None:
    """List the tasks of a database that look like they are running.

    Args:
        database: Path of the ``history.db`` to inspect.
        window: How recent a task's newest event has to be, in seconds,
            for the task to count as running.
        now: Present time as an epoch timestamp; defaults to
            :func:`time.time`.

    Returns:
        One ``(task_id, seconds_since_its_newest_event)`` pair per
        running task, the freshest first; empty when nothing is running,
        which includes there being no database and there being a file
        that is not one.  None when the question could not be answered --
        a database that exists but cannot be read may well be holding a
        running task.
    """
    return _query(database, window, now, unfinished=False)


def unfinished_tasks(
    database: str, window: float = DEFAULT_WINDOW, now: float | None = None
) -> list[tuple[str, float]] | None:
    """List the tasks of a database whose rows are not final yet.

    A task's row still carrying the result the web app writes at its
    start (:data:`UNFINISHED_RESULT`) will be rewritten -- when the task
    ends, or by the next start of the web app for a task whose process
    died -- and so must not be copied anywhere before that; a task that
    is running by the heartbeat of :func:`running_tasks` counts too.

    Args:
        database: Path of the ``history.db`` to inspect.
        window: How recent a task's newest event has to be, in seconds,
            for a task without the placeholder to count as running.
        now: Present time as an epoch timestamp; defaults to
            :func:`time.time`.

    Returns:
        One ``(task_id, age_in_seconds)`` pair per unfinished task, the
        freshest first -- the age of its newest event, or of the row
        itself when it has none; empty when every row is final, which
        includes there being no database.  None when the database exists
        but cannot be read.
    """
    return _query(database, window, now, unfinished=True)


def _query(
    database: str, window: float, now: float | None, unfinished: bool
) -> list[tuple[str, float]] | None:
    """Run the running-task query, widened to placeholder rows when asked.

    The query is built from what the database has: a database without
    the ``events`` table, or without the ``end_ts``, ``result`` or
    ``timestamp`` columns, cannot hold a task that is running or
    unfinished in a way anyone could tell, so the answer is that none
    is -- only a database that cannot be read answers None.
    """
    path = os.path.expanduser(database)
    if not os.path.isfile(path) or not is_sqlite_file(path):
        return []
    moment = time.time() if now is None else now
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    except sqlite3.Error:
        return None
    try:
        tables = {
            str(row[0])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if TASK_TABLE not in tables:
            return []
        columns = {str(row[1]) for row in con.execute(f'PRAGMA table_info("{TASK_TABLE}")')}
        if "id" not in columns:
            return []
        started = 't."timestamp"' if "timestamp" in columns else "NULL"
        ended = 'COALESCE(t."end_ts", 0) <= 0' if "end_ts" in columns else "1"
        tests: list[str] = []
        params: list[object] = []
        if unfinished and "result" in columns:
            tests.append('t."result" = ?')
            params.append(UNFINISHED_RESULT)
        if EVENT_TABLE in tables and window > 0:
            tests.append(f'({ended} AND MAX(e."timestamp") > ?)')
            params.append(moment - window)
            newest = f'COALESCE(MAX(e."timestamp"), {started})'
            source = f'{TASK_TABLE} t LEFT JOIN {EVENT_TABLE} e ON e."task_id" = t."id"'
        else:
            newest = started
            source = f"{TASK_TABLE} t"
        if not tests:
            return []
        rows = con.execute(
            f'SELECT t."id", {newest} FROM {source} GROUP BY t."id" HAVING {" OR ".join(tests)}',
            params,
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    return sorted(
        ((str(task_id), moment - float(last or 0.0)) for task_id, last in rows),
        key=lambda pair: pair[1],
    )


def main(argv: list[str] | None = None) -> int:
    """Print the running (or, with ``--unfinished``, the unfinished) tasks.

    Args:
        argv: Arguments ``[database, window_seconds, --unfinished]``, the
            last two optional; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 when the database could not
        be read, 2 on a usage error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    unfinished = "--unfinished" in args
    args = [arg for arg in args if arg != "--unfinished"]
    if not args or len(args) > 2:
        print(
            "usage: running_tasks.py DATABASE [WINDOW_SECONDS] [--unfinished]",
            file=sys.stderr,
        )
        return 2
    try:
        window = float(args[1]) if len(args) > 1 else DEFAULT_WINDOW
    except ValueError:
        print(f"running_tasks: not a number of seconds: {args[1]}", file=sys.stderr)
        return 2
    query = unfinished_tasks if unfinished else running_tasks
    tasks = query(args[0], window)
    if tasks is None:
        print("unknown")
        print(f"running_tasks: cannot read {args[0]}", file=sys.stderr)
        return 1
    print(len(tasks))
    for task_id, age in tasks:
        print(f"{task_id} {age:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

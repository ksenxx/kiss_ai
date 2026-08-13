#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Report the tasks of a ``sorcar.db`` that are still running.

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
    python3 running_tasks.py DATABASE [WINDOW_SECONDS]

The number of running tasks is printed on the first line, followed by
one ``<task_id> <seconds_since_last_event>`` line each.

A machine with no database, or whose ``sorcar.db`` is not an SQLite file
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
        database: Path of the ``sorcar.db`` to inspect.
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
    path = os.path.expanduser(database)
    if window <= 0 or not os.path.isfile(path) or not is_sqlite_file(path):
        return []
    moment = time.time() if now is None else now
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    except sqlite3.Error:
        return None
    try:
        rows = con.execute(
            f'SELECT t."id", MAX(e."timestamp") FROM {TASK_TABLE} t'
            f' JOIN {EVENT_TABLE} e ON e."task_id" = t."id"'
            ' WHERE COALESCE(t."end_ts", 0) <= 0'
            f' GROUP BY t."id" HAVING MAX(e."timestamp") > ?',
            (moment - window,),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    fresh = sorted(
        ((str(task_id), moment - float(last)) for task_id, last in rows),
        key=lambda pair: pair[1],
    )
    return fresh


def main(argv: list[str] | None = None) -> int:
    """Print the running tasks of a database from the command line.

    Args:
        argv: Arguments ``[database, window_seconds]``; defaults to
            ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 when the database could not
        be read, 2 on a usage error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or len(args) > 2:
        print("usage: running_tasks.py DATABASE [WINDOW_SECONDS]", file=sys.stderr)
        return 2
    try:
        window = float(args[1]) if len(args) > 1 else DEFAULT_WINDOW
    except ValueError:
        print(f"running_tasks: not a number of seconds: {args[1]}", file=sys.stderr)
        return 2
    tasks = running_tasks(args[0], window)
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

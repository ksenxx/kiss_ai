"""Backfill ``tags``, ``sea`` and ``chat_summaries`` in an existing ``history.db``.

Usage::

    uv run python -m kiss.scripts.backfill_task_metadata [--db PATH] [--refresh]

Adds the ``tags`` / ``sea`` columns and the ``chat_summaries`` table when
the database predates them (the same ``ALTER TABLE`` / ``CREATE TABLE IF
NOT EXISTS`` migration the daemon runs at start-up), then fills them for
every row that lacks a value with
:func:`kiss.agents.sorcar.task_metadata.backfill_task_metadata`.
``--refresh`` recomputes every task's tags and every chat summary first
(for a classifier change).  Safe to run while the daemon is up: writes
are committed in short batches and wait up to 30 s for a busy database.
"""

from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
import time
from pathlib import Path

from kiss.core.config import kiss_home


def _open(database: str) -> sqlite3.Connection:
    """Open a database that is already there, to update it.

    Args:
        database: Path of the database.

    Returns:
        The connection, its header read: a connection alone opens nothing,
        and a file that is no database would only fail at the first query.

    Raises:
        sqlite3.Error: If there is no such file or it cannot be read as a
            database.
    """
    uri = f"{Path(database).absolute().as_uri()}?mode=rw"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    try:
        conn.execute("PRAGMA schema_version").fetchone()
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def main(argv: list[str] | None = None) -> int:
    """Run the backfill on the given database and print the counts.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 when the database is missing
        or is not one.
    """
    from kiss.agents.sorcar.persistence import _init_tables
    from kiss.agents.sorcar.task_metadata import backfill_task_metadata

    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--db", default=str(kiss_home() / "history.db"))
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="recompute every task's tags and every chat summary (after a classifier change) "
        "instead of filling only the empty ones; inferred SEAs are kept",
    )
    args = parser.parse_args(argv)
    try:
        conn = _open(args.db)
    except sqlite3.Error as exc:
        print(f"cannot open {args.db}: {exc}", file=sys.stderr)
        return 1
    started = time.monotonic()
    with contextlib.closing(conn):
        with conn:
            _init_tables(conn)
            if args.refresh:
                conn.execute("UPDATE task_history SET tags = ''")
                # Blank the derived summary only: ``user_summary`` is the
                # title the user typed and nothing can recompute it.
                conn.execute("UPDATE chat_summaries SET summary = ''")
        counts = backfill_task_metadata(conn)
    print(
        f"{args.db}: tagged {counts['tags']} tasks, inferred the SEA of "
        f"{counts['sea']} sub-agent tasks, summarised {counts['chats']} chats "
        f"in {time.monotonic() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

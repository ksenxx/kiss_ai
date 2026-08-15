#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Re-point the work directories recorded in a ``sorcar.db`` copy.

Every task remembers the directory it ran in, and the History panel hides
tasks from other workspaces by default.  The same project checked out on
two machines lives at two different paths, so a database travelling from
one to the other has to have those paths translated, or the imported
history arrives complete and invisible.

Usage:
    python3 relocate_work_dir.py DATABASE FROM_DIR TO_DIR

Only paths inside ``FROM_DIR`` move -- the directory itself and anything
below it (the agent does most of its work in ``.kiss-worktrees/``).
Directories elsewhere, a lookalike sibling such as ``kiss_ai`` next to
``kiss``, and the empty path are left exactly as they are.  ``FROM_DIR``
or ``TO_DIR`` naming the root directory relocates nothing: ``/`` is not a
project.

The number of relocated tasks is printed on stdout.

This is meant to be run on a throw-away copy of a database -- never on a
live one, which would rewrite that machine's own history.  It is
stdlib-only and self-contained so that it can be piped straight into a
remote ``python3`` over ssh, and it touches nothing but the recorded work
directories.

Databases written before the flat ``work_dir`` column existed keep the
path inside their ``extra`` JSON instead; the migration that runs when
such a database is first opened copies that value across verbatim, so
those are rewritten too.
"""

from __future__ import annotations

import json
import sqlite3
import sys

TASK_TABLE = "task_history"


def relocate(path: str, from_dir: str, to_dir: str) -> str | None:
    """Translate one recorded work directory to the other checkout.

    Args:
        path: Work directory as recorded in the database.
        from_dir: Checkout the path may be inside, without a trailing slash.
        to_dir: Checkout it should point at, without a trailing slash.

    Returns:
        The translated path, or None when *path* is not inside
        *from_dir* and must be left alone.
    """
    if path == from_dir:
        return to_dir
    if path.startswith(from_dir + "/"):
        return to_dir + path[len(from_dir) :]
    return None


def relocate_database(database: str, from_dir: str, to_dir: str) -> int:
    """Re-point every work directory of a database at another checkout.

    Args:
        database: Path of the database copy to rewrite in place.
        from_dir: Checkout the recorded paths currently point at.
        to_dir: Checkout they should point at instead.

    Returns:
        The number of task rows whose work directory moved.
    """
    from_dir, to_dir = from_dir.rstrip("/"), to_dir.rstrip("/")
    if not from_dir or not to_dir or from_dir == to_dir:
        return 0  # "/" is not a project directory, and a no-op is a no-op
    con = sqlite3.connect(database, timeout=60.0)
    try:
        columns = {row[1] for row in con.execute(f"PRAGMA table_info({TASK_TABLE})")}
        if "work_dir" in columns:
            moved = _relocate_column(con, from_dir, to_dir)
        elif "extra" in columns:
            moved = _relocate_extra_json(con, from_dir, to_dir)
        else:
            moved = 0  # no work directory is recorded at all
        con.commit()
    finally:
        con.close()
    return moved


def _relocate_column(
    con: sqlite3.Connection, from_dir: str, to_dir: str
) -> int:
    """Rewrite the ``work_dir`` column of every task inside *from_dir*.

    Args:
        con: Open connection to the database being rewritten.
        from_dir: Checkout the recorded paths currently point at.
        to_dir: Checkout they should point at instead.

    Returns:
        The number of rows updated.
    """
    updates = [
        (moved_to, row_id)
        for row_id, work_dir in con.execute(
            f"SELECT rowid, work_dir FROM {TASK_TABLE} WHERE work_dir IS NOT NULL"
        ).fetchall()
        if (moved_to := relocate(str(work_dir), from_dir, to_dir)) is not None
    ]
    con.executemany(
        f"UPDATE {TASK_TABLE} SET work_dir = ? WHERE rowid = ?", updates
    )
    return len(updates)


def _relocate_extra_json(
    con: sqlite3.Connection, from_dir: str, to_dir: str
) -> int:
    """Rewrite the ``work_dir`` key inside the legacy ``extra`` JSON.

    Metadata that is not valid JSON, or that is JSON but not an object,
    is left untouched: it is not something this script can safely edit.

    Args:
        con: Open connection to the database being rewritten.
        from_dir: Checkout the recorded paths currently point at.
        to_dir: Checkout they should point at instead.

    Returns:
        The number of rows updated.
    """
    updates = []
    for row_id, extra in con.execute(
        f"SELECT rowid, extra FROM {TASK_TABLE} WHERE extra IS NOT NULL"
    ).fetchall():
        try:
            payload = json.loads(extra)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        moved_to = relocate(str(payload.get("work_dir", "")), from_dir, to_dir)
        if moved_to is None:
            continue
        payload["work_dir"] = moved_to
        updates.append((json.dumps(payload), row_id))
    con.executemany(
        f"UPDATE {TASK_TABLE} SET extra = ? WHERE rowid = ?", updates
    )
    return len(updates)


def main(argv: list[str] | None = None) -> int:
    """Run the relocation from the command line.

    Args:
        argv: Arguments ``[database, from_dir, to_dir]``; defaults to
            ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 2 on a usage error, 1 when the
        database cannot be rewritten.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(
            "usage: relocate_work_dir.py DATABASE FROM_DIR TO_DIR", file=sys.stderr
        )
        return 2
    try:
        print(relocate_database(*args))
    except sqlite3.Error as exc:
        print(f"relocate_work_dir: {args[0]}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

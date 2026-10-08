#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Give the task database of a ``~/.kiss`` directory its current name.

The database has been ``history.db`` since version 2026.10.2; it was
``sorcar.db`` from 2026-04-24 until then -- and ``history.db`` once
before, until 2026-04-24, when the rename to ``sorcar.db`` left the old
file where it was.  An install from the spring of 2026 therefore holds
BOTH names, and the leftover under the new one is not the database: it
has the schema of that time (integer task ids, none of today's columns),
which the current version cannot even open (``no such column:
parent_task_id``), while every task since lives in ``sorcar.db``.

The leftover is told apart by that schema: ``task_history.id`` is an
``INTEGER``, which no version since the UUID migration of 2026-06-28
writes, so a ``sorcar.db`` next to it is always the newer database.

Usage:
    python3 legacy_task_db.py KISS_DIR
        Print the name of the file that holds the history: ``sorcar.db``
        when it is still under the old name (``history.db`` absent, or
        such a leftover), ``history.db`` otherwise.
    python3 legacy_task_db.py KISS_DIR adopt
        Rename ``sorcar.db`` to ``history.db`` when the first form would
        print ``sorcar.db``: a leftover ``history.db`` is first moved
        out of the way to ``history.db.stale-<UTC time>`` (kept, with
        its ``-wal``/``-shm`` sidecars, never deleted), the sidecars of
        ``sorcar.db`` are renamed before the file itself, and the old
        name is left as a symlink to the new one so that a web app still
        running the previous version keeps writing to the same file.
        Prints one line per rename; nothing when there was nothing to do.

It is stdlib-only and self-contained so that it can be piped straight
into a remote ``python3`` over ssh.  The persistence layer
(``kiss.agents.sorcar.persistence``) does the same adoption, under a
lock, through ``kiss.core.config.adopt_legacy_file``.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows has no fcntl (and gets no symlink either)
    fcntl = None  # type: ignore[assignment]

CURRENT_NAME = "history.db"
LEGACY_NAME = "sorcar.db"
SIDECARS = ("-wal", "-shm", "")


def is_pre_2026_04_db(path: Path) -> bool:
    """Tell whether the database at *path* was last written before 2026-04-24.

    Same probe as :func:`kiss.core.config.is_pre_2026_04_db`, repeated
    here because this file is piped standalone into a remote ``python3 -``
    and cannot import the package.

    Args:
        path: The database file to inspect; it is opened read-only.

    Returns:
        True for the pre-rename schema (``task_history.id`` is an
        ``INTEGER``); False for the current schema, for a file without a
        ``task_history`` table, and for one SQLite cannot read (left as
        it is: nothing is set aside on a guess).
    """
    try:
        conn = sqlite3.connect(f"{Path(os.path.abspath(path)).as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:  # pragma: no cover — unreadable file
        return False
    try:
        cols = {
            r[1]: (r[2] or "").upper()
            for r in conn.execute("PRAGMA table_info(task_history)").fetchall()
        }
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return cols.get("id") == "INTEGER"


def live_db_name(kiss_dir: Path) -> str:
    """Return the name of the file in *kiss_dir* that holds the task history.

    ``sorcar.db`` when the database is still under its pre-2026.10.2
    name -- ``history.db`` is absent, or is a leftover from before
    2026-04-24 (:func:`is_pre_2026_04_db`) -- and ``history.db``
    otherwise, including when ``sorcar.db`` is the symlink an earlier
    adoption left behind.
    """
    current = kiss_dir / CURRENT_NAME
    legacy = kiss_dir / LEGACY_NAME
    if not legacy.exists():
        return CURRENT_NAME
    if not current.exists():
        return LEGACY_NAME
    if current.samefile(legacy) or not is_pre_2026_04_db(current):
        return CURRENT_NAME
    return LEGACY_NAME


def adopt_legacy_db(kiss_dir: Path) -> list[str]:
    """Rename ``sorcar.db`` in *kiss_dir* to ``history.db`` when that is where the history is.

    Serialises with the web app's own adoption
    (``kiss.core.config.adopt_legacy_file``) on the same lock file,
    ``history.db.rename.lock``, and decides under the lock: a web app
    starting at this very moment must not find its freshly adopted
    database set aside as the leftover.

    Returns:
        One line per rename performed, in order; empty when ``history.db``
        already held the history.
    """
    if not (kiss_dir / LEGACY_NAME).exists():
        return []
    lock_path = kiss_dir / (CURRENT_NAME + ".rename.lock")
    with open(lock_path, "a+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _adopt_legacy_db_locked(kiss_dir)


def _adopt_legacy_db_locked(kiss_dir: Path) -> list[str]:
    """The renames of :func:`adopt_legacy_db`, with the lock held."""
    if live_db_name(kiss_dir) != LEGACY_NAME:
        return []
    current = kiss_dir / CURRENT_NAME
    legacy = kiss_dir / LEGACY_NAME
    done: list[str] = []
    if current.exists():
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        set_aside = current.with_name(f"{CURRENT_NAME}.stale-{stamp}")
        for suffix in SIDECARS:
            if current.with_name(CURRENT_NAME + suffix).exists():
                os.replace(
                    current.with_name(CURRENT_NAME + suffix),
                    set_aside.with_name(set_aside.name + suffix),
                )
        done.append(
            f"{CURRENT_NAME} is a leftover from before 2026-04-24; kept as {set_aside.name}"
        )
    for suffix in SIDECARS:
        if legacy.with_name(LEGACY_NAME + suffix).exists():
            os.replace(
                legacy.with_name(LEGACY_NAME + suffix), current.with_name(CURRENT_NAME + suffix)
            )
    if os.name != "nt":
        os.symlink(CURRENT_NAME, legacy)
    done.append(f"{LEGACY_NAME} renamed to {CURRENT_NAME}")
    return done


def main(argv: list[str]) -> int:
    """Run the command line described in the module docstring."""
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[2] != "adopt"):
        print(f"usage: {Path(argv[0]).name} KISS_DIR [adopt]", file=sys.stderr)
        return 2
    kiss_dir = Path(argv[1])
    if len(argv) == 2:
        print(live_db_name(kiss_dir))
        return 0
    for line in adopt_legacy_db(kiss_dir):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The tests' reset of the history database's connections.

No product code closes the database for the whole process, so this
reset lives with the tests that need it: they redirect
``persistence._DB_PATH`` to a private database and must not keep a
connection, the event writer or a cache of the previous one.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.sorcar import persistence
from kiss.core.locked import locked_call


def close_db(db: Any = persistence) -> None:
    """Close the calling thread's connection and invalidate cached handles.

    The event writer is drained and stopped first.  The current thread's
    connection, and those of threads that have already exited, are
    closed eagerly.  Bumping the generation counter invalidates every
    live thread's cached connection *lazily*: each is detected as stale
    on that thread's next ``_get_db()`` call and replaced (and closed)
    there.  A dead thread never makes that call, and its handle would
    otherwise keep the database file open (and, on Windows, undeletable)
    until some thread opens a new connection.

    Args:
        db: The persistence module to reset; a test that re-imports the
            module passes its own copy.
    """
    db._stop_event_writer()
    db._close_thread_db()
    locked_call(db._init_tables_lock, _retire_other_threads_conns, db)


def _retire_other_threads_conns(db: Any) -> None:
    """Bump the generation and close dead threads' connections (``_init_tables_lock`` held)."""
    db._bump_db_generation()
    db._prune_dead_thread_conns()

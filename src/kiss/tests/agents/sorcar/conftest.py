# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared pytest hooks for the sorcar test suite.

Orphan-sweep join hook
----------------------

Every :class:`kiss.server.server.VSCodeServer` constructor
starts a daemon thread named ``orphan-task-sweep`` that runs SQL on a
per-thread SQLite connection aliased into ``persistence._db_conn``.
Many tests in this folder construct a ``VSCodeServer``, then in
``teardown_method`` close ``th._db_conn``, restore the redirected DB
globals, and ``rmtree`` the temp KISS_HOME — WITHOUT joining the sweep
thread.  If the sweep is still executing when the connection is closed
/ the DB file is deleted, the C-level ``pysqlite_connection_execute``
call dereferences a freed connection and the interpreter dies with
SIGSEGV (observed intermittently in e.g.
``test_restore_tabs_with_subagents.py``, crashing in
``_log_orphaned_task_forensics`` while the main thread executed the
test's ``teardown_method``).

The server intentionally keeps the thread handle
(``_orphan_sweep_thread``) "so tests can join it deterministically".
Rather than editing ~60 test files, the ``pytest_runtest_call``
hookwrapper below joins every live ``orphan-task-sweep`` thread right
after the test body finishes and BEFORE ``teardown_method`` runs —
exactly the window in which the DB connection is still valid.  This
mirrors the identical hook in ``tests/agents/vscode/conftest.py``.

Persistence connection-cache reset
----------------------------------

Many tests in this folder exercise the real sqlite persistence layer
(:mod:`kiss.agents.sorcar.persistence`).  When they run in the same
process AFTER tests from other suites that swap or delete the database
file at ``persistence._DB_PATH``, a stale per-thread connection (or a
still-running background event-writer thread) left behind by those
tests can keep writing into the orphaned old file while these tests
read a fresh, empty database — producing order-dependent failures such
as ``sqlite3.OperationalError: no such table: task_history``.  The
autouse reset below mirrors ``tests/server/conftest.py`` and
``tests/agents/vscode/conftest.py``; many persistence-layer tests
relocated here from ``tests/server`` relied on its identical autouse
reset there.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from kiss.agents.sorcar import persistence


@pytest.fixture(autouse=True)
def _reset_persistence_connection_cache() -> Iterator[None]:
    """Reset the persistence layer's cached sqlite connections before each test.

    ``persistence._close_db()`` stops and drains the background event
    writer, clears the sequence/HasEvents caches, bumps the global
    connection generation counter (lazily invalidating EVERY thread's
    cached connection, not just this thread's), and closes the current
    thread's connection.  The next ``_get_db()`` call therefore
    reconnects against the CURRENT ``persistence._DB_PATH`` and
    re-creates the schema, eliminating the test-order-dependent
    ``task_history`` failures described in the module docstring.
    """
    persistence._close_db()
    yield


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    """Join lingering ``orphan-task-sweep`` threads after the test body.

    Runs after the test's call phase but before its teardown phase, so
    the sweep thread finishes while the per-test SQLite database and
    connection are still alive — preventing the use-after-close SIGSEGV
    described in the module docstring.
    """
    try:
        yield
    finally:
        for thread in threading.enumerate():
            if thread.name == "orphan-task-sweep" and thread.is_alive():
                thread.join(timeout=30)

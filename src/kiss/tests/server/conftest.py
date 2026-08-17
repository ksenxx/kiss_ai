# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared fixtures for the kiss.server test suite.

These tests exercise the real sqlite persistence layer
(:mod:`kiss.agents.sorcar.persistence`).  When they run in the same
process AFTER tests from other suites that swap or delete the database
file at ``persistence._DB_PATH``, a stale per-thread connection (or a
still-running background event-writer thread) left behind by those
tests can keep writing into the orphaned old file while these tests
read a fresh, empty database — producing order-dependent failures such
as ``sqlite3.OperationalError: no such table: task_history`` or rows
that silently never appear in ``_get_history()``.

This mirrors ``tests/vscode/conftest.py``: many tests relocated here
from that directory relied on its identical autouse reset, and the
same order-dependence protection is appropriate for every server test.

The orphan-sweep join guard for ``VSCodeServer``'s ``orphan-task-sweep``
threads lives in the ROOT ``tests/conftest.py`` and protects this
directory too; see its module docstring for the mechanism.
"""

from __future__ import annotations

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

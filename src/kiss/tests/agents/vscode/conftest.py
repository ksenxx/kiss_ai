# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared pytest fixtures for VS Code / kiss-web web_server tests.

These tests instantiate the real :class:`RemoteAccessServer` and call
the module-level helpers that read/write ``remote-url.json``.  The
default file lives at ``~/.kiss/remote-url.json`` (or
``$KISS_HOME/remote-url.json``), which is watched every 10 seconds by:

* the running ``kiss-web`` daemon (it re-reads the file to publish the
  active URL), and
* the VS Code extension (it polls the file and, on certain transitions,
  invokes ``restartKissWebDaemon`` which kills the daemon).

When the test process and a live ``kiss-web`` daemon / VS Code extension
share the same path, writes and unlinks from tests can sever the live
agent's transport.  The root ``tests/conftest.py`` already isolates the
process by setting ``KISS_HOME`` to a per-process ``tempfile.mkdtemp``
so the shared path is no longer the live one.

For per-test isolation beyond that, individual tests can pass an
explicit ``url_file=tmp_path / "remote-url.json"`` to
:class:`RemoteAccessServer` (added by the constructor refactor) and use
that same path for direct file inspection.

Orphan-sweep join guard
-----------------------

Many tests in this folder construct a ``VSCodeServer`` — which starts
the ``orphan-task-sweep`` daemon thread — and then close
``persistence._db_conn`` in their teardown, which used to crash the
interpreter with SIGSEGV whenever the sweep was still running SQL on
that connection.  The guard against that race lives in the ROOT
``tests/conftest.py`` (``join_orphan_sweeps``), so it protects every
test directory rather than this one only; see its module docstring for
the mechanism.

Persistence connection reset
----------------------------

Many tests in this folder exercise the real sqlite persistence layer
(:mod:`kiss.agents.sorcar.persistence`).  When they run in the same
process AFTER tests from other suites that swap or delete the database
file at ``persistence._DB_PATH``, a stale per-thread connection (or a
still-running background event-writer thread) left behind by those
tests can keep writing into the orphaned old file while these tests
read a fresh, empty database — producing order-dependent failures such
as ``sqlite3.OperationalError: no such table: task_history``.  The
autouse reset below mirrors ``tests/server/conftest.py`` (the tests
formerly in ``tests/vscode/`` relied on an identical reset in that
directory's conftest before they were relocated here).
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

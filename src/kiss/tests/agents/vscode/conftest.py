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

Orphan-sweep join hook
----------------------

Every :class:`VSCodeServer` constructor starts a daemon thread named
``orphan-task-sweep`` (see ``server.py``) that runs SQL on a
per-thread SQLite connection aliased into
``persistence._db_conn``.  Many tests in this folder construct a
``VSCodeServer``, then in ``teardown_method`` close ``th._db_conn``,
restore the redirected DB globals, and ``rmtree`` the temp KISS_HOME
— WITHOUT joining the sweep thread.  If the sweep is still executing
when the connection is closed / the DB file is deleted, the C-level
``pysqlite_connection_execute`` call dereferences a freed connection
and the interpreter dies with SIGSEGV (observed intermittently under
parallel load in e.g. ``test_replay_event_coalescing.py``).

The server intentionally keeps the thread handle
(``_orphan_sweep_thread``) "so tests can join it deterministically".
Rather than editing ~50 test files, the ``pytest_runtest_call``
hookwrapper below joins every live ``orphan-task-sweep`` thread right
after the test body finishes and BEFORE ``teardown_method`` runs —
exactly the window in which the DB connection is still valid.  Tests
that assert the sweep's asynchrony (e.g.
``test_web_server_startup_orphan_sweep_nonblocking.py``) are
unaffected: they make their timing assertions inside the test body,
and joining an already-finished thread is a no-op.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest


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

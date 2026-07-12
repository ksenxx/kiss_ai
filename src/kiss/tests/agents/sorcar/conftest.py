# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared pytest hooks for the sorcar test suite.

Orphan-sweep join hook
----------------------

Every :class:`kiss.agents.vscode.server.VSCodeServer` constructor
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

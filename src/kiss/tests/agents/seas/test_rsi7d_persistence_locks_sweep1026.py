# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The rsi7d readers open their connection BEFORE taking ``persistence._rw_lock``.

``persistence._rw_lock`` is writer-preferring: while one writer waits,
every new reader waits too, so anything slow done under the lock stalls
the whole persistence layer of the process.  ``_get_db()`` on a thread's
first call opens a connection and runs the migration check (which can
sit in SQLite's 30 s busy wait behind another process) and the
orphaned-sidecar recovery, so it belongs before the lock, as in
``persistence.py`` and ``task_digest.py``.  ``rsi7d_sea._rows_since``,
``_events_like`` and ``_task_row`` used to call it inside the read lock.

Same probe as ``tests/agents/sorcar/test_task_digest_sweep1026.py``: the
main thread holds the write lock, the reader runs on a fresh thread, and
the test waits for that thread's connection to appear in
``persistence._open_conns`` while the lock is still held.  A reader that
takes the lock first never gets there.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from typing import Any

from kiss.agents.seas.rsi7d import rsi7d_sea
from kiss.agents.sorcar import persistence
from kiss.tests.agents.seas.conftest import persist_task

PROBE_TIMEOUT_S = 10.0


def _collect(body: Callable[[], Any], result: Future[Any]) -> None:
    """Run *body* and store its return value, or the exception it raised, in *result*."""
    try:
        result.set_result(body())
    except BaseException as error:
        result.set_exception(error)


def _owns_a_conn(worker: threading.Thread) -> bool:
    return worker in {owner for _conn, owner, _path in list(persistence._open_conns.values())}


def _run_while_write_locked(body: Callable[[], Any]) -> tuple[bool, Any]:
    """Run *body* on a fresh thread while this thread holds the write lock.

    Returns ``(observed, result)``: whether the worker registered its
    connection while the write lock was still held, and what it returned
    (an exception raised on the worker is re-raised here).
    """
    result: Future[Any] = Future()
    worker = threading.Thread(target=_collect, args=(body, result))
    with persistence._rw_lock.write_lock():
        worker.start()
        deadline = time.monotonic() + PROBE_TIMEOUT_S
        observed = False
        while time.monotonic() < deadline:
            if _owns_a_conn(worker):
                observed = True
                break
            time.sleep(0.005)
    worker.join(timeout=PROBE_TIMEOUT_S)
    assert not worker.is_alive(), "the worker never got the lock after it was released"
    return observed, result.result(timeout=0)


def test_rows_since_opens_its_connection_before_the_read_lock() -> None:
    task_id = persist_task("rows since probe")
    observed, rows = _run_while_write_locked(partial(rsi7d_sea._rows_since, 1.0))
    assert observed, "_rows_since took the read lock before opening its connection"
    assert task_id in [r["id"] for r in rows]


def test_events_like_opens_its_connection_before_the_read_lock() -> None:
    task_id = persist_task(
        "events like probe",
        [{"type": "agent_text", "text": "needle-sweep1026"}],
    )
    observed, found = _run_while_write_locked(
        partial(rsi7d_sea._events_like, [task_id], "%needle-sweep1026%"),
    )
    assert observed, "_events_like took the read lock before opening its connection"
    assert [(tid, ev["text"]) for tid, ev in found] == [(task_id, "needle-sweep1026")]


def test_task_row_opens_its_connection_before_the_read_lock() -> None:
    task_id = persist_task("task row probe")
    observed, row = _run_while_write_locked(partial(rsi7d_sea._task_row, task_id))
    assert observed, "_task_row took the read lock before opening its connection"
    assert row is not None and row["id"] == task_id
    observed, missing = _run_while_write_locked(partial(rsi7d_sea._task_row, "no-such-task"))
    assert observed and missing is None

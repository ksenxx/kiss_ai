"""A Stop landing anywhere in a waiting acquisition never strands the RW lock's mutex.

CPython 3.13 leaves the back-edge ``JUMP_BACKWARD`` of a ``while`` loop
outside the exception table of the handlers of its own frame (cpython
gh-130279), so an injected ``KeyboardInterrupt`` raised on the back-edge of
a wait loop written inside the acquisition's ``try`` skipped the
``finally`` and left the condition's mutex held by the thread that then
died: every later reader and writer of the daemon's database waited on
it forever.  The wait loop now runs in a frame of its own
(``_RWLock._await``), from which the escaped stop reaches the
acquisition's covered ``CALL``.

The waiter is kept CPU-bound (``_WAIT_RECHECK_S = 0`` makes its
``Condition.wait`` a non-blocking acquire) so the stop is raised at a
bytecode check point of the waiting thread itself, back-edge included,
rather than after a blocking C call; hundreds of real injections at
random moments are the only way to reach that instruction.
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from kiss.agents.sorcar.persistence import _RWLock
from kiss.core.tool_interrupt import inject_async_exception

_ROUNDS = 150


def _hold_write_lock(lock: _RWLock, holding: threading.Event, release: threading.Event) -> None:
    """Hold *lock* for writing until *release* is set."""
    with lock.write_lock():
        holding.set()
        release.wait()


def _wait_for_read_lock(lock: _RWLock, entered: threading.Event) -> None:
    """Take a read lock (blocks behind the writer) and record success.

    The stop is caught here, one frame above the acquisition, as the
    task runner catches it above a task's own frames.
    """
    try:
        with lock.read_lock():
            entered.set()
    except KeyboardInterrupt:
        pass


@pytest.mark.timeout(120)
def test_a_stop_during_a_lock_wait_never_strands_the_mutex() -> None:
    """Every injected stop into a spinning waiter leaves the mutex free afterwards."""
    for n in range(_ROUNDS):
        lock = _RWLock()
        lock._WAIT_RECHECK_S = 0  # type: ignore[misc]  # a CPU-bound waiter
        holding, release, entered = threading.Event(), threading.Event(), threading.Event()
        writer = threading.Thread(
            target=_hold_write_lock, args=(lock, holding, release), daemon=True
        )
        writer.start()
        assert holding.wait(5)
        waiter = threading.Thread(
            target=_wait_for_read_lock, args=(lock, entered), daemon=True
        )
        waiter.start()
        time.sleep(random.uniform(0.0005, 0.003))
        assert waiter.ident is not None
        if inject_async_exception(waiter.ident, KeyboardInterrupt) != 1:
            pytest.fail(f"round {n}: the waiter was gone before the stop")
        waiter.join(2)
        if waiter.is_alive():
            # Dropped (it landed in a finalizer): the daemon's watchdog retries too.
            inject_async_exception(waiter.ident, KeyboardInterrupt)
            waiter.join(5)
        assert not waiter.is_alive(), f"round {n}: the stopped waiter did not finish"
        assert not entered.is_set(), f"round {n}: the stopped waiter took the lock"
        # The waiter died: its mutex level must be released, not stranded.
        if not lock._mutex.acquire(timeout=2):
            pytest.fail(f"round {n}: the mutex stayed held by the stopped waiter")
        lock._mutex.release()
        release.set()
        writer.join(5)
        assert not writer.is_alive(), f"round {n}: the writer could not leave its lock"

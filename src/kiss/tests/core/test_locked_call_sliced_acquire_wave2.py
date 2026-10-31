# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop reaches a thread waiting in :func:`kiss.core.locked.locked_call` on a held lock.

A thread parked in the untimed C-level ``RLock.acquire()`` reaches no
bytecode check point, so an injected ``KeyboardInterrupt`` (the daemon's
Stop) used to land only once the holder released, which for a merge's
conflict resolver under ``repo_lock`` can be minutes.  ``locked_call``
now acquires in ``ACQUIRE_SLICE_SECONDS`` slices: a holder keeps the lock
for ``_HOLD_SECONDS``, the waiting thread is stopped after one second and
must end well before the holder releases, without running the critical
section or leaking the lock.
"""

from __future__ import annotations

import threading
import time

import pytest

from kiss.core.locked import ACQUIRE_SLICE_SECONDS, locked_call
from kiss.core.tool_interrupt import inject_async_exception

_HOLD_SECONDS = 8.0


def _hold(lock: threading.RLock | threading.Condition, held: threading.Event,
          release: threading.Event) -> None:
    """Keep *lock* until *release* is set (at most ``_HOLD_SECONDS``)."""
    lock.acquire()
    try:
        held.set()
        release.wait(_HOLD_SECONDS)
    finally:
        lock.release()


def _take(lock: threading.RLock | threading.Condition, started: threading.Event,
          ran: list[bool], outcome: dict[str, float]) -> None:
    """Wait for *lock* through ``locked_call``; record when a Stop ends the wait."""
    started.set()
    try:
        locked_call(lock, ran.append, True)
    except KeyboardInterrupt:
        outcome["stopped_at"] = time.monotonic()


@pytest.mark.parametrize("make_lock", [threading.RLock, threading.Condition])
def test_stop_lands_while_another_thread_holds_the_lock(make_lock: type) -> None:
    """The injected Stop ends the wait within a slice; the holder's lock is untouched."""
    lock = make_lock()
    held, release, started = threading.Event(), threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold, args=(lock, held, release), daemon=True)
    holder.start()
    assert held.wait(5)
    ran: list[bool] = []
    outcome: dict[str, float] = {}
    taker = threading.Thread(target=_take, args=(lock, started, ran, outcome), daemon=True)
    taker.start()
    try:
        assert started.wait(5)
        time.sleep(1.0)
        assert taker.is_alive()
        assert taker.ident is not None
        injected_at = time.monotonic()
        assert inject_async_exception(taker.ident, KeyboardInterrupt) == 1
        taker.join(_HOLD_SECONDS)
        assert not taker.is_alive()
        assert holder.is_alive(), "the Stop landed only once the holder released"
        assert outcome["stopped_at"] - injected_at < ACQUIRE_SLICE_SECONDS + 1.0
        assert ran == []
    finally:
        release.set()
        holder.join(5)
    assert lock.acquire(blocking=False), "the stopped waiter leaked the lock"
    lock.release()

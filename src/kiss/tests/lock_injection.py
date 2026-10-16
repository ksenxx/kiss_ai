# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared helpers for the Stop-injection lock hammers.

The runner's Stop delivers ``KeyboardInterrupt`` to a task thread with
``PyThreadState_SetAsyncExc``, which CPython raises at the next check
point of the bytecode, including the one right after a ``with lock:``'s
``__enter__`` returned (gh-74174): a plain ``with`` on a task thread
leaks the lock.  Every lock a task thread can take therefore goes
through :func:`~kiss.core.locked.locked_call`.

:func:`hammer` drives a real code path from a worker thread while
hundreds of interrupts are injected, then asserts the worker exited
and that every lock the path takes is free.  The server, sorcar and
core injection tests all build on it; the per-path drivers stay in
the test files.
"""

from __future__ import annotations

import ctypes
import gc
import sys
import threading
import time
from collections.abc import Callable

from kiss.server import agent_state

INJECTIONS = 300


def inject(thread: threading.Thread) -> int:
    """Deliver ``KeyboardInterrupt`` asynchronously to *thread*, as the runner's stop does."""
    assert thread.ident is not None
    return int(ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident), ctypes.py_object(KeyboardInterrupt),
    ))


def ignore_thread_interrupt(args: threading.ExceptHookArgs) -> None:
    """``threading.excepthook`` for hammered workers: their interrupts are expected."""
    if args.exc_type is not KeyboardInterrupt and args.exc_value is not None:
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)


class Hammered(threading.Thread):
    """A thread repeating *call* until stopped, absorbing injected interrupts.

    An interrupt landing in the loop itself rather than in *call* ends
    the thread; the hammer then starts a fresh one.  ``in_call`` is
    true only while *call* runs: the hammer injects nothing while the
    thread is still in its bootstrap or already on its way out, where
    ``threading._bootstrap_inner`` and ``Thread._delete`` take
    ``threading._active_limbo_lock`` with a plain ``with`` (the same
    gh-74174 check point as every other lock), and an interrupt landing
    there leaves that lock owned by a dead thread and every later
    ``Thread.start()``, including the hammer's own restart, blocked
    for good.  ``Thread.start`` returns once ``_started`` is set, before
    the bootstrap's ``with``, so a freshly started worker was in exactly
    that window when the first injection reached it under load.
    """

    def __init__(self, call: Callable[[], object]) -> None:
        super().__init__(daemon=True)
        self.call = call
        self.stop = threading.Event()
        self.in_call = False

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.in_call = True
                self.call()
            except KeyboardInterrupt:
                pass
            finally:
                self.in_call = False


class Contender(threading.Thread):
    """A thread holding *lock* for 1 ms at a time until stopped.

    It makes the worker's ``acquire()`` block, so an injected interrupt
    lands on the return from a blocking acquire: the reliable gh-74174
    route for a lock the worker's path otherwise holds only briefly.
    """

    def __init__(self, lock: object) -> None:
        super().__init__(daemon=True)
        self.lock = lock
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            with self.lock:  # type: ignore[attr-defined]
                time.sleep(0.001)
            time.sleep(0.0002)


def lock_is_free(lock: object) -> bool:
    """Whether *lock* can be taken right now (no thread holds it)."""
    if not lock.acquire(blocking=False):  # type: ignore[attr-defined]
        return False
    lock.release()  # type: ignore[attr-defined]
    return True


def hammer(
    call: Callable[[], object],
    locks: dict[str, object],
    contend: object | None = None,
) -> None:
    """Inject ``INJECTIONS`` interrupts into a thread looping *call*; *locks* stay free.

    ``agent_state.STATE_LOCK`` is always checked too.  With *contend*,
    a :class:`Contender` holds that lock in 1 ms bursts for the whole
    run and is stopped before the locks are checked.

    Args:
        call: The code path to hammer, invoked repeatedly on the worker.
        locks: Name → lock; every one must be free after the hammer.
        contend: A lock to keep contended during the run, or ``None``.
    """
    previous_hook = threading.excepthook
    threading.excepthook = ignore_thread_interrupt
    # An interrupt landing in a finalizer the cyclic collector runs on
    # the worker (some earlier test's leftovers) is only noise.
    gc.collect()
    gc.disable()
    delivered = 0
    contender = Contender(contend) if contend is not None else None
    if contender is not None:
        contender.start()
    worker = Hammered(call)
    worker.start()
    attempts = 0
    try:
        # A slice that finds the worker outside *call* (its bootstrap, the
        # loop, a restart) injects nothing and is not counted against the
        # interrupts; the attempt cap only bounds a worker that never
        # reaches *call* at all.
        while delivered < INJECTIONS and attempts < 10 * INJECTIONS:
            attempts += 1
            time.sleep(0.0005)
            if not worker.is_alive():
                worker = Hammered(call)
                worker.start()
            if worker.in_call:
                delivered += inject(worker)
        worker.stop.set()
        worker.join(5.0)
        if contender is not None:
            contender.stop.set()
            contender.join(5.0)
    finally:
        gc.enable()
        threading.excepthook = previous_hook
    assert not worker.is_alive(), "the worker hung on a lock"
    assert contender is None or not contender.is_alive(), "the contender hung on a leaked lock"
    assert delivered > INJECTIONS // 2, "the interrupts did not reach the worker"
    locks = {"STATE_LOCK": agent_state.STATE_LOCK, **locks}
    leaked = [name for name, lock in locks.items() if not lock_is_free(lock)]
    assert not leaked, f"an interrupt leaked {leaked}"

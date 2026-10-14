# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Take a shared lock on a thread that can receive an injected exception.

The daemon stops a task by injecting ``KeyboardInterrupt`` into its
thread (``PyThreadState_SetAsyncExc``), and the per-panel Stop injects
``ToolCallInterrupted`` the same way.  CPython raises the injected
exception at the next check point of the bytecode, and one of those
sits right after a ``with lock:``'s ``__enter__`` CALL, outside the
``with`` block's exception region (CPython gh-74174): an exception
delivered there skips ``__exit__`` and the lock stays held by a thread
that then dies.  A thread blocked in a contended ``acquire`` receives
a pending injection exactly there.  :func:`locked_call` and
:func:`locked` put the ``acquire`` call itself inside the ``try`` so
the ``finally`` always releases.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any


def locked_call[T](
    lock: threading.RLock | threading.Condition, fn: Callable[..., T], *args: Any,
) -> T:
    """Call ``fn(*args)`` holding *lock*; the lock is released on every exit.

    For code that runs on a task thread, where the runner's Stop
    injects ``KeyboardInterrupt`` at any check point of the bytecode,
    including the one right after ``lock.__enter__`` returns: an
    exception delivered there skips ``__exit__`` (CPython gh-74174),
    so a plain ``with lock:`` leaks the lock for good and every later
    taker, which may be every later task of the daemon, hangs on it.
    Here the ``acquire`` call itself is inside the ``try``, so an
    interrupt landing right after it still reaches the ``finally``,
    and the ``finally`` releases before anything else can raise.  An
    interrupt landing before the acquisition took effect leaves the
    lock free or another thread's: ``RLock.release`` refuses it with
    ``RuntimeError``, which is swallowed.  This needs a reentrant lock
    (or a ``Condition`` over one, the default) for that owner check.
    A re-entrant hold (the printer's ``delivery_lock`` is held by the
    runner's setup-failure broadcast while ``broadcast`` takes it
    again) is safe: the ``finally`` runs only once the ``acquire``
    call returned, so its ``release`` undoes this call's own
    acquisition and never the outer one.

    Args:
        lock: The shared reentrant lock, or a condition over one.
        fn: The critical section.
        *args: Its positional arguments.

    Returns:
        ``fn``'s return value.
    """
    try:
        lock.acquire()
        return fn(*args)
    finally:
        try:
            lock.release()
        except RuntimeError:
            pass  # the interrupt landed before the acquisition took effect


def locked[**P, T](
    lock: threading.RLock | threading.Condition,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator form of :func:`locked_call` for a function that is one critical section.

    ``@locked(LOCK)`` on ``def f(...)`` makes every call of ``f`` run as
    ``locked_call(LOCK, f, ...)``, so a Stop injection can never leak
    ``LOCK`` through ``f``.  Keyword arguments are forwarded.

    Args:
        lock: The shared reentrant lock, or a condition over one.

    Returns:
        The decorator.
    """

    def decorate(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def call_locked(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                lock.acquire()
                return fn(*args, **kwargs)
            finally:
                try:
                    lock.release()
                except RuntimeError:
                    pass  # the interrupt landed before the acquisition took effect

        return call_locked

    return decorate

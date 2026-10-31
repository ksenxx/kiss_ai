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
the ``finally`` always releases, and acquire in short slices so a
thread waiting on a lock another task holds for minutes still gets
its injected exception within one slice.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any, cast

#: Length of one ``acquire`` slice of :func:`locked_call`: the longest an
#: injected Stop waits before a thread blocked on a held lock notices it.
ACQUIRE_SLICE_SECONDS = 0.5


class LockTimeoutError(TimeoutError):
    """Another thread held the lock for the whole ``acquire_timeout`` of :func:`locked_call`."""


def locked_call[T](
    lock: threading.RLock | threading.Condition,
    fn: Callable[..., T],
    /,
    *args: Any,
    acquire_timeout: float | None = None,
    **kwargs: Any,
) -> T:
    """Call ``fn(*args, **kwargs)`` holding *lock*; the lock is released on every exit.

    ``lock`` and ``fn`` are positional-only so ``fn``'s own keyword
    arguments named ``lock`` or ``fn`` reach ``fn`` instead of colliding
    with them; ``acquire_timeout`` is the one keyword ``fn`` never
    receives (:func:`locked`'s wrapper binds every argument with
    ``functools.partial``, so a decorated function receives that one too).

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
    A thread parked in an untimed C-level ``acquire()`` reaches no
    check point, so a Stop would land only when the holder releases
    (a merge's conflict resolver can hold ``repo_lock`` for minutes):
    the acquire is taken in :data:`ACQUIRE_SLICE_SECONDS` slices, each
    failed slice a check point.
    A re-entrant take (the printer's ``delivery_lock`` is held by the
    runner's setup-failure broadcast while ``broadcast`` takes it
    again) runs ``fn`` without acquiring: the outer frame owns the
    release, and an interrupt landing between the ``try`` and the
    inner ``acquire`` would otherwise make the ``finally`` release
    the outer hold.

    Args:
        lock: The shared reentrant lock, or a condition over one.
        fn: The critical section.
        *args: Its positional arguments.
        acquire_timeout: Seconds to wait for *lock* at most (``0``: only
            take it when it is free); ``None`` waits until it is free.
        **kwargs: Its keyword arguments.

    Returns:
        ``fn``'s return value.

    Raises:
        LockTimeoutError: When another thread held *lock* for the whole
            *acquire_timeout*; ``fn`` did not run.
    """
    if cast(Any, lock)._is_owned():
        return fn(*args, **kwargs)
    deadline = None if acquire_timeout is None else time.monotonic() + acquire_timeout
    try:
        while True:
            wait = ACQUIRE_SLICE_SECONDS
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - time.monotonic()))
            if lock.acquire(timeout=wait):
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise LockTimeoutError(f"the lock stayed held for {acquire_timeout:g}s")
        return fn(*args, **kwargs)
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
            return locked_call(lock, functools.partial(fn, *args, **kwargs))

        return call_locked

    return decorate

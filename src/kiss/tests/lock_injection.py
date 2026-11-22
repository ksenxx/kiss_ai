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

The standard library's own waits (``Event.wait``, ``Queue.get``,
``Future.result``, ``Thread.start``) take their lock through
``threading.Condition.__enter__``, a Python function with the same
check point, and are not the product's to rewrite: a task thread makes
them through :func:`~kiss.core.tool_interrupt.call_without_injection`.
:func:`stop_as_it_takes_its_lock` lands a Stop exactly there.
"""

from __future__ import annotations

import ctypes
import functools
import gc
import sys
import threading
import time
from collections.abc import Callable, Iterable
from types import CodeType, FrameType
from typing import Any

from kiss.core import tool_interrupt
from kiss.server import agent_state

INJECTIONS = 300

_CONDITION_ENTER = threading.Condition.__enter__.__code__
_THREAD_START = threading.Thread.start.__code__


def stop_as_it_takes_its_lock(
    taker: Any, *under: Any, stop: type[BaseException] = KeyboardInterrupt, settle: float = 0.0,
) -> Callable[[FrameType, str, Any], None]:
    """Return a profile hook delivering *stop* where the stdlib call *taker* has taken its lock.

    For ``sys.setprofile`` on the thread that makes the call.  *taker*
    is the function whose ``with`` enters the condition
    (``threading.Event.wait``, ``queue.Queue.get``, ``Future.done``),
    and *under* the functions it must be running in, innermost first
    (``threading.Thread.start`` for the wait that ends a thread's start,
    then the product function).  A thread's start is no match unless
    *under* names it.  The Stop is delivered once, with the product's
    injector, to the calling thread itself: unheld, CPython raises it as
    ``Condition.__enter__`` returns, the lock taken and no ``with``
    block to release it (gh-74174).  *settle* seconds pass first with
    the lock held, for whoever publishes to the wait (a reader its
    output, a watcher its verdict) to be waiting for that lock by then.
    """
    chain = tuple(function.__code__ for function in under)
    return functools.partial(_stop_as_it_takes_its_lock, taker.__code__, chain, stop, settle)


def _stop_as_it_takes_its_lock(
    taker: CodeType, chain: tuple[CodeType, ...], stop: type[BaseException], settle: float,
    frame: FrameType, event: str, arg: Any,
) -> None:
    walk = frame.f_back
    if event != "return" or frame.f_code is not _CONDITION_ENTER:
        return
    if walk is None or walk.f_code is not taker:
        return
    for code in chain:
        while walk is not None and walk.f_code is not code:
            if walk.f_code is _THREAD_START and _THREAD_START not in chain:
                return
            walk = walk.f_back
        if walk is None:
            return
    sys.setprofile(None)
    time.sleep(settle)
    tool_interrupt.inject_async_exception(threading.get_ident(), stop)


def stranded_threads(
    before: Iterable[threading.Thread], seconds: float = 10.0,
) -> list[threading.Thread]:
    """Return the threads started since *before* was listed that are still there after *seconds*.

    A thread whose ``start()`` was cut where it had taken the lock of
    the thread's ``_started`` event never runs its target: it stays
    blocked setting that event, listed by ``threading.enumerate()`` for
    good and never alive.  One blocked on a lock a Stop left held (a
    queue's, a future's) is alive for good.
    """
    known = set(before)
    deadline = time.monotonic() + seconds
    while True:
        left = [thread for thread in threading.enumerate() if thread not in known]
        if not left or time.monotonic() >= deadline:
            return left
        time.sleep(0.05)


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


def lock_is_free(lock: object, within: float = 0.0) -> bool:
    """Whether *lock* can be taken *within* seconds; by default right now (no thread holds it)."""
    if not lock.acquire(timeout=within):  # type: ignore[attr-defined]
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
    run and is stopped before the locks are checked.  A lock is leaked
    when it stays held: a thread the hammer does not stop (a manager's
    loop running the callbacks the hammered path queued) may hold one
    for a moment more, so each lock is given 5 s to come free.

    Args:
        call: The code path to hammer, invoked repeatedly on the worker.
        locks: Name → lock; every one must come free after the hammer.
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
    leaked = [name for name, lock in locks.items() if not lock_is_free(lock, within=5.0)]
    assert not leaked, f"an interrupt leaked {leaked}"

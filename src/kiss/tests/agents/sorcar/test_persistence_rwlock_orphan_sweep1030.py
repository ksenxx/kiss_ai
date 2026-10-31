"""Orphaned ``persistence._RWLock`` acquisitions (audit S1, 2026-10-30).

An orphan is an acquisition whose context manager was entered and never
exited: a Stop injected between ``__enter__`` returning and the ``with``
block installing leaves the acquisition generator suspended at its
``yield``, holding the lock.  The tests make one the same way, by
calling ``__enter__`` and dropping the context manager.

Three defects, each reproduced against the real lock:

1. The generator kept itself alive through its own frame (a ``gen``
   local and the ``box`` argument), so every orphan was cyclic garbage
   and kept blocking other threads until a garbage collection happened
   to run, instead of dying with its context manager.
2. A collection that reclaimed an orphan inside another critical section
   of the lock (a waiter evaluating its predicate under the mutex) ran
   the orphan's teardown there, and the teardown released every mutex
   level that thread held.  The waiter's next ``release``/``wait`` then
   raised ``RuntimeError``.  This happened on the orphan's own thread
   too, so a check of the owning thread alone would not fix it.
3. The same teardown discarding from the reader set while a writer's
   predicate iterated it raised ``RuntimeError: Set changed size during
   iteration`` out of ``write_lock()``.

Orphans that only a collection can reclaim are built by parking the
context manager in a reference cycle, as a stored exception's traceback
can.  Automatic collection is off during each test; the one collection
that matters runs from a ``sys.settrace`` hook at the first line of
``_token_active``, which only runs inside a wait predicate under the
lock's mutex.  That is the boundary an allocation-triggered collection
lands on, which no timed test can hit deterministically.  No production
code is replaced: the trace only chooses when the collection runs.
"""

from __future__ import annotations

import gc
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import persistence

_WAIT_S = 10.0

_Acquire = Callable[[], AbstractContextManager[None]]

_TOKEN_ACTIVE_CODE = persistence._RWLock._token_active.__code__


@pytest.fixture(autouse=True)
def _no_automatic_gc() -> Iterator[None]:
    """Keep automatic garbage collection off for the whole test.

    Only the collections a test runs on purpose may reclaim an orphan.
    The final collection frees whatever a failing test left behind, so a
    thread blocked by a surviving orphan can finish.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


class _CollectInPredicate:
    """Trace function running one ``gc.collect()`` inside a wait predicate.

    It fires at the first line of ``_RWLock._token_active``, which the
    lock calls only while evaluating a wait predicate under its mutex.
    """

    def __init__(self) -> None:
        self.fired = False

    def __call__(self, frame: FrameType, event: str, arg: object) -> _CollectInPredicate:
        if not self.fired and event == "line" and frame.f_code is _TOKEN_ACTIVE_CODE:
            self.fired = True
            gc.collect()
        return self


def _orphan(acquire: _Acquire, cyclic: bool) -> None:
    """Enter the context manager *acquire* returns and drop it unexited.

    With *cyclic* the context manager is first parked in a reference
    cycle, so only a garbage collection can reclaim the acquisition.
    """
    cm = acquire()
    cm.__enter__()
    if cyclic:
        cycle: list[Any] = [cm]
        cycle.append(cycle)


def _orphan_and_stay(acquire: _Acquire, made: threading.Event, stay: threading.Event) -> None:
    """Thread body: orphan an acquisition, then keep the thread alive."""
    _orphan(acquire, cyclic=False)
    made.set()
    stay.wait(_WAIT_S * 2)


def _acquire_once(acquire: _Acquire, acquired: threading.Event) -> None:
    """Thread body: take the lock once and report it."""
    with acquire():
        acquired.set()


def _acquire_traced(
    acquire: _Acquire,
    orphan_first: _Acquire | None,
    tracer: _CollectInPredicate,
    acquired: threading.Event,
    errors: list[BaseException],
) -> None:
    """Thread body: take the lock once with *tracer* installed.

    Args:
        acquire: The acquisition to make.
        orphan_first: When given, this thread first orphans an
            acquisition of its own (cyclic, so it awaits a collection).
        tracer: Runs the collection inside this thread's wait predicate.
        acquired: Set once the acquisition got in.
        errors: Receives whatever the acquisition raised.
    """
    if orphan_first is not None:
        _orphan(orphan_first, cyclic=True)
    sys.settrace(tracer)
    try:
        with acquire():
            acquired.set()
    except BaseException as exc:  # noqa: BLE001 — the RuntimeError under test
        errors.append(exc)
    finally:
        sys.settrace(None)


def _start(target: Callable[..., None], *args: object) -> threading.Thread:
    """Run *target* with *args* on a new daemon thread."""
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


@pytest.mark.parametrize(
    ("orphaned", "contender"), [("read_lock", "write_lock"), ("write_lock", "read_lock")],
)
def test_orphan_dies_with_its_context_manager(orphaned: str, contender: str) -> None:
    """An orphan stops blocking the moment its context manager is dropped.

    Its owner thread lives on (a task thread that caught the Stop) and
    no garbage collection runs; before the fix the contender stayed
    blocked until one did.
    """
    lock = persistence._RWLock()
    made, stay, acquired = threading.Event(), threading.Event(), threading.Event()
    owner = _start(_orphan_and_stay, getattr(lock, orphaned), made, stay)
    try:
        assert made.wait(_WAIT_S)
        contending = _start(_acquire_once, getattr(lock, contender), acquired)
        assert acquired.wait(_WAIT_S), (
            f"{contender} stayed blocked by an orphaned {orphaned} awaiting a collection"
        )
    finally:
        stay.set()
        gc.collect()
        owner.join(_WAIT_S)
    contending.join(_WAIT_S)


@pytest.mark.parametrize("orphan_owner", ["another thread", "the waiting thread"])
def test_collection_inside_a_wait_keeps_the_waiters_mutex(orphan_owner: str) -> None:
    """A reader blocked by an orphaned writer survives the collection that
    reclaims the orphan inside its predicate, and gets in."""
    lock = persistence._RWLock()
    if orphan_owner == "another thread":
        _start(_orphan, lock.write_lock, True).join(_WAIT_S)
        orphan_first = None
    else:
        orphan_first = lock.write_lock
    tracer = _CollectInPredicate()
    acquired = threading.Event()
    errors: list[BaseException] = []
    waiter = _start(_acquire_traced, lock.read_lock, orphan_first, tracer, acquired, errors)
    waiter.join(_WAIT_S)
    assert tracer.fired, "the collection never ran inside the wait predicate"
    assert errors == []
    assert acquired.is_set()
    writer_in = threading.Event()
    _start(_acquire_once, lock.write_lock, writer_in)
    assert writer_in.wait(_WAIT_S), "the reclaimed orphan still blocks writers"


def test_collection_inside_a_writers_predicate_keeps_its_iteration() -> None:
    """A writer whose predicate walks the reader set survives a collection
    that withdraws an orphaned reader from that set mid-walk."""
    lock = persistence._RWLock()
    _start(_orphan, lock.read_lock, True).join(_WAIT_S)
    with lock._cond:
        assert len(lock._reader_refs) == 1, "the orphan was reclaimed before the walk"
    tracer = _CollectInPredicate()
    acquired = threading.Event()
    errors: list[BaseException] = []
    writer = _start(_acquire_traced, lock.write_lock, None, tracer, acquired, errors)
    writer.join(_WAIT_S)
    assert tracer.fired, "the collection never ran inside the wait predicate"
    assert errors == []
    assert acquired.is_set()
    reader_in = threading.Event()
    _start(_acquire_once, lock.read_lock, reader_in)
    assert reader_in.wait(_WAIT_S), "the writer's acquisition was left behind"

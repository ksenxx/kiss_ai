# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``locked_call(..., acquire_timeout=...)``: a bounded take of a lock another thread holds.

``MachineLink.request`` (a link's turn), ``SshTunnel.close_unless_opening``
(a non-blocking take) and the SQLite side-car recovery (an idle handle)
used to hand-inline this acquire.  Real threads and real locks; nothing
is mocked.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from kiss.core.locked import ACQUIRE_SLICE_SECONDS, LockTimeoutError, locked, locked_call


def _hold(lock: threading.RLock, taken: threading.Event, release: threading.Event) -> None:
    with lock:
        taken.set()
        release.wait(10)


@pytest.fixture
def held_lock() -> Iterator[tuple[threading.RLock, threading.Event]]:
    """An ``RLock`` another thread holds until the yielded event is set."""
    lock = threading.RLock()
    taken, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold, args=(lock, taken, release), daemon=True)
    holder.start()
    assert taken.wait(5)
    try:
        yield lock, release
    finally:
        release.set()
        holder.join(5)


def _record(calls: list[str], name: str) -> str:
    calls.append(name)
    return name


def test_a_held_lock_times_out_without_running_fn(
    held_lock: tuple[threading.RLock, threading.Event],
) -> None:
    """The take gives up after *acquire_timeout*; ``fn`` never ran and the holder keeps the lock."""
    lock, _release = held_lock
    calls: list[str] = []
    started = time.monotonic()
    with pytest.raises(LockTimeoutError, match="held for 0.3s"):
        locked_call(lock, _record, calls, "late", acquire_timeout=0.3)
    assert 0.25 <= time.monotonic() - started < 0.3 + ACQUIRE_SLICE_SECONDS
    assert calls == []
    assert not lock.acquire(timeout=0)


def test_zero_takes_a_free_lock_and_refuses_a_held_one(
    held_lock: tuple[threading.RLock, threading.Event],
) -> None:
    """``acquire_timeout=0`` is a non-blocking take."""
    lock, _release = held_lock
    with pytest.raises(LockTimeoutError):
        locked_call(lock, _record, [], "held", acquire_timeout=0)
    free = threading.RLock()
    assert locked_call(free, _record, [], "free", acquire_timeout=0) == "free"
    assert not free._is_owned()  # type: ignore[attr-defined]


def test_a_lock_freed_within_the_bound_is_taken(
    held_lock: tuple[threading.RLock, threading.Event],
) -> None:
    """A wait longer than one slice keeps trying until the holder lets go."""
    lock, release = held_lock
    threading.Timer(ACQUIRE_SLICE_SECONDS + 0.2, release.set).start()
    calls: list[str] = []
    assert locked_call(lock, _record, calls, "taken", acquire_timeout=5.0) == "taken"
    assert calls == ["taken"]
    assert not lock._is_owned()  # type: ignore[attr-defined]


def test_a_reentrant_take_ignores_the_bound() -> None:
    """The owner runs ``fn`` at once, and its own hold survives the call."""
    lock = threading.RLock()
    with lock:
        assert locked_call(lock, _record, [], "inner", acquire_timeout=0) == "inner"
        assert lock._is_owned()  # type: ignore[attr-defined]


def test_fn_keyword_arguments_still_reach_fn() -> None:
    """``acquire_timeout`` is taken out; every other keyword goes to ``fn``."""
    lock = threading.RLock()
    calls: list[str] = []
    assert locked_call(lock, _record, calls, name="kw", acquire_timeout=1.0) == "kw"
    assert calls == ["kw"]


_DECORATED_LOCK = threading.RLock()


@locked(_DECORATED_LOCK)
def _decorated(lock: str, fn: str, acquire_timeout: float) -> tuple[str, str, float, bool]:
    return lock, fn, acquire_timeout, _DECORATED_LOCK._is_owned()  # type: ignore[attr-defined]


def test_a_decorated_function_receives_every_keyword() -> None:
    """``@locked`` holds its lock and passes even ``acquire_timeout`` through to the function."""
    assert _decorated(lock="l", fn="f", acquire_timeout=2.5) == ("l", "f", 2.5, True)
    assert not _DECORATED_LOCK._is_owned()  # type: ignore[attr-defined]

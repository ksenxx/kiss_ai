# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-13 (core): ``lock_exclusive`` owns the bounded wait.

``exclusive_file_lock(timeout=...)`` used to carry its own poll loop next
to the one the Windows branch of ``lock_exclusive`` already had; now
``lock_exclusive`` takes ``timeout`` itself and both callers share one
loop.  These tests take a REAL lock from another thread on a second
descriptor of the same file (advisory locks are per open file, so two
opens in one process contend exactly as two processes do).

The Windows-only branch of the same change (``msvcrt.locking`` errors
other than EACCES/EDEADLOCK now propagate instead of being retried
forever) cannot run here: this box has no ``msvcrt``.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from kiss.core.file_lock import exclusive_file_lock, lock_exclusive, unlock


def _hold(lock: Path, held: threading.Event, release: threading.Event) -> threading.Thread:
    """Start a thread that holds *lock* until *release* is set; *held* fires once taken."""

    def run() -> None:
        with exclusive_file_lock(lock):
            held.set()
            release.wait(10)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert held.wait(5)
    return thread


def test_lock_exclusive_timeout_raises_without_the_path(tmp_path: Path) -> None:
    """The raw-descriptor form reports the elapsed budget; it knows no path."""
    lock = tmp_path / "x.lock"
    held, release = threading.Event(), threading.Event()
    holder = _hold(lock, held, release)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="^still locked by another process after 0.2s$"):
            lock_exclusive(fd, timeout=0.2)
        assert 0.2 <= time.monotonic() - started < 3.0
    finally:
        os.close(fd)
        release.set()
        holder.join(5)


def _take_raw_descriptor(lock: Path, acquired: threading.Event, timeout: float | None) -> None:
    """Take *lock* through ``lock_exclusive(fd, timeout=...)``; set *acquired* once held."""
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert lock_exclusive(fd, timeout=timeout) is True
        acquired.set()
        unlock(fd)
    finally:
        os.close(fd)


def _take_context_manager(lock: Path, acquired: threading.Event, timeout: float | None) -> None:
    """Take *lock* through ``exclusive_file_lock``; set *acquired* once held."""
    with exclusive_file_lock(lock, timeout=timeout) as locked:
        assert locked is True
        acquired.set()


def _assert_waiter_blocks_until_released(
    lock: Path, take: Callable[[Path, threading.Event, float | None], None], timeout: float | None,
) -> None:
    """A waiter must stay blocked while the holder keeps *lock* and acquire once it lets go.

    This is a handshake, not a stopwatch: the waiter cannot set
    ``acquired`` while the holder keeps the lock (the lock is really
    held, so ``acquired.wait(0.2)`` can only succeed if the lock is
    broken), and must set it after ``release`` whatever the scheduler
    does in between.
    """
    held, release, acquired = threading.Event(), threading.Event(), threading.Event()
    holder = _hold(lock, held, release)
    waiter = threading.Thread(target=take, args=(lock, acquired, timeout), daemon=True)
    waiter.start()
    try:
        assert not acquired.wait(0.2)
        release.set()
        assert acquired.wait(5)
    finally:
        release.set()
        holder.join(5)
        waiter.join(5)


def test_lock_exclusive_timeout_takes_the_lock_once_released(tmp_path: Path) -> None:
    _assert_waiter_blocks_until_released(tmp_path / "x.lock", _take_raw_descriptor, timeout=5.0)


def test_lock_exclusive_non_blocking_ignores_timeout(tmp_path: Path) -> None:
    """``blocking=False`` gives up at once even when a timeout is passed."""
    lock = tmp_path / "x.lock"
    held, release = threading.Event(), threading.Event()
    holder = _hold(lock, held, release)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        started = time.monotonic()
        assert lock_exclusive(fd, blocking=False, timeout=5.0) is False
        assert time.monotonic() - started < 1.0
    finally:
        os.close(fd)
        release.set()
        holder.join(5)


def test_exclusive_file_lock_timeout_names_the_lock_file(tmp_path: Path) -> None:
    """The context-manager form prefixes the same error with the lock path."""
    lock = tmp_path / "x.lock"
    held, release = threading.Event(), threading.Event()
    holder = _hold(lock, held, release)
    try:
        with pytest.raises(TimeoutError) as info:
            with exclusive_file_lock(lock, timeout=0.2):
                pass
        assert str(info.value) == f"{lock} still locked by another process after 0.2s"
        assert info.value.__cause__ is None
    finally:
        release.set()
        holder.join(5)


def test_unbounded_blocking_wait_still_blocks_in_the_kernel(tmp_path: Path) -> None:
    """Without a timeout the POSIX path is the plain blocking ``flock``."""
    _assert_waiter_blocks_until_released(tmp_path / "x.lock", _take_context_manager, timeout=None)

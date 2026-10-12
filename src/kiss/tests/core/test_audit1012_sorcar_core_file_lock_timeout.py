# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-12 (sorcar-core): ``exclusive_file_lock(timeout=...)``.

The bounded wait that ``local_endpoint._endpoint_lock`` (and the TLS
lock in ``web_server``) used to hand-roll lives in ``file_lock`` now.
Every test takes a REAL lock on a temporary file from another thread
(advisory locks are per open file, so two opens in one process contend
exactly as two processes do).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from kiss.core.file_lock import exclusive_file_lock


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


def test_timeout_elapses_while_another_holder_keeps_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    held, release = threading.Event(), threading.Event()
    holder = _hold(lock, held, release)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="still locked by another process after 0.3s"):
            with exclusive_file_lock(lock, timeout=0.3):
                pass
        elapsed = time.monotonic() - started
        assert 0.3 <= elapsed < 3.0
    finally:
        release.set()
        holder.join(5)


def test_timeout_wait_takes_the_lock_once_released(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    held, release = threading.Event(), threading.Event()
    holder = _hold(lock, held, release)
    threading.Timer(0.2, release.set).start()
    try:
        started = time.monotonic()
        with exclusive_file_lock(lock, timeout=5.0) as locked:
            assert locked is True
            elapsed = time.monotonic() - started
        assert 0.15 <= elapsed < 4.0
    finally:
        release.set()
        holder.join(5)


def test_timeout_is_ignored_without_blocking(tmp_path: Path) -> None:
    """``blocking=False`` keeps its immediate give-up even with a timeout."""
    lock = tmp_path / "x.lock"
    held, release = threading.Event(), threading.Event()
    holder = _hold(lock, held, release)
    try:
        started = time.monotonic()
        with exclusive_file_lock(lock, blocking=False, timeout=5.0) as locked:
            assert locked is False
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
        holder.join(5)


def test_uncontended_timeout_lock_is_immediate(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    started = time.monotonic()
    with exclusive_file_lock(lock, timeout=5.0) as locked:
        assert locked is True
    assert time.monotonic() - started < 1.0

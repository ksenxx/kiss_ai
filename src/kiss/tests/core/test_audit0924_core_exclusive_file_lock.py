# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-24 (core): ``file_lock.exclusive_file_lock`` is the one
cross-process lock helper.

``vscode_config`` used to inline the open / ``lock_exclusive`` / ``try:
... finally: unlock`` dance three times (``save_config``,
``_api_keys_store_flock`` and the RC-file lock in ``save_api_key``)
while the identical helper ``exclusive_file_lock`` sat unused in
``file_lock.py``.  The three sites now use the helper, so its contract
is pinned here with real processes and real files:

* two processes doing a read-modify-write under the lock never lose an
  update (the exclusion is cross-process, not just cross-thread);
* the lock is released when the block exits, normally or by exception,
  so a later holder is not blocked forever;
* a missing parent directory of the lock file is created.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kiss.core.file_lock import exclusive_file_lock

_WORKER = r"""
import os, sys, time
from pathlib import Path
from kiss.core.file_lock import exclusive_file_lock
lock_path, counter = Path(sys.argv[1]), Path(sys.argv[2])
start_file, n = sys.argv[3], int(sys.argv[4])
deadline = time.time() + 10
while not os.path.exists(start_file):
    if time.time() > deadline:
        sys.exit(2)
    time.sleep(0.001)
for _ in range(n):
    with exclusive_file_lock(lock_path):
        value = int(counter.read_text() or "0")
        time.sleep(0.0005)  # widen the read-modify-write window
        counter.write_text(str(value + 1))
"""


def _run_workers(tmp: Path, n: int, workers: int) -> int:
    """Run *workers* processes that each increment a counter *n* times under the lock."""
    lock_path = tmp / "locks" / "counter.lock"
    counter = tmp / "counter.txt"
    counter.write_text("0")
    start_file = tmp / "start"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(lock_path), str(counter), str(start_file), str(n)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(workers)
    ]
    time.sleep(0.5)  # let the interpreters import before releasing them together
    start_file.write_text("go")
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode(errors="replace")
    return int(counter.read_text())


def test_two_processes_never_lose_an_increment() -> None:
    """Cross-process read-modify-write under the lock is serialized."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        total = _run_workers(Path(tmp_dir), n=150, workers=2)
    assert total == 300


def test_lock_released_after_block_and_after_exception() -> None:
    """A later holder is not blocked once the block exits, even by an exception."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        lock_path = Path(tmp_dir) / "x.lock"
        with exclusive_file_lock(lock_path):
            pass
        with pytest.raises(RuntimeError), exclusive_file_lock(lock_path):
            raise RuntimeError("boom")

        # The lock excludes a second holder while held and admits it afterwards.
        held = threading.Event()
        release = threading.Event()
        acquired_at: list[float] = []

        def holder() -> None:
            with exclusive_file_lock(lock_path):
                held.set()
                release.wait(10)

        def waiter() -> None:
            held.wait(10)
            with exclusive_file_lock(lock_path):
                acquired_at.append(time.monotonic())

        holder_thread = threading.Thread(target=holder)
        waiter_thread = threading.Thread(target=waiter)
        holder_thread.start()
        waiter_thread.start()
        assert held.wait(10)
        time.sleep(0.3)
        assert not acquired_at, "waiter acquired the lock while the holder still had it"
        released_at = time.monotonic()
        release.set()
        waiter_thread.join(10)
        holder_thread.join(10)
        assert acquired_at and acquired_at[0] >= released_at
        assert lock_path.exists(), "the lock file must never be deleted"

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Stop-injection hammer only interrupts a worker that is inside *call*.

``threading._bootstrap_inner`` and ``Thread._delete`` take
``threading._active_limbo_lock`` with a plain ``with``, the same
gh-74174 check point every hammered lock has.  ``Thread.start``
returns as soon as the new thread set ``_started``, before that
``with``, so an interrupt fired right after ``start()`` could land on
the return from the limbo lock's ``__enter__``: the worker dies owning
the lock and every later ``Thread.start()``, including the hammer's
own restart of the worker, blocks for good (seen as a 180 s child
timeout of the ``sweep`` git-worktree hammer in the 2026-10-15 full run,
reproduced deterministically here by holding the limbo lock while the
worker starts).  :class:`~kiss.tests.lock_injection.Hammered` now
reports ``in_call`` only while *call* runs and
:func:`~kiss.tests.lock_injection.hammer` injects nothing otherwise.

The check runs in a child process: a leaked limbo lock would wedge the
pytest process itself.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

from kiss.tests.lock_injection import Hammered, hammer


def _noop() -> None:
    """The hammered path: returns at once, so the worker is outside *call* most of the time.

    That is the worst case for the hammer: most slices find the worker
    between two calls and inject nothing, and the thread dies and
    restarts as often as it can.
    """


def run_child() -> None:
    """Child-process body: a starting worker is not ``in_call``; the hammer then runs clean."""
    worker = Hammered(_noop)
    # Holding the limbo lock parks the worker on its bootstrap ``with``
    # after ``start()`` returned: exactly where an injection leaks it.
    limbo_lock: threading.RLock = getattr(threading, "_active_limbo_lock")
    with limbo_lock:
        worker.start()
        time.sleep(0.05)
        assert not worker.in_call, "a worker still in its bootstrap claims to be in call"
    deadline = time.monotonic() + 5.0
    while not worker.in_call and time.monotonic() < deadline:
        time.sleep(0.001)
    assert worker.in_call, "the worker never reached call"
    worker.stop.set()
    worker.join(5.0)
    assert not worker.is_alive()
    hammer(_noop, {})
    # Starting one more thread proves the limbo lock is still free.
    tail = threading.Thread(target=_noop, daemon=True)
    tail.start()
    tail.join(5.0)
    assert not tail.is_alive()


def test_hammer_injects_only_inside_call() -> None:
    """The hammer leaves ``threading._active_limbo_lock`` free: threads still start after it."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_child()"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


class _MomentaryHolder(threading.Thread):
    """A thread the hammer does not stop: it holds *lock* until a second after the last call.

    The MCP manager's loop was such a thread in the Linux full run of 2026-11-22 (load
    average 85): it was still running the ``_forget_orphan`` callbacks the hammered
    ``_reap_straggler`` had queued, each under ``MCPManager._lock``, when the hammer read
    that lock once and reported it leaked, though the hammered path never takes it.
    """

    def __init__(self, lock: threading.RLock) -> None:
        super().__init__(daemon=True)
        self.lock = lock
        self.last_call = time.monotonic()

    def call(self) -> None:
        """The hammered path: it tells the holder that the hammer still runs.

        The sleep keeps the worker inside *call* for most slices, so the hammer is over in
        about as many slices as it has interrupts to deliver.
        """
        self.last_call = time.monotonic()
        time.sleep(0.001)

    def run(self) -> None:
        with self.lock:
            while time.monotonic() - self.last_call < 1.0:
                time.sleep(0.01)


def run_momentary_holder() -> None:
    """Child-process body: a lock another thread holds for a moment at the check is no leak."""
    lock = threading.RLock()
    holder = _MomentaryHolder(lock)
    holder.start()
    hammer(holder.call, {"held for a moment": lock})
    holder.join(5.0)
    assert not holder.is_alive()


def test_a_lock_another_thread_holds_for_a_moment_is_not_leaked() -> None:
    """The hammer waits for a lock that a thread it does not stop still holds for a moment."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_momentary_holder()"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

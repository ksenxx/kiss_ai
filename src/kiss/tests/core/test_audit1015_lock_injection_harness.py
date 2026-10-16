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

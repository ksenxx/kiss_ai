# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A re-entrant :func:`kiss.core.locked.locked_call` keeps the outer hold under injection.

The printer's ``delivery_lock`` is taken by ``locked_call`` while the
same thread already holds it (the runner's setup-failure broadcast
calls ``broadcast``, which takes it again).  Hundreds of Stop
interrupts are injected into a thread doing exactly that; whenever one
lands in the inner call, the outer frame must still own the lock when
it handles the interrupt, and the lock must be free once the thread is
gone.  The hammer runs in a subprocess: it swaps ``threading.excepthook``
and pauses the cyclic collector, which must not leak into other tests.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import Any, cast

import pytest

from kiss.core.locked import locked_call
from kiss.tests.lock_injection import hammer

LOCK = threading.RLock()
# Whether the outer frame still owned ``LOCK`` when an interrupt that
# landed in the inner ``locked_call`` reached it.
_OWNED_AT_INTERRUPT: list[bool] = []


def _inner() -> None:
    """The inner critical section: a few check points for an interrupt to land on."""
    for _ in range(3):
        pass


def _outer() -> None:
    """Take ``LOCK`` again while holding it; record ownership when the inner call is cut short."""
    try:
        locked_call(LOCK, _inner)
    except KeyboardInterrupt:
        _OWNED_AT_INTERRUPT.append(cast(Any, LOCK)._is_owned())
        raise


def _nested() -> None:
    locked_call(LOCK, _outer)


def run_hammer() -> None:
    """Hammer the nested take and check the outer hold survived every inner interrupt."""
    hammer(_nested, {"LOCK": LOCK})
    assert _OWNED_AT_INTERRUPT, "no interrupt landed in the inner locked_call"
    assert all(_OWNED_AT_INTERRUPT), "an inner interrupt released the outer hold"


def test_reentrant_take_keeps_the_outer_hold_under_injection() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer()"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


def test_reentrant_take_runs_the_section_and_leaves_the_lock_held_by_the_outer_frame() -> None:
    """Without interrupts: the inner call runs, and the outer release frees the lock."""
    seen: list[bool] = []

    def inner() -> str:
        seen.append(cast(Any, LOCK)._is_owned())
        return "ok"

    def outer() -> str:
        result = locked_call(LOCK, inner)
        seen.append(cast(Any, LOCK)._is_owned())
        return result

    assert locked_call(LOCK, outer) == "ok"
    assert seen == [True, True]
    assert LOCK.acquire(blocking=False), "the outer release must leave the lock free"
    LOCK.release()


if __name__ == "__main__":
    pytest.main([__file__])

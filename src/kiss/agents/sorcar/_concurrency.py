# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared concurrency primitives for the sorcar package.

One copy of the ``KISS_RACE_DELAY`` test hook that widens
read-modify-write windows in concurrency tests, for every module
(sorcar and server) that coordinates across threads or processes.
:func:`locked_call` / :func:`locked`, the way a task thread takes a
shared lock, live in :mod:`kiss.core.locked` (core must not import
sorcar) and are re-exported here under their established names;
:func:`interruptible_locked_call` is their form for a lock another
task may hold for minutes.  The ``pid_alive`` liveness probe lives in
:mod:`kiss.core.processes` and cross-process file locks in
:mod:`kiss.core.file_lock`.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any

from kiss.core.locked import locked, locked_call

__all__ = ["interruptible_locked_call", "locked", "locked_call"]

#: Length of one ``acquire`` slice of :func:`interruptible_locked_call`:
#: the longest an injected Stop waits before the task thread notices it.
ACQUIRE_SLICE_SECONDS = 0.5


def interruptible_locked_call[T](
    lock: threading.RLock,
    fn: Callable[..., T],
    /,
    *args: Any,
    **kwargs: Any,
) -> T:
    """:func:`locked_call` for a lock another task may hold for minutes.

    A task thread parked in the untimed C-level ``RLock.acquire()``
    reaches no bytecode check point, so an injected Stop
    (``KeyboardInterrupt`` from the daemon, ``ToolCallInterrupted``
    from a panel) is delivered only once the holder releases: a merge
    whose conflict resolver runs under ``repo_lock`` for minutes keeps
    every other tab's worktree setup, merge and discard of that
    repository un-stoppable for as long.  Acquiring in
    :data:`ACQUIRE_SLICE_SECONDS` slices gives the interrupt a check
    point after every slice.  The ``acquire`` loop stays inside the
    ``try`` (CPython gh-74174, see :func:`locked_call`): an interrupt
    landing right after a successful slice still reaches the
    ``finally``, and one landing after a failed slice makes ``release``
    raise ``RuntimeError``, which is swallowed.

    Args:
        lock: The shared reentrant lock.
        fn: The critical section.
        *args: Its positional arguments.
        **kwargs: Its keyword arguments.

    Returns:
        ``fn``'s return value.
    """
    try:
        while not lock.acquire(timeout=ACQUIRE_SLICE_SECONDS):
            pass
        return fn(*args, **kwargs)
    finally:
        try:
            lock.release()
        except RuntimeError:
            pass  # the interrupt landed before the acquisition took effect


def _race_delay() -> None:
    """Sleep briefly when ``KISS_RACE_DELAY`` is set (no-op by default).

    Concurrency tests need to widen a read-modify-write window to make
    a cross-process race deterministic.  The delay is opt-in via an
    environment variable that production never sets, and is capped at
    100 ms so a stray value can never stall a real run.
    """
    raw = os.environ.get("KISS_RACE_DELAY")
    if not raw:
        return
    try:
        time.sleep(min(float(raw), 0.1))
    except ValueError:
        pass

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
sorcar) and are re-exported here under their established names.
The ``pid_alive`` liveness probe lives in :mod:`kiss.core.processes`
and cross-process file locks in :mod:`kiss.core.file_lock`.
"""

from __future__ import annotations

import os
import time

from kiss.core.locked import locked, locked_call

__all__ = ["locked", "locked_call"]


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

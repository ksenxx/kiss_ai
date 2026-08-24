# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared concurrency primitives for the sorcar package.

Holds the pieces that ``persistence.py`` and ``git_worktree.py`` (and
any other sorcar module coordinating across threads or processes)
previously duplicated byte-for-byte: the optional ``fcntl`` import
used for cross-process ``flock`` locks, and the ``KISS_RACE_DELAY``
test hook that widens read-modify-write windows in concurrency tests.
"""

from __future__ import annotations

import os
import time

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover — Windows has no fcntl
    _fcntl = None  # type: ignore[assignment]


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

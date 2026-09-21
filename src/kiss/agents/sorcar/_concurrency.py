# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared concurrency primitives for the sorcar package.

Holds the pieces that ``persistence.py``, ``git_worktree.py``,
``web_use_tool.py`` and ``kiss.server.web_server`` (and any other
module coordinating across threads or processes) previously duplicated
with drifting semantics: the ``KISS_RACE_DELAY`` test hook that widens
read-modify-write windows in concurrency tests.  The ``pid_alive``
liveness probe is re-exported from :mod:`kiss.core.processes`, and
cross-process file locks live in :mod:`kiss.core.file_lock`.
"""

from __future__ import annotations

import os
import time

from kiss.core.processes import pid_alive as pid_alive  # noqa: F401 — shared probe


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

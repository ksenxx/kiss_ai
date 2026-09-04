# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared concurrency primitives for the sorcar package.

Holds the pieces that ``persistence.py``, ``git_worktree.py``,
``web_use_tool.py`` and ``kiss.server.web_server`` (and any other
module coordinating across threads or processes) previously duplicated
with drifting semantics: the optional ``fcntl`` import used for
cross-process ``flock`` locks, the ``KISS_RACE_DELAY`` test hook that
widens read-modify-write windows in concurrency tests, and the
``pid_alive`` liveness probe.
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


def pid_alive(pid: int) -> bool:
    """Return whether the OS process *pid* currently exists.

    Probes with ``os.kill(pid, 0)``, which sends no signal but performs
    the kernel's existence and permission checks:

    * ``pid <= 0`` is never a single process (``0`` is the caller's own
      process group, negative values address whole groups or every
      process), so it is reported dead without signalling anything;
    * success means the process exists and is ours;
    * ``ProcessLookupError`` means it is gone;
    * ``PermissionError`` means it exists but belongs to another user —
      still alive;
    * any other ``OSError`` (an invalid pid on an exotic platform) is
      reported dead.

    Args:
        pid: The process id to probe.

    Returns:
        True when the process exists, False when it does not.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # pragma: no cover — not reachable without doubles on Linux
        return False
    return True

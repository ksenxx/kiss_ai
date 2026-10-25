# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Read the live occupancy of a ``persistence._RWLock`` for assertions.

The lock keeps weak references to the generator frames of its live
acquisitions; these helpers count the ones still alive, which is what
a test means by "no reader left behind" after an injected stop.
"""

from __future__ import annotations

from kiss.agents.sorcar.persistence import _RWLock


def readers(lock: _RWLock) -> int:
    """Return the number of live acquisitions holding read access."""
    with lock._cond:
        return sum(1 for ref in lock._reader_refs if lock._token_active(ref))


def pending_writers(lock: _RWLock) -> int:
    """Return the number of live writers waiting for the lock."""
    with lock._cond:
        return sum(1 for ref in lock._pending_refs if lock._token_active(ref))


def writer(lock: _RWLock) -> bool:
    """Return whether a live writer holds exclusive access."""
    with lock._cond:
        return lock._writer_alive()

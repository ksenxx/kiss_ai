# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Exclusive inter-process file locks that work on POSIX and Windows.

``fcntl`` does not exist on Windows, so a module that imports it at the
top level cannot even be imported there — and because ``kiss.core.base``
imports ``model_info``, one such import made the whole package unusable
on Windows.  Every ``flock`` user goes through this module instead:
``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows.  Both are
advisory, per-open-file locks released automatically when the file is
closed or the process dies, so callers keep their existing
open-lock-work-unlock-close shape.

The lock is not reentrant across separate opens on either platform.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows has no fcntl
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # POSIX has no msvcrt
    msvcrt = None  # type: ignore[assignment]

# Windows locks a byte range rather than the whole file; the first byte
# is the range every locker agrees on (it may lie beyond the end of an
# empty lock file, which Windows allows).
_WINDOWS_LOCK_BYTES = 1
_WINDOWS_RETRY_INTERVAL = 0.05


def _fileno(target: int | IO[str] | IO[bytes]) -> int:
    return target if isinstance(target, int) else target.fileno()


def lock_exclusive(target: int | IO[str] | IO[bytes], blocking: bool = True) -> bool:
    """Take an exclusive advisory lock on the open file *target*.

    Args:
        target: A file descriptor or an open file object.  The lock
            belongs to that open file and is released by :func:`unlock`,
            by closing it, or by the process ending.
        blocking: Whether to wait for another holder to release the
            lock.  ``False`` gives up immediately instead.

    Returns:
        ``True`` when the lock is held.  ``False`` only when *blocking*
        is ``False`` and another process holds the lock.  On a platform
        with neither ``fcntl`` nor ``msvcrt`` the lock is a no-op and
        ``True`` is returned.
    """
    fd = _fileno(target)
    if fcntl is not None:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            return False
        return True
    if msvcrt is not None:  # pragma: no cover — Windows-only branch
        # Always the non-blocking ``LK_NBLCK`` probe: ``LK_LOCK`` sleeps a
        # full second between each of its ten internal attempts, so a
        # holder that keeps the lock for a millisecond still cost every
        # waiter one second (two cron jobs finishing together took over
        # 2 s to record their results).  The blocking case polls at
        # ``_WINDOWS_RETRY_INTERVAL`` until the lock is held.
        while True:
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, _WINDOWS_LOCK_BYTES)  # pyright: ignore[reportAttributeAccessIssue]
                return True
            except OSError:
                if not blocking:
                    return False
                time.sleep(_WINDOWS_RETRY_INTERVAL)
    return True  # pragma: no cover — platform without either primitive


def unlock(target: int | IO[str] | IO[bytes]) -> None:
    """Release the lock :func:`lock_exclusive` took on *target*.

    Args:
        target: The file descriptor or open file object that was locked.
    """
    fd = _fileno(target)
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover — Windows-only branch
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, _WINDOWS_LOCK_BYTES)  # pyright: ignore[reportAttributeAccessIssue]
        except OSError:
            pass


@contextmanager
def exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive lock on the file at *lock_path* for the block.

    Creates the file (and its parent directories) when missing, locks
    it, and releases and closes it afterwards.  The lock file is never
    deleted: unlinking it would let a later opener lock a different
    inode and defeat the exclusion.

    Args:
        lock_path: The lock file to hold.

    Yields:
        None while the lock is held.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        lock_exclusive(lock_file)
        try:
            yield
        finally:
            unlock(lock_file)

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

import errno
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
# ``msvcrt.locking`` errnos that mean "another holder has it": EACCES
# for the ``LK_NBLCK`` probe, EDEADLK (Windows' EDEADLOCK, the same
# value 36) for ``LK_LOCK``'s own timeout.  ``errno.EDEADLOCK`` is only
# defined where the C library has that alias — Linux and Windows, not
# macOS — so naming it here made ``import kiss`` fail on a Mac.
# Any other errno (EBADF for a closed descriptor, EINVAL) is a caller
# bug and propagates, as it does on POSIX.
_WINDOWS_CONTENDED_ERRNOS = (errno.EACCES, errno.EDEADLK)
# How often a polled wait re-probes the lock.
_POLL_INTERVAL = 0.05


def _fileno(target: int | IO[str] | IO[bytes]) -> int:
    return target if isinstance(target, int) else target.fileno()


def _try_lock(fd: int) -> bool:
    """Probe the lock once without waiting; ``True`` when it is now held."""
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    if msvcrt is not None:  # pragma: no cover — Windows-only branch
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, _WINDOWS_LOCK_BYTES)  # pyright: ignore[reportAttributeAccessIssue]
        except OSError as e:
            if e.errno in _WINDOWS_CONTENDED_ERRNOS:
                return False
            raise
        return True
    return True  # pragma: no cover — platform without either primitive


def lock_exclusive(
    target: int | IO[str] | IO[bytes], blocking: bool = True, timeout: float | None = None
) -> bool:
    """Take an exclusive advisory lock on the open file *target*.

    Args:
        target: A file descriptor or an open file object.  The lock
            belongs to that open file and is released by :func:`unlock`,
            by closing it, or by the process ending.
        blocking: Whether to wait for another holder to release the
            lock.  ``False`` gives up immediately instead.
        timeout: With *blocking*, the most seconds to wait; ``None``
            waits without limit.  Every wait polls the non-blocking
            lock every ``_POLL_INTERVAL`` instead of blocking in the
            kernel: a holder that wedged costs the waiter at most
            *timeout*, and a task Stop (a ``KeyboardInterrupt`` injected
            into the waiting thread, which a blocking ``flock`` would
            hold off until the lock is granted) lands between two
            probes.  Windows always polls this way: ``LK_LOCK`` sleeps
            a full second between each of its ten internal attempts, so
            a holder that kept the lock for a millisecond still cost
            every waiter one second (two cron jobs finishing together
            took over 2 s to record their results).

    Returns:
        ``True`` when the lock is held.  ``False`` only when *blocking*
        is ``False`` and another process holds the lock.  On a platform
        with neither ``fcntl`` nor ``msvcrt`` the lock is a no-op and
        ``True`` is returned.

    Raises:
        TimeoutError: When *timeout* elapsed with the lock still held
            by another process.
    """
    fd = _fileno(target)
    deadline = None if timeout is None else time.monotonic() + timeout
    while not _try_lock(fd):
        if not blocking:
            return False
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"still locked by another process after {timeout:g}s")
        time.sleep(_POLL_INTERVAL)
    return True


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
def exclusive_file_lock(
    lock_path: Path, blocking: bool = True, timeout: float | None = None
) -> Iterator[bool]:
    """Hold an exclusive lock on the file at *lock_path* for the block.

    Serializes check-then-use sequences on resources shared by every
    kiss process on the machine (config files, MCP and OAuth token
    stores, the cron job store, the Chromium profile directory) across
    daemons, CLI runs, channel-agent processes and event loops; a
    ``threading`` lock cannot, since the resources live on disk.
    Creates the file (mode ``0600``, with its parent directories) when
    missing, locks it, and releases and closes it afterwards.  The lock
    file is never deleted: unlinking it would let a later opener lock a
    different inode and defeat the exclusion.

    Args:
        lock_path: The lock file to hold.
        blocking: Whether to wait for the lock.  ``False`` gives up
            immediately when another process holds it (the cron
            scheduler's overlapping-tick skip) instead of waiting.
        timeout: With *blocking*, the most seconds to wait.  The wait
            polls the non-blocking lock instead of blocking in the
            kernel, so a holder that wedged (or a lock taken from an
            executor thread the event loop can no longer cancel) costs
            the waiter at most *timeout* instead of forever.  ``None``
            waits without limit.

    Yields:
        ``True`` while the lock is held; ``False`` only when *blocking*
        is ``False`` and another process holds the lock.

    Raises:
        TimeoutError: When *timeout* elapsed with the lock still held
            by another process.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        try:
            locked = lock_exclusive(descriptor, blocking, timeout)
        except TimeoutError as e:
            raise TimeoutError(f"{lock_path} {e}") from None
        yield locked
    finally:
        try:
            if locked:
                unlock(descriptor)
        finally:
            os.close(descriptor)

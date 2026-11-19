# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The agents' temp directory: ``<OS temp dir>/kiss-tmp``.

Every agent keeps its temporary files in a folder of this one directory
(a task's scratch files, its tools' spilled output and job logs, a
sub-agent's private folder), never in the work tree.  The directory
belongs to the user alone, and what is in it does not survive a reboot.
The operating system cannot be relied on for that (macOS and Windows keep
their temp directories across a reboot, Linux distributions differ), so
the first use after a reboot empties the directory (:func:`kiss_tmp_dir`;
on Windows the first use after a shutdown or restart, see :func:`_boot_id`).
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from kiss.core.file_lock import exclusive_file_lock
from kiss.core.utils import rmtree_force

logger = logging.getLogger(__name__)

KISS_TMP_NAME = "kiss-tmp"
"""The directory's name in the OS temp directory."""
SUBAGENT_DIR_PREFIX = "kiss-sub-"
"""How the name of a sub-agent's private folder in the directory starts.  The folder is
the sub-agent's scratch dir, holds its job logs and spilled output, is the temp directory
of every command its tools start, and is deleted when the sub-agent ends."""
CONTAINER_TMP_DIR = f"/tmp/{KISS_TMP_NAME}"
"""The directory in a container the tools run in (its temp directory is ``/tmp``)."""
_BOOT_MARKER = ".boot"
"""The file of the directory that names the boot its content belongs to (:func:`_boot_id`)."""
_BOOT_LOCK = ".boot.lock"
"""The lock file the marker is checked and the directory emptied under."""


def kiss_tmp_dir() -> Path:
    """Return the agents' temp directory, ``<OS temp dir>/kiss-tmp``, made when missing.

    The OS temp directory (:func:`_os_temp_dir`) is
    :func:`tempfile.gettempdir` (``$TMPDIR``, ``%TEMP%``, ``/tmp``),
    resolved so that every tool compares one path (macOS keeps it behind
    a symlink).  The directory is the user's own
    (:func:`_own_dir`).  Its content is that of the current boot: the
    marker file in it names the boot that wrote it, and the first call
    after a reboot, which finds another boot's name there, deletes
    everything else in the directory.  Processes that start together
    decide that under a file lock, so none deletes what another has made
    since.  A directory without a marker (a new one, or one the system
    emptied) and a machine with no boot id to read lose nothing, and
    neither does a directory that takes no lock or marker.  The one
    exception is a directory an earlier boot's marker stays in (the lock
    cannot be taken, or the marker can be neither replaced nor deleted):
    the call fails rather than hand out a directory that the next call
    would empty of what is made now.

    Raises:
        PermissionError: When the directory belongs to another user (see :func:`_own_dir`).
        OSError: When an earlier boot's marker stays in the directory.
    """
    path = _own_dir(_os_temp_dir())
    boot = _boot_id()
    marker = path / _BOOT_MARKER
    if boot and _recorded_boot(marker) != boot:
        try:
            with exclusive_file_lock(path / _BOOT_LOCK):
                _enter_boot(path, marker, boot)
        except OSError:
            if _recorded_boot(marker) not in ("", boot):
                raise
            # A temp directory that takes no lock or marker keeps what it has.
            logger.warning("Could not record the boot in %s", path, exc_info=True)
    return path


def _enter_boot(path: Path, marker: Path, boot: str) -> None:
    """Empty *path* of what an earlier boot left in it and record *boot* in *marker*.

    Called with the marker's lock held.  An earlier boot's marker that
    cannot be replaced is deleted, still under the lock: one that stayed
    would have the next call empty the directory again, of what this
    boot has made since.

    Raises:
        OSError: When the marker could not be written.
    """
    recorded = _recorded_boot(marker)
    if recorded == boot:
        return
    if recorded:
        _empty(path)
    try:
        marker.write_text(boot, encoding="utf-8")
    except OSError:
        if recorded:
            marker.unlink(missing_ok=True)
        raise


def _os_temp_dir() -> Path:
    """Return the OS temp directory the agents' one is in, resolved.

    A command of a sub-agent runs with the sub-agent's private folder,
    ``<agents' temp dir>/kiss-sub-…``, as its temp directory.  The
    agents' temp directory is then the one that folder is in, not a
    second one inside it: that would go with the sub-agent, and with it
    what commands keep there for later runs (the test durations, the
    check cache).
    """
    base = Path(tempfile.gettempdir()).resolve()
    if base.name.startswith(SUBAGENT_DIR_PREFIX) and base.parent.name.startswith(KISS_TMP_NAME):
        return base.parent.parent
    return base


def _own_dir(base: Path) -> Path:
    """Make the user's own ``kiss-tmp`` in *base* (mode ``0700``) and return it.

    A temp directory that several users share (``/tmp`` on Linux) may
    already hold another user's ``kiss-tmp``, or a link or a file
    someone planted under that name: scratch files written there could
    be read or replaced by that user.  The directory is then
    ``kiss-tmp-<uid>``.  A directory of the user's that others can write
    to is passed over like another user's (what is in it may be theirs),
    and one they can only enter is closed to them.  Windows gives every
    user a temp directory, so there the name is taken as it is.

    Raises:
        PermissionError: When both names belong to someone else.
    """
    getuid = getattr(os, "getuid", None)
    names = [KISS_TMP_NAME]
    if getuid is not None:  # pragma: no branch
        names.append(f"{KISS_TMP_NAME}-{getuid()}")
    for name in names:
        path = base / name
        try:
            path.mkdir(mode=0o700, exist_ok=True)
        except FileExistsError:
            continue  # A file, or a link to nothing, under that name.
        if getuid is None:  # pragma: no cover
            return path
        status = path.lstat()
        if (
            stat.S_ISDIR(status.st_mode) and status.st_uid == getuid()
            and not status.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            if stat.S_IMODE(status.st_mode) != 0o700:
                path.chmod(0o700)
            return path
    raise PermissionError(
        f"{base / names[-1]} belongs to another user; point TMPDIR at a directory of your own"
    )


@functools.cache
def _boot_id() -> str:
    """Return a name for the current boot of the machine, or ``""`` when it has none to read.

    The name is new after every reboot and the same for every process
    until the next one, whatever happens to the clock in between (a boot
    time computed from the clock and the uptime moves when the clock is
    set).  Linux: the kernel's random ``boot_id``.  macOS:
    ``kern.bootsessionuuid`` (``sysctl`` by its path: the ``PATH`` of a
    cron job has no ``/usr/sbin``).  Windows: the ``ShutdownTime`` the system
    recorded when it last shut down or restarted, which no running
    process outlives (after a crash it is unchanged, and the files stay
    until the next shutdown; the ``BootId`` boot counter is not used,
    since nothing says it cannot change while processes run).
    """
    try:
        if sys.platform == "win32":  # pragma: no cover
            winreg: Any = importlib.import_module("winreg")
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Windows",
            ) as key:
                return f"win32:{bytes(winreg.QueryValueEx(key, 'ShutdownTime')[0]).hex()}"
        if sys.platform == "darwin":
            name = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.strip()
            return f"darwin:{name}" if name else ""
        name = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        return f"linux:{name}" if name else ""
    except (OSError, subprocess.SubprocessError, ValueError):
        logger.debug("No boot id on this machine", exc_info=True)
        return ""


def _recorded_boot(marker: Path) -> str:
    """Return the boot name *marker* holds, ``""`` when there is no marker to read."""
    try:
        return marker.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


def _empty(path: Path) -> None:
    """Delete what an earlier boot left in *path*: everything but the marker and its lock.

    What cannot be deleted (a file another program holds open on Windows)
    stays, with a warning, for the next reboot.
    """
    for entry in path.iterdir():
        if entry.name in (_BOOT_MARKER, _BOOT_LOCK):
            continue
        try:
            if entry.is_dir() and not entry.is_symlink():
                rmtree_force(entry)
            else:
                entry.unlink()
        except OSError:
            logger.warning("Could not delete %s, left by an earlier boot", entry, exc_info=True)

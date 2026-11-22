#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Set the web app's password and work directory in ``config.json``.

Run on the machine a deploy is publishing.  ``~/.kiss/config.json``
holds everything the web app remembers between runs -- the model last
used, notification settings, the tunnel password -- so a deploy has to
edit it rather than write it:

* every other key is kept exactly as it was;
* a file that cannot be read or parsed is *kept* under a
  ``.unreadable-<time>`` name before a fresh one is written, because a
  file that today's json refuses may still be the only copy of settings
  that took a while to get right (and the next release may well read it
  again);
* the new file is written beside the old one and renamed over it, so a
  deploy that is interrupted mid-write leaves the previous
  configuration, never a half-written one;
* a ``config.json`` that is a link into a dotfiles repository is written
  where it really lives: replacing the link with a regular file would
  quietly disconnect the file somebody maintains from the web app that
  reads it.

Usage:
    python3 remote_config.py CONFIG_JSON WORK_DIR [PASSWORD]

The password in force afterwards is printed as
``SORCAR_PASSWORD=<password>``: the one given, else the one already
configured, else a fresh random one (the web app only opens a public
tunnel when a password is set).  It is stored as typed under
``remote_password`` (the key ``kiss.core.remote_password`` uses; keep
the two in step), 20 letters and digits when generated here.  It is
stdlib-only and self-contained, so ./sorcar-cloud ships this one file to
the machine it configures rather than relying on the checkout there
being new enough to have it.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import secrets
import shutil
import stat
import string
import sys
import tempfile
import time
from collections.abc import Iterator
from typing import IO, Any

# Self-contained on purpose (see the module docstring): this file travels
# alone to the machine it configures, so it cannot use kiss.core.file_lock.
try:
    import fcntl
except ImportError:  # pragma: no cover — Windows has no fcntl
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # POSIX has no msvcrt
    msvcrt = None  # type: ignore[assignment]


# Longest wait for another writer's ``.config.lock`` (the web app's own bound on
# it, kiss.core.user_config._LOCK_TIMEOUT_S: a holder is done in well under a
# second, so one that outlasts this is wedged) and the time between two tries.
LOCK_TIMEOUT = 30.0
LOCK_POLL = 0.05


def _try_lock(lock_file: IO[str]) -> bool:
    """Probe the lock on *lock_file* once without waiting; True when it is now held.

    False means another process holds it; any other failure of the lock call
    is permanent and is raised (as in ``kiss.core.file_lock``).
    """
    if fcntl is not None:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    elif msvcrt is not None:  # pragma: no cover — Windows-only branch
        lock_file.seek(0)
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EDEADLK):
                raise
            return False
    return True


def _lock_exclusive(lock_file: IO[str]) -> None:
    """Take an exclusive lock on *lock_file*: ``flock`` on POSIX, ``msvcrt`` on Windows.

    Raises:
        TimeoutError: When another process still holds the lock after
            :data:`LOCK_TIMEOUT` seconds.
    """
    deadline = time.monotonic() + LOCK_TIMEOUT
    while not _try_lock(lock_file):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{lock_file.name} still locked by another process after {LOCK_TIMEOUT:g}s"
            )
        time.sleep(LOCK_POLL)


def _replace_waiting_for_readers(source: str, target: str) -> None:
    """``os.replace`` that waits out a Windows reader holding *target* open (up to 1 s)."""
    deadline = time.monotonic() + 1.0
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if msvcrt is None or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _unlock(lock_file: IO[str]) -> None:
    """Release the lock :func:`_lock_exclusive` took on *lock_file*."""
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover — Windows-only branch
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # pyright: ignore[reportAttributeAccessIssue]


@contextlib.contextmanager
def _locked(lock_path: str) -> Iterator[None]:
    """Hold the exclusive lock on the file *lock_path*, created when missing, for the block."""
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        _lock_exclusive(lock_file)
        try:
            yield
        finally:
            _unlock(lock_file)


def _target_and_locks(path: str) -> tuple[str, dict[str, str]]:
    """Return the file a deploy writes for *path* and the lock files of that write.

    The file is the one a link names, else *path*: renaming over a link
    replaces the link, and the file it points at -- the one somebody
    maintains, and the one the web app follows the link to read -- would
    keep the settings this is meant to change.

    The locks, by the real path of each, are the ``.config.lock`` files
    the web app's save_config takes (``kiss.core.user_config``): the one
    beside the path as *named*, link or not, and the one beside the file
    itself, which a writer that reaches the file by another path holds
    too.  Both under one real path are one lock.  A link into a
    directory that does not exist has no second lock yet: nobody holds
    one there, and :func:`_config_locks` takes it as it makes the
    directory.
    """
    named = os.path.join(os.path.dirname(path) or ".", ".config.lock")
    lock_paths = {os.path.realpath(named): named}
    target = path
    if os.path.islink(path):
        target = os.path.realpath(path)
        if os.path.isdir(os.path.dirname(target)):
            beside = os.path.join(os.path.dirname(target), ".config.lock")
            lock_paths.setdefault(os.path.realpath(beside), beside)
    return target, lock_paths


@contextlib.contextmanager
def _config_locks(path: str) -> Iterator[str]:
    """Hold the locks of one load-modify-replace of the configuration *path*.

    Yields the file to read and to replace (:func:`_target_and_locks`).
    The locks are taken in the order of their real paths, as save_config
    takes them, so a deploy and a settings save never wait for each
    other.  That order may put the lock beside the linked file first, so
    the link is read before any lock is held; it is read again once they
    all are, and when it names another file by then (save_config
    replaces a link whose target is gone with a regular file) the locks
    are given up and taken anew for that file, once.

    A link into a directory that was not there had no lock beside the
    file among those waited for.  The directory is made with the lock
    beside the link held, where the link is known to name it still, and
    the lock in it is taken at once, without a wait: the read and the
    replace then hold it like any other, against a writer whose own
    file the linked one is.  When such a writer has it already, the
    locks are given up and taken anew in their order as well.  Whether
    that lock is held decides, not whether the directory is there by
    now: such a writer may have made it a moment ago.

    Raises:
        TimeoutError: When another process still holds a lock after
            :data:`LOCK_TIMEOUT` seconds, or the locks were not those of
            the file *path* names after the second wait either.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for _ in range(2):
        target, lock_paths = _target_and_locks(path)
        with contextlib.ExitStack() as locks:
            for real in sorted(lock_paths):
                locks.enter_context(_locked(lock_paths[real]))
            if _target_and_locks(path) != (target, lock_paths):
                continue
            directory = os.path.dirname(target) or "."
            beside = os.path.join(directory, ".config.lock")
            if os.path.realpath(beside) not in lock_paths:
                os.makedirs(directory, exist_ok=True)
                lock_file = locks.enter_context(open(beside, "w", encoding="utf-8"))
                if not _try_lock(lock_file):
                    continue
                locks.callback(_unlock, lock_file)
            yield target
            return
    raise TimeoutError(f"{path} named another file after each wait for its locks")


PASSWORD_LENGTH = 20
PASSWORD_ALPHABET = string.ascii_letters + string.digits


def new_password() -> str:
    """Return a fresh random password for the web app.

    Returns:
        A 20-character string of letters and digits.
    """
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(PASSWORD_LENGTH))


def _open_without_waiting(path: str, flags: int) -> int:
    """Opener for ``open``: a named pipe nobody writes to opens at once instead of never."""
    return os.open(path, flags | getattr(os, "O_NONBLOCK", 0))


def load_config(path: str) -> dict[str, Any]:
    """Read a configuration file, keeping an unusable one as a backup.

    Args:
        path: Path of the configuration file; it need not exist.

    Returns:
        The settings the file holds, or an empty mapping when there is
        no file or its content cannot be used.

    Raises:
        OSError: If the path names something other than a regular file
            (a named pipe, a device: nothing to read, and nothing to
            keep), or an unusable file could not be kept.  Writing a
            fresh configuration over the only copy of settings that took
            a while to get right is worse than stopping the deploy.
    """
    if not os.path.exists(path):
        return {}
    reason = ""
    loaded: Any = None
    regular = True
    try:
        # What was opened is what is examined: a check of the path before a
        # plain open would leave a pipe put there in between to wait on.
        with open(path, encoding="utf-8", opener=_open_without_waiting) as handle:
            regular = stat.S_ISREG(os.fstat(handle.fileno()).st_mode)
            if regular:
                loaded = json.load(handle)
    except (OSError, ValueError) as exc:
        reason = str(exc)
    if not regular:
        raise OSError(errno.EINVAL, "not a regular file", path)
    if not reason and isinstance(loaded, dict):
        return loaded
    # Valid json that is not an object is as unusable as json that does not
    # parse -- and just as much somebody's file.
    reason = reason or f"top level is {type(loaded).__name__}, not an object"
    backup = f"{path}.unreadable-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    while os.path.exists(backup):
        backup += "+"
    shutil.copy2(path, backup)
    print(f"kept the previous {path} as {backup} ({reason})", file=sys.stderr)
    return {}


def configure(path: str, work_dir: str, password: str = "") -> str:
    """Set the password and work directory, keeping every other setting.

    Args:
        path: Path of ``config.json``; created when missing.
        work_dir: Directory the web app should work in.
        password: Password to install; empty keeps the configured one
            and generates a fresh one when there is none.

    Returns:
        The password in force after the update.

    Raises:
        OSError: When the file cannot be read, kept or written, or (a
            ``TimeoutError``) when its locks could not be had within
            :data:`LOCK_TIMEOUT` seconds (:func:`_config_locks`).
    """
    # The load-modify-replace runs under the same ``.config.lock`` flocks
    # the web app's save_config takes for this file, so a deploy and a
    # settings save (or two concurrent deploys) serialize instead of the
    # later replace silently discarding the earlier writer's changes.
    # The real file is what gets read, backed up and written.
    with _config_locks(path) as path:
        directory = os.path.dirname(path) or "."
        config = load_config(path)
        configured = config.get("remote_password")
        chosen = password or (configured if isinstance(configured, str) else "")
        chosen = chosen or new_password()
        config["remote_password"] = chosen
        config["work_dir"] = work_dir
        # mkstemp gives every writer its own staging file; a fixed
        # name would let two concurrent runs truncate and publish
        # each other's half-written JSON.  It is created readable by
        # its owner only, as the file holding the password must be.
        fd, temporary = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".sorcar-new-", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_waiting_for_readers(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    return chosen


def main(argv: list[str] | None = None) -> int:
    """Update a configuration file from the command line.

    Args:
        argv: Arguments ``[config_json, work_dir, password]``; defaults
            to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 when the file could not be
        updated, 2 on a usage error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not 2 <= len(args) <= 3:
        print(
            "usage: remote_config.py CONFIG_JSON WORK_DIR [PASSWORD]", file=sys.stderr
        )
        return 2
    try:
        password = configure(args[0], args[1], args[2] if len(args) > 2 else "")
    except OSError as exc:
        print(f"remote_config: {args[0]}: {exc}", file=sys.stderr)
        return 1
    print("SORCAR_PASSWORD=" + password)
    return 0


if __name__ == "__main__":
    sys.exit(main())

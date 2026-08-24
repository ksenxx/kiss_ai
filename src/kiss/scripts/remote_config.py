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
tunnel when a password is set).  It is stdlib-only and self-contained, so
./sorcar-cloud ships this one file to the machine it configures rather
than relying on the checkout there being new enough to have it.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import shutil
import string
import sys
import tempfile
import time
from typing import Any

PASSWORD_LENGTH = 12
PASSWORD_ALPHABET = string.ascii_lowercase + string.digits


def new_password() -> str:
    """Return a fresh random password for the web app.

    Returns:
        A lower-case alphanumeric string.
    """
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(PASSWORD_LENGTH))


def load_config(path: str) -> dict[str, Any]:
    """Read a configuration file, keeping an unusable one as a backup.

    Args:
        path: Path of the configuration file; it need not exist.

    Returns:
        The settings the file holds, or an empty mapping when there is
        no file or its content cannot be used.

    Raises:
        OSError: If an unusable file could not be kept.  Writing a fresh
            configuration over the only copy of settings that took a
            while to get right is worse than stopping the deploy.
    """
    if not os.path.exists(path):
        return {}
    reason = ""
    loaded: Any = None
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError) as exc:
        reason = str(exc)
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
    """
    # The lock is keyed to the path as *named* -- the web app's save_config
    # flocks ``.config.lock`` beside the ``config.json`` it was told about,
    # link or not -- so it must be derived BEFORE the link is resolved:
    # locking beside the link's target (say ``~/dotfiles/.config.lock``)
    # while the web app locks ``~/.kiss/.config.lock`` would give two
    # writers two different locks and no mutual exclusion at all.
    lock_directory = os.path.dirname(path) or "."
    # Renaming over a link replaces the link; the file it points at -- the one
    # somebody maintains, and the one the web app follows the link to read --
    # would keep the settings this is meant to change.  So the real file is
    # what gets read, backed up and written.
    if os.path.islink(path):
        path = os.path.realpath(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(lock_directory, exist_ok=True)
    os.makedirs(directory, exist_ok=True)
    # The load-modify-replace runs under the same ``.config.lock`` flock
    # the web app's save_config takes for this file, so a deploy and a
    # settings save (or two concurrent deploys) serialize instead of the
    # later replace silently discarding the earlier writer's changes.
    with open(
        os.path.join(lock_directory, ".config.lock"), "w", encoding="utf-8"
    ) as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            config = load_config(path)
            configured = config.get("remote_password")
            chosen = password or (configured if isinstance(configured, str) else "")
            chosen = chosen or new_password()
            config["remote_password"] = chosen
            config["work_dir"] = work_dir
            # mkstemp gives every writer its own staging file; a fixed
            # name would let two concurrent runs truncate and publish
            # each other's half-written JSON.
            fd, temporary = tempfile.mkstemp(
                prefix=os.path.basename(path) + ".sorcar-new-", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(config, handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, path)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
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

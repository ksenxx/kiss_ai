# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The restart of kiss-web that ``install.sh`` deferred.

``install.sh`` never restarts a daemon that reports a task in flight.
When it defers, it records what the restart needs -- the installer that
ran, the state directory and the work dir the daemon is to serve -- as a
JSON object in the file the busy daemon named in its
``activeTasksResponse`` (:data:`RESTART_PENDING_FILE` in the daemon's
home; ``record_deferred_restart`` in ``install.sh``).  The daemon polls
that file and, once no task is in flight, runs the installer with
``--restart-only`` (``web_server._deferred_restart_loop``): the restart
onto the build the deferred run left on disk.  Every restart by
``install.sh`` removes the file.
"""

import contextlib
import json
import logging
import os
from pathlib import Path

from kiss.core.processes import find_bash

logger = logging.getLogger(__name__)

RESTART_PENDING_FILE = ".kiss-web.restart-pending"
"""Basename, in the daemon's home, of the deferred-restart record."""


def claim(path: Path) -> tuple[list[str], str, dict[str, str]] | None:
    """Take the record at *path* and return the command that finishes the restart.

    The record is moved away before it is read, so one a later install
    writes meanwhile is left for the next poll, and it is removed
    whatever it holds, so a bad record is not retried.

    Args:
        path: The deferred-restart record.

    Returns:
        ``(argv, cwd, env)``: the recorded ``install.sh`` run with
        ``--restart-only`` and ``KISS_HOME`` set to the recorded state
        directory, from the recorded work dir (``$HOME`` when it is
        gone); ``None`` when there is no record or it names no existing
        installer.
    """
    claimed = path.with_name(f"{path.name}.{os.getpid()}")
    try:
        os.replace(path, claimed)
        record = json.loads(claimed.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("Unreadable deferred-restart record %s", path, exc_info=True)
        return None
    finally:
        with contextlib.suppress(OSError):
            claimed.unlink()
    fields = record if isinstance(record, dict) else {}
    installer, home, workdir = (fields.get(k) for k in ("installer", "home", "workdir"))
    if not (
        isinstance(installer, str) and os.path.isfile(installer)
        and isinstance(home, str) and home and isinstance(workdir, str)
    ):
        logger.warning("Ignoring the deferred-restart record %s: %r", path, record)
        return None
    argv = [find_bash() or "bash", installer, "--restart-only", "--non-interactive"]
    cwd = workdir if os.path.isdir(workdir) else str(Path.home())
    return argv, cwd, dict(os.environ, KISS_HOME=home)

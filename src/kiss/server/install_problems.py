# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The web app's notice about a failed or partly failed install.

``install.sh`` records every failure of a run, one line each (its
``warn`` and ``fail`` helpers and its EXIT trap: a post-install hook
that exited non-zero, a brand home whose nested install failed, a
daemon that did not come up, a ``uv sync`` error, ...), in
``$KISS_HOME/.install-problems`` of the home whose daemon runs after
the install, and removes the file at the start of every run.  Without
this notice such a failure went unseen: the installer exits 0 after a
warning, and when it restarts the daemon the daemon's own watcher of
the update (``web_server._watch_update_exit``) dies with it.

The daemon sends the notice (a sticky ``notification`` of severity
``error``) with the ``ready`` bootstrap of every page while the file is
less than :data:`NOTICE_DAYS` old, and the watcher sends it when the
installer exits without restarting the daemon.  A later run with no
failure leaves no file, which ends the notice.
"""

import time
from pathlib import Path
from typing import Any

from kiss.core.brand import PRODUCT_NAME
from kiss.core.config import kiss_home

PROBLEMS_FILE = ".install-problems"
"""Basename, under ``$KISS_HOME``, of the file ``install.sh`` writes."""

NOTICE_DAYS = 7
"""How long after the install a marker ``install.sh`` wrote keeps its
notice appearing (this one and :mod:`kiss.server.vscode_retired`'s)."""

NOTICE_ID = "install-problems"
"""The notification id: one toast per page, replaced on reconnect."""


def marker_is_fresh(path: Path, since: float = 0.0) -> bool:
    """Whether the install marker *path* still warrants its notice.

    ``True`` when the file exists, was written at or after *since* and
    is at most :data:`NOTICE_DAYS` old; ``False`` otherwise, including
    when it cannot be ``stat``-ed.

    Args:
        path: The marker file ``install.sh`` writes under ``$KISS_HOME``.
        since: A Unix time; a file written before it (an earlier run's)
            does not count.
    """
    try:
        written = path.stat().st_mtime
    except OSError:
        return False
    return written >= since and time.time() - written <= NOTICE_DAYS * 86400


def install_problems(since: float = 0.0) -> list[str]:
    """The failures the last install recorded, oldest first.

    Empty when no install failed, when the last failed install is older
    than :data:`NOTICE_DAYS` or was recorded before *since*, or when the
    file cannot be read.

    Args:
        since: As for :func:`marker_is_fresh`.
    """
    path = kiss_home() / PROBLEMS_FILE
    if not marker_is_fresh(path, since):
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip()]


def install_problems_notice(since: float = 0.0) -> dict[str, Any] | None:
    """The sticky ``notification`` event for the recorded failures, else ``None``.

    Args:
        since: As for :func:`install_problems`.
    """
    problems = install_problems(since)
    if not problems:
        return None
    count = f"{len(problems)} problem" + ("s" if len(problems) > 1 else "")
    return {
        "type": "notification",
        "id": NOTICE_ID,
        "severity": "error",
        "sticky": True,
        "title": f"{PRODUCT_NAME} update: {count}",
        "message": (
            f"The last {PRODUCT_NAME} install ran into {count}:\n"
            + "\n".join(f"\u2022 {problem}" for problem in problems)
            + "\nThe full output is in ~/.kiss/install.log."
        ),
    }

#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Probe kiss-web's UDS for in-flight agent tasks.

Called by ``scripts/build-extension.sh`` and ``install.sh`` BEFORE they
SIGTERM the running ``kiss-web`` daemon.  Without this guard, those
scripts silently kill any in-flight agent task — the bug that turned a
multi-step task into ``"Task interrupted by server restart/shutdown"``
on task_history rows 3233/3234 (see also the same regression on
row 3192 that prompted the matching guard in
``src/kiss/agents/vscode/src/DependencyInstaller.ts``).

Wire protocol
=============
Speaks the SAME newline-delimited JSON protocol as
:meth:`RemoteAccessServer._uds_handler`:

* Request:  ``{"type":"activeTasksQuery"}\\n``
* Response: ``{"type":"activeTasksResponse","count":<int>,"tabs":[...]}\\n``

Exit codes
==========
* ``0`` — safe to kill: the socket file is absent, the connect was
  actively refused (stale socket but no listener), OR the daemon
  reported ``count == 0``.
* ``1`` — NOT safe to kill: ``count > 0`` (in-flight tasks present), OR
  the probe could not be completed (timeout, malformed response,
  unexpected message type).  Matches the conservative "alive +
  active-tasks uncertain → skip" policy in ``daemonHealth.js``.

Environment
===========
* ``KISS_SORCAR_SOCK`` — override the UDS path (default
  ``~/.kiss/sorcar.sock``).  Used by the integration test in
  ``test_check_active_tasks_script.py`` to point at a per-test socket.
* ``KISS_ACTIVE_TASKS_TIMEOUT`` — connect+read timeout in seconds
  (default ``2.0``).
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

DEFAULT_SOCK = Path.home() / ".kiss" / "sorcar.sock"
DEFAULT_TIMEOUT = 2.0


def _probe_active_tasks(
    sock_path: Path, timeout: float,
) -> tuple[int, str]:
    """Probe the kiss-web UDS for active tasks.

    Returns a ``(exit_code, message)`` tuple.  ``exit_code`` follows
    the module docstring's contract (0 safe, 1 unsafe).  ``message``
    is a single human-readable line written to stderr by ``main``.
    """
    if not sock_path.exists():
        return 0, f"kiss-web socket not present at {sock_path}; nothing to defer to."
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(sock_path))
            s.sendall(b'{"type":"activeTasksQuery"}\n')
            buf = b""
            deadline_remaining_chunks = 64
            while b"\n" not in buf and deadline_remaining_chunks > 0:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
                deadline_remaining_chunks -= 1
    except ConnectionRefusedError:
        return 0, (
            f"kiss-web UDS at {sock_path} refused connection "
            "(stale socket, daemon dead); safe to kill."
        )
    except (TimeoutError, OSError) as exc:
        return 1, (
            f"kiss-web UDS probe at {sock_path} failed "
            f"({exc.__class__.__name__}: {exc}); refusing to kill — set "
            "KISS_FORCE_RESTART=1 to override."
        )

    line, _, _ = buf.partition(b"\n")
    try:
        msg = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 1, (
            f"kiss-web UDS probe at {sock_path} returned malformed JSON "
            f"({line!r}); refusing to kill."
        )
    if not isinstance(msg, dict) or msg.get("type") != "activeTasksResponse":
        return 1, (
            f"kiss-web UDS probe at {sock_path} returned unexpected "
            f"message {msg!r}; refusing to kill."
        )
    count = msg.get("count")
    if not isinstance(count, int) or count < 0:
        return 1, (
            f"kiss-web UDS probe at {sock_path} returned non-integer "
            f"count {count!r}; refusing to kill."
        )
    if count > 0:
        tabs_raw = msg.get("tabs") or []
        tabs = [t for t in tabs_raw if isinstance(t, str)]
        return 1, (
            f"kiss-web has {count} in-flight task(s): "
            f"{', '.join(tabs) if tabs else '<unnamed>'}.  Refusing to kill — "
            "set KISS_FORCE_RESTART=1 to override."
        )
    return 0, f"kiss-web is idle (count=0) at {sock_path}; safe to kill."


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write a status line to stderr."""
    del argv  # No CLI args; configured via environment variables.
    sock_env = os.environ.get("KISS_SORCAR_SOCK")
    sock_path = Path(sock_env) if sock_env else DEFAULT_SOCK
    try:
        timeout = float(os.environ.get(
            "KISS_ACTIVE_TASKS_TIMEOUT", str(DEFAULT_TIMEOUT),
        ))
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    exit_code, message = _probe_active_tasks(sock_path, timeout)
    print(message, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

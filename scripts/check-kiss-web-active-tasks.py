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
import time
from pathlib import Path

DEFAULT_SOCK = Path.home() / ".kiss" / "sorcar.sock"
DEFAULT_TIMEOUT = 2.0


def _classify_message(msg: object) -> str:
    """Classify a parsed JSON line from the UDS into one of:

    * ``"response"`` — the awaited ``activeTasksResponse``.
    * ``"old-daemon"`` — an ``{"type":"error","text":"Unknown command:
      activeTasksQuery"}`` broadcast from a pre-``activeTasksQuery``
      daemon.  Treated as "safe to kill" because such a daemon predates
      the in-flight-task accounting we are gating on; killing it cannot
      abort a task that the new accounting would have flagged.
    * ``"skip"`` — any other broadcast line (event, log, etc.) the
      daemon happened to push onto every UDS writer.  The caller should
      keep reading.
    """
    if not isinstance(msg, dict):
        return "skip"
    msg_type = msg.get("type")
    if msg_type == "activeTasksResponse":
        return "response"
    if msg_type == "error":
        text = msg.get("text")
        if isinstance(text, str) and "Unknown command: activeTasksQuery" in text:
            return "old-daemon"
    return "skip"


def _probe_active_tasks(
    sock_path: Path, timeout: float,
) -> tuple[int, str]:
    """Probe the kiss-web UDS for active tasks.

    Returns a ``(exit_code, message)`` tuple.  ``exit_code`` follows
    the module docstring's contract (0 safe, 1 unsafe).  ``message``
    is a single human-readable line written to stderr by ``main``.

    The reader is a line loop rather than a single-shot ``recv`` because
    :meth:`RemoteAccessServer._uds_handler` registers every connected
    client as a broadcast destination via
    :meth:`MessagePrinter.add_uds_writer`.  Unrelated events
    (e.g. ``{"type":"error","text":"Unknown command: ..."}`` emitted
    when an OLD daemon doesn't recognise ``activeTasksQuery``, or any
    other broadcast that races with the response) can therefore land
    on the wire BEFORE our ``activeTasksResponse``.  We consume lines
    until we see either the response we asked for, the specific
    ``Unknown command: activeTasksQuery`` error (which means the
    daemon predates ``activeTasksQuery`` and is therefore not running
    in-flight-task accounting we need to defer to), or the deadline
    elapses.
    """
    if not sock_path.exists():
        return 0, f"kiss-web socket not present at {sock_path}; nothing to defer to."
    deadline = time.monotonic() + timeout
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(sock_path))
            s.sendall(b'{"type":"activeTasksQuery"}\n')
            buf = b""
            msg: object = None
            while True:
                # Consume any complete lines already buffered.
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    try:
                        candidate = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # Malformed/empty lines in the broadcast stream
                        # are skipped rather than failing the probe;
                        # the daemon may legitimately emit non-JSON
                        # debug noise in future protocol versions.
                        continue
                    kind = _classify_message(candidate)
                    if kind == "response":
                        msg = candidate
                        break
                    if kind == "old-daemon":
                        return 0, (
                            f"kiss-web daemon at {sock_path} predates the "
                            "activeTasksQuery handler (responded 'Unknown "
                            "command: activeTasksQuery'); cannot have "
                            "in-flight-task accounting to defer to — safe "
                            "to kill so install.sh can replace it."
                        )
                    # kind == "skip": keep draining.
                if msg is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 1, (
                        f"kiss-web UDS probe at {sock_path} timed out "
                        "waiting for activeTasksResponse; refusing to "
                        "kill — set KISS_FORCE_RESTART=1 to override."
                    )
                s.settimeout(remaining)
                chunk = s.recv(4096)
                if not chunk:
                    return 1, (
                        f"kiss-web UDS probe at {sock_path} closed "
                        "before sending activeTasksResponse; refusing "
                        "to kill — set KISS_FORCE_RESTART=1 to override."
                    )
                buf += chunk
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

    assert isinstance(msg, dict)  # _classify_message returns "response" only for dicts.
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

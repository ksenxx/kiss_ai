#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Probe kiss-web's local WSS endpoint for in-flight agent tasks.

Called by ``scripts/build-extension.sh`` and ``install.sh`` BEFORE they
SIGTERM the running ``kiss-web`` daemon.  Without this guard, those
scripts silently kill any in-flight agent task — the bug that turned a
multi-step task into ``"Task interrupted by server restart/shutdown"``
on task_history rows 3233/3234 (see also the same regression on
row 3192 that prompted the matching guard in
``src/kiss/agents/vscode/src/DependencyInstaller.ts``).

Wire protocol
=============

The daemon publishes its URL and a local token in
``~/.kiss/sorcar-local.json`` (see ``kiss.agents.sorcar.local_endpoint``).
The probe connects to that URL, authenticates as a local client and
speaks the SAME JSON protocol as the VS Code extension:

* Request:  ``{"type":"activeTasksQuery"}``
* Response: ``{"type":"activeTasksResponse","count":<int>,"tabs":[...]}``

Exit codes
==========

* ``0`` — safe to kill: the endpoint file is absent, the connect was
  actively refused (stale file but no listener), OR the daemon
  reported ``count == 0``.
* ``1`` — NOT safe to kill: ``count > 0`` (in-flight tasks present), OR
  the probe could not be completed (timeout, malformed response,
  unexpected message type).  Matches the conservative "alive +
  active-tasks uncertain → skip" policy in ``daemonHealth.js``.

Environment
===========

* ``KISS_SORCAR_LOCAL`` — override the endpoint file path (default
  ``~/.kiss/sorcar-local.json``).  Used by the integration test in
  ``test_check_active_tasks_script.py`` to point at a per-test daemon.
* ``KISS_ACTIVE_TASKS_TIMEOUT`` — connect+read timeout in seconds
  (default ``2.0``).
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
from pathlib import Path

DEFAULT_ENDPOINT_FILE = Path.home() / ".kiss" / "sorcar-local.json"
DEFAULT_TIMEOUT = 2.0


def _classify_message(msg: object) -> str:
    """Classify a parsed JSON frame from the daemon into one of:

    * ``"response"`` — the awaited ``activeTasksResponse``.
    * ``"old-daemon"`` — an ``{"type":"error","text":"Unknown command:
      activeTasksQuery"}`` broadcast from a pre-``activeTasksQuery``
      daemon.  Treated as "safe to kill" because such a daemon predates
      the in-flight-task accounting we are gating on; killing it cannot
      abort a task that the new accounting would have flagged.
    * ``"skip"`` — any other broadcast frame (event, log, etc.) the
      daemon happened to push to every connected client.  The caller
      should keep reading.
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


def _read_endpoint(endpoint_file: Path) -> tuple[str, str, str | None] | None:
    """Return ``(url, token, ca)`` from the endpoint file, or ``None``."""
    try:
        data = json.loads(endpoint_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    url, token, ca = data.get("url"), data.get("token"), data.get("ca")
    if not isinstance(url, str) or not isinstance(token, str) or not url or not token:
        return None
    return url, token, ca if isinstance(ca, str) else None


def _probe_active_tasks(
    endpoint_file: Path, timeout: float,
) -> tuple[int, str]:
    """Probe the kiss-web daemon for active tasks.

    Returns a ``(exit_code, message)`` tuple.  ``exit_code`` follows
    the module docstring's contract (0 safe, 1 unsafe).  ``message``
    is a single human-readable line written to stderr by ``main``.

    The reader is a loop rather than a single ``recv`` because the
    daemon registers every connected client as a broadcast
    destination: unrelated events can land on the wire BEFORE our
    ``activeTasksResponse``.  Frames are consumed until the response
    we asked for arrives, the specific ``Unknown command:
    activeTasksQuery`` error shows the daemon predates the accounting
    we need to defer to, or the deadline elapses.
    """
    endpoint = _read_endpoint(endpoint_file)
    if endpoint is None:
        return 0, (
            f"kiss-web endpoint file not present at {endpoint_file}; "
            "nothing to defer to."
        )
    url, token, ca = endpoint
    try:
        from websockets.exceptions import ConnectionClosed, InvalidHandshake
        from websockets.sync.client import connect
    except ImportError as exc:
        return 1, (
            f"kiss-web probe cannot import websockets ({exc}); refusing to "
            "kill — set KISS_FORCE_RESTART=1 to override."
        )
    ssl_ctx = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
    deadline = time.monotonic() + timeout
    try:
        with connect(
            url, ssl=ssl_ctx, open_timeout=timeout, close_timeout=1.0,
            compression=None,
        ) as ws:
            ws.send(json.dumps({"type": "auth", "token": token}))
            ws.send(json.dumps({"type": "activeTasksQuery"}))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 1, (
                        f"kiss-web probe at {url} timed out waiting for "
                        "activeTasksResponse; refusing to kill — set "
                        "KISS_FORCE_RESTART=1 to override."
                    )
                try:
                    raw = ws.recv(timeout=remaining)
                except TimeoutError:
                    continue
                try:
                    candidate = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                kind = _classify_message(candidate)
                if kind == "response":
                    msg = candidate
                    break
                if kind == "old-daemon":
                    return 0, (
                        f"kiss-web daemon at {url} predates the "
                        "activeTasksQuery handler (responded 'Unknown "
                        "command: activeTasksQuery'); cannot have "
                        "in-flight-task accounting to defer to — safe "
                        "to kill so install.sh can replace it."
                    )
    except ConnectionRefusedError:
        return 0, (
            f"kiss-web at {url} refused connection (stale endpoint file, "
            "daemon dead); safe to kill."
        )
    except ConnectionClosed:
        return 1, (
            f"kiss-web probe at {url} closed before sending "
            "activeTasksResponse; refusing to kill — set "
            "KISS_FORCE_RESTART=1 to override."
        )
    except (TimeoutError, OSError, InvalidHandshake) as exc:
        return 1, (
            f"kiss-web probe at {url} failed "
            f"({exc.__class__.__name__}: {exc}); refusing to kill — set "
            "KISS_FORCE_RESTART=1 to override."
        )

    assert isinstance(msg, dict)  # _classify_message returns "response" only for dicts.
    count = msg.get("count")
    if not isinstance(count, int) or count < 0:
        return 1, (
            f"kiss-web probe at {url} returned non-integer "
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
    return 0, f"kiss-web is idle (count=0) at {url}; safe to kill."


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write a status line to stderr."""
    del argv  # No CLI args; configured via environment variables.
    env = os.environ.get("KISS_SORCAR_LOCAL")
    endpoint_file = Path(env) if env else DEFAULT_ENDPOINT_FILE
    try:
        timeout = float(os.environ.get(
            "KISS_ACTIVE_TASKS_TIMEOUT", str(DEFAULT_TIMEOUT),
        ))
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    exit_code, message = _probe_active_tasks(endpoint_file, timeout)
    print(message, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

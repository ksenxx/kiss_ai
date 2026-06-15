# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Best-effort one-way bridge from the ``sorcar`` CLI to the daemon.

The CLI agent loop runs in its own process and emits display events via
:class:`~kiss.agents.sorcar.cli_printer.RecordingConsolePrinter`.  That
printer records and persists every event to the chat DB but has no
transport: a webview already open at the task's chat id therefore sees
nothing until the next page reload replays from the DB.

This module gives the CLI printer a transport: a cached AF_UNIX
connection to the local daemon's UDS endpoint (default
``~/.kiss/sorcar.sock``; override via the ``KISS_SORCAR_SOCK`` env var
for tests).  Each broadcast sends a newline-delimited
``{"type": "cliEvent", "event": <event>}`` envelope which the daemon
dispatcher relays to every subscriber tab via
``RemoteAccessServer._relay_cli_event``.  Connection failures are
absorbed silently (CLI must still work when no daemon is running) and
the cached socket is dropped on the first send error so a later event
will transparently reconnect.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

_DEFAULT_SOCK_PATH = Path.home() / ".kiss" / "sorcar.sock"

_LOCK = threading.Lock()
_WRITER: socket.socket | None = None


def _sock_path() -> Path:
    """Return the UDS path the bridge should connect to.

    Reads the ``KISS_SORCAR_SOCK`` environment variable on every call
    (so tests can repoint the bridge by setting an env var and calling
    :func:`reset_for_tests`).
    """
    env = os.environ.get("KISS_SORCAR_SOCK")
    return Path(env) if env else _DEFAULT_SOCK_PATH


def _connect() -> socket.socket | None:
    """Open a best-effort AF_UNIX connection to the daemon.

    Returns ``None`` when the socket file does not exist, the daemon
    refuses the connection, or any other I/O error occurs — the CLI
    must keep running normally even when no daemon is listening.
    """
    path = _sock_path()
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(str(path))
    except OSError:
        return None
    return s


def send_event(event: dict[str, Any]) -> None:
    """Forward *event* to the daemon for live fan-out to webviews.

    Wraps *event* in a ``cliEvent`` envelope and writes one newline-
    delimited JSON line on the cached UDS connection.  Lazily opens
    the connection on first use.  On any write error, closes and
    drops the cached socket so the next call will try to reconnect
    (one-shot retry semantics: a transient daemon restart resumes
    streaming on the next event).
    """
    global _WRITER
    payload = json.dumps({"type": "cliEvent", "event": event}).encode() + b"\n"
    with _LOCK:
        if _WRITER is None:
            _WRITER = _connect()
            if _WRITER is None:
                return
        try:
            _WRITER.sendall(payload)
        except OSError:
            try:
                _WRITER.close()
            except OSError:
                pass
            _WRITER = None


def reset_for_tests() -> None:
    """Drop the cached UDS connection.

    Called by tests between scenarios so a new daemon (bound to a
    fresh temp UDS path) is contacted instead of a stale cached
    socket from a previous test.
    """
    global _WRITER
    with _LOCK:
        if _WRITER is not None:
            try:
                _WRITER.close()
            except OSError:
                pass
            _WRITER = None

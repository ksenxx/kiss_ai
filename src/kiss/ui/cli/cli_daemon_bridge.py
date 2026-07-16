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
``$KISS_HOME/sorcar.sock``, falling back to ``~/.kiss/sorcar.sock``;
override via the ``KISS_SORCAR_SOCK`` env var for tests).  Each broadcast sends a newline-delimited
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

from kiss.agents.sorcar.persistence import _default_kiss_dir

_LOCK = threading.Lock()
_WRITER: socket.socket | None = None
# Connect / send timeout (seconds) applied to the cached UDS socket.
# The bridge is best-effort by contract: a daemon that accepted the
# connection but stopped reading (wedged event loop, paused process)
# must NOT freeze the CLI forever once the kernel socket buffer
# fills.  ``socket.timeout`` is an ``OSError`` subclass, so a timed-out
# send flows through the existing drop-and-reconnect error path.
_SEND_TIMEOUT = 2.0
# UDS path the cached ``_WRITER`` is connected to.  ``_sock_path()``
# re-reads ``KISS_SORCAR_SOCK`` on every call (its documented
# contract), so ``_send_envelope`` must compare the cached
# connection's path against the current one and reconnect when they
# differ — otherwise events emitted after the env var changed would
# keep flowing to the OLD daemon socket.
_WRITER_PATH: Path | None = None


def _sock_path() -> Path:
    """Return the UDS path the bridge should connect to.

    Reads the ``KISS_SORCAR_SOCK`` environment variable on every call.
    When unset, derives the default from
    :func:`persistence._default_kiss_dir` so the bridge targets the
    same ``$KISS_HOME/sorcar.sock`` the daemon binds
    (``web_server._UDS_PATH``) instead of a hard-coded
    ``~/.kiss/sorcar.sock`` that ignores ``KISS_HOME`` (w3 C-5).
    """
    env = os.environ.get("KISS_SORCAR_SOCK")
    return Path(env) if env else _default_kiss_dir() / "sorcar.sock"


def _connect() -> socket.socket | None:
    """Open a best-effort AF_UNIX connection to the daemon.

    Returns ``None`` when the socket file does not exist, the daemon
    refuses the connection, or any other I/O error occurs — the CLI
    must keep running normally even when no daemon is listening.
    """
    path = _sock_path()
    s: socket.socket | None = None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(_SEND_TIMEOUT)
        s.connect(str(path))
    except OSError:
        # Close the already-created socket object so a daemon-less CLI
        # run does not leak one fd per broadcast until GC (w3 C-1).
        if s is not None:
            s.close()
        return None
    return s


def _send_envelope(envelope: dict[str, Any]) -> None:
    """Write one newline-delimited JSON envelope on the cached UDS socket.

    Shared transport for :func:`send_event`, :func:`send_cli_task_start`,
    and :func:`send_cli_task_end`: lazily opens the connection on first
    use, drops it on any write error (one-shot reconnect on the next
    call), and swallows all I/O failures so the CLI keeps working when
    no daemon is listening.
    """
    global _WRITER, _WRITER_PATH
    payload = json.dumps(envelope).encode() + b"\n"
    with _LOCK:
        path = _sock_path()
        if _WRITER is not None and _WRITER_PATH != path:
            # ``KISS_SORCAR_SOCK`` changed since the connection was
            # cached: drop the stale writer so this event reaches the
            # daemon now listening at the new path.
            try:
                _WRITER.close()
            except OSError:
                pass
            _WRITER = None
        if _WRITER is None:
            _WRITER = _connect()
            if _WRITER is None:
                _WRITER_PATH = None
                return
            _WRITER_PATH = path
        try:
            _WRITER.sendall(payload)
        except OSError:
            try:
                _WRITER.close()
            except OSError:
                pass
            _WRITER = None
            _WRITER_PATH = None


def send_event(event: dict[str, Any]) -> None:
    """Forward *event* to the daemon for live fan-out to webviews.

    Wraps *event* in a ``cliEvent`` envelope so the daemon's
    :meth:`RemoteAccessServer._relay_cli_event` can fan it out over
    WSS / UDS to every subscribed chat webview.
    """
    _send_envelope({"type": "cliEvent", "event": event})


def send_cli_task_start(task_id: str) -> None:
    """Announce that the CLI process has begun running *task_id*.

    The daemon records the task id in
    :attr:`RemoteAccessServer._cli_running_tasks` so a webview tab
    later resuming the task from the history panel can be subscribed
    to the live event stream and shown the blinking-green-circle
    "running" indicator in its tab title.
    """
    _send_envelope({"type": "cliTaskStart", "taskId": str(task_id)})


def send_cli_task_end(task_id: str) -> None:
    """Announce that the CLI process has finished running *task_id*.

    The daemon drops the task id from
    :attr:`RemoteAccessServer._cli_running_tasks` and broadcasts a
    ``status:running=false`` event to every subscribed webview so
    the blinking-green-circle indicator stops.
    """
    _send_envelope({"type": "cliTaskEnd", "taskId": str(task_id)})

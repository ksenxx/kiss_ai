# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An injected Stop never leaks the lock of an ``AbortableNetworkBackend``.

The backend's ``track`` runs on the thread that opens a connection,
which for every streamed OpenAI-compatible request is the task thread,
and ``abort`` runs on the task thread when a blocking request is
stopped.  The runner's Stop injects ``KeyboardInterrupt`` into that
thread, and CPython raises it at the next check point, including the
one right after a ``with lock:``'s ``__enter__`` returned (gh-74174).
The backend took a plain ``threading.Lock`` with ``with``, so such a
Stop left the lock held for good: the next connection of the reused
client, and a later Stop's ``abort``, then waited on it forever.

:func:`kiss.tests.lock_injection.hammer` loops ``track`` and ``abort``
on a worker thread while hundreds of interrupts are injected and the
lock is kept contended (so the worker's acquire blocks and receives the
pending interrupt right there), in a child process so that a leaked
lock wedges only the child.
"""

from __future__ import annotations

import socket
import subprocess
import sys

from httpcore._backends.sync import SyncBackend, SyncStream

from kiss.core.models.stream_abort import AbortableNetworkBackend
from kiss.tests.lock_injection import hammer


class _TrackThenAbort:
    """Keep a stream, then abort the backend: both of the backend's locked sections."""

    def __init__(self) -> None:
        self.backend = AbortableNetworkBackend(SyncBackend())
        # One end of a connected pair: its shutdown always succeeds, even repeated.
        self.sockets = socket.socketpair()
        self.stream = SyncStream(self.sockets[0])

    def __call__(self) -> None:
        self.backend.track(self.stream)
        self.backend.abort()


def run_hammer() -> None:
    """The child-process body: hammer the backend and check its lock is free after."""
    call = _TrackThenAbort()
    lock = call.backend._lock
    hammer(call, {"AbortableNetworkBackend._lock": lock}, contend=lock)


def test_an_injected_stop_leaks_no_backend_lock() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer()"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``talk`` fan-out of the daemon's printer survives the Stop injection.

A ``talk`` tool call broadcasts its event from the task thread.  The
printer then asks whether a local client (a ``sorcar`` CLI or
``daemon_client`` run) addressed one of the event's tabs, which reads
the local-tab registrations under ``WebPrinter._ws_lock``.  The
runner's Stop delivers ``KeyboardInterrupt`` to that thread with
``PyThreadState_SetAsyncExc``, which CPython can raise right after a
``with lock:``'s ``__enter__`` returned (gh-74174): a plain ``with``
there leaks ``_ws_lock``, and the daemon has ONE printer, so every
later send and client registration wedges.

The hammer (:func:`kiss.tests.lock_injection.hammer`) runs the real
fan-out from a worker thread under hundreds of injections in a child
process, with a contender keeping ``_ws_lock`` busy so the worker's
``acquire`` blocks and an injection lands on its return, and asserts
that every lock the path took is free afterwards.  Until the audit of
2026-10-15 ``_local_client_addressed`` took ``_ws_lock`` with a plain
``with``, the only such site reachable from a task thread; it now
goes through ``locked_call`` like every other printer lock, and this
hammer keeps the path covered (the 2026-10-13 hammers never send a
``talk`` event).
"""

from __future__ import annotations

import subprocess
import sys

from kiss.server.web_server import WebPrinter
from kiss.tests.lock_injection import hammer


class _TalkBroadcaster:
    """``broadcast`` of a ``talk`` event whose tab a local client addressed.

    The event carries no clip (``audioB64``), so the fan-out never
    starts an audio player: the path under test is the local-client
    check and the stamped send.
    """

    def __init__(self, printer: WebPrinter) -> None:
        self.printer = printer
        printer.subscribe_tab("audit1015-talk", "tab-1")
        printer.register_local_tab("conn-1", "tab-1")

    def __call__(self) -> None:
        self.printer._thread_local.task_id = "audit1015-talk"
        self.printer.broadcast({"type": "talk", "text": "hello"})


def run_hammer() -> None:
    """The child-process body: hammer the talk fan-out of a real ``WebPrinter``."""
    printer = WebPrinter()
    hammer(_TalkBroadcaster(printer), {
        "_lock": printer._lock,
        "delivery_lock": printer.delivery_lock,
        "_ws_lock": printer._ws_lock,
    }, contend=printer._ws_lock)


def test_talk_fanout_leaks_no_lock() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer()"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

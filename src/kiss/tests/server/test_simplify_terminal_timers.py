# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a terminal tab's grace timer dies with the re-attach, not 60 s later.

:meth:`kiss.server.terminal_tab.TerminalService.viewer_gone` arms one
``threading.Timer`` (``GRACE_SECONDS``, 60 s) per shell the dropped
connection owned.  Before the fix the timer was never stored, so a page
that re-attached at once (``open`` with the same tab id) left the timer
thread sleeping until its 60 s ran out; a flaky connection reconnecting
every second piled up ~60 live threads per terminal tab, and ``shutdown``
left them all behind.  Now the session keeps its timer and cancels it
on re-attach and on hang-up, and a cancelled ``Timer`` exits at once.

Both tests fail on the old code: twenty disconnect/re-attach cycles
leave twenty extra threads alive, and a shut-down service with one
detached shell leaves one.

Two sibling fixes in ``browser_tab.py`` have no practical e2e test and
are recorded here instead:

* ``BrowserTabService.shutdown`` no longer calls ``loop.close()`` when
  the loop thread survived ``join(timeout=10)`` (``loop.close()`` raises
  ``RuntimeError: Cannot close a running event loop`` and aborted the
  rest of the daemon's ``stop_async``).  Reaching it needs a callback
  wedged for 10 s on the service's private loop, which only a test
  double could inject.
* ``open_for_agent`` / ``open_for_user`` wrap their coroutine in
  ``asyncio.wait_for(_OPEN_TIMEOUT)``, so a launch that outlives the 90 s
  the caller waits is cancelled instead of announcing a tab nobody asked
  for any more.  Demonstrating it needs a browser launch slower than
  90 s.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.server.terminal_tab import TerminalService
from kiss.tests.server.test_terminal_tab_service import ConnPrinter, hermetic_shell

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="terminal tabs need a pty",
)

_CYCLES = 20


@pytest.fixture(autouse=True)
def _bash_without_dotfiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", str(hermetic_shell(tmp_path)))


def _timers_of(svc: TerminalService) -> list[threading.Timer]:
    """The live ``threading.Timer`` threads that will call back into *svc*."""
    return [
        t
        for t in threading.enumerate()
        if isinstance(t, threading.Timer) and getattr(t.function, "__self__", None) is svc
    ]


def _settled_timers(svc: TerminalService, timeout: float = 5.0) -> list[threading.Timer]:
    """Return *svc*'s live timers once there are none, or those left after *timeout*."""
    deadline = time.monotonic() + timeout
    while _timers_of(svc) and time.monotonic() < deadline:
        time.sleep(0.02)
    return _timers_of(svc)


def test_reattach_cancels_the_grace_timer_instead_of_leaking_its_thread(
    tmp_path: Path,
) -> None:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    try:
        svc.open("tab-flaky", "conn-0", str(tmp_path), 80, 24)
        svc.input("tab-flaky", "conn-0", "echo ready-$((1+1))\n")
        printer.wait_for(lambda: "ready-2" in printer.output("tab-flaky"))

        for i in range(_CYCLES):
            svc.viewer_gone(f"conn-{i}")
            svc.open("tab-flaky", f"conn-{i + 1}", str(tmp_path), 80, 24)
        # One shell throughout: the re-attaches were all accepted.
        assert svc.session_count() == 1
        attached = [e for e in printer.of_type("terminalOpened") if e["attached"]]
        assert len(attached) == _CYCLES

        # A cancelled timer thread exits at once; a leaked one sleeps for
        # its 60 s (twenty of them before the fix).
        assert _settled_timers(svc) == []

        # The shell is still live and owned by the final connection.
        svc.input("tab-flaky", f"conn-{_CYCLES}", "echo still-$((20+2))\n")
        printer.wait_for(lambda: "still-22" in printer.output("tab-flaky"))
    finally:
        svc.shutdown()


def test_shutdown_cancels_the_grace_timer_of_a_detached_shell(tmp_path: Path) -> None:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    svc.open("tab-gone", "conn-1", str(tmp_path), 80, 24)
    printer.wait_for(lambda: printer.of_type("terminalOpened", "conn-1"))
    svc.viewer_gone("conn-1")
    svc.shutdown()
    assert svc.session_count() == 0
    printer.wait_for(lambda: printer.of_type("terminalExit"))
    # The 2 s hang-up kill timer runs out; the 60 s grace timer was
    # cancelled by the hang-up (before the fix it slept on for a minute).
    assert _settled_timers(svc) == []

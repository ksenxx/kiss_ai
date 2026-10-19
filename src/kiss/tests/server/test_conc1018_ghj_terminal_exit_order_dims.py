# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: ``terminalExit`` ordering against a re-open, and NaN/Infinity sizes.

Two ``TerminalService`` fixes, each with a test that fails on the old code:

* ``_pump`` used to pop the finished session under the service lock and
  emit ``terminalExit`` after releasing it.  A ``terminalOpen`` for the
  same tab that ran in between (a page reload while the shell exits)
  spawned and announced a NEW shell, and the client then got the old
  shell's ``terminalExit`` after the new shell's ``terminalOpened``; both
  are keyed by tab id only, so the page showed "The shell exited" for a
  shell that was alive.  The exit is now announced under the lock, in the
  same step as the pop.  The window (the reader thread between its pop
  and its emit) is a few bytecodes long and never showed up in 200
  stress rounds under the GIL, so the test pins the reader there
  instead: it holds the session's input-queue mutex, which the reader
  takes right after the pop (``writes.put(None)``), opens the tab again
  and only then lets the reader go.  No code is replaced or patched;
  only the scheduling is forced.

* ``_clamp_dim`` did ``int(value)`` on a float, and ``json.loads``
  accepts ``NaN`` and ``Infinity``: ``int(nan)`` raises ``ValueError``,
  ``int(inf)`` ``OverflowError``, which aborted the ``terminalOpen`` or
  ``terminalResize`` command.  Both now fall back to the default size,
  checked through ``stty size`` in the real shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kiss.server.terminal_tab import TerminalService
from kiss.tests.server.test_terminal_tab_service import ConnPrinter, hermetic_shell

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="terminal tabs need a pty")


@pytest.fixture(autouse=True)
def _bash_without_dotfiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", str(hermetic_shell(tmp_path)))


def _session_gone(svc: TerminalService, tab_id: str) -> bool:
    with svc._lock:
        return tab_id not in svc._sessions


def test_exit_of_the_old_shell_is_announced_before_the_new_shell_opens(tmp_path: Path) -> None:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    try:
        svc.open("t", "c1", str(tmp_path), 80, 24)
        printer.wait_for(lambda: printer.of_type("terminalOpened"))
        with svc._lock:
            writes = svc._sessions["t"].writes
        # The reader thread takes this mutex for ``writes.put(None)``
        # right after it deregisters the finished session.
        writes.mutex.acquire()
        try:
            svc.close("t")
            printer.wait_for(lambda: _session_gone(svc, "t"))
            # A page re-opening the tab now gets a fresh shell.
            svc.open("t", "c2", str(tmp_path), 80, 24)
            printer.wait_for(
                lambda: (
                    len([e for e in printer.of_type("terminalOpened") if not e["attached"]]) == 2
                )
            )
        finally:
            writes.mutex.release()
        printer.wait_for(lambda: printer.of_type("terminalExit"))
        with printer.lock:
            order = [
                (e["type"], e.get("attached"))
                for e in printer.events
                if e["type"] in ("terminalOpened", "terminalExit")
            ]
        assert order == [
            ("terminalOpened", False),
            ("terminalExit", None),
            ("terminalOpened", False),
        ], order
        assert printer.of_type("terminalExit")[0]["connId"] == "c1"
    finally:
        svc.shutdown()


def test_nan_and_infinity_dimensions_fall_back_instead_of_aborting(tmp_path: Path) -> None:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    try:
        svc.open("t", "c1", str(tmp_path), float("nan"), float("inf"))
        printer.wait_for(lambda: printer.of_type("terminalOpened"))
        svc.input("t", "c1", "stty size\n")
        printer.wait_for(lambda: "24 80" in printer.output("t"))
        svc.resize("t", "c1", float("-inf"), 30)
        svc.input("t", "c1", "stty size\n")
        printer.wait_for(lambda: "30 80" in printer.output("t"))
        svc.resize("t", "c1", 100, float("nan"))
        svc.input("t", "c1", "stty size\n")
        printer.wait_for(lambda: "24 100" in printer.output("t"))
        assert printer.of_type("terminalError") == []
    finally:
        svc.shutdown()

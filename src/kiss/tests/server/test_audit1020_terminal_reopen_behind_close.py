# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a ``terminalOpen`` right behind the ``terminalClose`` of its tab starts a fresh shell.

A page reloading as the grace hang-up of its shell fires sends its
``terminalOpen`` while the hung-up shell is still registered under the
tab (it has ``SIGHUP`` and is on its way out).  ``TerminalService``
used to re-attach the new page to that dying shell: it got
``terminalOpened`` with ``attached`` and, moments later, the old
shell's ``terminalExit``, so the page showed "The shell exited" for a
tab it had just opened.  The open now waits for the hung-up shell to
be reaped and starts a fresh one.

The shell here ignores ``SIGHUP``, so the hang-up only ends it with
the ``SIGKILL`` that follows ``_HANGUP_TIMEOUT``: the second open is
certain to find the old shell still registered.
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


def test_an_open_behind_the_close_of_a_lingering_shell_starts_a_fresh_one(
    tmp_path: Path,
) -> None:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    try:
        svc.open("t", "c1", str(tmp_path), 80, 24)
        printer.wait_for(lambda: printer.of_type("terminalOpened", "c1"))
        svc.input("t", "c1", "trap '' HUP; echo trap-$((6*7))\n")
        printer.wait_for(lambda: "trap-42" in printer.output("t"))

        svc.close("t")
        svc.open("t", "c2", str(tmp_path), 80, 24)

        printer.wait_for(lambda: printer.of_type("terminalOpened", "c2"))
        (opened,) = printer.of_type("terminalOpened", "c2")
        assert opened["attached"] is False
        svc.input("t", "c2", "echo fresh-$((6*7))\n")
        printer.wait_for(lambda: "fresh-42" in printer.output("t"))
        # The old shell's exit went to the page that had it, before the
        # new shell was announced; the new page never hears of an exit.
        with printer.lock:
            order = [
                (e["type"], e["connId"])
                for e in printer.events
                if e["type"] in ("terminalOpened", "terminalExit")
            ]
        assert order == [
            ("terminalOpened", "c1"),
            ("terminalExit", "c1"),
            ("terminalOpened", "c2"),
        ]
    finally:
        svc.shutdown()

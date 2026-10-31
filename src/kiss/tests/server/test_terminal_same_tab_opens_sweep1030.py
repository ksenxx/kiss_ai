# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `service` and `bash_shell` fixtures are imported from
#   kiss.tests.server.test_audit1013_terminal_open_lock and are intentionally
#   shadowed by test parameters of the same name)
"""E2E: two opens of one terminal tab on one connection, closed while both wait.

On a machine's daemon every browser connection of the local daemon
forwards its ``terminalOpen`` over the machine's one stream, so two
opens of the same tab carry the same ``connId``.  Each open registers
an ``_Opening`` that ``close``, ``viewer_gone`` and ``shutdown`` flag,
and deregisters it when done.  The two are equal by value, so the
openings compare by identity: an open that finishes removes its own,
never the other one still waiting for its turn (which ``close`` would
then miss, and the waiting open would start a shell for the closed tab
or re-attach a detached one to a dead connection).

That harmful order cannot be forced end to end.  The open registered
second must take the spawn lock first, i.e. run in the few bytecodes
between the first open's registration and its request for the lock;
opens already waiting for the lock get it in their order of arrival
(CPython hands a released lock to its longest waiter), so reversing
them takes a thread-scheduling hook, a test double.  This test drives
the same tab and connection through the real service in the order that
can be forced: both opens queued behind another tab's spawn (held on a
FIFO, as in ``test_audit1013_terminal_open_lock``) and the tab closed
meanwhile.  Neither starts a shell, and a later open starts a fresh one.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.tests.server.test_audit1013_terminal_open_lock import (
    _open_in_thread,
    _release_fifo,
    bash_shell,  # noqa: F401  (fixture used by param name)
    service,  # noqa: F401  (fixture used by param name)
)
from kiss.tests.server.test_terminal_tab_service import session_count

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="terminal tabs need a pty")


def test_both_waiting_opens_of_a_closed_tab_start_no_shell(
    service: Any, bash_shell: Path, tmp_path: Path,
) -> None:
    svc, printer = service
    work = tmp_path / "work"
    work.mkdir()
    bash_shell.parent.mkdir(parents=True)
    os.mkfifo(bash_shell)
    # Another tab's spawn holds the spawn lock, stuck on the FIFO.
    holder = _open_in_thread(svc, "tab-x", "conn-x", work)
    time.sleep(0.3)
    first = _open_in_thread(svc, "tab-t", "stream", work)
    second = _open_in_thread(svc, "tab-t", "stream", work)
    try:
        time.sleep(0.3)
        assert holder.is_alive() and first.is_alive() and second.is_alive()
        svc.close("tab-t", "stream")
    finally:
        _release_fifo(bash_shell)
    for opener in (holder, first, second):
        opener.join(10.0)
        assert not opener.is_alive()

    assert [e["tab_id"] for e in printer.of_type("terminalOpened")] == ["tab-x"]
    assert session_count(svc) == 1
    # The closed tab opens afresh later, for the connection that asks.
    svc.open("tab-t", "stream", str(work), 80, 24)
    opened = printer.of_type("terminalOpened", "stream")
    assert len(opened) == 1 and opened[0]["attached"] is False
    svc.input("tab-t", "stream", "echo fresh-$((1+1))\n")
    printer.wait_for(lambda: "fresh-2" in printer.output("tab-t"), timeout=5.0)

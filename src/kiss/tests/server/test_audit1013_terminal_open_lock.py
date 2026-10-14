# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a slow shell spawn must not stall the other terminal tabs.

``TerminalService.open`` used to hold the service lock while it forked
the shell (``_spawn``: start-up files, ``pty.fork``, the first
``TIOCSWINSZ``).  ``input``/``resize``/``close``/``viewer_gone`` take
the same lock on the server loop, so every keystroke of every other
terminal tab waited for the fork to finish.

The spawn is held here for real, with no test double: the shell is
named ``bash`` so :func:`kiss.server.terminal_rc.with_persistent_history`
writes ``$KISS_HOME/terminal/bashrc``, and that path is a FIFO, so the
``read_text`` that checks the file's current content blocks in
``open(2)`` until the test opens the FIFO's write end.  Before the fix
the ``input`` on the other tab blocks on the lock for as long as the
FIFO is held; after it the keystrokes reach the first shell and its
output comes back while the second spawn is still stuck.

The second test (Linux only: it reads ``/proc``) drives two ``open``
calls for the same tab id that arrive while no session exists.  Opens
are serialized by the spawn lock, so the second waits for the first's
shell and re-attaches to it: one shell, one ``terminalOpened
attached=False`` and one ``attached=True``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.core.processes import find_bash
from kiss.server.terminal_tab import TerminalService
from kiss.tests.server.test_terminal_tab_service import ConnPrinter

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="terminal tabs need a pty",
)


@pytest.fixture
def bash_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``$SHELL`` at a wrapper NAMED ``bash`` and ``$KISS_HOME`` at a temp dir.

    The name makes the service write its ``bashrc`` start-up file under
    ``$KISS_HOME/terminal`` on every spawn; the wrapper itself runs bash
    without any dotfile (it drops the ``--rcfile`` the service passes,
    so the content of that file never matters).
    """
    bash = find_bash()
    assert bash, "the terminal tests need bash on PATH"
    wrapper = tmp_path / "shell" / "bash"
    wrapper.parent.mkdir()
    wrapper.write_text(f'#!/bin/sh\nexec "{bash}" --noprofile --norc\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("SHELL", str(wrapper))
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    return tmp_path / "home" / "terminal" / "bashrc"


@pytest.fixture
def service() -> Any:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    yield svc, printer
    svc.shutdown()


def _release_fifo(fifo: Path) -> None:
    """Give every reader blocked on *fifo* an immediate EOF."""
    fd = os.open(fifo, os.O_WRONLY)
    os.close(fd)


def _open_in_thread(svc: TerminalService, tab_id: str, conn_id: str, cwd: Path) -> threading.Thread:
    thread = threading.Thread(
        target=svc.open, args=(tab_id, conn_id, str(cwd), 80, 24), daemon=True,
    )
    thread.start()
    return thread


def _shells_in(cwd: Path) -> list[int]:
    """Return the pids of this process's live children whose cwd is *cwd*."""
    me = os.getpid()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="ascii") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            if int(fields[1]) != me or fields[0] == "Z":
                continue
            if os.readlink(f"/proc/{entry}/cwd") == str(cwd):
                found.append(int(entry))
        except OSError:
            continue
    return found


def test_input_on_another_tab_is_not_blocked_by_a_slow_spawn(
    service: Any, bash_shell: Path, tmp_path: Path,
) -> None:
    svc, printer = service
    work = tmp_path / "work"
    work.mkdir()
    svc.open("tab-a", "conn-1", str(work), 80, 24)
    assert len(printer.of_type("terminalOpened", "conn-1")) == 1
    assert bash_shell.is_file()

    # The next spawn blocks inside with_persistent_history's read_text.
    bash_shell.unlink()
    os.mkfifo(bash_shell)
    opener = _open_in_thread(svc, "tab-b", "conn-2", work)
    try:
        time.sleep(0.3)
        assert opener.is_alive(), "the spawn was expected to be stuck on the FIFO"
        typer = threading.Thread(
            target=svc.input, args=("tab-a", "conn-1", "echo marker-$((40+2))\n"),
            daemon=True,
        )
        typer.start()
        typer.join(3.0)
        assert not typer.is_alive(), "input on tab-a blocked behind tab-b's spawn"
        sizer = threading.Thread(target=svc.resize, args=("tab-a", "conn-1", 120, 40), daemon=True)
        sizer.start()
        sizer.join(3.0)
        assert not sizer.is_alive(), "resize on tab-a blocked behind tab-b's spawn"
        printer.wait_for(lambda: "marker-42" in printer.output("tab-a"), timeout=5.0)
        assert opener.is_alive(), "tab-b's spawn should still be held by the FIFO"
    finally:
        _release_fifo(bash_shell)
    opener.join(10.0)
    assert not opener.is_alive()
    assert bash_shell.is_file()
    opened = printer.of_type("terminalOpened", "conn-2")
    assert len(opened) == 1 and opened[0]["tab_id"] == "tab-b" and opened[0]["attached"] is False
    assert svc.session_count() == 2
    svc.input("tab-b", "conn-2", "echo from-b\n")
    printer.wait_for(lambda: "from-b" in printer.output("tab-b"), timeout=5.0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inspects shells through /proc")
def test_concurrent_opens_of_one_tab_keep_a_single_shell(
    service: Any, bash_shell: Path, tmp_path: Path,
) -> None:
    svc, printer = service
    work = tmp_path / "race"
    work.mkdir()
    bash_shell.parent.mkdir(parents=True)
    os.mkfifo(bash_shell)
    # The first open blocks in its spawn on the FIFO; the second waits
    # for it (one spawn at a time) and finds its shell.
    first = _open_in_thread(svc, "tab-r", "conn-1", work)
    second = _open_in_thread(svc, "tab-r", "conn-2", work)
    try:
        time.sleep(0.3)
        assert first.is_alive() and second.is_alive()
    finally:
        _release_fifo(bash_shell)
    first.join(10.0)
    second.join(10.0)
    assert not first.is_alive() and not second.is_alive()

    assert svc.session_count() == 1
    opened = printer.of_type("terminalOpened")
    assert sorted(e["attached"] for e in opened) == [False, True]
    assert {e["tab_id"] for e in opened} == {"tab-r"}
    # One shell was forked: one live child in the work dir, no zombie.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and len(_shells_in(work)) != 1:
        time.sleep(0.05)
    assert len(_shells_in(work)) == 1
    # The shell now belongs to the connection whose open came second (a re-attach).
    owner = [e["connId"] for e in opened if e["attached"]][0]
    svc.input("tab-r", owner, "echo survivor\n")
    printer.wait_for(lambda: "survivor" in printer.output("tab-r"), timeout=5.0)
    svc.close("tab-r")
    printer.wait_for(lambda: printer.of_type("terminalExit"), timeout=5.0)
    assert svc.session_count() == 0
    assert _shells_in(work) == []

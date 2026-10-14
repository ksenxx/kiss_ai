# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a terminal open that is still spawning is part of the service's state.

Moving the fork of ``TerminalService.open`` out of the service lock
(``test_audit1013_terminal_open_lock.py``) left the spawn invisible
until its session was inserted:

* ``close``, ``viewer_gone`` and ``shutdown`` found nothing to act on,
  and the shell that appeared afterwards ran attached and unexpiring;
* two spawns ran at once, so their start-up-file writes collided on the
  shared ``bashrc.tmp`` name (``os.replace`` raised ``ENOENT`` and one
  shell started without its history hook);
* a fork that ran while another spawn had not yet cleared the
  inheritable flag of its pty master handed that master to the new
  shell, which kept it open after the first tab was closed.

Now one spawn runs at a time (a spawn lock held off the server loop,
never together with the session lock across the fork), so the start-up
files are written one at a time and no terminal forks before the
previous spawn's master is marked non-inheritable; and the spawn in
progress is registered so the lifecycle commands can cancel or detach
it.

Every spawn here is held for real: the shell is named ``bash`` so
:func:`kiss.server.terminal_rc.with_persistent_history` reads
``$KISS_HOME/terminal/bashrc``, which the tests turn into a FIFO.
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
from kiss.server import terminal_tab
from kiss.server.terminal_tab import TerminalService
from kiss.tests.server.test_audit1013_terminal_open_lock import (
    _open_in_thread,
    _release_fifo,
    _shells_in,
)
from kiss.tests.server.test_terminal_tab_service import ConnPrinter

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="inspects the shells through /proc",
)


@pytest.fixture
def bash_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``$SHELL`` at a wrapper NAMED ``bash`` that keeps the service's arguments.

    Unlike the wrapper of ``test_audit1013_terminal_open_lock``, this one
    passes ``--rcfile`` on, so the shell runs the service's start-up
    file and gets its history hook.  ``$HOME`` is empty, so that file
    sources no dotfile of the developer.  Returns the start-up file's
    path (``$KISS_HOME/terminal/bashrc``).
    """
    bash = find_bash()
    assert bash, "the terminal tests need bash on PATH"
    wrapper = tmp_path / "shell" / "bash"
    wrapper.parent.mkdir()
    wrapper.write_text(f'#!/bin/sh\nexec "{bash}" --noprofile "$@"\n')
    wrapper.chmod(0o755)
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("SHELL", str(wrapper))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss"))
    rc = tmp_path / "kiss" / "terminal" / "bashrc"
    rc.parent.mkdir(parents=True)
    os.mkfifo(rc)
    return rc


@pytest.fixture
def service() -> Any:
    printer = ConnPrinter()
    svc = TerminalService(printer)
    yield svc, printer
    svc.shutdown()


def _call_in_thread(target: Any, *args: Any) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def _wait_until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def _held_open(svc: TerminalService, tab_id: str, conn_id: str, work: Path) -> threading.Thread:
    """Start an open of *tab_id* and wait until it is stuck on the FIFO."""
    opener = _open_in_thread(svc, tab_id, conn_id, work)
    time.sleep(0.3)
    assert opener.is_alive(), "the spawn was expected to be stuck on the FIFO"
    return opener


def test_close_during_a_pending_open_leaves_no_shell(
    service: Any, bash_shell: Path, tmp_path: Path,
) -> None:
    svc, printer = service
    work = tmp_path / "work"
    work.mkdir()
    # tab-a is forking (held on the FIFO); tab-b waits its turn behind it.
    opener_a = _held_open(svc, "tab-a", "conn", work)
    opener_b = _open_in_thread(svc, "tab-b", "conn", work)
    time.sleep(0.2)
    for tab in ("tab-a", "tab-b"):
        closer = _call_in_thread(svc.close, tab, "conn")
        closer.join(3.0)
        assert not closer.is_alive(), "close blocked behind the pending spawn"
    _release_fifo(bash_shell)
    opener_a.join(10.0)
    opener_b.join(10.0)
    assert not opener_a.is_alive() and not opener_b.is_alive()
    assert svc.session_count() == 0
    assert printer.of_type("terminalOpened") == []
    _wait_until(lambda: _shells_in(work) == [])


def test_disconnect_during_a_pending_open_carries_the_grace_forward(
    service: Any, bash_shell: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(terminal_tab, "GRACE_SECONDS", 0.5)
    svc, printer = service
    work = tmp_path / "work"
    work.mkdir()
    # tab-a is forking (held on the FIFO); tab-b waits its turn behind it.
    opener_a = _held_open(svc, "tab-a", "conn", work)
    opener_b = _open_in_thread(svc, "tab-b", "conn", work)
    time.sleep(0.2)
    gone = _call_in_thread(svc.viewer_gone, "conn")
    gone.join(3.0)
    assert not gone.is_alive(), "viewer_gone blocked behind the pending spawn"
    _release_fifo(bash_shell)
    opener_a.join(10.0)
    opener_b.join(10.0)
    assert not opener_a.is_alive() and not opener_b.is_alive()
    # Both shells came up detached: their grace runs out and they are hung up.
    opened = printer.of_type("terminalOpened", "conn")
    assert sorted(e["tab_id"] for e in opened) == ["tab-a", "tab-b"]
    assert all(e["attached"] is False for e in opened)
    printer.wait_for(lambda: len(printer.of_type("terminalExit", "conn")) == 2, timeout=5.0)
    assert svc.session_count() == 0
    _wait_until(lambda: _shells_in(work) == [])


def test_reconnect_during_a_pending_open_claims_the_shell(
    service: Any, bash_shell: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(terminal_tab, "GRACE_SECONDS", 0.5)
    svc, printer = service
    work = tmp_path / "work"
    work.mkdir()
    opener = _held_open(svc, "tab", "conn-1", work)
    svc.viewer_gone("conn-1")
    # The page is back on a new connection before the shell is up: its
    # open waits for the spawn and re-attaches, cancelling the grace.
    reopener = _open_in_thread(svc, "tab", "conn-2", work)
    time.sleep(0.3)
    assert reopener.is_alive()
    _release_fifo(bash_shell)
    opener.join(10.0)
    reopener.join(10.0)
    assert not opener.is_alive() and not reopener.is_alive()
    opened = printer.of_type("terminalOpened", "conn-2")
    assert len(opened) == 1 and opened[0]["attached"] is True
    time.sleep(1.0)
    assert svc.session_count() == 1 and printer.of_type("terminalExit") == []
    svc.input("tab", "conn-2", "echo claimed-$((40+2))\n")
    printer.wait_for(lambda: "claimed-42" in printer.output("tab"), timeout=5.0)
    assert len(_shells_in(work)) == 1


def test_shutdown_during_a_pending_open_leaves_no_shell(
    service: Any, bash_shell: Path, tmp_path: Path,
) -> None:
    svc, printer = service
    work = tmp_path / "work"
    work.mkdir()
    # tab-a is forking (held on the FIFO); tab-b waits its turn behind it.
    opener_a = _held_open(svc, "tab-a", "conn", work)
    opener_b = _open_in_thread(svc, "tab-b", "conn", work)
    time.sleep(0.2)
    stopper = _call_in_thread(svc.shutdown)
    stopper.join(5.0)
    assert not stopper.is_alive(), "shutdown blocked behind the pending spawn"
    _release_fifo(bash_shell)
    opener_a.join(10.0)
    opener_b.join(10.0)
    assert not opener_a.is_alive() and not opener_b.is_alive()
    assert svc.session_count() == 0
    assert printer.of_type("terminalOpened") == []
    _wait_until(lambda: _shells_in(work) == [])
    # The service is reusable after a shutdown (a test harness hangs up
    # leftover shells that way): a later open starts a shell.
    svc.open("tab-c", "conn", str(work), 80, 24)
    assert svc.session_count() == 1 and len(_shells_in(work)) == 1


def _pty_masters_of(pid: int) -> list[int]:
    """Return the fds of *pid* that are pty masters (``/dev/ptmx``)."""
    found = []
    for entry in os.listdir(f"/proc/{pid}/fd"):
        try:
            if os.readlink(f"/proc/{pid}/fd/{entry}") == "/dev/ptmx":
                found.append(int(entry))
        except OSError:
            continue
    return found


def test_concurrent_opens_keep_every_history_hook_and_share_no_master(
    service: Any, bash_shell: Path, tmp_path: Path,
) -> None:
    svc, printer = service
    work = tmp_path / "work"
    work.mkdir()
    tabs = [f"tab-{i}" for i in range(4)]
    openers = [_open_in_thread(svc, tab, "conn", work) for tab in tabs]
    time.sleep(0.3)
    assert all(t.is_alive() for t in openers)
    _release_fifo(bash_shell)
    for opener in openers:
        opener.join(10.0)
        assert not opener.is_alive()
    assert svc.session_count() == len(tabs)
    assert bash_shell.is_file() and not bash_shell.with_name("bashrc.tmp").exists()
    # Every shell got the start-up file: none lost it to a colliding write.
    for tab in tabs:
        svc.input(tab, "conn", "printf 'hook=<%s>\\n' \"${PROMPT_COMMAND-}\"\n")
    for tab in tabs:
        printer.wait_for(
            lambda: "hook=<>" in printer.output(tab) or "hook=<history -a>" in printer.output(tab),
            timeout=5.0,
        )
        assert "hook=<history -a>" in printer.output(tab), tab
    # No shell holds another terminal's pty master.
    _wait_until(lambda: len(_shells_in(work)) == len(tabs))
    for pid in _shells_in(work):
        assert _pty_masters_of(pid) == [], f"shell {pid} inherited a pty master"

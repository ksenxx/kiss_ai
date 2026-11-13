# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (S5 F2): an exception thrown into ``bridge_exec``'s wait kills the command.

``bridge_exec`` posts ``/exec`` from a helper thread and polls it in 0.2 s
slices.  A cooperative Stop (the stop event, the tool-panel interrupt)
posted ``/kill`` for the command's token, but an exception raised on the
calling thread inside that wait (a ``KeyboardInterrupt``, or the runner's
Stop injected as one) unwound without any kill: the detached request and
the remote command went on.  The wait is now wrapped so any exception
posts the same bounded ``/kill`` before it propagates.

The test runs the real bridge server over a local environment (asyncio
subprocesses) and throws the interrupt into the real calling thread with
``PyThreadState_SetAsyncExc``, as the runner's Stop does.  Nothing is mocked.
The command touches ``late`` only after :data:`_LATE_AFTER_S` seconds: the
kill (a ``/proc`` sweep run as one more subprocess) must land before that
even on a loaded machine, and once the exec has ended the shell is dead, so
``late`` either exists by then or never will.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

import pytest

from kiss.agents.sorcar.docker_manager import _KILL_WAIT_S
from kiss.agents.sorcar.exec_bridge import bridge_exec
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.tests.agents.seas.test_coding_sea_evidence_gate import (
    LocalEnvironment,
    _close,
    _Loop,
)
from kiss.tests.conftest import wait_for

#: Seconds the command waits before touching ``late``: the kill's budget.
_LATE_AFTER_S = 5

#: The kill finds the command's tree through ``/proc/<pid>/environ`` (the bridge serves
#: Linux sandboxes); macOS has no ``/proc``, so a bridged local command there cannot be killed.
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="the tagged-tree kill scans /proc")


@linux_only
def test_an_interrupt_in_the_wait_kills_the_remote_command(tmp_path: Path) -> None:
    """A ``KeyboardInterrupt`` thrown into the wait ends the command on the environment too."""
    loop = _Loop()
    server = ExecBridgeServer(LocalEnvironment(str(tmp_path)), loop.loop)
    started, late = tmp_path / "started", tmp_path / "late"
    errors: list[str] = []

    def caller() -> None:
        try:
            bridge_exec(
                server.url, f"touch {started}; sleep {_LATE_AFTER_S}; touch {late}; sleep 30", 60,
            )
        except BaseException as exc:
            errors.append(type(exc).__name__)

    thread = threading.Thread(target=caller)
    try:
        thread.start()
        assert wait_for(started.exists, 10)
        assert thread.ident is not None
        injected = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident), ctypes.py_object(KeyboardInterrupt),
        )
        assert injected == 1
        # The interrupt propagates once the bounded kill wait is over.
        thread.join(2 + _KILL_WAIT_S)
        assert not thread.is_alive() and errors == ["KeyboardInterrupt"], errors
        # The command was killed on the environment: its exec ends (the shell
        # is dead) before it reached the ``touch`` of ``late``.
        assert wait_for(lambda: not server.pending, timeout=10), server.pending
        assert not late.exists(), "the remote command outlived the call"
        # The bridge is still usable afterwards.
        assert bridge_exec(server.url, "echo still here", 10) == (0, "still here\n")
    finally:
        _close(server, loop)
        thread.join(3)

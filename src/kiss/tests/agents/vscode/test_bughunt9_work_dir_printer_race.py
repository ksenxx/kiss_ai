# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Race reproduction: ``setWorkDir`` must sync the printer atomically.

``_cmd_set_work_dir`` updates ``VSCodeServer.work_dir`` under
``_state_lock`` but (pre-fix) mirrored the value onto
``printer.work_dir`` OUTSIDE the lock.  Two racing ``setWorkDir``
commands (two VS Code windows reconnecting together) could therefore
interleave the printer writes in the OPPOSITE order of the server
writes, leaving ``printer.work_dir != server.work_dir`` permanently —
so global ``configData`` events kept reporting a folder the daemon no
longer used.

The test hammers the handler from two barrier-synchronised threads
through the real command dispatch.  The printer is a genuine
:class:`JsonPrinter` whose ``work_dir`` attribute is a plain property
with a sub-10ms random pause in its setter, widening the window
between the locked server write and the printer sync exactly where the
race lives (per the race-confirmation policy of adding a random sleep
before the suspected racing statement).  After both threads finish,
the invariant ``printer.work_dir == server.work_dir`` must hold on
every iteration.
"""

from __future__ import annotations

import random
import threading
import time

from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer


class _TimedWorkDirPrinter(JsonPrinter):
    """A JsonPrinter whose ``work_dir`` writes take a random pause.

    The pause sits BEFORE the underlying attribute write, modelling an
    OS scheduler preemption between the server's locked ``work_dir``
    update and the (pre-fix, unlocked) printer sync.  The attribute
    behaves exactly like ``WebPrinter.work_dir`` otherwise.
    """

    def __init__(self) -> None:
        """Initialise with an empty work_dir."""
        super().__init__()
        self._work_dir = ""

    @property
    def work_dir(self) -> str:
        """Return the last folder written to the printer."""
        return self._work_dir

    @work_dir.setter
    def work_dir(self, value: str) -> None:
        time.sleep(random.uniform(0.0, 0.008))
        self._work_dir = value


def test_racing_set_work_dir_keeps_printer_and_server_consistent() -> None:
    """Two racing ``setWorkDir`` commands must never desync the printer.

    Pre-fix this fails within a handful of iterations: the thread that
    loses the ``_state_lock`` race can still win the (unlocked) printer
    write, leaving ``printer.work_dir`` pointing at the folder that
    LOST the server-side update.
    """
    printer = _TimedWorkDirPrinter()
    server = VSCodeServer(printer=printer)
    server.work_dir = "/init"
    printer._work_dir = "/init"

    for i in range(40):
        dirs = (f"/ws/a{i}", f"/ws/b{i}")
        barrier = threading.Barrier(2)

        def worker(target_dir: str) -> None:
            barrier.wait()
            server._handle_command(
                {
                    "type": "setWorkDir",
                    "workDir": target_dir,
                    "connId": target_dir,
                }
            )

        threads = [
            threading.Thread(target=worker, args=(d,)) for d in dirs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with server._state_lock:
            server_dir = server.work_dir
        assert printer.work_dir == server_dir, (
            f"iteration {i}: printer.work_dir={printer.work_dir!r} "
            f"desynced from server.work_dir={server_dir!r} — the "
            "printer sync must happen inside _state_lock together "
            "with the server-side work_dir write"
        )

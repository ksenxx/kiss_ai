# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``ConsolePrinter`` under the runner's Stop injection and under a missing ``sys.stdout``.

C-1: a ``ConsolePrinter`` is shared by every sub-agent thread of a run,
and a task thread can receive an injected ``KeyboardInterrupt``
(``PyThreadState_SetAsyncExc``) at the check point right after a
``with lock:``'s ``__enter__`` returned (CPython gh-74174), where the
``with`` block's ``__exit__`` is skipped and the lock stays owned by a
thread that then dies.  Every later ``print`` on any other thread would
then hang for the printer's life.  :func:`hammer` drives real prints
from a worker thread while a contender keeps ``_lock`` busy (so the
worker's ``acquire`` blocks and the injection lands on its return) and
asserts the lock is free afterwards.

The hammered path is the raw ``bash_stream`` write only.  Every other
event renders through ``rich``, whose ``Console._write_buffer`` takes
its own ``RLock`` with a plain ``with`` and so has the same gh-74174
window; an interrupt landing there leaves rich's lock owned by the
worker and the main thread's closing print would hang on it.  That is
rich's limitation, not the printer's, and it is unreachable in
production: the daemon's Stop injection only ever targets tasks that
print through ``JsonPrinter``/``WebPrinter``, while a ``ConsolePrinter``
run (the CLI) is stopped by a terminating Ctrl-C.

C-5: the output stream is now ``rich``'s ``Console.file``, which falls
back to its ``NULL_FILE`` when ``sys.stdout`` is ``None`` (a ``pythonw``
process, or a host that closed the standard streams), so a printer with
no explicit file must not crash on its raw writes either.
"""

from __future__ import annotations

import io
import sys

from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.lock_injection import hammer, lock_is_free


class _Printing:
    """One raw bash-stream write on the worker: takes ``_lock`` and nothing of rich's."""

    def __init__(self, printer: ConsolePrinter) -> None:
        self.printer = printer

    def __call__(self) -> None:
        self.printer.print("line\n", type="bash_stream")


def test_injected_interrupt_never_leaks_the_printer_lock() -> None:
    printer = ConsolePrinter(file=io.StringIO())
    # The first bash_stream write draws a rich rule; do it here so the
    # worker's writes never enter rich (see the module docstring).
    printer.print("first\n", type="bash_stream")
    hammer(_Printing(printer), {"_lock": printer._lock}, contend=printer._lock)
    assert lock_is_free(printer._lock)
    # The printer is still usable from this thread afterwards.
    printer.print("after", type="text")


def test_prints_when_sys_stdout_is_none() -> None:
    saved = sys.stdout
    sys.stdout = None  # type: ignore[assignment]
    try:
        printer = ConsolePrinter()
        printer.print("line\n", type="bash_stream")
        printer.print("ls", type="tool_call", tool_input={"command": "ls"})
        printer.print("out\n", type="tool_result", tool_name="Bash", is_error=False)
        printer.token_callback("tok")
        printer.print("hello", type="text")
        printer.print("done", type="result", cost="$0.1", total_tokens=1, step_count=1)
    finally:
        sys.stdout = saved


def test_explicit_file_is_still_honoured() -> None:
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    printer.print("raw\n", type="bash_stream")
    printer.print("x", type="tool_result", tool_name="Bash", is_error=False)
    assert "raw" in buf.getvalue()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A ``Thread.join(timeout)`` that ends at its deadline on Windows.

CPython 3.13, 3.14 and 3.15 work out the time a timed join has left with
``Py_MAX(_PyDeadline_Get(deadline), 0)`` (``ThreadHandle_join`` in
``Modules/_threadmodule.c``).  The macro evaluates its argument twice,
so the clock is read twice.  When the deadline passes between the two
reads the result is negative, and the wait that follows takes a negative
timeout for "no limit": the join returns only when the thread ends
(https://github.com/python/cpython/issues/161712; 3.13.18, 3.14.10 and
3.15.2 still have it).

That line runs whenever a wait ends before the deadline.  A Windows wait
counts its timeout in timer ticks and often ends early (a third of the
100 ms waits measured), so the race is run all the time there.  On a
Windows machine busy with a parallel test run up to one ``join(0.1)``
in a thousand did not return.  The Stop wait of the Codex tool bridge
was caught stuck in such a join; the timeout of a cron command
(``Popen.communicate(timeout=...)`` joins its reader threads on
Windows) and the deadline of a CLI prompt write rest on timed joins
too.

:func:`bounded_join` waits with what the defect does not touch: a join
with a zero timeout (the faulty line then finds the deadline passed at
its first read of the clock), ``Thread.is_alive`` and ``time.sleep``.
:func:`install_bounded_join` makes it ``Thread.join`` on Windows, so
the product's joins, the standard library's and those of other packages
all get it.  Both can go once no supported interpreter has the defect.
"""

import functools
import sys
import threading
import time

_native_join = threading.Thread.join

# The pause between two looks at the thread doubles from the first value to
# the longest: a thread that ends at once is seen at once, and a long wait
# costs fifty looks a second.
_FIRST_PAUSE_SECONDS = 0.0005
_LONGEST_PAUSE_SECONDS = 0.02


@functools.cache
def _ended_thread() -> threading.Thread:
    """Return a thread that has ended.

    A timed join of it only converts the timeout, which raises what the
    interpreter raises for a timeout it cannot use (NaN, a value too
    large, not a number).
    """
    thread = threading.Thread(target=int, daemon=True)
    thread.start()
    _native_join(thread)
    return thread


def bounded_join(self: threading.Thread, timeout: float | None = None) -> None:
    """Wait until the thread ends or *timeout* seconds have passed.

    A replacement for ``threading.Thread.join`` with the same contract;
    the caller reads ``is_alive()`` afterwards to tell the two outcomes
    apart.  The first parameter is named ``self`` because the function
    is installed as that method.

    Args:
        self: The thread to wait for.
        timeout: The longest wait in seconds; zero or less only looks,
            ``None`` waits until the thread ends.

    Raises:
        RuntimeError: For a thread that has not been started and for the
            calling thread, as ``Thread.join`` raises it.
        ValueError, OverflowError, TypeError: For a timeout the
            interpreter cannot use, as ``Thread.join`` raises them.
    """
    if timeout is None:
        _native_join(self)
        return
    _native_join(self, 0)
    _native_join(_ended_thread(), timeout)
    deadline = time.monotonic() + timeout
    pause = _FIRST_PAUSE_SECONDS
    while self.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(pause, remaining))
        pause = min(pause * 2, _LONGEST_PAUSE_SECONDS)
    # The thread has run its last Python code: this reaps it, waiting for
    # no more than its native exit.
    _native_join(self)


def install_bounded_join() -> None:
    """Make :func:`bounded_join` the ``Thread.join`` of this process on Windows.

    A POSIX wait ends before its deadline only for a signal, so the
    interpreter's own join is kept there.  The ended thread is made
    here, not by the first timed join: that join may come at interpreter
    shutdown, when no thread can be started any more.
    """
    if sys.platform == "win32":  # pragma: no cover - one side per platform
        _ended_thread()
        setattr(threading.Thread, "join", bounded_join)

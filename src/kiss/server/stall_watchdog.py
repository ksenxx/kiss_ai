# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""GIL-independent stall detector for long-running server processes.

When one thread keeps the GIL inside a long C-level call (a regex with
super-linear backtracking, for instance), every other Python thread
freezes: nothing is logged, the asyncio loop stops answering
websockets, and the process looks dead while one core runs at 100%.
That is exactly what the ``kiss-web`` outage of 2026-09-12 looked like
— 83 minutes of silence in the log with nothing to point at the
culprit — and this module exists so the next stall leaves a trace.

:func:`faulthandler.dump_traceback_later` runs a C watchdog thread that
does not need the GIL.  A small Python heartbeat thread re-arms it
every *interval* seconds; when the heartbeat itself cannot get the GIL
for *timeout* seconds, the watchdog dumps every thread's stack to the
process's stderr (the systemd-appended ``kiss-web-stderr.log``),
naming the frame that is stuck.
"""

import faulthandler
import logging
import sys
import threading
import time
from typing import TextIO

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECS = 60.0
_DEFAULT_INTERVAL_SECS = 5.0


def start_stall_watchdog(
    timeout: float = _DEFAULT_TIMEOUT_SECS,
    interval: float = _DEFAULT_INTERVAL_SECS,
    file: TextIO | None = None,
) -> threading.Thread | None:
    """Arm a watchdog that dumps all thread stacks when Python stalls.

    Args:
        timeout: Seconds the heartbeat may be starved before the C
            watchdog dumps every thread's traceback.  Dumps repeat
            every *timeout* seconds for as long as the stall lasts,
            so a log reader can tell a spinning frame from a slowly
            progressing one.
        interval: Heartbeat period; must be well below *timeout*.
        file: Destination for the dumps.  Defaults to ``sys.stderr``.
            It must be backed by a real file descriptor: faulthandler
            writes with ``write(2)``, bypassing Python-level buffering
            (and the GIL).

    Returns:
        The daemon heartbeat thread, or ``None`` when *file* has no
        usable file descriptor (e.g. stderr replaced by a ``StringIO``
        under some test harnesses), in which case nothing is armed.
    """
    target = sys.stderr if file is None else file
    try:
        target.fileno()
    except (AttributeError, OSError, ValueError):
        logger.info("Stall watchdog not armed: output has no file descriptor")
        return None

    def _heartbeat() -> None:
        while True:
            faulthandler.dump_traceback_later(timeout, repeat=True, file=target)
            time.sleep(interval)

    thread = threading.Thread(target=_heartbeat, name="stall-watchdog", daemon=True)
    thread.start()
    logger.info(
        "Stall watchdog armed: all thread stacks are dumped to stderr if the "
        "interpreter is unresponsive for %.0fs",
        timeout,
    )
    return thread

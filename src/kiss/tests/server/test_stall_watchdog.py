# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for :mod:`kiss.server.stall_watchdog`.

Each test runs a real child interpreter: the watchdog is armed, then a
regex with quadratic backtracking (the shape of the 2026-09-12 outage)
holds the GIL for a few seconds.  Only a GIL-independent watchdog can
report anything while that happens, so the child's stderr is the
observable: it must contain faulthandler's ``Timeout`` header and the
stuck frame when the stall exceeds the timeout, and nothing when the
heartbeat keeps running.
"""

import subprocess
import sys

# Bare token of this length takes several seconds under the quadratic
# pattern (measured 2.4 s for 20k chars on a 2026 server core).
_HOG_TOKEN_CHARS = 20_000

_HOG_SCRIPT = """
import re, sys, time
from kiss.server.stall_watchdog import start_stall_watchdog

def hog_the_gil():
    # The pre-fix _IMAGE_PATH_RE: a match attempt at every offset of the
    # token, each backtracking to its end.  Never releases the GIL.
    quadratic = re.compile(r"[^\\s\\"'`<>|:;,()\\[\\]{{}}]+\\.(?:png|jpe?g)")
    quadratic.search("A" * {chars})

watchdog = start_stall_watchdog(timeout={timeout}, interval=0.05)
assert watchdog is not None and watchdog.thread.daemon
time.sleep(0.2)  # let the heartbeat arm the watchdog at least once
hog_the_gil()
print("done", flush=True)
"""

# Same stall, but the watchdog is disarmed first: ``stop()`` must cancel
# the pending C-level dump, not just end the heartbeat thread.
_STOP_SCRIPT = _HOG_SCRIPT.replace(
    "hog_the_gil()\n",
    "watchdog.stop()\nassert not watchdog.thread.is_alive()\nhog_the_gil()\n",
)

# The dump target is closed while the watchdog is armed (pytest closes
# its captured stderr at session end while in-process servers' watchdogs
# are still running).  The heartbeat must end quietly instead of dying
# with "ValueError: I/O operation on closed file" on the real stderr.
_CLOSED_TARGET_SCRIPT = """
import sys, tempfile, time
from kiss.server.stall_watchdog import start_stall_watchdog

target = tempfile.TemporaryFile("w+")
watchdog = start_stall_watchdog(timeout=0.3, interval=0.05)
closed = start_stall_watchdog(timeout=600, interval=0.05, file=target)
assert watchdog is not None and closed is not None
time.sleep(0.2)
target.close()
closed.thread.join(timeout=5)
print("alive" if closed.thread.is_alive() else "ended", flush=True)
watchdog.stop()
"""

_NO_FD_SCRIPT = """
import io, logging, sys
from kiss.server.stall_watchdog import start_stall_watchdog
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
assert start_stall_watchdog(file=io.StringIO()) is None
print("unarmed", flush=True)
"""


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_gil_stall_dumps_all_thread_stacks_to_stderr():
    """A stall longer than the timeout names the stuck frame on stderr."""
    proc = _run(_HOG_SCRIPT.format(chars=_HOG_TOKEN_CHARS, timeout=0.5))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "done"
    assert "Timeout (0:00:00.500000)!" in proc.stderr
    assert "most recent call first" in proc.stderr
    assert "hog_the_gil" in proc.stderr
    # The heartbeat thread is listed too, proving the dump covers every
    # thread, not just the one holding the GIL.  (Thread names in the
    # header are a Python 3.14 addition, so match the frame instead.)
    assert "in _heartbeat" in proc.stderr


def test_short_gil_stall_below_timeout_stays_silent():
    """A stall shorter than the timeout produces no dump."""
    proc = _run(_HOG_SCRIPT.format(chars=_HOG_TOKEN_CHARS, timeout=600))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "done"
    assert "Timeout (" not in proc.stderr


def test_stop_cancels_the_pending_dump():
    """After ``stop()`` a stall longer than the old timeout stays silent."""
    proc = _run(_STOP_SCRIPT.format(chars=_HOG_TOKEN_CHARS, timeout=0.5))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "done"
    assert "Timeout (" not in proc.stderr


def test_closed_target_ends_the_heartbeat_quietly():
    """Closing the dump target stops that heartbeat without a traceback."""
    proc = _run(_CLOSED_TARGET_SCRIPT)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ended"
    assert "Exception in thread stall-watchdog" not in proc.stderr
    assert "closed file" not in proc.stderr


def test_output_without_file_descriptor_is_not_armed():
    """faulthandler needs a real fd; a StringIO leaves the watchdog off."""
    proc = _run(_NO_FD_SCRIPT)
    assert proc.returncode == 0, proc.stderr
    assert "Stall watchdog not armed" in proc.stdout
    assert proc.stdout.rstrip().endswith("unarmed")

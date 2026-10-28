# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A capped command is reported as capped, never as timed out as well.

``processes.run_capped`` (behind ``fs_actions``) kills the child once its stdout passes the
cap (``truncated``) or the deadline passes (``timed_out``); its callers
report ``timed_out`` first.  The two decisions race on the timer
thread: the capped verdict must be taken under ``guard`` together with
``finished``, or a timer firing in between relabels a capped search as
a timeout.  The race window is a few bytecodes wide, so the test runs a
real child many times with a tiny thread switch interval and deadlines
spread around the time the child needs to pass the cap; before the fix
it tripped about 1 run in 300.
"""

from __future__ import annotations

import sys
import time
import unittest

from kiss.core.processes import run_capped

# A child that writes 300 kB to stdout: the same interpreter (bare, no
# site), so the test needs no ``sh``, ``head`` or ``/dev/zero``.
_EMIT_BYTES = [
    sys.executable, "-I", "-S", "-c", "import sys; sys.stdout.buffer.write(b'0' * 300000)",
]


class TestCapVersusTimeout(unittest.TestCase):
    """``truncated`` and ``timed_out`` are mutually exclusive."""

    def test_capped_run_is_never_also_timed_out(self) -> None:
        """A child that passes the cap right at the deadline is capped or timed out, not both."""
        # How long the child takes to pass the cap here; the deadlines
        # below are spread from half to one and a half times that.
        started = time.monotonic()
        self.assertTrue(run_capped(_EMIT_BYTES, limit=1000, timeout=30.0).truncated)
        to_cap = time.monotonic() - started
        saved = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            deadline = time.monotonic() + 30
            runs = 0
            while runs < 3000 and time.monotonic() < deadline:
                timeout = to_cap * (0.5 + (runs % 50) / 50)
                out = run_capped(_EMIT_BYTES, limit=1000, timeout=timeout)
                self.assertFalse(
                    out.truncated and out.timed_out,
                    f"run {runs} (timeout {timeout}): capped AND timed out",
                )
                self.assertTrue(out.truncated or out.timed_out)
                runs += 1
        finally:
            sys.setswitchinterval(saved)


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: worker stderr output must not corrupt the steering box.

Bug: :meth:`kiss.agents.sorcar.cli_steering.SteeringSession.run` swaps
only ``sys.stdout`` for the lock-guarded :class:`_StdoutProxy`, so
every console write lands in the DECSTBM scroll region above the box
via the ``ESC 8`` (restore output cursor) / ``ESC 7`` (re-save) dance.
``sys.stderr`` was never swapped: anything the worker (or a library it
calls — ``logging``'s default stderr handlers, ``warnings``, LLM SDK
noise) writes to stderr is emitted at the *visible* cursor position,
which is parked inside the box body row.  The text overprints the
input panel and never scrolls with the rest of the agent output.

The test drives the real interactive path end to end on a PTY: a child
interpreter runs ``run_with_steering`` with an agent that writes a
marker to ``sys.stderr`` while the box is active.  The parent asserts
the marker was routed through the same restore/emit/re-save dance as
stdout output (i.e. immediately preceded by ``ESC 8`` on the wire),
proving it landed in the scroll region instead of the box body.
"""

from __future__ import annotations

import os
import select
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires a POSIX pty",
)

_MARKER = b"XSTDERRNOISEX"


def _drain(fd: int, seconds: float) -> bytes:
    """Read whatever the child writes to the PTY for *seconds*."""
    out = b""
    deadline = time.time() + seconds
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 8192)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    return out


def test_worker_stderr_goes_to_scroll_region_not_box_body(
    tmp_path: Path,
) -> None:
    """Stderr output during steering must use the output-cursor dance.

    Every write routed through the steering session's stream proxy is
    emitted as ``ESC 8`` + text + ``ESC 7`` so it appends to the agent
    output inside the scroll region.  A raw (unproxied) stderr write
    instead appears at the parked cursor on the box body row,
    overprinting the input panel.
    """
    from kiss.tests.agents.sorcar._pty_helper import pty_spawn

    started = tmp_path / "started"
    kiss_home = tmp_path / ".kisshome"

    child_code = f"""
import os
os.environ["KISS_HOME"] = {str(kiss_home)!r}
os.environ["LINES"] = "24"
os.environ["COLUMNS"] = "80"
import sys, time
from pathlib import Path

started = Path({str(started)!r})
real_stderr = sys.stderr


class NoisyAgent:
    model_name = "demo-model"

    def run(self, **kwargs):
        started.write_text("1")
        time.sleep(1.0)  # let the box finish drawing
        sys.stderr.write({_MARKER.decode()!r} + "\\n")
        sys.stderr.flush()
        time.sleep(0.3)
        return "summary: done\\n"


from kiss.agents.sorcar.cli_steering import run_with_steering

run_with_steering(NoisyAgent(), {{}})
sys.stdout.write("RESTORED:" + str(sys.stderr is real_stderr) + "\\n")
sys.stdout.flush()
"""

    pid, fd = pty_spawn([sys.executable, "-c", child_code])

    try:
        deadline = time.time() + 30.0
        out = b""
        while time.time() < deadline and not started.exists():
            out += _drain(fd, 0.2)
        assert started.exists(), f"agent never started; output: {out!r}"

        deadline = time.time() + 15.0
        while time.time() < deadline and b"RESTORED:" not in out:
            out += _drain(fd, 0.3)

        assert _MARKER in out, f"stderr marker never appeared: {out!r}"
        assert b"\x1b8" + _MARKER in out, (
            "the worker's stderr write was NOT routed through the "
            "steering session's output-cursor restore (ESC 8) — it was "
            "emitted raw at the parked cursor inside the box body row, "
            f"overprinting the input panel: {out!r}"
        )
        assert b"RESTORED:True" in out, (
            f"sys.stderr was not restored after the session: {out!r}"
        )
    finally:
        try:
            os.write(fd, b"\x03")
        except OSError:
            pass
        _drain(fd, 2.0)
        try:
            os.close(fd)
        except OSError:
            pass
        os.waitpid(pid, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

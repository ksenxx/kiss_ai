# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: Ctrl+C while the main thread waits on the terminal lock.

Bug: ``SteeringSession._loop`` only caught ``KeyboardInterrupt`` around
its ``select.select`` call.  A ``SIGINT`` that arrives while the main
thread is instead blocked on the shared terminal lock (inside
``box.feed`` → ``redraw``, waiting for the worker's ``_StdoutProxy``
write to finish) raised ``KeyboardInterrupt`` *outside* that handler.
The exception escaped ``_loop`` without ``_on_abort`` ever running, so
``SteeringSession.run`` re-raised without injecting the abort into the
worker — the "interrupted" task kept executing ``agent.run`` (and
spending budget) in the background, the very leak iteration 1 fixed for
the in-``select`` case.

The test makes that window deterministic on a real PTY: the agent
floods stdout while the parent stops draining, so the worker blocks
mid-write *holding the terminal lock*; a typed key then parks the main
thread on the lock inside ``redraw``, and only then is Ctrl+C sent.
"""

from __future__ import annotations

import os
import re
import select
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires a POSIX pty",
)


def _drain(fd: int, seconds: float, stop: re.Pattern[str] | None = None) -> str:
    """Read PTY output for *seconds* (early-out when *stop* matches)."""
    out = b""
    deadline = time.time() + seconds
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        if stop is not None and stop.search(out.decode("utf-8", "ignore")):
            break
    return out.decode("utf-8", "ignore")


def test_ctrl_c_during_lock_wait_still_stops_the_worker(
    tmp_path: Path,
) -> None:
    """SIGINT while blocked on the terminal lock must abort the worker."""
    import pty

    started = tmp_path / "started"
    result = tmp_path / "result"
    kiss_home = tmp_path / ".kisshome"

    child_code = f"""
import os
os.environ["KISS_HOME"] = {str(kiss_home)!r}
os.environ["LINES"] = "24"
os.environ["COLUMNS"] = "80"
import sys, time
from pathlib import Path

started = Path({str(started)!r})
result = Path({str(result)!r})


class FloodingAgent:
    model_name = "demo-model"

    def run(self, **kwargs):
        started.write_text("1")
        try:
            for _ in range(100):
                sys.stdout.write("X" * 65536)
        except KeyboardInterrupt:
            result.write_text("interrupted")
            raise
        result.write_text("completed")
        return "summary: done\\n"


from kiss.agents.sorcar.cli_steering import run_with_steering

try:
    run_with_steering(FloodingAgent(), {{}})
    sys.stdout.write("NO_INTERRUPT\\n")
except KeyboardInterrupt:
    deadline = time.time() + 5.0
    while time.time() < deadline and not result.exists():
        time.sleep(0.05)
    text = result.read_text() if result.exists() else "missing"
    sys.stdout.write(f"\\nWORKER[{{text}}]\\n")
sys.stdout.flush()
"""

    pid, fd = pty.fork()
    if pid == 0:  # child: fresh interpreter attached to the PTY slave
        os.execvp(sys.executable, [sys.executable, "-c", child_code])
        os._exit(0)  # pragma: no cover - exec never returns

    out = ""
    try:
        deadline = time.time() + 30.0
        while time.time() < deadline and not started.exists():
            out += _drain(fd, 0.2)
        assert started.exists(), f"agent never started; output: {out!r}"
        # Stop draining: the worker fills the PTY buffer and blocks
        # mid-write while *holding the terminal lock*.
        time.sleep(1.0)
        # A typed key wakes the select loop; feed() -> redraw() then
        # parks the main thread on the lock held by the blocked worker.
        os.write(fd, b"a")
        time.sleep(0.5)
        # Ctrl+C now interrupts the lock wait, not the select call.
        os.write(fd, b"\x03")
        # Resume draining so the blocked writes can complete.
        out += _drain(fd, 25.0, stop=re.compile(r"WORKER\[\w+\]"))
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        os.waitpid(pid, 0)

    marker = re.search(r"WORKER\[(\w+)\]", out)
    assert marker, f"child never reported worker status; tail: {out[-2000:]!r}"
    assert marker.group(1) == "interrupted", (
        "Ctrl+C delivered during a terminal-lock wait aborted the session "
        "but left the agent's run() executing in the background "
        f"(worker status: {marker.group(1)!r})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

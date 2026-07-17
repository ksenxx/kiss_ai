# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt integration tests for the sorcar CLI steering session.

Reproduces a real bug in :mod:`kiss.ui.cli.cli_steering`: pressing
``Ctrl+C`` while a task runs inside the anchored steering box aborted the
*waiting* loop (``run_with_steering`` raised ``KeyboardInterrupt`` and the
REPL printed "Task interrupted"), but the daemon worker thread kept
running ``agent.run`` in the background — the agent went on printing over
the next idle prompt and spending budget after the user had aborted it.

The test drives the real interactive code path end to end on a
pseudo-terminal (no mocks): a fresh child interpreter runs
:func:`run_with_steering` with a slow agent attached to the PTY slave,
the parent sends a real ``Ctrl+C`` byte through the PTY master (the
box keeps ``ISIG`` on, so the kernel delivers ``SIGINT``), and the child
reports whether the agent's ``run`` was actually interrupted.
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
                chunk = os.read(fd, 8192)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        if stop is not None and stop.search(out.decode("utf-8", "ignore")):
            break
    return out.decode("utf-8", "ignore")


def _abort_steering_over_pty(tmp_path: Path) -> str:
    """Run a slow agent under ``run_with_steering`` on a PTY and Ctrl+C it.

    Forks a child interpreter attached to a pseudo-terminal.  The child's
    agent writes a ``started`` marker file when its ``run`` begins, then
    sleeps in short slices for ~20s; if a ``KeyboardInterrupt`` reaches it
    the agent records ``interrupted`` in a ``result`` marker file (and
    ``completed`` if the loop ever finishes).  Once ``run_with_steering``
    raises ``KeyboardInterrupt`` in the child's main thread, the child
    waits up to 15 seconds for the worker to be interrupted and prints a
    ``WORKER[...]`` marker with the result-file contents (or ``missing``).

    Args:
        tmp_path: Directory for the marker files and isolated KISS_HOME.

    Returns:
        The full decoded PTY output of the child (contains the
        ``WORKER[...]`` marker).
    """
    from kiss.tests.agents.sorcar._pty_helper import pty_spawn

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


class SlowAgent:
    model_name = "demo-model"

    def run(self, **kwargs):
        started.write_text("1")
        try:
            for _ in range(400):
                time.sleep(0.05)
        except KeyboardInterrupt:
            result.write_text("interrupted")
            raise
        result.write_text("completed")
        return "summary: done\\n"


from kiss.ui.cli.cli_steering import run_with_steering

try:
    run_with_steering(SlowAgent(), {{}})
    sys.stdout.write("NO_INTERRUPT\\n")
except KeyboardInterrupt:
    deadline = time.time() + 15.0
    while time.time() < deadline and not result.exists():
        time.sleep(0.05)
    text = result.read_text() if result.exists() else "missing"
    sys.stdout.write(f"WORKER[{{text}}]\\n")
sys.stdout.flush()
"""

    pid, fd = pty_spawn([sys.executable, "-c", child_code])

    out = ""
    try:
        # Wait for the agent's run() to start inside the steering session.
        deadline = time.time() + 30.0
        while time.time() < deadline and not started.exists():
            out += _drain(fd, 0.2)
        assert started.exists(), f"agent never started; output: {out!r}"
        # The main thread is now in the steering select loop: send a real
        # Ctrl+C through the PTY (ISIG is kept on, so SIGINT is raised).
        os.write(fd, b"\x03")
        out += _drain(fd, 20.0, stop=re.compile(r"WORKER\[\w+\]"))
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        os.waitpid(pid, 0)
    return out


def test_ctrl_c_abort_actually_stops_the_running_agent(tmp_path: Path) -> None:
    """Ctrl+C in the steering box must interrupt the agent's run.

    Bug: ``SteeringSession.run`` raised ``KeyboardInterrupt`` to its
    caller on abort but left the daemon worker thread running
    ``agent.run`` — the "aborted" task kept executing (and spending
    budget) in the background.  After the fix the worker receives a
    ``KeyboardInterrupt`` injection, so the agent observes the abort.
    """
    out = _abort_steering_over_pty(tmp_path)
    marker = re.search(r"WORKER\[(\w+)\]", out)
    assert marker, f"child never reported worker status; output: {out!r}"
    assert marker.group(1) == "interrupted", (
        "Ctrl+C aborted the steering session but the agent's run() was "
        f"left running in the background (worker status: {marker.group(1)!r})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

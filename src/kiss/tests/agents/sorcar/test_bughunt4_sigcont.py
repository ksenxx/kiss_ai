# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: the steering box must restore itself after SIGCONT.

Bug: :class:`kiss.agents.sorcar.cli_steering._InputBox` installed no
``SIGCONT`` handler.  When the user suspends the CLI (Ctrl+Z) and
resumes it (``fg``), the raw terminal mode, bracketed-paste mode and
the DECSTBM scroll region were never re-asserted and the box was not
redrawn — the screen stayed corrupted (shell prompt over the box, agent
output scrolling over the box rows) until the next keypress, resize or
agent write.

The test drives the real interactive path end to end on a PTY: a child
interpreter runs ``run_with_steering`` with a quiet slow agent, the
parent delivers ``SIGCONT`` and asserts the child re-anchors the scroll
region and redraws the box.
"""

from __future__ import annotations

import os
import select
import signal
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires a POSIX pty",
)


def _drain(fd: int, seconds: float) -> str:
    """Read whatever the child writes to the PTY for *seconds*."""
    out = b""
    deadline = time.time() + seconds
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 8192)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    return out.decode("utf-8", "ignore")


def test_sigcont_reanchors_scroll_region_and_redraws_box(
    tmp_path: Path,
) -> None:
    """After SIGCONT the box must re-emit DECSTBM and redraw itself."""
    import pty

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


class QuietSlowAgent:
    model_name = "demo-model"

    def run(self, **kwargs):
        started.write_text("1")
        try:
            for _ in range(400):
                time.sleep(0.05)
        except KeyboardInterrupt:
            raise
        return "summary: done\\n"


from kiss.agents.sorcar.cli_steering import run_with_steering

try:
    run_with_steering(QuietSlowAgent(), {{}})
except KeyboardInterrupt:
    pass
sys.stdout.write("CHILD_DONE\\n")
sys.stdout.flush()
"""

    pid, fd = pty.fork()
    if pid == 0:  # child: fresh interpreter attached to the PTY slave
        os.execvp(sys.executable, [sys.executable, "-c", child_code])
        os._exit(0)  # pragma: no cover - exec never returns

    try:
        deadline = time.time() + 30.0
        out = ""
        while time.time() < deadline and not started.exists():
            out += _drain(fd, 0.2)
        assert started.exists(), f"agent never started; output: {out!r}"
        # Let the box finish drawing, then capture only post-SIGCONT output.
        out += _drain(fd, 1.0)

        os.kill(pid, signal.SIGCONT)
        after = _drain(fd, 2.0)

        region = f"\x1b[1;{24 - 3}r"  # rows=24, _BOX_H=3
        assert region in after, (
            "SIGCONT did not re-anchor the scroll region / redraw the "
            f"steering box; post-resume output: {after!r}"
        )
        assert "╭" in after, "SIGCONT did not redraw the box top border"
    finally:
        try:
            os.write(fd, b"\x03")  # abort the task so the child exits
        except OSError:
            pass
        _drain(fd, 5.0)
        try:
            os.close(fd)
        except OSError:
            pass
        os.waitpid(pid, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

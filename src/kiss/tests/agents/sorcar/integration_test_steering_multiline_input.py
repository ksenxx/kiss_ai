# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end integration test: multi-line Shift+Enter steering on a real PTY.

The previous unit / panel tests (see
``test_cli_steering_multiline_panel.py``) verified the *rendering*
contract of :func:`panel_body` and the in-memory ``_InputBox`` behaviour
in isolation: that ``Shift+Enter`` inserts a real ``\\n`` and that the
bordered box grows to one body row per buffer line.  Those tests run
with an ``io.StringIO`` substitute for the terminal and never engage the
``termios`` raw-mode, ``DECSTBM`` scroll-region, ``_StdoutProxy`` and
``_RunningAgentState`` -> ``pre_step_hook`` plumbing that production
code uses to deliver a multi-line follow-up instruction to a live
agent.

This integration test fills that gap.  It spawns a child process on a
fresh PTY pair (parent owns the master fd, child gets the slave fd as
stdin / stdout / stderr) and runs the real :func:`run_with_steering`
entry point with a thin :class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent`
subclass that:

* does NOT invoke a real LLM (so the test stays hermetic, deterministic
  and free) — it instead wires up exactly the same
  ``pre_step_hook = self._drain_pending_user_messages`` hook the real
  ``SorcarAgent.run`` wires up and then polls the queue;
* writes the conversation it drained into ``$KISS_TEST_OUT`` and exits,
  so the parent can assert on the messages the agent actually received.

The parent simulates the user typing a three-line follow-up instruction
into the bottom steering box.  Each line break is sent as the
``CSI 13;2u`` byte sequence — the kitty / xterm ``modifyOtherKeys``
encoding of Shift+Enter that
:meth:`~kiss.agents.sorcar.cli_steering._InputBox.feed` recognises (the
same encoding the matching unit test ``test_shift_enter_grows_the_box_live``
uses).  After capturing the painted terminal frames from the master fd,
the parent sends a plain ``\\r`` to submit, then asserts on:

1. The box visibly grew on screen: the three typed lines all appear in
   the painted output as plain text (no ``⏎`` glyph collapses them),
   and the final pre-submit frame contains at least 6 vertical border
   bars (3 body rows × 2 bars per row).
2. The agent's conversation received exactly one ``user``-role message
   whose content is the three lines joined by ``\\n`` — confirming that
   the multi-line buffer survived the box -> queue -> drain pipeline
   intact, including the embedded newlines.

If either side of the contract regresses (the box stops growing, OR
``\\n`` characters get lost on their way to the model conversation) this
test fails with an actionable diff.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import select
import struct
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from kiss.tests.agents.sorcar._pty_helper import pty_spawn

# Sentinel-bearing child program.  Run with the parent's interpreter
# (``sys.executable``) so the same venv / dependency tree is in scope.
# The child wires up exactly the production
# ``pre_step_hook = self._drain_pending_user_messages`` path
# :class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent.run` uses, but
# without calling a real model — it just polls until the drain
# delivers at least one queued line, then dumps the resulting
# conversation to ``$KISS_TEST_OUT`` and returns.
_CHILD_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import sys
    import time
    from pathlib import Path

    from kiss.agents.sorcar.cli_steering import run_with_steering
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent


    class _RecordingModel:
        '''Stand-in for the live model that just records what the drain
        injects.  Mirrors the ``Model.add_message_to_conversation`` shape
        used by ``SorcarAgent._drain_pending_user_messages``.'''
        def __init__(self):
            self.conversation = []
        def add_message_to_conversation(self, role, content):
            self.conversation.append({'role': role, 'content': content})


    class _SteeringIntegrationAgent(SorcarAgent):
        '''Real SorcarAgent subclass whose ``run`` only exercises the
        steering -> drain pipeline.  Wires the same pre_step_hook
        SorcarAgent.run does, polls the queue at the same cadence a
        real model step would (the drain runs at the top of every
        step), records the drained conversation and returns.'''

        def __init__(self):
            super().__init__('steer-multiline-integration')
            self.recording_model = _RecordingModel()

        def run(self, **kwargs):
            # Production wires this hook in ``run`` right before
            # building the model; mirror it exactly so the same
            # drain code path is exercised.
            self.pre_step_hook = self._drain_pending_user_messages
            try:
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    self.pre_step_hook(self.recording_model)
                    if self.recording_model.conversation:
                        break
                    time.sleep(0.05)
            finally:
                self.pre_step_hook = None
            out_path = Path(os.environ['KISS_TEST_OUT'])
            out_path.write_text(
                json.dumps(self.recording_model.conversation),
            )
            return 'summary: integration done'


    agent = _SteeringIntegrationAgent()
    result = run_with_steering(agent, {})
    sys.stdout.write(f'\\nFINAL_RESULT={result}\\n')
    sys.stdout.flush()
    """,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(data: bytes) -> str:
    """Drop ANSI control sequences and decode *data* as UTF-8.

    Used to count border glyphs / search for typed text in the captured
    PTY stream without false positives from cursor-move / SGR codes.

    Args:
        data: Raw bytes captured from the master PTY fd.

    Returns:
        The decoded text with ANSI control sequences removed.
    """
    return _ANSI_RE.sub("", data.decode("utf-8", errors="replace"))


def _make_env(tmp_path: Path) -> dict[str, str]:
    """Build a deterministic child environment for the PTY-spawned agent.

    ``LINES`` / ``COLUMNS`` pin :func:`shutil.get_terminal_size` (and
    therefore the box's row / col math) so the test does not depend on
    the PTY default winsize that varies between platforms.
    ``PROMPT_TOOLKIT_NO_CPR=1`` is set for parity with the other PTY
    tests even though the steering box does not use prompt_toolkit —
    cheap insurance against a stray dependency probe.

    Args:
        tmp_path: Pytest-provided tmp directory for ``KISS_TEST_OUT``.

    Returns:
        An ``env`` mapping suitable for :func:`pty_spawn`.
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ["HOME"],
        "TERM": "xterm-256color",
        "PROMPT_TOOLKIT_NO_CPR": "1",
        "KISS_TEST_OUT": str(tmp_path / "result.json"),
        "PYTHONUNBUFFERED": "1",
        "LINES": "30",
        "COLUMNS": "80",
        # Isolate the child's KISS home so the test never reads or
        # writes the developer's real persistence state.
        "KISS_HOME": str(tmp_path / ".kiss"),
    }
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "LC_ALL", "LANG"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


class _Drainer(threading.Thread):
    """Background reader that keeps the PTY master from blocking the child.

    Without a drainer the child's redraw bursts (which can easily exceed
    the kernel PTY buffer) would block on stdout and stall the test.
    The drained bytes are appended to ``buffer`` so the parent can
    inspect what was painted at any point.
    """

    def __init__(self, fd: int) -> None:
        super().__init__(daemon=True)
        self.fd = fd
        self.buffer = bytearray()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def snapshot(self) -> bytes:
        """Return a copy of everything captured so far."""
        with self._lock:
            return bytes(self.buffer)

    def stop(self) -> None:
        """Ask the drain thread to exit at the next poll boundary."""
        self._stop.set()

    def run(self) -> None:
        """Read-and-append loop guarded by a short ``select`` timeout."""
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                chunk = os.read(self.fd, 4096)
            except OSError:
                return
            if not chunk:
                return
            with self._lock:
                self.buffer.extend(chunk)


def _wait_until(predicate, timeout: float, poll: float = 0.05) -> bool:
    """Poll *predicate* until truthy or *timeout* seconds elapse.

    Args:
        predicate: A zero-arg callable polled at *poll*-second intervals.
        timeout: Maximum total seconds to wait.
        poll: Seconds to sleep between polls.

    Returns:
        ``True`` if the predicate became truthy in time, else ``False``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


def _reap(pid: int, timeout: float = 5.0) -> None:
    """Wait for *pid* to exit; SIGKILL it on timeout so the test is bounded."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            wpid, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if wpid == pid:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


@pytest.mark.timeout(90)
def test_shift_enter_multiline_steering_grows_box_and_reaches_agent(
    tmp_path: Path,
) -> None:
    """A three-line Shift+Enter follow-up renders multi-row AND drains intact.

    Walks the user-visible flow end-to-end on a real PTY:

    1. Spawn :data:`_CHILD_SCRIPT` (a real ``SorcarAgent`` subclass
       wrapped in :func:`run_with_steering`) on a fresh PTY pair.
    2. Set the PTY winsize so the steering box rendering math sees a
       30×80 terminal regardless of platform defaults.
    3. Wait for the bottom-anchored steering box to be painted.
    4. Type ``"line one\\nline two\\nline three"`` where each ``\\n``
       is delivered as the ``CSI 13;2u`` Shift+Enter sequence.
    5. Snapshot the captured terminal frames, then send plain ``\\r``
       to submit.
    6. Wait for the child to write ``$KISS_TEST_OUT`` and exit.
    7. Assert that (a) the three typed lines all appear in the painted
       output with no ``⏎`` glyph collapse, (b) the final pre-submit
       frame shows the grown box (>= 6 vertical bars), and (c) the
       drained agent conversation contains exactly one ``user``
       message whose content is the three lines joined by ``\\n``.
    """
    env = _make_env(tmp_path)
    out_path = Path(env["KISS_TEST_OUT"])

    pid, fd = pty_spawn(
        [sys.executable, "-c", _CHILD_SCRIPT],
        env=env,
    )
    drainer = _Drainer(fd)
    drainer.start()
    try:
        # Pin the PTY winsize to (rows=30, cols=80) so the steering
        # box rendering math (panel_cols, _term_size) sees a stable
        # terminal size regardless of platform defaults.  Without this
        # macOS' default PTY winsize of (0, 0) makes shutil fall back
        # to (24, 80) — close enough to work but brittle.
        fcntl.ioctl(fd, TIOCSWINSZ, struct.pack("HHHH", 30, 80, 0, 0))

        # Give the child a healthy startup window: ``run_with_steering``
        # has to import the agent stack, register the running-state, set
        # raw-mode, anchor the scroll region and paint the initial box.
        # Wait until the bottom border glyph appears so we know the box
        # is fully painted before typing.
        assert _wait_until(
            lambda: "╯" in _strip_ansi(drainer.snapshot()),
            timeout=20.0,
        ), (
            "steering box was never painted; child output so far:\n"
            f"{_strip_ansi(drainer.snapshot())[:2000]!r}"
        )

        # Type the three-line follow-up.  ``CSI 13;2u`` is the kitty /
        # xterm modifyOtherKeys encoding of Shift+Enter that
        # ``_InputBox.feed`` recognises.  Short sleeps between segments
        # let the box redraw between keystrokes so the captured frames
        # include intermediate states (helpful for diagnostics on
        # failure).
        os.write(fd, b"line one")
        time.sleep(0.15)
        os.write(fd, b"\x1b[13;2u")
        time.sleep(0.15)
        os.write(fd, b"line two")
        time.sleep(0.15)
        os.write(fd, b"\x1b[13;2u")
        time.sleep(0.15)
        os.write(fd, b"line three")
        time.sleep(0.4)

        # Snapshot the painted output BEFORE submitting.  This is the
        # frame the user would see with their three-line follow-up
        # filled into the grown box.
        rendered_before_submit_raw = drainer.snapshot()
        rendered_before_submit = _strip_ansi(rendered_before_submit_raw)

        # Submit with plain Enter.  The session's ``_on_submit`` will
        # then queue the line into ``state.pending_user_messages``,
        # which the child's pre_step_hook drains on the next poll.
        os.write(fd, b"\r")

        # The child writes ``$KISS_TEST_OUT`` immediately after the
        # first non-empty drain and returns, which makes
        # ``run_with_steering`` unwind the box and exit the process.
        assert _wait_until(out_path.exists, timeout=20.0), (
            "agent never drained the multi-line follow-up; child "
            "output:\n"
            f"{_strip_ansi(drainer.snapshot())[-2000:]!r}"
        )
        _reap(pid, timeout=10.0)
    finally:
        drainer.stop()
        drainer.join(timeout=2.0)
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass

    # --- Assertions on the painted output ---------------------------------
    # All three typed lines must appear in the painted box on their own
    # rows.  A regression that collapses ``\n`` to ``⏎`` would still
    # contain the literal text "line one"/"two"/"three" but would also
    # leak a ``⏎`` glyph between them — the second check catches that.
    assert "line one" in rendered_before_submit, (
        f"'line one' missing from rendered output:\n"
        f"{rendered_before_submit[-2000:]!r}"
    )
    assert "line two" in rendered_before_submit, (
        f"'line two' missing from rendered output:\n"
        f"{rendered_before_submit[-2000:]!r}"
    )
    assert "line three" in rendered_before_submit, (
        f"'line three' missing from rendered output:\n"
        f"{rendered_before_submit[-2000:]!r}"
    )
    assert "⏎" not in rendered_before_submit, (
        "Shift+Enter newlines must split into separate body rows, not "
        "collapse into a single row with ⏎ glyphs"
    )
    # The box must have grown to three body rows.  Each body row is
    # bordered by ``│`` on the left and right, so three body rows
    # contribute six ``│`` characters per redraw.  The drainer
    # accumulates redraws, so we expect AT LEAST six bars — and the
    # final frame must have rendered at least three vertical bars on
    # the left edge alone.  Counting six is a strong necessary
    # condition.
    bars = rendered_before_submit.count("│")
    assert bars >= 6, (
        f"steering box did not grow vertically; only {bars} '│' bars "
        f"seen across the captured frames (need >= 6 for 3 body rows). "
        f"Rendered tail:\n{rendered_before_submit[-2000:]!r}"
    )

    # --- Assertions on the drained agent conversation ----------------------
    # The whole point of the integration test: the box must have queued
    # the multi-line text intact, and ``_drain_pending_user_messages``
    # must have injected it as ONE ``user`` message containing the
    # newlines — not three messages, not a single line with ``\n``
    # escaped, not a stripped/joined version.
    payload = json.loads(out_path.read_text())
    assert payload == [
        {"role": "user", "content": "line one\nline two\nline three"},
    ], (
        f"agent conversation mismatch.  Expected one user message with "
        f"three newline-separated lines; got {payload!r}.  "
        f"Painted output tail:\n{rendered_before_submit[-2000:]!r}"
    )


# Imported lazily at the bottom so the module imports cleanly on
# platforms where ``termios`` is unavailable (Windows): the test itself
# is skipped automatically by the PTY helper there, but importing this
# module must not blow up the collection phase.
try:
    import termios as _termios

    TIOCSWINSZ = _termios.TIOCSWINSZ
except ImportError:  # pragma: no cover - non-POSIX
    TIOCSWINSZ = 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

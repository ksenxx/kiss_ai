# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end reproduction of the Shift+Enter multi-line submission bug.

Root cause of the bug the user reported ("as soon as Shift+Enter is
pressed, the text is sent to the agent"):

* Most terminals collapse Shift+Enter to a bare ``\\r`` byte — the same
  bytes as a plain Enter — **unless** the application opts into an
  extended-keyboard protocol on startup.
* The two portable opt-in sequences are:

  - ``ESC[>4;2m`` — xterm modifyOtherKeys level 2 (Shift+Enter then
    emits ``ESC[27;2;13~``).
  - ``ESC[>1u``   — Kitty keyboard protocol push flag 1 / disambiguate
    escape codes (Shift+Enter then emits ``ESC[13;2u``).

* Both encodings are already recognised by
  :data:`~kiss.agents.sorcar.cli_steering._NEWLINE_AFTER_ESC` and
  handled by :meth:`~kiss.agents.sorcar.cli_steering._InputBox.feed`.
  What was missing was writing the enable sequences on entry and the
  matching disable sequences on exit.

These tests pin the fix in three complementary ways:

1. A direct assertion that :meth:`_InputBox.start` emits
   ``ESC[>4;2m`` and ``ESC[>1u``, and :meth:`_InputBox.stop` emits the
   matching ``ESC[>4;0m`` and ``ESC[<u``.
2. A ``SIGCONT`` resume path assertion that re-arms the modes after a
   Ctrl+Z / ``fg`` cycle.
3. A tmux-driven end-to-end test (skipped unless a real ``tmux``
   binary and a ``pty`` are available) which launches a tiny Python
   driver around :class:`_InputBox` inside a tmux session with
   ``extended-keys always`` + ``extkeys`` terminal feature.  Under
   that configuration tmux emits the modifyOtherKeys=2 form for
   ``S-Enter``, and the driver must record ONE multi-line submission
   for a three-line prompt separated by two ``S-Enter`` presses.
"""

from __future__ import annotations

import io
import os
import pty
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.cli_steering import _InputBox

# ---------------------------------------------------------------------------
# 1. start()/stop() emit the enable / disable CSI sequences
# ---------------------------------------------------------------------------


class TestStartEmitsEnableSequences:
    """The core reproduction: without these bytes, terminals collapse
    Shift+Enter to a bare ``\\r`` and the steering box submits.

    Historical regression: the previous fix added the parsing table
    but never had the box opt into either extended-keyboard protocol,
    so real terminals never emitted the distinct encodings.
    """

    def _run_start_and_stop(self) -> tuple[str, str]:
        """Return the output emitted during ``start()`` and then
        during ``stop()`` in order to assert on each phase separately.
        """
        master, slave = pty.openpty()
        stdin_file = os.fdopen(slave, "r", closefd=False)
        orig_stdin = sys.stdin
        sys.stdin = stdin_file
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        try:
            box.start()
            start_output = out.getvalue()
            out.truncate(0)
            out.seek(0)
            box.stop()
            stop_output = out.getvalue()
        finally:
            sys.stdin = orig_stdin
            stdin_file.close()
            os.close(master)
            # slave was closed via stdin_file's underlying fd; do not
            # double-close if closefd=False left it open.
            try:
                os.close(slave)
            except OSError:
                pass
        return start_output, stop_output

    def test_start_emits_modify_other_keys_level_2(self) -> None:
        start_out, _ = self._run_start_and_stop()
        assert "\x1b[>4;2m" in start_out, (
            "start() must opt into xterm modifyOtherKeys=2 so "
            "Shift/Ctrl/Alt+Enter emit ESC[27;<m>;13~ instead of "
            "collapsing to a bare CR"
        )

    def test_start_emits_kitty_keyboard_push_flag_1(self) -> None:
        start_out, _ = self._run_start_and_stop()
        assert "\x1b[>1u" in start_out, (
            "start() must push Kitty keyboard flag 1 so Shift+Enter "
            "emits ESC[13;<m>u under kitty / WezTerm / ghostty / foot"
        )

    def test_stop_emits_modify_other_keys_level_0(self) -> None:
        _, stop_out = self._run_start_and_stop()
        assert "\x1b[>4;0m" in stop_out, (
            "stop() must restore modifyOtherKeys to level 0 so a "
            "subsequently-run child process does not inherit our mode"
        )

    def test_stop_emits_kitty_keyboard_pop(self) -> None:
        _, stop_out = self._run_start_and_stop()
        assert "\x1b[<u" in stop_out, (
            "stop() must pop the Kitty keyboard flag we pushed in "
            "start() so we don't leak a stack entry into the shell"
        )

    def test_enable_written_after_bracketed_paste_enable(self) -> None:
        """Sanity: the enable sequences must be emitted *inside*
        start(), not before bracketed-paste enable (which would look
        like the app never turned itself on and off in the right
        order).
        """
        start_out, _ = self._run_start_and_stop()
        paste_on = start_out.find("\x1b[?2004h")
        modify_on = start_out.find("\x1b[>4;2m")
        kitty_on = start_out.find("\x1b[>1u")
        assert paste_on >= 0 and modify_on > paste_on and kitty_on > paste_on

    def test_disable_written_after_bracketed_paste_disable(self) -> None:
        _, stop_out = self._run_start_and_stop()
        paste_off = stop_out.find("\x1b[?2004l")
        modify_off = stop_out.find("\x1b[>4;0m")
        kitty_off = stop_out.find("\x1b[<u")
        assert paste_off >= 0 and modify_off > paste_off and kitty_off > paste_off


# ---------------------------------------------------------------------------
# 2. Resume from Ctrl+Z re-arms the enable sequences
# ---------------------------------------------------------------------------


class TestSigcontReEnablesExtendedKeys:
    """A shell suspend (``Ctrl+Z``) followed by ``fg`` hands the
    terminal back to us with modifyOtherKeys popped off.  ``_on_sigcont``
    must re-arm both extended-keyboard protocols alongside the
    bracketed-paste and scroll-region rearming already there.
    """

    def test_sigcont_re_pushes_modify_and_kitty(self) -> None:
        master, slave = pty.openpty()
        stdin_file = os.fdopen(slave, "r", closefd=False)
        orig_stdin = sys.stdin
        sys.stdin = stdin_file
        out = io.StringIO()
        box = _InputBox(threading.RLock(), out)
        try:
            box.start()
            out.truncate(0)
            out.seek(0)
            box._on_sigcont(0, None)
            resumed = out.getvalue()
            assert "\x1b[>4;2m" in resumed
            assert "\x1b[>1u" in resumed
        finally:
            if box._active:
                box.stop()
            sys.stdin = orig_stdin
            stdin_file.close()
            os.close(master)
            try:
                os.close(slave)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 3. Realistic tmux-driven end-to-end reproduction / regression check
# ---------------------------------------------------------------------------


_TMUX = shutil.which("tmux")


def _has_tmux() -> bool:
    return _TMUX is not None and os.name == "posix"


_DRIVER_SRC = textwrap.dedent(
    """
    import json
    import sys
    import threading
    from pathlib import Path

    from kiss.agents.sorcar.cli_steering import _InputBox

    OUT_FILE = Path(sys.argv[1])
    OUT_FILE.write_text("[]")

    submissions: list[str] = []

    def on_submit(text: str) -> None:
        submissions.append(text)
        OUT_FILE.write_text(json.dumps(submissions))
        # Exit after the first submission — the test drives exactly
        # one multi-line submit.
        raise SystemExit(0)

    def on_abort() -> None:
        raise SystemExit(1)

    box = _InputBox(threading.RLock(), sys.stdout)
    box.start()
    try:
        import os, select
        fd = sys.stdin.fileno()
        while True:
            r, _, _ = select.select([fd], [], [], 5.0)
            if not r:
                break
            data = os.read(fd, 4096)
            if not data:
                break
            box.feed(data, on_submit, on_abort)
    finally:
        try:
            box.stop()
        except Exception:
            pass
    """
).strip()


@pytest.mark.skipif(not _has_tmux(), reason="tmux binary required")
class TestTmuxShiftEnterMultiLine:
    """End-to-end: with ``extended-keys always`` tmux emits the
    modifyOtherKeys=2 form for S-Enter, and the steering box records
    ONE multi-line submission for a three-line prompt.

    This test intentionally does NOT launch the real ``sorcar`` CLI —
    that would require a live LLM backend which is out of scope for
    unit tests.  Instead it drives the identical parsing / termios /
    submit path inside a tiny Python driver.  A full ``sorcar`` run
    is done manually as part of the fix verification.
    """

    def _tmux_config(self, tmp_path: Path) -> Path:
        conf = tmp_path / "tmux.conf"
        conf.write_text(
            "set -s extended-keys always\n"
            "set -as terminal-features ',*:extkeys'\n"
        )
        return conf

    def _run_driver(
        self, tmp_path: Path, key_between_lines: str
    ) -> list[str]:
        # Narrow ``_TMUX`` from ``str | None`` to ``str`` for the type
        # checker; the ``@pytest.mark.skipif(not _has_tmux())`` on the
        # class already guarantees this at runtime.
        assert _TMUX is not None
        tmux: str = _TMUX
        conf = self._tmux_config(tmp_path)
        driver = tmp_path / "driver.py"
        driver.write_text(_DRIVER_SRC)
        out_json = tmp_path / "submissions.json"
        session = f"stest{os.getpid()}"
        # Kill any stale session with the same name first.
        subprocess.run(
            [tmux, "-f", str(conf), "kill-session", "-t", session],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cmd_line = (
            f"{sys.executable} -m coverage run --parallel-mode "
            f"{driver} {out_json}"
            if os.environ.get("COVERAGE_RUN")
            else f"{sys.executable} {driver} {out_json}"
        )
        subprocess.run(
            [
                tmux,
                "-f",
                str(conf),
                "new-session",
                "-d",
                "-s",
                session,
                "-x",
                "220",
                "-y",
                "50",
                cmd_line,
            ],
            check=True,
        )
        try:
            # Wait for the driver to draw its initial box (small sleep).
            time.sleep(0.8)
            subprocess.run(
                [tmux, "send-keys", "-t", session, "line1"], check=True
            )
            time.sleep(0.15)
            subprocess.run(
                [tmux, "send-keys", "-t", session, key_between_lines],
                check=True,
            )
            time.sleep(0.15)
            subprocess.run(
                [tmux, "send-keys", "-t", session, "line2"], check=True
            )
            time.sleep(0.15)
            subprocess.run(
                [tmux, "send-keys", "-t", session, key_between_lines],
                check=True,
            )
            time.sleep(0.15)
            subprocess.run(
                [tmux, "send-keys", "-t", session, "line3"], check=True
            )
            time.sleep(0.15)
            # Plain Enter submits.
            subprocess.run(
                [tmux, "send-keys", "-t", session, "Enter"], check=True
            )
            # Wait for the driver to record its submission and exit.
            deadline = time.time() + 5.0
            data: list[str] = []
            while time.time() < deadline:
                try:
                    data = __import__("json").loads(
                        out_json.read_text() or "[]"
                    )
                except Exception:
                    data = []
                if data:
                    break
                time.sleep(0.1)
            return data
        finally:
            subprocess.run(
                [tmux, "kill-session", "-t", session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def test_shift_enter_between_lines_yields_single_multiline_submit(
        self, tmp_path: Path
    ) -> None:
        submissions = self._run_driver(tmp_path, "S-Enter")
        assert submissions == ["line1\nline2\nline3"], (
            "Shift+Enter must insert newlines and only a plain Enter "
            f"should submit; got submissions={submissions!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

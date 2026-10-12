# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A KISS process never blocks on a Windows error dialog when an executable is corrupt.

A process started from a hidden window (the kiss-web daemon launched by
the editor, a service, the full-suite driver) inherits error mode 0,
under which ``CreateProcess`` of a file that is not a valid executable
shows a modal "not compatible with the version of Windows" dialog and
blocks until it is dismissed.  :func:`suppress_error_dialogs` opts the
process out, the way a console shell already has, so the call fails
with ``OSError`` at once -- which is what ``heif_to_jpeg`` and every
other converter/CLI launch rely on.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from kiss.core.processes import IS_WINDOWS, suppress_error_dialogs

# The child mimics a hidden-window launch (error mode 0), opts out the
# way the daemon does, then starts a garbage executable and reports how
# long the failed start took.
_CHILD = textwrap.dedent(
    """\
    import msvcrt, subprocess, sys, time
    from kiss.core.processes import suppress_error_dialogs
    msvcrt.SetErrorMode(0)
    suppress_error_dialogs()
    started = time.monotonic()
    try:
        subprocess.run([sys.argv[1]], capture_output=True)
    except OSError as e:
        print("OSERROR", e.winerror, round(time.monotonic() - started, 3))
    """
)


@pytest.mark.skipif(not IS_WINDOWS, reason="the dialog is a Windows behaviour")
def test_a_corrupt_executable_fails_at_once_instead_of_showing_a_dialog(tmp_path: Path) -> None:
    garbage = tmp_path / "convert.exe"
    garbage.write_bytes(b"\x7fELF not a program")
    child = tmp_path / "child.py"
    child.write_text(_CHILD, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(child), str(garbage)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.stdout.startswith("OSERROR "), (proc.stdout, proc.stderr)
    _tag, _winerror, seconds = proc.stdout.split()
    assert float(seconds) < 5.0, proc.stdout


@pytest.mark.skipif(IS_WINDOWS, reason="no-op branch is the POSIX one")
def test_is_a_no_op_elsewhere() -> None:
    suppress_error_dialogs()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``install.sh`` MUST survive losing its terminal.

Run from the webapp's Terminal tab, ``install.sh`` loses its terminal
when step [5/5] restarts the kiss-web daemon that hosts it: the PTY
master closes, the tab's shell (the session leader) exits, and the PTY
is revoked, so every fd still on it fails (EBADF on macOS, EIO on
Linux).  The ``kiss-hup-trap`` block re-routes stdout/stderr to the
log and must also move stdin and the saved terminal fd 3 off the dead
PTY: otherwise ``kiss-app`` (a Python program) dies at start-up with
``init_sys_streams ... Bad file descriptor`` and the password box's
``echo >&3`` fails, which ``set -e`` turns into a failed install.

Each test extracts the block verbatim and runs it, under ``bash -eu``,
inside a real PTY whose session leader (a Python process with the
default SIGHUP disposition, standing in for the Terminal tab's shell)
exits when the master closes, exactly as the daemon restart does.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from kiss.tests.conftest import posix_only
from kiss.tests.test_install_open_webapp import install_sh_block, install_sh_function

pytestmark = posix_only("runs the HUP trap of install.sh under bash in a PTY")


def _hup_script(log: Path, outcome: Path) -> str:
    # After the HUP: a Python child must start with the script's stdin,
    # fd 3 must take a write, and the password box must land somewhere
    # that is not the log.
    return "\n".join(
        [
            "set -eu",
            f"LOG_FILE={log}",
            install_sh_block("kiss-hup-trap"),
            install_sh_function("print_box"),
            install_sh_function("print_password_box"),
            "exec 3>&1",
            "echo ready",
            "i=0",
            # The hang-up also kills the sleep in flight (exit 129).
            'while [ "$TERMINAL_GONE" = 0 ] && [ "$i" -lt 100 ]; do',
            "    sleep 0.1 || true; i=$((i + 1))",
            "done",
            f'"{sys.executable}" -c \'import sys; print("stdin read:", repr(sys.stdin.read()))\'',
            'echo "fd3 write ok" >&3',
            "print_password_box s3cr3t-pw",
            f'echo "exit-status 0" > "{outcome}"',
        ]
    )


def _run_in_pty_and_hang_up(script: Path, outcome: Path) -> None:
    # The session leader is a Python process running the script as a
    # child; its default SIGHUP disposition makes it exit when the master
    # closes, which revokes the PTY under the script.
    leader = f"import subprocess; subprocess.run(['bash', {str(script)!r}])"
    import pty  # POSIX only; imported here so collection succeeds where the module is skipped

    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        # An ignored SIGHUP survives exec (pytest under ``nohup`` ignores it):
        # the leader would then outlive the hang-up and bash, which cannot
        # trap a signal ignored at entry, would never see it.
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        os.execv(sys.executable, [sys.executable, "-c", leader])
    seen = b""
    deadline = time.time() + 20
    while b"ready" not in seen and time.time() < deadline:
        readable, _, _ = select.select([master], [], [], 0.5)
        if readable:
            seen += os.read(master, 4096)
    assert b"ready" in seen, seen
    os.close(master)
    os.waitpid(pid, 0)
    deadline = time.time() + 20
    while not outcome.exists() and time.time() < deadline:
        time.sleep(0.1)


def test_children_and_fd3_survive_the_terminal_closing(tmp_path: Path) -> None:
    log = tmp_path / "install.log"
    outcome = tmp_path / "outcome"
    script = tmp_path / "script.sh"
    script.write_text(_hup_script(log, outcome), encoding="utf-8")

    _run_in_pty_and_hang_up(script, outcome)

    assert outcome.read_text(encoding="utf-8") == "exit-status 0\n"
    text = log.read_text(encoding="utf-8")
    assert text.count("Controlling terminal closed (SIGHUP)") == 1, text
    assert "stdin read: ''" in text, text
    assert "Bad file descriptor" not in text, text
    assert "write error" not in text, text
    assert "fd3 write ok" not in text, text  # fd 3 went to /dev/null, not the log
    assert "s3cr3t-pw" not in text, text
    assert "password was not shown because the terminal closed" in text, text


def test_password_box_goes_to_the_terminal_while_it_is_there(tmp_path: Path) -> None:
    log = tmp_path / "install.log"
    script = "\n".join(
        [
            "set -eu",
            f"LOG_FILE={log}",
            install_sh_block("kiss-hup-trap"),
            install_sh_function("print_box"),
            install_sh_function("print_password_box"),
            "exec 3>&1",
            f'exec >>"{log}" 2>&1',
            "print_password_box s3cr3t-pw",
        ]
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "Remote access password:  s3cr3t-pw" in result.stdout
    assert "s3cr3t-pw" not in log.read_text(encoding="utf-8")

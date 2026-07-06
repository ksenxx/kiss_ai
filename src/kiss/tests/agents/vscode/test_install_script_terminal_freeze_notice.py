# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: ``install.sh`` must warn the user BEFORE step [6/6] that the
terminal may freeze, and the install must still complete when the terminal
dies mid-step.

Background — the bug
====================
A user ran ``./install.sh`` from a VS Code integrated terminal.  The last
output they ever saw was::

    >>> [6/6] Installing VS Code extension...
    ...
       Extension installed into VS Code
    kiss-web is idle (count=0) at /Users/ksen/.kiss/sorcar.sock; safe to kill.
       Stopping old kiss-web daemon (PIDs: 64585)...

and the terminal then appeared "stuck" forever — no further output, no
prompt.  ``~/.kiss/install.log`` proves the install actually COMPLETED
("=== Source bootstrap complete ===" a few seconds later): the
``--install-extension --force`` in step [6/6] makes VS Code detect the
on-disk extension update and reload, which disposes / stops rendering the
very terminal the script is printing to.  The new-session (setsid)
detachment at the top of ``install.sh`` keeps the install alive, but the
user has no way to tell a frozen terminal from a hung install.

The fix
=======
1. ``install.sh`` prints an explicit NOTE at the start of step [6/6] —
   BEFORE the disruptive ``--install-extension`` — telling the user the
   terminal may stop updating, that the install keeps running detached,
   and how to follow/verify completion via ``~/.kiss/install.log``.
2. (pre-existing, verified here end-to-end) the install body survives the
   terminal's PTY being closed mid-step and still writes the completion
   banner to the log.

What these tests do
===================
Each test extracts the step [6/6] block verbatim from ``install.sh``
(between ``# BEGIN: kiss-step-6-6-terminal-freeze`` and the matching END
marker), wraps it in a harness that reproduces install.sh's real
``exec > >(tee -a "$LOG_FILE") 2>&1`` redirection, and runs it attached to
a REAL PTY with:

* a stub ``CODE_CLI`` that behaves like ``code --install-extension``,
* a stub ``lsof`` backed by a real dummy "daemon" process (a ``sleep``)
  so the "Stopping old kiss-web daemon (PIDs: ...)" path is exercised
  for real (the dummy process is really SIGTERMed),
* stub ``launchctl``/``systemctl`` so the REAL kiss-web service on the
  development machine is never touched,
* a sandboxed ``$HOME`` so marker files land in a tmp dir.

No mocks or patches — real bash, real tee, real PTY, real signals.
"""

from __future__ import annotations

import errno
import os
import pty
import subprocess
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"
ACTIVE_TASKS_SCRIPT = REPO / "scripts" / "check-kiss-web-active-tasks.py"

_BEGIN_MARKER = "# BEGIN: kiss-step-6-6-terminal-freeze"
_END_MARKER = "# END: kiss-step-6-6-terminal-freeze"

# The last line the user saw before the terminal froze (bug report).
_STOPPING_LINE = "Stopping old kiss-web daemon"
_COMPLETE_LINE = "=== Source bootstrap complete ==="
_CODE_STUB_DONE = "Extension 'kiss-sorcar.vsix' was successfully installed."

# The user-facing heads-up that must precede the disruptive extension
# install.  Matched loosely so wording can be polished without breaking
# the test, but the load-bearing concepts must be present.
_NOTICE_SNIPPET = "NOT stuck"


def _extract_step_6_6_block() -> str:
    """Return the verbatim step [6/6] block from ``install.sh``."""
    assert INSTALL_SCRIPT.exists(), f"install.sh not found at {INSTALL_SCRIPT}"
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert _BEGIN_MARKER in src, (
        f"install.sh missing '{_BEGIN_MARKER}'; the step [6/6] block must "
        "be bracketed by BEGIN/END markers for verbatim extraction."
    )
    begin_idx = src.index(_BEGIN_MARKER)
    end_idx = src.index(_END_MARKER, begin_idx)
    begin_eol = src.index("\n", begin_idx) + 1
    return src[begin_eol:end_idx]


def _build_sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """Create the sandbox (stubs, fake HOME, project dir, dummy daemon).

    Returns:
        ``(harness_path, log_path)`` — the harness script to run under
        bash and the install log it tees into.
    """
    home = tmp_path / "home"
    (home / ".kiss").mkdir(parents=True)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    # Real active-tasks probe script: with the sandboxed HOME there is no
    # ~/.kiss/sorcar.sock, so it reports "safe to kill" (exit 0) — the
    # exact state from the bug report ("kiss-web is idle ... safe to kill").
    (project / "scripts" / "check-kiss-web-active-tasks.py").write_text(
        ACTIVE_TASKS_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    log = tmp_path / "install.log"
    vsix = project / "kiss-sorcar.vsix"
    vsix.write_bytes(b"dummy vsix")
    dummy_pidfile = tmp_path / "dummy-daemon.pid"

    # Stub VS Code CLI: mimics `code --install-extension <vsix> --force`.
    code_cli = stubs / "code-stub"
    code_cli.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            echo "Installing extensions..."
            echo "{_CODE_STUB_DONE}"
            """
        ),
        encoding="utf-8",
    )
    code_cli.chmod(0o755)

    # Stub lsof: reports the dummy daemon's PID while it is alive, exactly
    # like `lsof -ti :8787` reports the real kiss-web daemon.  Once the
    # extracted block SIGTERMs it, the stub prints nothing (exit 1).
    (stubs / "lsof").write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            pid=$(cat {dummy_pidfile.as_posix()!r} 2>/dev/null) || exit 1
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "$pid"
                exit 0
            fi
            exit 1
            """
        ),
        encoding="utf-8",
    )
    (stubs / "lsof").chmod(0o755)

    # Stub the service supervisors so the REAL kiss-web daemon on this
    # machine is never kickstarted/restarted by the test.
    for supervisor in ("launchctl", "systemctl"):
        (stubs / supervisor).write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        (stubs / supervisor).chmod(0o755)

    block = _extract_step_6_6_block()
    harness = tmp_path / "harness.sh"
    harness.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -eo pipefail
            export HOME={home.as_posix()!r}
            export PATH={stubs.as_posix()!r}:"$PATH"
            CODE_CLI={code_cli.as_posix()!r}
            VSIX={vsix.as_posix()!r}
            PROJECT_DIR={project.as_posix()!r}
            LOG_FILE={log.as_posix()!r}
            # Dummy long-lived "kiss-web daemon" the extracted block will
            # discover via the lsof stub and really SIGTERM.
            sleep 300 &
            echo $! > {dummy_pidfile.as_posix()!r}
            # Same stdout/stderr plumbing as install.sh: everything the
            # block prints goes through tee to BOTH the terminal (the
            # test's PTY) and the log file.
            exec > >(tee -a "$LOG_FILE") 2>&1
            {{
            """
        )
        + block
        + textwrap.dedent(
            """\
            }
            """
        ),
        encoding="utf-8",
    )
    harness.chmod(0o755)
    return harness, log


def _spawn_on_pty(harness: Path) -> tuple[subprocess.Popen[bytes], int]:
    """Run *harness* with stdout/stderr attached to a fresh PTY slave.

    Returns the process and the PTY master fd (caller must close it).
    """
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        ["bash", str(harness)],
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave)
    return proc, master


def _read_until(master: int, needle: bytes, timeout: float = 30.0) -> bytes:
    """Read from the PTY master until *needle* appears or *timeout*."""
    buf = b""
    deadline = time.monotonic() + timeout
    while needle not in buf and time.monotonic() < deadline:
        try:
            chunk = os.read(master, 4096)
        except OSError as exc:
            if exc.errno == errno.EIO:  # writer side gone
                break
            raise
        if not chunk:
            break
        buf += chunk
    return buf


def _drain(master: int, timeout: float = 30.0) -> bytes:
    """Read from the PTY master until EOF/EIO or *timeout*."""
    return _read_until(master, b"\x00__never__\x00", timeout)


def _wait_for_line(log: Path, needle: str, timeout: float = 30.0) -> None:
    """Poll *log* until *needle* appears (tee flush is asynchronous)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log.exists() and needle in log.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.05)
    raise AssertionError(
        f"{needle!r} never appeared in {log}:\n"
        + (log.read_text(encoding="utf-8", errors="replace") if log.exists() else "<missing>")
    )


def test_freeze_notice_printed_before_disruptive_extension_install(
    tmp_path: Path,
) -> None:
    """The user must be told BEFORE ``--install-extension`` runs that the
    terminal may stop updating and the install continues detached.

    This is the reproduction test for the "install.sh gets stuck at
    'Stopping old kiss-web daemon (PIDs: ...)'" report: without the
    notice, a user whose terminal is disposed by the VS Code reload has
    no way to know the install is still running.  The notice must appear
    strictly BEFORE the stub VS Code CLI's output (the disruptive step)
    so it is guaranteed to be rendered before any reload can kill the
    terminal.
    """
    harness, log = _build_sandbox(tmp_path)
    proc, master = _spawn_on_pty(harness)
    try:
        out = _drain(master)
        rc = proc.wait(timeout=30)
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    text = out.decode("utf-8", errors="replace")
    assert rc == 0, f"step [6/6] harness failed rc={rc}:\n{text}"
    assert _NOTICE_SNIPPET in text, (
        "install.sh step [6/6] must print a heads-up that a frozen "
        "terminal does not mean the install is stuck.  Expected a NOTE "
        f"containing {_NOTICE_SNIPPET!r} in:\n{text}"
    )
    notice_idx = text.index(_NOTICE_SNIPPET)
    install_idx = text.index(_CODE_STUB_DONE)
    stopping_idx = text.index(_STOPPING_LINE)
    assert notice_idx < install_idx, (
        "the freeze notice must be printed BEFORE `--install-extension` "
        "(the step that triggers the VS Code reload which can dispose "
        "the terminal); printing it later means a disposed terminal "
        "never shows it."
    )
    assert notice_idx < stopping_idx, (
        "the freeze notice must precede the 'Stopping old kiss-web "
        "daemon' line — the last line users saw before the freeze."
    )
    # The notice must point the user at the log file so they can follow
    # the detached install after the terminal dies.
    assert str(log) in text, (
        "the freeze notice must mention the install log path so the "
        "user can follow the detached install after the terminal dies."
    )


def test_install_completes_when_terminal_dies_mid_step_6_6(
    tmp_path: Path,
) -> None:
    """Closing the PTY right after 'Extension installed into VS Code'
    (simulating VS Code disposing the terminal during the reload) must
    NOT prevent the rest of step [6/6] from completing.

    The dummy daemon must really be SIGTERMed, the ``.extension-updated``
    marker must be written into the sandboxed ``$HOME``, and the
    completion banner must reach the log — proving the "stuck" terminal
    from the bug report was cosmetic, and stays that way.
    """
    harness, log = _build_sandbox(tmp_path)
    proc, master = _spawn_on_pty(harness)
    try:
        out = _read_until(master, b"Extension installed into VS Code")
        assert b"Extension installed into VS Code" in out, (
            "harness never reached the post-install point:\n"
            + out.decode("utf-8", errors="replace")
        )
    finally:
        # Simulate VS Code disposing the integrated terminal: the PTY
        # master goes away while the install is still mid-step.
        os.close(master)

    try:
        rc = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise AssertionError(
            "step [6/6] did not finish after the terminal PTY was closed "
            "— the install really would hang for users."
        )

    assert rc == 0, f"step [6/6] harness exited rc={rc} after PTY close"
    # tee keeps writing the log even though its terminal fd is dead.
    _wait_for_line(log, _STOPPING_LINE)
    _wait_for_line(log, _COMPLETE_LINE)
    home = tmp_path / "home"
    marker = home / ".kiss" / ".extension-updated"
    assert marker.exists(), (
        "the .extension-updated marker was not written after the "
        "terminal died — the extension reload would never be triggered."
    )
    # The dummy daemon must actually have been stopped.
    dummy_pid = int((tmp_path / "dummy-daemon.pid").read_text().strip())
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(dummy_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(dummy_pid, 9)
        raise AssertionError("dummy kiss-web daemon was never SIGTERMed")

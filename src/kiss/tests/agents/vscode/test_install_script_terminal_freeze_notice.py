# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``install.sh`` step [5/5].

Covered contracts
=================
1. ``install.sh`` must warn the user BEFORE the disruptive
   ``code --install-extension --force`` call that the terminal may freeze
   (the VS Code reload triggered by the on-disk extension update can
   dispose the very terminal the script prints to), and where to follow
   progress (``~/.kiss/install.log``).
2. The install must still complete (marker written, completion banner
   logged) when the terminal dies mid-step.
3. ``install.sh`` must NEVER touch the kiss-web daemon: no kill, no
   ``~/.kiss/sorcar.sock`` removal, no ``launchctl kickstart`` /
   ``systemctl restart``.  Restarting kiss-web is owned entirely by the
   VS Code extension's DependencyInstaller (``restartKissWebDaemon``,
   fingerprint mismatch after the reload), which also defers while tasks
   are in flight — so running ``install.sh`` can never clobber an
   in-flight agent run.

What these tests do
===================
Each test extracts the step [5/5] block verbatim from ``install.sh``
(between ``# BEGIN: kiss-step-5-5-terminal-freeze`` and the matching END
marker), wraps it in a harness that reproduces install.sh's real
``exec > >(tee -a "$LOG_FILE") 2>&1`` redirection, and runs it attached to
a REAL PTY with:

* a stub ``CODE_CLI`` that behaves like ``code --install-extension``,
* a real dummy "kiss-web daemon" process (a ``sleep``) plus a stub
  ``lsof`` reporting it, so any regression that reintroduces the old
  kill-by-port logic would really SIGTERM the dummy and be caught,
* stub ``launchctl``/``systemctl`` that LOG their invocations (and keep
  the REAL kiss-web service on the development machine safe),
* a sandboxed ``$HOME`` so marker/socket files land in a tmp dir.

No mocks or patches — real bash, real tee, real PTY, real processes.
"""

from __future__ import annotations

import errno
import os
import pty
import select
import subprocess
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

_BEGIN_MARKER = "# BEGIN: kiss-step-5-5-terminal-freeze"
_END_MARKER = "# END: kiss-step-5-5-terminal-freeze"

_COMPLETE_LINE = "=== Source bootstrap complete ==="
_CODE_STUB_DONE = "Extension 'kiss-sorcar.vsix' was successfully installed."
# Final echo of the extracted block — used to detect the end of the run
# without waiting for PTY EOF (the surviving dummy daemon keeps the PTY
# open, so EOF never comes).
_LAST_BLOCK_LINE = "remote access auth, and kiss-web."

# The user-facing heads-up that must precede the disruptive extension
# install.  Matched loosely so wording can be polished without breaking
# the test, but the load-bearing concept must be present.
_NOTICE_SNIPPET = "NOT stuck"


def _extract_step_5_5_block() -> str:
    """Return the verbatim step [5/5] block from ``install.sh``."""
    assert INSTALL_SCRIPT.exists(), f"install.sh not found at {INSTALL_SCRIPT}"
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert _BEGIN_MARKER in src, (
        f"install.sh missing '{_BEGIN_MARKER}'; the step [5/5] block must "
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
    # Pre-existing UDS socket file of the running daemon: the block must
    # leave it alone (the old code ``rm -f``-ed it).
    (home / ".kiss" / "sorcar.sock").write_bytes(b"")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    project = tmp_path / "project"
    project.mkdir(parents=True)
    log = tmp_path / "install.log"
    vsix = project / "kiss-sorcar.vsix"
    vsix.write_bytes(b"dummy vsix")
    dummy_pidfile = tmp_path / "dummy-daemon.pid"
    supervisor_log = tmp_path / "supervisor-calls.log"

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
    # like `lsof -ti :8787` reports the real kiss-web daemon.  The block
    # must never consult it, but if a regression reintroduces the old
    # kill-by-port logic the dummy daemon would really be SIGTERMed and
    # the never-touches-kiss-web test below would fail.
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

    # Stub the service supervisors: they log every invocation (so the
    # tests can assert none happened) and keep the REAL kiss-web daemon
    # on this machine safe from kickstart/restart.
    for supervisor in ("launchctl", "systemctl"):
        (stubs / supervisor).write_text(
            f'#!/bin/bash\necho "{supervisor} $*" >> {supervisor_log.as_posix()!r}\nexit 0\n',
            encoding="utf-8",
        )
        (stubs / supervisor).chmod(0o755)

    # Supervisor configs pointing at an existing, executable daemon binary
    # — i.e. a restart WOULD be possible right now.  The block must still
    # not touch the daemon.
    daemon_bin = stubs / "kiss-web-daemon-stub"
    daemon_bin.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    daemon_bin.chmod(0o755)
    launch_agents = home / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True)
    (launch_agents / "com.kiss.web-server.plist").write_text(
        textwrap.dedent(
            f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <plist version="1.0">
            <dict>
                <key>Label</key>
                <string>com.kiss.web-server</string>
                <key>ProgramArguments</key>
                <array>
                    <string>{daemon_bin.as_posix()}</string>
                </array>
            </dict>
            </plist>
            """
        ),
        encoding="utf-8",
    )
    systemd_dir = home / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True)
    (systemd_dir / "kiss-web.service").write_text(
        f"[Service]\nExecStart={daemon_bin.as_posix()}\n", encoding="utf-8"
    )

    block = _extract_step_5_5_block()
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
            # Dummy long-lived "kiss-web daemon".  The block must leave it
            # completely alone.
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
    """Read from the PTY master until *needle* appears or *timeout*.

    Uses ``select`` so a PTY held open by a surviving background process
    (the dummy daemon) cannot block ``os.read`` past the deadline.
    """
    buf = b""
    deadline = time.monotonic() + timeout
    while needle not in buf and time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.2)
        if not ready:
            continue
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


def _dummy_daemon_pid(tmp_path: Path) -> int:
    """Return the PID of the sandbox's dummy kiss-web daemon."""
    return int((tmp_path / "dummy-daemon.pid").read_text().strip())


def _kill_dummy_daemon(tmp_path: Path) -> None:
    """Best-effort cleanup of the sandbox's dummy daemon."""
    try:
        os.kill(_dummy_daemon_pid(tmp_path), 9)
    except (ProcessLookupError, FileNotFoundError, ValueError):
        pass


def test_freeze_notice_printed_before_disruptive_extension_install(
    tmp_path: Path,
) -> None:
    """The user must be told BEFORE ``--install-extension`` runs that the
    terminal may stop updating and the install continues detached.

    Without the notice, a user whose terminal is disposed by the VS Code
    reload has no way to know the install is still running.  The notice
    must appear strictly BEFORE the stub VS Code CLI's output (the
    disruptive step) so it is guaranteed to be rendered before any reload
    can kill the terminal.
    """
    harness, log = _build_sandbox(tmp_path)
    proc, master = _spawn_on_pty(harness)
    try:
        out = _read_until(master, _LAST_BLOCK_LINE.encode())
        rc = proc.wait(timeout=30)
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        _kill_dummy_daemon(tmp_path)

    text = out.decode("utf-8", errors="replace")
    assert rc == 0, f"step [5/5] harness failed rc={rc}:\n{text}"
    assert _NOTICE_SNIPPET in text, (
        "install.sh step [5/5] must print a heads-up that a frozen "
        "terminal does not mean the install is stuck.  Expected a NOTE "
        f"containing {_NOTICE_SNIPPET!r} in:\n{text}"
    )
    notice_idx = text.index(_NOTICE_SNIPPET)
    install_idx = text.index(_CODE_STUB_DONE)
    assert notice_idx < install_idx, (
        "the freeze notice must be printed BEFORE `--install-extension` "
        "(the step that triggers the VS Code reload which can dispose "
        "the terminal); printing it later means a disposed terminal "
        "never shows it."
    )
    # The notice must point the user at the log file so they can follow
    # the detached install after the terminal dies.
    assert str(log) in text, (
        "the freeze notice must mention the install log path so the "
        "user can follow the detached install after the terminal dies."
    )


def test_install_completes_when_terminal_dies_mid_step_5_5(
    tmp_path: Path,
) -> None:
    """Closing the PTY right after 'Extension installed into VS Code'
    (simulating VS Code disposing the terminal during the reload) must
    NOT prevent the rest of step [5/5] from completing.

    The ``.extension-updated`` marker must be written into the sandboxed
    ``$HOME`` and the completion banner must reach the log — proving a
    "stuck" terminal is cosmetic.  The kiss-web daemon must survive
    untouched throughout.
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
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise AssertionError(
                "step [5/5] did not finish after the terminal PTY was "
                "closed — the install really would hang for users."
            )

        assert rc == 0, f"step [5/5] harness exited rc={rc} after PTY close"
        # tee keeps writing the log even though its terminal fd is dead.
        _wait_for_line(log, _COMPLETE_LINE)
        home = tmp_path / "home"
        marker = home / ".kiss" / ".extension-updated"
        assert marker.exists(), (
            "the .extension-updated marker was not written after the "
            "terminal died — the extension reload would never be triggered."
        )
        # The daemon must still be alive — install.sh never touches it.
        os.kill(_dummy_daemon_pid(tmp_path), 0)
    finally:
        _kill_dummy_daemon(tmp_path)


def test_step_5_5_never_touches_kiss_web(tmp_path: Path) -> None:
    """``install.sh`` must not stop, restart, or otherwise touch kiss-web.

    Even with a supervisor config whose binary exists and is executable
    (i.e. a restart WOULD succeed right now), the block must leave the
    running daemon, its UDS socket file, and the supervisors alone.
    Restarting kiss-web is owned by the extension's DependencyInstaller
    (``restartKissWebDaemon``) during extension installation/activation.
    """
    harness, log = _build_sandbox(tmp_path)
    proc, master = _spawn_on_pty(harness)
    try:
        out = _read_until(master, _LAST_BLOCK_LINE.encode())
        rc = proc.wait(timeout=30)
        text = out.decode("utf-8", errors="replace")
        assert rc == 0, f"step [5/5] harness failed rc={rc}:\n{text}"
        # The old daemon must still be alive.
        os.kill(_dummy_daemon_pid(tmp_path), 0)
        # Its UDS socket file must not have been removed.
        sock = tmp_path / "home" / ".kiss" / "sorcar.sock"
        assert sock.exists(), (
            "install.sh removed ~/.kiss/sorcar.sock — it must leave the "
            "daemon's socket alone."
        )
        # No supervisor (launchctl/systemctl) may have been invoked.
        supervisor_log = tmp_path / "supervisor-calls.log"
        assert not supervisor_log.exists(), (
            "install.sh invoked a service supervisor:\n"
            + supervisor_log.read_text(encoding="utf-8", errors="replace")
        )
        assert "Stopping old kiss-web daemon" not in text, (
            f"install.sh tried to stop the kiss-web daemon:\n{text}"
        )
        _wait_for_line(log, _COMPLETE_LINE)
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        _kill_dummy_daemon(tmp_path)

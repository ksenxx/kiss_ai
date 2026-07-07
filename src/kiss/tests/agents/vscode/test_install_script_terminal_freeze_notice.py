# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: ``install.sh`` must warn the user BEFORE step [5/5] that the
terminal may freeze, and the install must still complete when the terminal
dies mid-step.

Background — the bug
====================
A user ran ``./install.sh`` from a VS Code integrated terminal.  The last
output they ever saw was::

    >>> [5/5] Installing VS Code extension...
    ...
       Extension installed into VS Code
    kiss-web is idle (count=0) at /Users/ksen/.kiss/sorcar.sock; safe to kill.
       Stopping old kiss-web daemon (PIDs: 64585)...

and the terminal then appeared "stuck" forever — no further output, no
prompt.  ``~/.kiss/install.log`` proves the install actually COMPLETED
("=== Source bootstrap complete ===" a few seconds later): the
``--install-extension --force`` in step [5/5] makes VS Code detect the
on-disk extension update and reload, which disposes / stops rendering the
very terminal the script is printing to.  The new-session (setsid)
detachment at the top of ``install.sh`` keeps the install alive, but the
user has no way to tell a frozen terminal from a hung install.

The fix
=======
1. ``install.sh`` prints an explicit NOTE at the start of step [5/5] —
   BEFORE the disruptive ``--install-extension`` — telling the user the
   terminal may stop updating, that the install keeps running detached,
   and how to follow/verify completion via ``~/.kiss/install.log``.
2. (pre-existing, verified here end-to-end) the install body survives the
   terminal's PTY being closed mid-step and still writes the completion
   banner to the log.

What these tests do
===================
Each test extracts the step [5/5] block verbatim from ``install.sh``
(between ``# BEGIN: kiss-step-5-5-terminal-freeze`` and the matching END
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
import select
import subprocess
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"
ACTIVE_TASKS_SCRIPT = REPO / "scripts" / "check-kiss-web-active-tasks.py"

_BEGIN_MARKER = "# BEGIN: kiss-step-5-5-terminal-freeze"
_END_MARKER = "# END: kiss-step-5-5-terminal-freeze"

# The last line the user saw before the terminal froze (bug report).
_STOPPING_LINE = "Stopping old kiss-web daemon"
_COMPLETE_LINE = "=== Source bootstrap complete ==="
_CODE_STUB_DONE = "Extension 'kiss-sorcar.vsix' was successfully installed."

# The user-facing heads-up that must precede the disruptive extension
# install.  Matched loosely so wording can be polished without breaking
# the test, but the load-bearing concepts must be present.
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


def _build_sandbox(
    tmp_path: Path,
    *,
    supervisor_ok: bool = True,
    with_ext_uv: bool = False,
) -> tuple[Path, Path]:
    """Create the sandbox (stubs, fake HOME, project dir, dummy daemon).

    Args:
        tmp_path: pytest tmp dir for the whole sandbox.
        supervisor_ok: when True the sandbox's supervisor configs (macOS
            LaunchAgent plist + Linux systemd unit) point at an existing
            executable daemon binary, so the step [5/5] block is allowed
            to stop the old daemon.  When False they point at a missing
            binary — the block must then SKIP the kill (killing without a
            respawnable binary caused the ~90 s "Server is starting ..."
            outage).
        with_ext_uv: when True the sandbox simulates the real
            post-``--install-extension --force`` state: a freshly
            installed extension dir exists under ``$HOME/.vscode/
            extensions`` but its ``kiss_project/.venv`` (and the daemon
            binary the supervisor points to) does NOT exist until the
            stub ``uv`` recreates it — verifying the pre-build ordering.

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

    # Supervisor configs inside the sandbox HOME.  The step [5/5] block
    # only stops the old daemon when the binary the supervisor points to
    # exists and is executable — a kill without a respawnable binary
    # left users with no server (launchd failure-throttling) for ~90 s.
    daemon_bin = stubs / "kiss-web-daemon-stub"
    if supervisor_ok and not with_ext_uv:
        daemon_bin.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        daemon_bin.chmod(0o755)
    if with_ext_uv:
        # Reproduce the real post-``--force``-install state: the extension
        # dir exists but kiss_project/.venv (and thus the daemon binary
        # the supervisor points to) is GONE until ``uv sync`` rebuilds it.
        ext_project = (
            home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-9.9.9" / "kiss_project"
        )
        ext_project.mkdir(parents=True)
        daemon_bin = ext_project / ".venv" / "bin" / "kiss-web"
        pkg_dir = project / "src" / "kiss" / "agents" / "vscode"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "package.json").write_text('{\n  "version": "9.9.9"\n}\n', encoding="utf-8")
        uv_log = tmp_path / "uv-calls.log"
        (stubs / "uv").write_text(
            textwrap.dedent(
                f"""\
                #!/bin/bash
                echo "$@" >> {uv_log.as_posix()!r}
                if [ "$1" = "sync" ]; then
                    mkdir -p {daemon_bin.parent.as_posix()!r}
                    printf '#!/bin/bash\\nexit 0\\n' > {daemon_bin.as_posix()!r}
                    chmod +x {daemon_bin.as_posix()!r}
                fi
                exit 0
                """
            ),
            encoding="utf-8",
        )
        (stubs / "uv").chmod(0o755)
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
            # Keep the bounded "waiting for the daemon to come back" poll
            # short — the stub launchctl/systemctl never respawn anything.
            export KISS_DAEMON_WAIT_SECS=1
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
    """Read from the PTY master until *needle* appears or *timeout*.

    Uses ``select`` so a PTY held open by a surviving background process
    (e.g. the dummy daemon in the skip-restart test) cannot block
    ``os.read`` past the deadline.
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
    assert rc == 0, f"step [5/5] harness failed rc={rc}:\n{text}"
    assert _NOTICE_SNIPPET in text, (
        "install.sh step [5/5] must print a heads-up that a frozen "
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


def test_install_completes_when_terminal_dies_mid_step_5_5(
    tmp_path: Path,
) -> None:
    """Closing the PTY right after 'Extension installed into VS Code'
    (simulating VS Code disposing the terminal during the reload) must
    NOT prevent the rest of step [5/5] from completing.

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
            "step [5/5] did not finish after the terminal PTY was closed "
            "— the install really would hang for users."
        )

    assert rc == 0, f"step [5/5] harness exited rc={rc} after PTY close"
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


def test_daemon_survives_when_supervisor_cannot_respawn(tmp_path: Path) -> None:
    """Regression for the ~90 s "KISS Sorcar Server is starting ..." outage.

    ``--install-extension --force`` wipes the extension's
    ``kiss_project/.venv``, so the supervisor's program
    (``.venv/bin/kiss-web``) may not exist when step [5/5] reaches the
    daemon-stop.  Killing the daemon then leaves the user with NO server:
    launchd/systemd respawns fail on the missing binary and get
    failure-throttled until the extension rebuilds the venv after the
    window reload (91 s in the reported trace, kiss-web-stderr.log
    20:28:45 SIGTERM → 20:30:16 next start).  The block must instead SKIP
    the kill and leave the old daemon serving.
    """
    harness, log = _build_sandbox(tmp_path, supervisor_ok=False)
    proc, master = _spawn_on_pty(harness)
    dummy_pid = None
    try:
        # Read to the block's final line instead of EOF: the (deliberately)
        # surviving dummy daemon keeps the PTY open, so EOF never comes.
        out = _read_until(master, b"remote access auth, and kiss-web.")
        rc = proc.wait(timeout=30)
        text = out.decode("utf-8", errors="replace")
        assert rc == 0, f"step [5/5] harness failed rc={rc}:\n{text}"
        dummy_pid = int((tmp_path / "dummy-daemon.pid").read_text().strip())
        assert "Skipping kiss-web daemon restart" in text, (
            "step [5/5] must explain that the daemon restart is skipped "
            f"when the supervisor binary is missing:\n{text}"
        )
        assert _STOPPING_LINE not in text, (
            "step [5/5] killed the kiss-web daemon even though its "
            "supervisor cannot respawn it — this is the ~90 s "
            f"'Server is starting ...' outage regression:\n{text}"
        )
        # The old daemon must still be alive — it keeps serving until the
        # extension restarts it with a working environment.
        os.kill(dummy_pid, 0)  # raises ProcessLookupError if it was killed
        _wait_for_line(log, _COMPLETE_LINE)
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if dummy_pid is None:
            dummy_pid = int((tmp_path / "dummy-daemon.pid").read_text().strip())
        try:
            os.kill(dummy_pid, 9)
        except ProcessLookupError:
            pass


def test_venv_prebuilt_before_daemon_is_stopped(tmp_path: Path) -> None:
    """The freshly-installed extension's venv must be rebuilt (``uv sync``)
    BEFORE the old daemon is stopped, so the supervisor's respawn succeeds
    immediately instead of failure-throttling for ~90 s.

    The sandbox reproduces the real post-install state: the extension dir
    exists under ``$HOME/.vscode/extensions/ksenxx.kiss-sorcar-9.9.9`` but
    ``kiss_project/.venv/bin/kiss-web`` (which the sandbox plist/systemd
    unit point to) only comes into existence when the stub ``uv sync``
    runs.  The dummy daemon must be SIGTERMed only AFTER that.
    """
    harness, log = _build_sandbox(tmp_path, with_ext_uv=True)
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
    assert rc == 0, f"step [5/5] harness failed rc={rc}:\n{text}"
    # uv must have been asked to sync the freshly-installed extension's
    # kiss_project (the one the supervisor's daemon binary lives in).
    uv_calls = (tmp_path / "uv-calls.log").read_text(encoding="utf-8")
    ext_project = (
        tmp_path / "home" / ".vscode" / "extensions"
        / "ksenxx.kiss-sorcar-9.9.9" / "kiss_project"
    )
    assert f"sync --project {ext_project.as_posix()}" in uv_calls, (
        f"uv sync was not run against the new extension dir:\n{uv_calls}"
    )
    # Ordering: the pre-build message (and hence uv sync) must precede the
    # daemon stop — stopping first is exactly the outage bug.
    prebuild_idx = text.index("Pre-building the new extension's Python environment")
    stopping_idx = text.index(_STOPPING_LINE)
    assert prebuild_idx < stopping_idx, (
        "uv sync must run BEFORE the old kiss-web daemon is stopped"
    )
    # With the binary rebuilt, the daemon stop must actually proceed.
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
    _wait_for_line(log, _COMPLETE_LINE)

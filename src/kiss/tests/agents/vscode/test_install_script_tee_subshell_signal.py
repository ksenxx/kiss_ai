# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: ``install.sh`` must keep its SIGINT/SIGTERM trap effective for
the entire install body, and must survive the VS Code "Update" button's
PTY teardown (SIGHUP).

Background — the bug
====================
When the user clicked the settings Update button, ``install.sh`` ran inside
a VS Code integrated terminal.  Step ``[6/6]`` invoked
``code --install-extension --force``; VS Code's extension manager detected
the on-disk update, deactivated the running extension, and (per the
documented contract) disposed the integrated terminal — which first
injects ``\\x03`` (SIGINT) into the PTY and then closes it (SIGHUP).

``install.sh`` had ``trap handle_interrupt INT TERM`` at the top, but the
entire install body was wrapped in ``{ ... } 2>&1 | tee "$LOG_FILE"`` — a
pipeline subshell.  POSIX bash specifies::

    Trapped signals that are not being ignored are reset to their original
    values in a subshell or subshell environment when one is created.

So the trap was completely ineffective inside the pipeline.  When VS Code
injected ``\\x03``, default SIGINT terminated the subshell instantly with
a bare ``^C``, leaving kiss-web alive on the old code path and the
``.extension-updated`` marker unwritten — exactly the symptom users saw.

Fix
===
1. Replace the ``{ ... } | tee`` wrapper with ``exec > >(tee -a "$LOG_FILE") 2>&1``
   so the install body runs in the OUTER (trap-handled) shell.
2. Add ``handle_hup`` + ``trap handle_hup HUP`` to re-route stdout/stderr
   to the log file when the controlling terminal closes — so PTY closure
   no longer kills bash mid-step.

The tests below assert both the structural invariants and the behavioural
ones (SIGINT delivered to install.sh's body fires the trap; SIGHUP runs
``handle_hup`` and the script keeps going).
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"


def _read_install_sh() -> str:
    """Return ``install.sh`` source text, ensuring the script is present."""
    assert INSTALL_SCRIPT.exists(), f"install.sh not found at {INSTALL_SCRIPT}"
    return INSTALL_SCRIPT.read_text(encoding="utf-8")


def test_install_sh_uses_exec_tee_not_pipeline_subshell() -> None:
    """The install body must run in the outer (trap-handled) shell.

    Wrapping the install body in ``{ ... } 2>&1 | tee "$LOG_FILE"`` forks
    a pipeline subshell, and bash resets all trapped signals to their
    default disposition inside subshells.  That made the existing
    ``trap handle_interrupt INT TERM`` ineffective and let a single PTY
    ``\\x03`` from VS Code's terminal teardown kill the script.
    """
    src = _read_install_sh()
    # ``exec > >(tee -a "$LOG_FILE") 2>&1`` must be present at module scope
    # (not inside a function body) so it applies to the install body.
    exec_pattern = r'exec\s*>\s*>\(\s*tee\s+-a\s+"?\$LOG_FILE"?\s*\)\s*2>&1'
    assert re.search(exec_pattern, src), (
        "install.sh must use `exec > >(tee -a \"$LOG_FILE\") 2>&1` so the "
        "install body runs in the outer trap-handled shell instead of a "
        "pipeline subshell.  Without this fix the script's SIGINT trap is "
        "reset by bash inside the subshell and a stray ^C from VS Code's "
        "terminal teardown silently aborts the update."
    )
    # The previous pattern — ``{ ... } 2>&1 | tee "$LOG_FILE"`` — must be
    # gone from the executable code.  The string may legitimately appear
    # inside the explanatory comment immediately above the ``exec ...``
    # line; strip lines whose first non-whitespace character is ``#``
    # before checking.
    code_only = "\n".join(
        line
        for line in src.splitlines()
        if not line.lstrip().startswith("#")
    )
    forbidden = r'\}\s*2>&1\s*\|\s*tee\s+"?\$LOG_FILE"?'
    m = re.search(forbidden, code_only)
    assert m is None, (
        "install.sh still wraps the install body in `{ ... } 2>&1 | tee "
        "\"$LOG_FILE\"` — that runs the body in a pipeline subshell and "
        "neutralises the SIGINT/SIGTERM trap.  Found: "
        f"{m.group(0) if m else ''!r}"
    )


def test_install_sh_traps_sighup_for_pty_teardown() -> None:
    """install.sh must trap SIGHUP so PTY closure does not kill the script.

    When VS Code disposes the integrated terminal that ran ``install.sh``
    (because ``code --install-extension --force`` deactivated the
    extension that owned the terminal), the documented teardown sequence
    is ``\\x03`` (Ctrl+C) followed by closing the PTY (SIGHUP).  Without
    a SIGHUP trap, bash's default disposition terminates the script —
    leaving kiss-web alive on the old code path and the
    ``.extension-updated`` marker unwritten.
    """
    src = _read_install_sh()
    assert re.search(r"^\s*handle_hup\s*\(\)\s*\{", src, flags=re.MULTILINE), (
        "install.sh must define a `handle_hup()` function that re-routes "
        "stdout/stderr to the log file when the controlling terminal is "
        "closed (SIGHUP from VS Code's terminal disposal)."
    )
    assert re.search(r"^\s*trap\s+handle_hup\s+HUP\b", src, flags=re.MULTILINE), (
        "install.sh must install `trap handle_hup HUP` so PTY closure runs "
        "the re-route handler instead of killing bash mid-step."
    )
    # The handler must redirect to the log file so subsequent writes do
    # not crash with EBADF/ENXIO on the closed PTY.
    handler_match = re.search(
        r"handle_hup\s*\(\)\s*\{([^}]*)\}", src, flags=re.DOTALL
    )
    assert handler_match is not None
    body = handler_match.group(1)
    assert "exec" in body and "$LOG_FILE" in body, (
        "handle_hup must `exec >>$LOG_FILE 2>&1` so the script keeps "
        "logging after the controlling PTY closes."
    )


def _reset_signals() -> None:
    """Restore default terminal-signal dispositions in the harness child.

    POSIX requires a non-interactive shell to start asynchronous
    (``cmd &``) children with SIGINT/SIGQUIT *ignored*, and ignored
    dispositions survive fork+exec — so when the pytest session itself is
    launched as a background job (CI runners, ``nohup pytest &``, parallel
    test drivers), the harness bash would inherit ``SIG_IGN`` for SIGINT.
    Bash cannot trap a signal that was ignored on entry to the shell, so
    ``trap handle_interrupt INT`` would silently never install and the
    ``os.killpg(..., SIGINT)`` below would be a no-op.  Resetting to
    ``SIG_DFL`` (via ``preexec_fn``, i.e. post-fork/pre-exec) restores the
    disposition the harness — and the real VS Code integrated terminal —
    assumes.
    """
    for sig in (signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_DFL)


def _wait_for_log(log: Path, needle: str, timeout: float = 10.0) -> str:
    """Re-read *log* until *needle* appears or *timeout* elapses.

    The SIGINT diagnostic reaches the log through the process-substitution
    ``tee -a`` child, which may not have been scheduled to drain its pipe
    by the time the harness shell itself has exited — a flake seen only
    under heavy parallel test load.  Polling removes that window.  Returns
    the last-read log text either way so assertion messages stay useful.
    """
    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        text = (
            log.read_text(encoding="utf-8", errors="replace")
            if log.exists()
            else ""
        )
        if needle in text:
            return text
        time.sleep(0.1)
    return text


def _outer_trap_harness(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a harness mirroring install.sh's outer trap + exec-tee pattern.

    Returns (harness_path, log_path, marker_path, ready_path).  The
    harness sets up ``handle_interrupt`` (single-signal: ignore +
    diagnostic), pipes output through ``tee -a`` via ``exec``, touches
    the ready marker (traps installed, redirection live), then enters a
    long ``sleep`` in the outer shell.  If the trap fires the sleep is
    interrupted and the script continues to touch the marker.
    """
    log = tmp_path / "install.log"
    marker = tmp_path / "post_signal_marker"
    ready = tmp_path / "harness_ready"
    harness = tmp_path / "harness.sh"
    harness.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -eo pipefail
            LOG_FILE="{log}"
            mkdir -p "$(dirname "$LOG_FILE")"
            handle_interrupt() {{
                echo "Interrupt received but ignored"
            }}
            handle_hup() {{
                exec >>"$LOG_FILE" 2>&1 || true
                echo "Controlling terminal closed (SIGHUP)"
            }}
            trap handle_interrupt INT TERM
            trap handle_hup HUP
            exec > >(tee -a "$LOG_FILE") 2>&1
            : > "{ready}"
            # Mimic install.sh's body — a long-running step that must
            # survive a stray PTY-injected signal.
            for _ in 1 2 3 4 5 6 7 8 9 10; do
                sleep 0.5 || true
            done
            : > "{marker}"
            """
        ),
        encoding="utf-8",
    )
    harness.chmod(0o755)
    return harness, log, marker, ready


@pytest.mark.process_killer
def test_install_sh_outer_trap_survives_sigint(tmp_path: Path) -> None:
    """SIGINT to the install body must fire the trap and keep going.

    Reproduces the post-fix invariant: with ``exec > >(tee -a $LOG_FILE) 2>&1``
    instead of ``{ ... } | tee``, the install body runs in the outer
    shell whose ``trap handle_interrupt INT TERM`` is still active.  A
    single stray SIGINT (the kind VS Code injects when disposing the
    integrated terminal) prints the trap diagnostic and the script keeps
    going to write the post-signal marker.

    Before the fix this same harness — but with ``{ ... } | tee`` — would
    die instantly with a bare ``^C`` and the marker would never appear.
    """
    harness, log, marker, ready = _outer_trap_harness(tmp_path)
    proc = subprocess.Popen(
        ["bash", str(harness)],
        start_new_session=True,
        preexec_fn=_reset_signals,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Wait until the harness has installed its traps and passed
        # through the `exec > >(tee ...)` redirection (it touches the
        # ready marker right after) before delivering the signal.  A
        # fixed sleep raced bash's startup under heavy parallel load.
        deadline = time.time() + 15.0
        while time.time() < deadline and not ready.exists():
            time.sleep(0.05)
        assert ready.exists(), "harness never reached the install body"
        os.killpg(proc.pid, signal.SIGINT)
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    log_text = _wait_for_log(log, "Interrupt received but ignored")
    assert rc == 0, (
        "harness aborted on a single stray SIGINT — the outer trap is no "
        f"longer effective.  rc={rc!r}\nLog:\n{log_text}"
    )
    assert marker.exists(), (
        "the install body did not run to completion after the SIGINT — "
        f"the trap did not fire or the script was killed.\nLog:\n{log_text}"
    )
    assert "Interrupt received but ignored" in log_text, (
        "the handle_interrupt diagnostic was not logged — the outer trap "
        f"did not fire.\nLog:\n{log_text}"
    )


@pytest.mark.process_killer
def test_install_sh_outer_trap_survives_sighup(tmp_path: Path) -> None:
    """SIGHUP must trigger the re-route handler and the script must keep going.

    When VS Code closes the integrated terminal that hosted install.sh,
    bash receives SIGHUP.  ``handle_hup`` ``exec``s stdout/stderr onto the
    log file and returns.  The remaining install steps then keep running
    (writing into the log only) — which is what allows the kiss-web kill
    and the marker write to actually finish.
    """
    harness, log, marker, ready = _outer_trap_harness(tmp_path)
    proc = subprocess.Popen(
        ["bash", str(harness)],
        start_new_session=True,
        preexec_fn=_reset_signals,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Wait until the harness has installed its traps and passed
        # through the `exec > >(tee ...)` redirection (it touches the
        # ready marker right after) before delivering the signal.  A
        # fixed sleep raced bash's startup under heavy parallel load.
        deadline = time.time() + 15.0
        while time.time() < deadline and not ready.exists():
            time.sleep(0.05)
        assert ready.exists(), "harness never reached the install body"
        # Send SIGHUP to bash specifically (not killpg) — that mirrors the
        # kernel's "controlling-terminal hangup" delivery to the session
        # leader.  Hitting the whole pgrp would also kill the ``tee -a``
        # child of the process substitution before bash's trap can run
        # ``exec >>$LOG_FILE``, racing in SIGPIPE on the broken pipe.
        os.kill(proc.pid, signal.SIGHUP)
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    log_text = _wait_for_log(log, "Controlling terminal closed (SIGHUP)")
    assert rc == 0, (
        "harness aborted on SIGHUP — handle_hup did not absorb the PTY "
        f"closure.  rc={rc!r}\nLog:\n{log_text}"
    )
    assert marker.exists(), (
        "the install body did not run to completion after SIGHUP — the "
        f"script was killed before the post-signal step.\nLog:\n{log_text}"
    )
    assert "Controlling terminal closed (SIGHUP)" in log_text, (
        "handle_hup did not print its diagnostic into the log file after "
        f"the PTY-close signal.\nLog:\n{log_text}"
    )


def test_install_sh_explains_subshell_trap_reset() -> None:
    """A comment in install.sh explains the bash subshell-trap-reset rule.

    A future maintainer must not "simplify" the install by reintroducing
    the ``{ ... } | tee`` wrapper.  The narrative comment documenting why
    that pattern is unsafe is the canonical rationale.
    """
    src = _read_install_sh()
    assert "subshell" in src, (
        "install.sh should retain a comment mentioning the bash subshell "
        "trap-reset rule that justified replacing `{ ... } | tee` with "
        "`exec > >(tee -a ...) 2>&1`."
    )
    assert "reset" in src.lower() and "trap" in src.lower(), (
        "install.sh should retain a comment explaining that bash resets "
        "trapped signals to default disposition inside pipeline subshells."
    )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: ``install.sh`` must run its install body inside a brand-new
POSIX session (no controlling terminal) so terminal-driven signals
(SIGINT from ``\\x03``, SIGHUP from PTY close, SIGTERM from a stray kill
of the foreground process group) cannot reach it from the original
VS Code integrated terminal.

Background — the bug
====================
A user clicked the VS Code "Update" button.  ``runUpdate()`` in
``SorcarSidebarView.ts`` opened a VS Code integrated terminal and
``sendText``'d ``bash '/Users/ksen/kiss_ai/install.sh'``.  The script
ran through Xcode CLT / Homebrew / git / node / VS Code CLI / Claude
skills, then died in the middle of the TypeScript compile::

    >>> [5/6] Building VS Code extension...
       Compiling extension TypeScript...

    > kiss-sorcar@2026.6.38 compile
    > tsc -p ./

    ^C
       ⚠ Interrupt received but ignored — long npm/git steps can sit
          silent for 30-60 s while they download or extract.  Press
          Ctrl+C again within 3 s to really abort.
    ksen@Mac kiss_ai %

The user did NOT press Ctrl-C — something delivered SIGINT (or
``\\x03``) into the PTY.  The outer ``handle_interrupt`` trap fired
(the diagnostic printed) but the script STILL exited.

Why traps and ``run_with_heartbeat``'s SIG_IGN subshell were not enough
======================================================================
* SIGINT delivered to the terminal foreground process group is delivered
  to EVERY process in that group — npm, node, tsc — simultaneously.
* Node.js installs its own SIGINT handling in some configurations and
  does not always honour an inherited SIG_IGN, so ``tsc`` (which runs
  on Node) can still die on a stray SIGINT even when bash's wrapper
  exec'd it with ``trap '' INT TERM``.
* Several install steps — ``bash scripts/fetch-claude-skills.sh``,
  ``python3 scripts/check-kiss-web-active-tasks.py``,
  ``"$CODE_CLI" --install-extension``, ``xargs kill`` — are NOT wrapped
  in ``run_with_heartbeat`` at all and were therefore unprotected.

The fix
=======
At the top of ``install.sh`` (before any other work) the script
re-execs itself into a brand-new session via ``perl + POSIX::setsid``.
The kernel only delivers terminal-driven signals to the process
group(s) of the controlling terminal's session — so a session with no
controlling TTY cannot receive SIGINT/SIGHUP from any terminal at all.

The re-exec uses ``perl`` because:

* ``setsid(2)`` fails with EPERM when called by a process-group leader
  (which bash on ``install.sh`` is); we therefore must ``fork`` first
  and call ``setsid`` in the child.  ``exec setsid bash install.sh``
  would EPERM on the spot.
* ``perl`` is at ``/usr/bin/perl`` on every macOS release and every
  standard Linux distro the install supports, with ``POSIX::setsid``
  in core (no CPAN deps).

Sentinel ``_KISS_NEW_SESSION=1`` is exported before the re-exec so the
re-execd child does not fork again (no infinite loop).

If perl is missing, the script falls through and continues with the
trap-only behaviour (graceful fallback).

What these tests assert
=======================
Each test below is an independent end-to-end behavioural check — no
mocks, no patches.  Tests 3-5 spawn a real bash harness containing the
verbatim re-exec block extracted from ``install.sh`` (the top-level
``if [ -z "${_KISS_NEW_SESSION:-}" ] …`` guard up to, but excluding,
``set -eo pipefail``) followed by a long-running ``sleep`` and a
post-sleep marker file; they then send SIGINT / SIGHUP / SIGTERM
repeatedly to the harness's process group and verify the marker is
still written.  Without the re-exec the unwrapped sleep dies on the
first signal.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

_GUARD_PREFIX = 'if [ -z "${_KISS_NEW_SESSION:-}" ]'
_SET_E_LINE = "set -eo pipefail"


def _read_install_sh() -> str:
    """Return ``install.sh`` source text, ensuring the script is present."""
    assert INSTALL_SCRIPT.exists(), f"install.sh not found at {INSTALL_SCRIPT}"
    return INSTALL_SCRIPT.read_text(encoding="utf-8")


def _extract_reexec_block() -> str:
    """Return the verbatim re-exec block from the top of ``install.sh``.

    The block starts at the top-level ``if [ -z "${_KISS_NEW_SESSION:-}"
    ] …`` guard line and ends just before the ``set -eo pipefail`` line
    (the first non-comment statement of the install body).  Tests paste
    this into a tmp ``harness.sh`` so they can exercise the detachment
    behaviour without running the full installer.
    """
    src = _read_install_sh()
    begin_idx = src.index(_GUARD_PREFIX)
    end_idx = src.index(_SET_E_LINE, begin_idx)
    return src[begin_idx:end_idx]


def _spawn_harness(
    harness: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Spawn *harness* under bash in its own POSIX session.

    ``start_new_session=True`` places the harness's bash in a fresh
    session and process group so the test can ``os.killpg(proc.pid,
    signum)`` without touching the pytest worker process.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        ["bash", str(harness)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def _write_signal_harness(
    tmp_path: Path,
    marker: Path,
    log: Path,
    sleep_seconds: float = 5.0,
) -> Path:
    """Build a harness that exercises the re-exec block + long sleep.

    The harness contains the verbatim re-exec block from ``install.sh``
    followed by a ``sleep`` simulating ``tsc`` and a ``touch`` of the
    marker file.  After the re-exec the body runs inside the new
    session (created by ``POSIX::setsid`` in the child), so signals
    sent to the harness's original process group cannot reach it.
    """
    block = _extract_reexec_block()
    harness = tmp_path / "harness.sh"
    harness.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            # Mirror install.sh's top: the re-exec block runs FIRST, before
            # anything else.  Any signal delivered before the block has had
            # a chance to fork+setsid would still hit the original bash —
            # but the block is unconditional and fast (perl fork is well
            # under a millisecond) so the test's first signal (sent after
            # a generous warm-up delay) always lands on the detached body.
            LOG_FILE={log.as_posix()!r}
            exec_log() {{ printf '%s\\n' "$*" >> "$LOG_FILE" 2>/dev/null || true; }}
            exec_log "harness pre-reexec pid=$$ sid=via-python"
            """
        )
        + block
        + textwrap.dedent(
            f"""\

            # Post-reexec install body.  Sleep simulates the long ``tsc``
            # compile from the bug report.  If a signal reaches us, bash's
            # default disposition would kill us before the marker write.
            exec_log "harness post-reexec pid=$$"
            sleep {sleep_seconds}
            : > {marker.as_posix()!r}
            exec_log "harness marker written"
            """
        ),
        encoding="utf-8",
    )
    harness.chmod(0o755)
    return harness


def _kill_pgrp_repeatedly(
    proc: subprocess.Popen[bytes], sig: signal.Signals, count: int = 3
) -> None:
    """Send *sig* to ``proc``'s process group *count* times, ~300 ms apart.

    The intent is to mimic the VS Code PTY teardown burst — VS Code can
    inject ``\\x03`` and then close the PTY in rapid succession, and a
    laptop wake-from-sleep can also produce multiple signals.  If even
    one of these makes it through to the install body the test fails.
    """
    for _ in range(count):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        time.sleep(0.3)


def _cleanup(proc: subprocess.Popen[bytes]) -> None:
    """Force-kill the harness's process group if it is still alive."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def test_install_sh_reexecs_in_new_session_via_perl() -> None:
    """``install.sh`` must contain the perl-based new-session re-exec block.

    Structural invariant: the very top of the script (before
    ``set -eo pipefail``) must guard a ``perl + POSIX::setsid`` re-exec
    behind the ``_KISS_NEW_SESSION`` sentinel.  Without this block the
    install body inherits the VS Code terminal's controlling TTY and
    every terminal-driven signal reaches it.
    """
    src = _read_install_sh()

    # The guard must exist and appear before ``set -eo pipefail``.
    assert _GUARD_PREFIX in src, (
        f"install.sh missing the {_GUARD_PREFIX!r} guard; the new-session "
        "re-exec block must be present at the top of the script so a "
        "re-execd child does not fork again."
    )
    begin_idx = src.index(_GUARD_PREFIX)
    set_e_idx = src.index(_SET_E_LINE)
    assert begin_idx < set_e_idx, (
        "the new-session re-exec block must precede `set -eo pipefail`; "
        "otherwise a signal arriving between `set -e` and the fork could "
        "still abort the install."
    )

    block = _extract_reexec_block()

    # Guard against infinite re-exec.
    assert "_KISS_NEW_SESSION" in block, (
        "the re-exec block must export and check `_KISS_NEW_SESSION` to "
        "prevent the re-execd child from forking again."
    )

    # The block must use perl and POSIX::setsid in the child.
    assert "perl" in block, (
        "the re-exec block must invoke perl (the only universally "
        "available helper that exposes setsid(2) via POSIX::setsid)."
    )
    assert "POSIX::setsid" in block, (
        "the perl child must call POSIX::setsid() to create the new "
        "session with no controlling TTY."
    )

    # The block must fork — calling setsid without forking would EPERM
    # because bash on install.sh is its own process-group leader.
    assert "fork()" in block, (
        "perl must fork before calling setsid (a process-group leader "
        "cannot call setsid; we must fork to escape that restriction)."
    )

    # The block must `exec` perl to replace bash, so stray signals to the
    # original PTY's process group never reach a bash that would default-
    # terminate on them.
    assert re.search(r"\bexec\s+/usr/bin/env\s+perl\b", block), (
        "the block must `exec /usr/bin/env perl …` so the bash process "
        "is REPLACED by perl; otherwise a stray SIGINT to the original "
        "pgrp would default-terminate bash even before fork()."
    )

    # The parent perl must IGNORE INT/TERM/HUP so a stray Ctrl-C cannot
    # take down the wait loop and leave the child orphaned.
    assert re.search(r'\$SIG\{INT\}\s*=\s*"IGNORE"', block), (
        'parent perl must set $SIG{INT} = "IGNORE" to absorb stray '
        "SIGINTs from the original terminal."
    )
    assert re.search(r'\$SIG\{TERM\}\s*=\s*"IGNORE"', block), (
        'parent perl must set $SIG{TERM} = "IGNORE".'
    )
    assert re.search(r'\$SIG\{HUP\}\s*=\s*"IGNORE"', block), (
        'parent perl must set $SIG{HUP} = "IGNORE" so PTY hangup does '
        "not kill the parent."
    )


@pytest.mark.process_killer
def test_install_sh_perl_reexec_creates_new_session_id_for_child(
    tmp_path: Path,
) -> None:
    """The re-exec child must end up in a NEW POSIX session.

    Run a harness whose body (post-re-exec) writes its own PID to a
    file, then sleeps.  Compare ``os.getsid(child_pid)`` to
    ``os.getsid(proc.pid)`` (the parent perl's SID).  They must differ
    — meaning ``POSIX::setsid()`` actually created a new session for
    the install body.
    """
    if shutil.which("perl") is None:
        # If perl is missing the re-exec block falls through; the
        # post-reexec body still runs but in the SAME session as the
        # harness, so this test would be a no-op.  Skip it.
        pytest.skip("perl not available; new-session detachment cannot run")

    child_pid_file = tmp_path / "child.pid"
    done = tmp_path / "done"
    block = _extract_reexec_block()
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        + block
        + textwrap.dedent(
            f"""\

            # After the re-exec we are the child bash inside the new
            # session.  Record our PID for the python test to inspect,
            # then sleep long enough for the test to read it and call
            # os.getsid() before we exit.
            echo $$ > {child_pid_file.as_posix()!r}
            sleep 3
            : > {done.as_posix()!r}
            """
        ),
        encoding="utf-8",
    )
    harness.chmod(0o755)

    proc = _spawn_harness(harness)
    try:
        # Poll for the child PID file with a tight bound.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not child_pid_file.exists():
            time.sleep(0.05)
        assert child_pid_file.exists(), (
            "harness did not write its post-re-exec PID; the re-exec "
            "block either failed or never ran"
        )
        child_pid = int(child_pid_file.read_text().strip())

        # Parent perl ended up at proc.pid (we exec'd bash → perl).
        parent_sid = os.getsid(proc.pid)
        child_sid = os.getsid(child_pid)

        assert parent_sid != child_sid, (
            "POSIX::setsid did not create a new session for the install "
            f"body: parent perl SID={parent_sid}, child bash SID="
            f"{child_sid}.  Without a new session the install body still "
            "shares the original VS Code PTY's controlling terminal and "
            "every stray SIGINT/SIGHUP/SIGTERM kills it."
        )
        # Furthermore, the child should be its own session leader.
        assert child_sid == child_pid, (
            "the re-exec'd child should be the leader of the new session "
            f"(child_pid={child_pid}, child_sid={child_sid})"
        )

        proc.wait(timeout=15)
        assert done.exists(), (
            "harness did not complete normally after the re-exec"
        )
    finally:
        _cleanup(proc)


def _run_signal_immunity_test(
    tmp_path: Path, sig: signal.Signals
) -> None:
    """Shared body: spawn re-exec harness, blast *sig* x3, expect rc=0."""
    marker = tmp_path / "marker"
    log = tmp_path / "harness.log"
    harness = _write_signal_harness(tmp_path, marker, log, sleep_seconds=5.0)

    proc = _spawn_harness(harness)
    try:
        # Let the harness reach the post-re-exec sleep.  The fork+setsid
        # path completes in well under 100 ms; 1 s is a generous warm-up.
        time.sleep(1.0)
        _kill_pgrp_repeatedly(proc, sig, count=3)
        try:
            rc = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise
    finally:
        _cleanup(proc)

    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    stdout = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
    assert rc == 0, (
        f"harness aborted on a stray {sig.name} burst — the new-session "
        f"re-exec did not detach the install body.  rc={rc!r}\n"
        f"Log:\n{log_text}\nStdout:\n{stdout}"
    )
    assert marker.exists(), (
        f"the install body did not run to completion after {sig.name} — "
        f"the kernel still delivered the signal to the install body.\n"
        f"Log:\n{log_text}\nStdout:\n{stdout}"
    )


@pytest.mark.process_killer
def test_install_sh_survives_sigint_during_long_running_child(
    tmp_path: Path,
) -> None:
    """A SIGINT burst during the install body must NOT abort the script.

    This is the exact failure mode reported by the user: ``\\x03`` was
    injected into the VS Code PTY during ``tsc`` and the install died.
    With the new-session re-exec, ``tsc`` runs in a session without a
    controlling TTY and the kernel cannot deliver terminal SIGINT to it.
    """
    if shutil.which("perl") is None:
        pytest.skip("perl not available; new-session detachment cannot run")
    _run_signal_immunity_test(tmp_path, signal.SIGINT)


@pytest.mark.process_killer
def test_install_sh_survives_sighup_during_long_running_child(
    tmp_path: Path,
) -> None:
    """A SIGHUP burst (PTY close) must NOT abort the install body.

    When VS Code disposes the integrated terminal that hosted the
    Update, the documented teardown sequence is ``\\x03`` followed by
    PTY close (SIGHUP).  Inside a fresh session with no controlling
    TTY there is no SIGHUP delivery to the install body.
    """
    if shutil.which("perl") is None:
        pytest.skip("perl not available; new-session detachment cannot run")
    _run_signal_immunity_test(tmp_path, signal.SIGHUP)


@pytest.mark.process_killer
def test_install_sh_survives_sigterm_during_long_running_child(
    tmp_path: Path,
) -> None:
    """A SIGTERM burst to the original pgrp must NOT abort the install.

    A stray kill of the foreground process group (e.g. ``pkill bash``
    from the user's other terminal, or a watchdog) used to take out
    the install.  After the new-session re-exec the install body is
    in a different process group and is unaffected.
    """
    if shutil.which("perl") is None:
        pytest.skip("perl not available; new-session detachment cannot run")
    _run_signal_immunity_test(tmp_path, signal.SIGTERM)


@pytest.mark.process_killer
def test_install_sh_perl_fallback_when_unavailable(tmp_path: Path) -> None:
    """When perl is missing the script must continue gracefully.

    Two invariants:

    1. **No infinite re-exec.** The block must terminate naturally and
       run the rest of the harness body.
    2. **No abort.** The fallback path must not return non-zero or
       leave a half-baked state.

    We simulate "perl missing" by setting ``PATH`` inside the harness
    to a directory that contains nothing — so ``command -v perl``
    returns false and the ``if`` falls through to the install body.
    """
    block = _extract_reexec_block()
    marker = tmp_path / "fallback_marker"
    empty_dir = tmp_path / "empty_path"
    empty_dir.mkdir()

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -eo pipefail\n"
        # Restrict PATH to a directory with NO commands at all.  bash's
        # builtins (`command`, `export`, `[`, etc.) still work, but
        # `command -v perl` returns nothing and the if-guard fails.
        f'export PATH={empty_dir.as_posix()!r}\n'
        + block
        + textwrap.dedent(
            f"""\

            # If the re-exec block aborted or looped we never get here.
            : > {marker.as_posix()!r}
            """
        ),
        encoding="utf-8",
    )
    harness.chmod(0o755)

    proc = subprocess.Popen(
        ["bash", str(harness)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise AssertionError(
                "fallback path did not finish within 10 s — the re-exec "
                "block may be looping when perl is unavailable"
            )
    finally:
        _cleanup(proc)

    stdout = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
    assert rc == 0, (
        "fallback path exited non-zero when perl was unavailable; the "
        f"block must degrade gracefully.  rc={rc!r}\nStdout:\n{stdout}"
    )
    assert marker.exists(), (
        "the install body did not run after the perl-unavailable "
        f"fallback — the script aborted prematurely.\nStdout:\n{stdout}"
    )


def test_install_sh_explanatory_comment_present() -> None:
    """The re-exec block's comments must explain the fix.

    A future maintainer must not "simplify" install.sh by removing the
    re-exec block.  The comments embedded in the block are the
    canonical rationale and must mention every load-bearing concept:
    ``setsid``, ``session``, the controlling ``terminal``, and the
    terminal-driven ``signal`` delivery it defeats.
    """
    block = _extract_reexec_block()
    comment = "\n".join(
        line for line in block.splitlines() if line.lstrip().startswith("#")
    )
    lowered = comment.lower()
    assert "setsid" in lowered, (
        "the re-exec block's comments must mention `setsid` — the system "
        "call that creates the new session without a controlling TTY."
    )
    assert "session" in lowered, (
        "the re-exec block's comments must mention `session` — the POSIX "
        "abstraction that scopes terminal-driven signal delivery."
    )
    assert "terminal" in lowered, (
        "the re-exec block's comments must mention the controlling "
        "`terminal` — the reason the detached session cannot receive "
        "SIGINT/SIGHUP from the VS Code PTY."
    )
    assert "signal" in lowered, (
        "the re-exec block's comments must mention `signal` — the "
        "terminal-driven delivery (SIGINT/SIGHUP/SIGTERM) the new "
        "session defeats."
    )

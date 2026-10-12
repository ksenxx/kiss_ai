# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-12 (sorcar-core): a CLI turn that is cut off must not
orphan the CLI's own subprocesses.

The real CLIs run tool commands (shells, tests, servers) as
grandchildren.  ``_CLIProcess.close()`` used to ``terminate()`` /
``kill()`` the direct child only on POSIX, so after a stall timeout or a
Stop those grandchildren kept running with no owner.  The child is now
started in its own process group and the whole group is signalled.

Every test drives a REAL stand-in ``claude`` (a Python script on
``PATH``) that forks a long-lived grandchild with its stdio on
``/dev/null`` (so the pipes play no part) and then stalls.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest

from kiss.core.models import claude_code_model as cc_module
from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_claude_cli
from kiss.core.models.model import _CLIProcess
from kiss.core.processes import kill_process_group, pid_alive
from kiss.tests.conftest import install_cli_script


@pytest.mark.parametrize("pid", [0, -1])
def test_kill_process_group_refuses_a_non_pid(pid: int) -> None:
    """``os.killpg(0, ...)`` is the caller's own group: the helper must raise, not kill us."""
    with pytest.raises(ProcessLookupError):
        kill_process_group(pid, signal.SIGTERM)


_GRANDCHILD_LIFETIME = 300
_STALL_TIMEOUT = 1


def _kill(pid_file: Path) -> None:
    """Kill the grandchild whose pid the stand-in recorded in *pid_file*."""
    if pid_file.exists():
        pid = int(pid_file.read_text())
        if pid_alive(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def _forks_then_stalls(
    pid_file: Path,
    ignore_term: bool,
    grandchild_ignores_term: bool = False,
    stall: bool = True,
) -> str:
    """Stand-in body: fork a detached grandchild, record its pid, then stall forever or exit.

    Args:
        pid_file: Where the grandchild's pid is written.
        ignore_term: Whether the stand-in itself ignores SIGTERM.
        grandchild_ignores_term: Whether the grandchild ignores SIGTERM.
        stall: Whether the stand-in sleeps after forking (``False``
            exits at once, leaving the grandchild behind).
    """
    grandchild_code = (
        "import signal, time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if grandchild_ignores_term else "")
        + f"time.sleep({_GRANDCHILD_LIFETIME})"
    )
    return f"""
    import signal
    import subprocess
    import sys
    import time

    if {ignore_term!r} and hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    grandchild = subprocess.Popen(
        [sys.executable, "-c", {grandchild_code!r}],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with open({str(pid_file)!r}, "w") as f:
        f.write(str(grandchild.pid))
    if {stall!r}:
        time.sleep({_GRANDCHILD_LIFETIME})
    """


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    """Return whether *pid* disappears within *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not pid_alive(pid)


def test_stall_timeout_kills_the_cli_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ClaudeCodeModel.generate()`` raising on a stalled turn leaves no grandchild behind."""
    pid_file = tmp_path / "grandchild.pid"
    install_cli_script(
        tmp_path / "claude",
        f"#!{sys.executable}\n" + textwrap.dedent(_forks_then_stalls(pid_file, False)),
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(cc_module, "_find_claude_cli", _find_claude_cli)
    model = ClaudeCodeModel("cc/opus", model_config={"timeout": _STALL_TIMEOUT})
    model.initialize("hi")
    try:
        with pytest.raises(TimeoutError, match="produced no output"):
            model.generate()
        assert pid_file.exists(), "the stand-in never forked its grandchild"
        gpid = int(pid_file.read_text())
        assert _wait_dead(gpid), f"grandchild {gpid} survived the timed-out turn"
    finally:
        _kill(pid_file)


@pytest.mark.parametrize("ignore_term", [False, True], ids=["sigterm", "sigkill-escalation"])
def test_close_on_a_live_child_kills_its_grandchild(tmp_path: Path, ignore_term: bool) -> None:
    """Leaving the ``with`` block while the child runs takes its whole tree down.

    With ``ignore_term`` the child ignores SIGTERM, so ``close()`` must
    escalate to SIGKILL for the group, not just for the child.
    """
    pid_file = tmp_path / "grandchild.pid"
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(_forks_then_stalls(pid_file, ignore_term)))
    try:
        with _CLIProcess([sys.executable, str(script)], "probe", 30) as proc:
            deadline = time.monotonic() + 30
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert pid_file.exists(), "the stand-in never forked its grandchild"
            child_pid = proc._proc.pid
        gpid = int(pid_file.read_text())
        assert proc._proc.poll() is not None, "close() did not reap the child"
        assert not pid_alive(child_pid)
        assert _wait_dead(gpid), f"grandchild {gpid} survived close()"
    finally:
        _kill(pid_file)


@pytest.mark.parametrize("stall", [True, False], ids=["child-exits-on-sigterm", "child-exited"])
def test_close_kills_a_grandchild_that_ignores_sigterm_whatever_the_child_did(
    tmp_path: Path, stall: bool
) -> None:
    """SIGKILL escalation covers the group, not only a child that survived SIGTERM.

    The grandchild traps SIGTERM while the child either dies on it
    (``stall``) or had already exited on its own before ``close()``;
    either way the child's status alone says nothing about the group,
    which must be SIGKILLed after the grace.
    """
    pid_file = tmp_path / "grandchild.pid"
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(_forks_then_stalls(
        pid_file, False, grandchild_ignores_term=True, stall=stall,
    )))
    try:
        with _CLIProcess([sys.executable, str(script)], "probe", 30) as proc:
            deadline = time.monotonic() + 30
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert pid_file.exists(), "the stand-in never forked its grandchild"
            if not stall:
                assert proc.wait_for_exit() == 0, "the stand-in did not exit on its own"
            started = time.monotonic()
        gpid = int(pid_file.read_text())
        assert proc._proc.poll() is not None, "close() did not reap the child"
        assert _wait_dead(gpid), f"grandchild {gpid} survived close()"
        assert time.monotonic() - started < 10
    finally:
        _kill(pid_file)

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit H fixes: the CLI auth probes are bounded, and the Codex child
environment is built from one reading of the subscription switch.

F1/F2 -- ``claude_auth_status`` and ``codex_auth_status`` run the CLI
through :func:`kiss.core.processes.run_process_group`: a CLI that hangs
past the timeout is killed with its whole process group (so a grandchild
holding the pipes dies too) and reaped with a bounded wait.  A stand-in
``claude`` / ``codex`` on a temporary PATH spawns a grandchild that
inherits the output pipes and sleeps for a minute; the probe must still
return within the timeout plus a small grace, and the grandchild must be
gone.  Before the fix the Claude probe was a plain ``subprocess.run``,
whose timeout kills the direct child only: the grandchild lived on.

F2 (detached) -- a grandchild that moved into its own session is beyond
the group's reach and keeps the pipes open.  The probe must still return
once the stand-in itself is reaped: ``subprocess.run`` parked until the
grandchild closed the pipes, a minute later.  The test kills that
grandchild itself.

Stop -- ``run_process_group`` kills the child on *any* exception raised
while it waits, not only on the timeout: a Stop (KeyboardInterrupt, an
injected exception) that landed in the wait used to propagate with the
child still running.  A ``SIGALRM`` handler raises in the main thread
while the call waits for a sleeping child; the exception propagates and
the child is dead and reaped.

F3 -- ``codex_cli_env`` read ``DEFAULT_CONFIG.codex_use_subscription``
twice; a settings save flipping it between the reads produced an
environment with the API key variables kept *and* ``CODEX_ACCESS_TOKEN``
added.  A thread flips the switch while the environment is built in a
loop; no environment may hold both.

Liveness is judged by *termination*, not by ``pid_alive``: a killed
grandchild is a zombie until its new parent waits for it, and under a
``PR_SET_CHILD_SUBREAPER`` ancestor that never does, ``os.kill(pid, 0)``
reports it alive forever.  On Linux the test makes itself the subreaper
for the probe's duration, so the orphan becomes its child and is reaped
here explicitly.
"""

import contextlib
import ctypes
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.claude_code_model import claude_auth_status
from kiss.core.models.codex_model import codex_auth_status, codex_cli_env
from kiss.core.processes import pid_alive, run_process_group
from kiss.tests.conftest import install_cli_script, posix_only

PROBE_TIMEOUT = 0.5
# Slack for the kill and the reap of the stand-in after the timeout.
RETURN_GRACE = 2.0
# A killed grandchild is a zombie until its new parent reaps it.
REAP_WAIT = 3.0

_PR_SET_CHILD_SUBREAPER = 36

# The stand-in CLI: starts a grandchild that inherits stdout/stderr and
# sleeps, records the grandchild's pid, and waits for it (so the stand-in
# itself never answers the probe).  ``detached`` moves the grandchild
# into its own session, out of the stand-in's process group.
_HANGING_CLI = """#!/usr/bin/env python3
import pathlib, subprocess, sys
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session={detached!r}
)
pathlib.Path({pid_path!r}).write_text(str(child.pid))
child.wait()
"""


def _adopt_orphans(enabled: bool) -> None:
    """Make this process the parent of every descendant whose parent dies (Linux only)."""
    if sys.platform == "linux":
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, int(enabled), 0, 0, 0)


def _terminated(pid: int) -> bool:
    """Whether *pid* has exited: already reaped, or a zombie its parent has not waited for."""
    if not pid_alive(pid):
        return True
    try:
        with open(f"/proc/{pid}/stat") as stat:  # Linux; elsewhere a zombie reads as alive
            return stat.read().rpartition(")")[2].split()[0] == "Z"
    except OSError:
        return False


def _kill_and_reap(pid: int) -> None:
    """Kill *pid* if it still runs, then reap it when it is a child of this process."""
    with contextlib.suppress(OSError):
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    with contextlib.suppress(OSError):  # not (or no longer) a child of this process
        os.waitpid(pid, 0)


class _HangingCli:
    """A stand-in *name* on PATH whose grandchild holds the pipes."""

    def __init__(
        self, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detached: bool
    ) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        self.pid_path = tmp_path / "grandchild.pid"
        source = _HANGING_CLI.format(pid_path=str(self.pid_path), detached=detached)
        install_cli_script(bin_dir / name, source)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def grandchild_pid(self) -> int:
        """The pid the stand-in recorded (it is written before the probe can time out)."""
        return int(self.pid_path.read_text())

    def grandchild_terminated(self) -> bool:
        """Whether the grandchild has exited (polled up to ``REAP_WAIT``)."""
        pid = self.grandchild_pid()
        deadline = time.monotonic() + REAP_WAIT
        while not _terminated(pid):
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def cleanup(self) -> None:
        """Kill and reap a grandchild the probe left behind, so it does not outlive the test."""
        if self.pid_path.exists():
            _kill_and_reap(self.grandchild_pid())


def _probe_with_hanging_cli(
    name: str,
    probe: Callable[[float], dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detached: bool = False,
) -> dict[str, Any]:
    """Run *probe* against a hanging stand-in; assert it is bounded and leaves nothing behind."""
    cli = _HangingCli(name, tmp_path, monkeypatch, detached)
    _adopt_orphans(True)
    try:
        started = time.monotonic()
        status = probe(PROBE_TIMEOUT)
        elapsed = time.monotonic() - started
        assert elapsed < PROBE_TIMEOUT + RETURN_GRACE, f"probe took {elapsed:.2f}s"
        if detached:
            assert pid_alive(cli.grandchild_pid()), "a detached grandchild is beyond the group"
        else:
            assert cli.grandchild_terminated(), "the grandchild holding the pipes survived"
    finally:
        cli.cleanup()
        _adopt_orphans(False)
    assert status["installed"] is True
    assert "error" in status
    return status


def test_claude_auth_status_kills_the_grandchild_holding_the_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: the Claude probe returns within its timeout and its whole group is gone."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", True)
    status = _probe_with_hanging_cli("claude", claude_auth_status, tmp_path, monkeypatch)
    assert "failed" in status["error"]
    assert "timed out" in status["error"]


def test_codex_auth_status_kills_the_grandchild_holding_the_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2: the Codex probe returns within its timeout and its whole group is gone."""
    monkeypatch.setattr("kiss.core.models.codex_model._UI_CANDIDATE_PATHS", ())
    status = _probe_with_hanging_cli("codex", codex_auth_status, tmp_path, monkeypatch)
    assert "timed out" in status["error"]


@posix_only("start_new_session moves the grandchild out of the group")
def test_codex_auth_status_returns_while_a_detached_grandchild_holds_the_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2 (detached): the probe does not wait for pipes a grandchild outside the group keeps."""
    monkeypatch.setattr("kiss.core.models.codex_model._UI_CANDIDATE_PATHS", ())
    status = _probe_with_hanging_cli(
        "codex", codex_auth_status, tmp_path, monkeypatch, detached=True
    )
    assert "timed out" in status["error"]


class _StopError(Exception):
    """Stands in for the exception a Stop raises in a waiting task thread."""


def _raise_stop(signum: int, frame: Any) -> None:
    raise _StopError


# Seconds the child gets to start and record its pid before the wait is interrupted.
INTERRUPT_AFTER = 1.0


@posix_only("SIGALRM interrupts the wait in the main thread")
def test_run_process_group_kills_the_child_when_its_wait_is_interrupted(tmp_path: Path) -> None:
    """Stop: an exception raised while the call waits leaves no running child behind."""
    pid_path = tmp_path / "child.pid"
    program = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    previous = signal.signal(signal.SIGALRM, _raise_stop)
    signal.setitimer(signal.ITIMER_REAL, INTERRUPT_AFTER)
    started = time.monotonic()
    try:
        with pytest.raises(_StopError):
            run_process_group([sys.executable, "-c", program], timeout=30)
        elapsed = time.monotonic() - started
        pid = int(pid_path.read_text())
        assert elapsed < INTERRUPT_AFTER + RETURN_GRACE, f"call took {elapsed:.2f}s"
        assert not pid_alive(pid), "the child outlived the interrupted wait"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if pid_path.exists():
            _kill_and_reap(int(pid_path.read_text()))


@pytest.fixture
def switch_flipper(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """A thread flipping ``codex_use_subscription`` as fast as it can until the event is set."""
    stop = threading.Event()

    def flip() -> None:
        while not stop.is_set():
            DEFAULT_CONFIG.codex_use_subscription = not DEFAULT_CONFIG.codex_use_subscription

    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", True)
    thread = threading.Thread(target=flip, name="switch-flipper", daemon=True)
    thread.start()
    try:
        yield stop
    finally:
        stop.set()
        thread.join(timeout=5)


def test_codex_cli_env_never_mixes_key_and_token_while_the_switch_flips(
    switch_flipper: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3: one reading of the switch per build -- keys or token, never both."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("CODEX_API_KEY", "sk-codex-test")
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(DEFAULT_CONFIG, "CODEX_ACCESS_TOKEN", "codex-token-panel")
    deadline = time.monotonic() + 1.5
    seen_on = seen_off = False
    while time.monotonic() < deadline:
        env = codex_cli_env()
        has_key = "CODEX_API_KEY" in env or "OPENAI_API_KEY" in env
        has_token = "CODEX_ACCESS_TOKEN" in env
        assert not (has_key and has_token), "environment built from two switch states"
        assert has_key or has_token, "environment has neither credential"
        seen_on |= has_token
        seen_off |= has_key
    switch_flipper.set()
    assert seen_on and seen_off, "the flipper never exposed both states"

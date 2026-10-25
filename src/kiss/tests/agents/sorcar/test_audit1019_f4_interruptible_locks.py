# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-19 (F4): every ``repo_lock`` taker is stoppable and every flock wait is bounded.

* S2-1 — ``_do_merge``, ``_discard_one`` and the ``GitWorktreeOps`` task-thread
  paths used to park in the untimed C-level ``RLock.acquire()`` while
  another tab's merge held ``repo_lock`` for minutes, so the daemon's
  Stop (an injected ``KeyboardInterrupt``) reached them only when the
  holder released.  They now acquire through
  ``interruptible_locked_call`` (short slices); the shape of each test
  is ``test_conc1018_f_repo_lock_stop.py``: a holder thread keeps the
  lock for ``_HOLD_SECONDS``, the blocked thread is stopped after one
  second and must end well before the holder releases.
* S4-1 — ``_reclaim_process_lock`` and ``_file_lock`` waited without
  limit in the kernel's ``flock``; a wedged other process hung the
  waiter and nothing could stop it.  The wait is now polled (so a Stop
  lands) and bounded by ``PROCESS_LOCK_TIMEOUT_SECONDS`` with a clear
  ``KISSError``.  A real child process holds the flock here.
* S4-3 — ``_git_stdout_head`` read git's stdout with a plain ``read``,
  which a descendant that left git's process group (``setsid``) kept
  open past the watchdog kill; and a git that could not start raised
  out of it.  A git alias starting such a descendant shows the read
  giving up ``_PIPE_GIVE_UP_SECONDS`` after the kill, and an empty
  ``PATH`` shows the "no output" return.
* D2-1 — a re-entrant ``interruptible_locked_call`` (the thread already
  owns the ``RLock``) used to enter its ``try`` before acquiring, so a
  Stop landing between the two made the ``finally`` release the OUTER
  hold (no ``RuntimeError``: the recursion count just dropped to 0).
  It now runs the critical section directly when the thread owns the
  lock.  A ``sys.settrace`` line hook forces the interrupt onto the
  acquire line and a second thread checks the outer hold survived.
"""

from __future__ import annotations

import contextlib
import functools
import linecache
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar._concurrency import interruptible_locked_call, locked_call
from kiss.agents.sorcar.git_worktree import (
    _PIPE_GIVE_UP_SECONDS,
    GitWorktree,
    GitWorktreeOps,
    _git_common_dir,
    _git_stdout_head,
    _reclaim_process_lock,
    repo_lock,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.kiss_error import KISSError
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.conftest import posix_only

_HOLD_SECONDS = 8.0


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("x\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)
    return path.resolve()


@pytest.fixture
def repo() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        yield _make_repo(Path(tmp) / "repo")


def _hold_lock(repo: Path, held: threading.Event, release: threading.Event) -> None:
    """Tab A: the merge's conflict resolver, running under ``repo_lock``."""
    lock = repo_lock(repo)
    lock.acquire()
    try:
        held.set()
        release.wait(_HOLD_SECONDS)
    finally:
        lock.release()


def _run_until_stopped(
    taker: Callable[[], object], started: threading.Event, outcome: dict[str, object],
) -> None:
    """Tab B: a ``repo_lock`` taker on the same repository."""
    started.set()
    try:
        outcome["result"] = taker()
    except KeyboardInterrupt:
        outcome["stopped_at"] = time.monotonic()


def _assert_stop_lands(
    repo: Path, taker: Callable[[], object], blocker: Callable[[], None],
) -> None:
    """Run *taker* on a thread while *blocker* keeps it waiting; stop it after 1 s."""
    started = threading.Event()
    outcome: dict[str, object] = {}
    blocked = threading.Thread(
        target=_run_until_stopped, args=(taker, started, outcome), daemon=True,
    )
    blocked.start()
    assert started.wait(5)
    time.sleep(1.0)
    assert blocked.is_alive()
    t_inject = time.monotonic()
    assert blocked.ident is not None
    assert inject_keyboard_interrupt(blocked.ident) == 1
    blocked.join(timeout=_HOLD_SECONDS + 5)
    blocker()
    assert not blocked.is_alive()
    assert "result" not in outcome, outcome
    stopped_at = outcome["stopped_at"]
    assert isinstance(stopped_at, float)
    assert stopped_at - t_inject < 2.0, f"Stop took {stopped_at - t_inject:.1f}s"
    # The interrupted taker left the lock free (no leaked hold).
    lock = repo_lock(repo)
    assert lock.acquire(blocking=False)
    lock.release()


def _pending_worktree(repo: Path) -> tuple[SorcarAgent, GitWorktree]:
    """An agent with one real worktree pending on *repo*."""
    wt_dir = repo.parent / "wt"
    assert GitWorktreeOps.create(repo, "kiss/wt-f4", wt_dir)
    agent = SorcarAgent("audit1019-f4-tab-b")
    agent.work_dir = str(repo)
    wt = GitWorktree(repo_root=repo, branch="kiss/wt-f4", original_branch="master", wt_dir=wt_dir)
    agent._wt = wt
    return agent, wt


def _takers(repo: Path) -> dict[str, Callable[[], object]]:
    """Every ``repo_lock`` taker a task thread reaches, keyed by name."""
    agent, wt = _pending_worktree(repo)
    return {
        "_do_merge": functools.partial(agent._do_merge, wt),
        "_discard_one": functools.partial(agent._discard_one, wt, False),
        "create": functools.partial(
            GitWorktreeOps.create, repo, "kiss/wt-f4b", repo.parent / "wt2",
        ),
        "reattach": functools.partial(GitWorktreeOps.reattach, repo, wt.branch, wt.wt_dir),
        "remove": functools.partial(GitWorktreeOps.remove, repo, wt.wt_dir),
        "prune": functools.partial(GitWorktreeOps.prune, repo),
        "_append_info_line": functools.partial(
            GitWorktreeOps._append_info_line, repo, "exclude", "/f4-probe",
        ),
        "sweep_orphaned_state": functools.partial(GitWorktreeOps.sweep_orphaned_state, repo),
        "reclaim_orphaned_worktrees": functools.partial(
            GitWorktreeOps.reclaim_orphaned_worktrees, repo,
        ),
    }


@pytest.mark.parametrize(
    "name",
    [
        "_do_merge", "_discard_one", "create", "reattach", "remove", "prune",
        "_append_info_line", "sweep_orphaned_state", "reclaim_orphaned_worktrees",
    ],
)
def test_stop_lands_while_blocked_on_another_tabs_repo_lock(repo: Path, name: str) -> None:
    taker = _takers(repo)[name]
    held, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold_lock, args=(repo, held, release), daemon=True)
    holder.start()
    assert held.wait(5)

    def release_holder() -> None:
        release.set()
        holder.join(timeout=5)

    _assert_stop_lands(repo, taker, release_holder)


# --- S4-1: the cross-process flock wait is bounded and interruptible ---

_FLOCK_HOLDER = """
import fcntl, sys, time
handle = open(sys.argv[1], "a+")
fcntl.flock(handle, fcntl.LOCK_EX)
print("held", flush=True)
time.sleep(float(sys.argv[2]))
"""


def _hold_flock_in_child(lock_path: Path, seconds: float) -> subprocess.Popen[str]:
    """Another Sorcar process holding *lock_path* for *seconds*."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _FLOCK_HOLDER, str(lock_path), str(seconds)],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "held"
    return proc


def _end_child(proc: subprocess.Popen[str]) -> None:
    proc.kill()
    proc.wait(timeout=5)


@posix_only("fcntl.flock held by a child process")
def test_reclaim_lock_held_by_another_process_raises_after_the_bound(repo: Path) -> None:
    git_common = _git_common_dir(repo)
    assert git_common is not None
    lock_path = git_common / "kiss-reclaim.lock"
    child = _hold_flock_in_child(lock_path, 30)
    try:
        t0 = time.monotonic()
        with pytest.raises(KISSError, match="kiss-reclaim.lock.*another Sorcar process"):
            with _reclaim_process_lock(repo, timeout=1.0):
                pass
        assert 1.0 <= time.monotonic() - t0 < 5.0
    finally:
        _end_child(child)
    # Released by the child: the lock is taken at once and the normal path runs.
    t0 = time.monotonic()
    with _reclaim_process_lock(repo, timeout=1.0):
        pass
    assert time.monotonic() - t0 < 1.0


@posix_only("fcntl.flock held by a child process")
def test_stop_lands_while_waiting_for_another_processes_flock(repo: Path) -> None:
    """``GitWorktreeOps.create`` on a task thread, blocked by another process's flock."""
    git_common = _git_common_dir(repo)
    assert git_common is not None
    child = _hold_flock_in_child(git_common / "kiss-reclaim.lock", _HOLD_SECONDS)
    taker = functools.partial(GitWorktreeOps.create, repo, "kiss/wt-f4c", repo.parent / "wt3")
    _assert_stop_lands(repo, taker, functools.partial(_end_child, child))


# --- S4-3: ``_git_stdout_head`` cannot be held open by a stray descendant ---

_SESSION_LEADER = """
import os, sys, time
os.setsid()
with open(sys.argv[1], "w") as f:
    f.write(str(os.getpid()))
sys.stdout.write("hello\\n")
sys.stdout.flush()
time.sleep(float(sys.argv[2]))
"""


@posix_only("setsid and SIGKILL")
def test_stdout_head_gives_up_on_a_pipe_held_by_a_descendant_in_its_own_session(
    repo: Path, tmp_path: Path,
) -> None:
    script = tmp_path / "leader.py"
    script.write_text(_SESSION_LEADER)
    pid_file = tmp_path / "leader.pid"
    alias = "!" + " ".join(
        shlex.quote(a) for a in (sys.executable, str(script), str(pid_file), "60")
    )
    t0 = time.monotonic()
    try:
        text, truncated = _git_stdout_head(
            "-c", f"alias.hang={alias}", "hang", cwd=repo, max_bytes=1000, timeout=1.0,
        )
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
    elapsed = time.monotonic() - t0
    assert (text, truncated) == ("hello", True)
    assert 1.0 <= elapsed < 1.0 + _PIPE_GIVE_UP_SECONDS + 4.0, f"read held for {elapsed:.1f}s"


def test_stdout_head_without_git_on_the_path_reports_no_output(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert _git_stdout_head("status", cwd=repo, max_bytes=100) == ("", False)
    assert GitWorktreeOps.staged_diff(repo) == ""


# --- D2-1: a Stop before a re-entrant acquire must not release the outer hold ---

_INNER_CODE = interruptible_locked_call.__code__


def _interrupt_at_acquire(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace hook: ``KeyboardInterrupt`` on the acquire line of ``interruptible_locked_call``."""
    if frame.f_code is not _INNER_CODE:
        return None
    source_line = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
    if event == "line" and "lock.acquire(" in source_line:
        raise KeyboardInterrupt
    return _interrupt_at_acquire


def _inner() -> str:
    return "ran"


def _try_take(lock: threading.RLock, outcome: dict[str, bool]) -> None:
    outcome["taken"] = lock.acquire(blocking=False)
    if outcome["taken"]:
        lock.release()


def _other_thread_can_take(lock: threading.RLock) -> bool:
    outcome: dict[str, bool] = {}
    probe = threading.Thread(target=_try_take, args=(lock, outcome), daemon=True)
    probe.start()
    probe.join(timeout=5)
    return outcome["taken"]


def _outer(lock: threading.RLock, outcome: dict[str, object]) -> None:
    """The outer critical section: a nested take, with a Stop landing before its acquire."""
    sys.settrace(_interrupt_at_acquire)
    try:
        outcome["inner"] = interruptible_locked_call(lock, _inner)
    except KeyboardInterrupt:
        outcome["stopped"] = True
    finally:
        sys.settrace(None)
    # Still inside the outer hold: no other thread may take the lock.
    outcome["leaked"] = _other_thread_can_take(lock)


def test_stop_before_a_reentrant_acquire_keeps_the_outer_hold(repo: Path) -> None:
    lock = repo_lock(repo)
    outcome: dict[str, object] = {}
    locked_call(lock, _outer, lock, outcome)
    assert outcome["leaked"] is False, outcome
    assert outcome.get("inner") == "ran", outcome
    assert _other_thread_can_take(lock)

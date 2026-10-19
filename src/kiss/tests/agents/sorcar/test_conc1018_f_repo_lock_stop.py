# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop reaches a task blocked on another tab's ``repo_lock`` in ``create_worktree``.

Tab A merges a worktree whose conflict resolver runs for minutes under
``repo_lock(repo)``; tab B calls ``create_worktree`` on the same
repository and parks in the lock's ``acquire``.  The daemon stops B by
injecting ``KeyboardInterrupt`` (``inject_keyboard_interrupt``), which
CPython delivers only at a bytecode check point: a thread inside the
untimed C-level ``RLock.acquire()`` reaches none until A releases.
``_try_setup_worktree`` now acquires in short slices, so the injected
Stop lands within one slice.

The test is the same shape with real threads: the holder thread keeps
the repository's lock for ``_HOLD_SECONDS``; the blocked thread runs
``SorcarAgent._try_setup_worktree``; the interrupt is injected after
one second and the blocked thread must end well before the holder
releases.  Before the fix it ends only after ``_HOLD_SECONDS``.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.git_worktree import repo_lock
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.task_runner import inject_keyboard_interrupt

_HOLD_SECONDS = 8.0


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("x\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)
    return path


def _hold_lock(repo: Path, held: threading.Event, release: threading.Event) -> None:
    """Tab A: the merge's conflict resolver, running under ``repo_lock``."""
    lock = repo_lock(repo)
    lock.acquire()
    try:
        held.set()
        release.wait(_HOLD_SECONDS)
    finally:
        lock.release()


def _blocked_create(
    agent: SorcarAgent,
    repo: Path,
    started: threading.Event,
    outcome: dict[str, object],
) -> None:
    """Tab B: ``create_worktree`` on the same repository."""
    started.set()
    try:
        outcome["result"] = agent._try_setup_worktree(repo, str(repo))
    except KeyboardInterrupt:
        outcome["stopped_at"] = time.monotonic()


def test_stop_lands_while_blocked_on_another_tabs_repo_lock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo").resolve()
        agent = SorcarAgent("conc1018-f-tab-b")
        agent.work_dir = str(repo)
        held, release, started = threading.Event(), threading.Event(), threading.Event()
        outcome: dict[str, object] = {}
        holder = threading.Thread(target=_hold_lock, args=(repo, held, release), daemon=True)
        blocked = threading.Thread(
            target=_blocked_create,
            args=(agent, repo, started, outcome),
            daemon=True,
        )
        holder.start()
        assert held.wait(5)
        blocked.start()
        assert started.wait(5)
        time.sleep(1.0)
        assert blocked.is_alive()
        t_inject = time.monotonic()
        assert blocked.ident is not None
        assert inject_keyboard_interrupt(blocked.ident) == 1
        blocked.join(timeout=_HOLD_SECONDS + 5)
        release.set()
        holder.join(timeout=5)
        assert not blocked.is_alive()
        assert "result" not in outcome, outcome
        stopped_at = outcome["stopped_at"]
        assert isinstance(stopped_at, float)
        assert stopped_at - t_inject < 2.0, f"Stop took {stopped_at - t_inject:.1f}s"
        assert not agent._wts
        # The interrupted taker left the lock free (no leaked hold).
        lock = repo_lock(repo)
        assert lock.acquire(blocking=False)
        lock.release()

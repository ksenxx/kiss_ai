# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Worktree cleanup commits and removes at once.

Everything here is real: a real git repository and a real ``git
worktree``, a real (local, free) OpenAI-compatible endpoint, and the
real cleanup path.  Nothing is mocked or patched.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _WorktreeCleanupOutcome,
)
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    OfflineFastModel,
    StandInModelServer,
    finish_response,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo.

    The auto-commit message generator is kept offline for the whole
    test so the cleanup path costs nothing and its timing is that of
    plain git rather than of a network round trip.
    """
    isolated = IsolatedKissHome("kiss-late-subagent-")
    try:
        with OfflineFastModel():
            yield isolated
    finally:
        isolated.cleanup()


def _branch_file(repo: Path, branch: str, name: str) -> str:
    """Return *name*'s content on *branch*, or ``""`` when absent."""
    done = subprocess.run(
        ["git", "show", f"{branch}:{name}"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout if done.returncode == 0 else ""


def test_cleanup_commits_and_removes_the_worktree(
    env: IsolatedKissHome,
) -> None:
    """The cleanup commits the worktree's changes and removes it without waiting."""
    server = StandInModelServer(lambda request: finish_response("quick"))
    printer = CapturePrinter()
    parent = WorktreeSorcarAgent("clean-cleanup")
    parent.printer = printer
    parent.model_name = STANDIN_MODEL
    parent.model_config = server.model_config

    wt_work_dir = parent._try_setup_worktree(env.repo, str(env.repo))
    assert wt_work_dir is not None
    wt = parent._wt
    assert wt is not None
    parent.work_dir = str(wt_work_dir)
    (wt_work_dir / "only-output.txt").write_text("work\n", encoding="utf-8")

    try:
        started = time.monotonic()
        outcome, leftover = parent._commit_and_clean_worktree(wt)
        elapsed = time.monotonic() - started
    finally:
        server.stop()

    assert outcome is _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED, leftover
    assert not wt.wt_dir.exists()
    assert elapsed < 5.0, "a cleanup must not wait for anything"
    assert _branch_file(env.repo, wt.branch, "only-output.txt") == "work\n"

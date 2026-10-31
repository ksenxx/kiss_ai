# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-30, slice S3: ``git_worktree.py`` / ``merge_conflict_resolver.py`` fixes.

F1  A Stop (``KeyboardInterrupt``) landing inside
    ``begin_conflicted_merge`` after git applied the conflicting change
    left the user's checkout conflicted: ``resolve_merge_conflict``
    called it before entering the ``try`` whose handler restores the
    tree.
F2  ``git status --porcelain`` honours ``status.showUntrackedFiles=no``,
    so the user's untracked files were invisible to every clean/dirty
    decision: ``stash_if_dirty`` skipped the stash, and a merge
    resolution that was stopped ran ``git clean -fd`` over them.
F3  ``git commit -F -`` honours ``commit.cleanup=strip``: message lines
    starting with ``#`` were dropped, a message made only of such lines
    aborted the commit, and a squash merge stamped the task's prompt and
    result a second time because the branch HEAD's copy had lost its
    ``###`` heading.
F4  ``sweep_orphaned_state`` deleted a pending task branch whose
    worktree directory was already removed but whose merge had not run
    yet: one of a live agent of this process (``exclude_branches``) or
    one owned by another live process (its owner stamp).

The Stop is landed with a ``sys.settrace`` hook that raises
``KeyboardInterrupt`` at the first bytecode of a function of the real
code path, which is where the runner's asynchronously injected Stop can
land; nothing is replaced.  Real git repositories, real git config and
a real second process otherwise.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import merge_conflict_resolver
from kiss.agents.sorcar.commit_message import fallback_commit_message
from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SUBDIR,
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
)
from kiss.agents.sorcar.merge_conflict_resolver import resolve_merge_conflict


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "T")
    (path / "f.txt").write_text("a\nb\nc\n")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "initial")
    GitWorktreeOps.prepare_repo(path)
    return path


@pytest.fixture
def repo() -> Iterator[Path]:
    tmp = tempfile.mkdtemp(prefix="audit1030-s3-")
    try:
        yield _make_repo(Path(tmp) / "repo")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _new_worktree(repo: Path) -> tuple[str, Path]:
    """Create a task worktree the way a task does; return its branch and directory."""
    branch = GitWorktreeOps.new_task_branch(repo)
    wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    return branch, wt_dir


def _conflicting_worktree(repo: Path) -> GitWorktree:
    """A task worktree that edited line b of ``f.txt`` while the user edited it on main."""
    baseline = _git(repo, "rev-parse", "HEAD").strip()
    branch, wt_dir = _new_worktree(repo)
    (wt_dir / "f.txt").write_text("a\nB-agent\nc\n")
    assert GitWorktreeOps.commit_all(wt_dir, "agent work")
    (repo / "f.txt").write_text("a\nB-user\nc\n")
    _git(repo, "commit", "-qam", "user work")
    return GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch="main",
        wt_dir=wt_dir,
        baseline_commit=baseline,
    )


class _StopAt:
    """``sys.settrace`` hook raising the runner's Stop when *code* is entered.

    CPython uninstalls a trace function that raises, so the hook fires once.
    """

    def __init__(self, code: CodeType) -> None:
        self._code = code
        self.fired = False

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is self._code:
            self.fired = True
            raise KeyboardInterrupt("Stop")


def _resolve_with_stop_at(code: CodeType, wt: GitWorktree) -> None:
    """Run ``resolve_merge_conflict`` with a Stop landing at the entry of *code*."""
    stop = _StopAt(code)
    previous = sys.gettrace()  # coverage's tracer, when measuring
    sys.settrace(stop)
    try:
        with pytest.raises(KeyboardInterrupt):
            resolve_merge_conflict(None, wt, "change line b", "changed line b")
    finally:
        sys.settrace(previous)
    assert stop.fired, "the Stop never landed"


# --- F1 -----------------------------------------------------------------


def test_stop_inside_begin_conflicted_merge_restores_the_checkout(repo: Path) -> None:
    """A Stop after git applied the conflicting change leaves a clean checkout."""
    wt = _conflicting_worktree(repo)
    head = GitWorktreeOps.head_sha(repo)

    # ``unresolved_paths`` is first called inside ``begin_conflicted_merge``,
    # right after the cherry-pick wrote the conflict into the tree.
    _resolve_with_stop_at(GitWorktreeOps.unresolved_paths.__code__, wt)

    assert GitWorktreeOps.status_porcelain(repo) == ""
    assert (repo / "f.txt").read_text() == "a\nB-user\nc\n"
    assert GitWorktreeOps.head_sha(repo) == head
    assert GitWorktreeOps.branch_exists(repo, wt.branch), "the task's work must survive"


# --- F2 -----------------------------------------------------------------


def test_untracked_files_count_as_dirty_with_show_untracked_files_no(repo: Path) -> None:
    """``status.showUntrackedFiles=no`` does not hide the user's new files from the stash."""
    _git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "notes.txt").write_text("mine\n")

    assert GitWorktreeOps.status_porcelain(repo) == "?? notes.txt"
    assert GitWorktreeOps.has_uncommitted_changes(repo)
    assert GitWorktreeOps.stash_if_dirty(repo) is True
    assert not (repo / "notes.txt").exists()
    assert GitWorktreeOps.stash_pop(repo)
    assert (repo / "notes.txt").read_text() == "mine\n"


def test_stopped_merge_resolution_keeps_untracked_files_with_show_untracked_files_no(
    repo: Path,
) -> None:
    """The restore after a stopped merge agent does not ``git clean`` the user's files."""
    _git(repo, "config", "status.showUntrackedFiles", "no")
    wt = _conflicting_worktree(repo)
    (repo / "notes.txt").write_text("mine\n")

    _resolve_with_stop_at(merge_conflict_resolver.run_merge_sea.__code__, wt)

    assert (repo / "notes.txt").read_text() == "mine\n"
    assert (repo / "f.txt").read_text() == "a\nB-user\nc\n"
    assert GitWorktreeOps.status_porcelain(repo) == "?? notes.txt"


# --- F3 -----------------------------------------------------------------


def test_commit_keeps_hash_lines_with_commit_cleanup_strip(repo: Path) -> None:
    """``commit.cleanup=strip`` does not drop message lines starting with ``#``."""
    _git(repo, "config", "commit.cleanup", "strip")
    (repo / "f.txt").write_text("changed\n")
    GitWorktreeOps.stage_all(repo)
    message = "#42 fix the parser\n\n# Summary\nkept line\n"

    assert GitWorktreeOps.commit_staged(repo, message)

    assert _git(repo, "log", "-1", "--format=%B").rstrip("\n") == message.rstrip("\n")


def test_commit_of_hash_only_message_succeeds_with_commit_cleanup_strip(repo: Path) -> None:
    """A message made only of ``#`` lines is committed, not rejected as empty."""
    _git(repo, "config", "commit.cleanup", "strip")
    (repo / "f.txt").write_text("changed\n")
    GitWorktreeOps.stage_all(repo)

    assert GitWorktreeOps.commit_staged(repo, "# Release notes")

    assert _git(repo, "log", "-1", "--format=%B").strip() == "# Release notes"
    assert GitWorktreeOps.status_porcelain(repo) == ""


def test_squash_merge_stamps_task_metadata_once_with_commit_cleanup_strip(
    repo: Path,
) -> None:
    """The merge commit carries the prompt and the Markdown result once, heading kept.

    The squash merge re-stamps the branch HEAD's message only when it does
    not already end with the task's blocks; a stored copy that lost its
    ``### Summary`` line no longer matched and was stamped a second time.
    """
    _git(repo, "config", "commit.cleanup", "strip")
    baseline = _git(repo, "rev-parse", "HEAD").strip()
    branch, wt_dir = _new_worktree(repo)
    prompt, result = "fix the bug", "<h3>Summary</h3><p>Did it.</p>"
    (wt_dir / "f.txt").write_text("fixed\n")
    assert GitWorktreeOps.commit_all(wt_dir, fallback_commit_message(prompt, result))

    merged = GitWorktreeOps.squash_merge_from_baseline(
        repo, branch, baseline, user_prompt=prompt, task_result=result,
    )

    assert merged == MergeResult.SUCCESS
    message = _git(repo, "log", "-1", "--format=%B")
    assert message.count("User prompt:") == 1, message
    assert "### Summary" in message, message
    assert (repo / "f.txt").read_text() == "fixed\n"


# --- F4 -----------------------------------------------------------------


def test_sweep_keeps_an_excluded_branch_whose_worktree_was_removed(repo: Path) -> None:
    """A live agent's branch between worktree removal and merge is not debris."""
    branch, wt_dir = _new_worktree(repo)
    GitWorktreeOps.remove(repo, wt_dir)  # what ``_finalize_worktree`` does before the merge

    assert GitWorktreeOps.sweep_orphaned_state(repo, exclude_branches={branch}) == 0
    assert GitWorktreeOps.branch_exists(repo, branch)
    # Once no live agent claims it, the same branch is debris again.
    assert GitWorktreeOps.sweep_orphaned_state(repo) == 1
    assert not GitWorktreeOps.branch_exists(repo, branch)


_PEER = textwrap.dedent("""
    import sys
    from pathlib import Path
    from kiss.agents.sorcar.git_worktree import _WORKTREE_SUBDIR, GitWorktreeOps
    repo = Path(sys.argv[1])
    branch = GitWorktreeOps.new_task_branch(repo)
    wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.remove(repo, wt_dir)
    print(branch, flush=True)
    sys.stdin.readline()
""")


def test_sweep_keeps_a_live_peer_process_branch_whose_worktree_was_removed(
    repo: Path,
) -> None:
    """Another live process's branch between worktree removal and merge is not debris."""
    peer = subprocess.Popen(
        [sys.executable, "-c", _PEER, str(repo)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert peer.stdout is not None and peer.stdin is not None
        branch = peer.stdout.readline().strip()
        assert branch.startswith("kiss/wt-"), branch

        assert GitWorktreeOps.sweep_orphaned_state(repo) == 0
        assert GitWorktreeOps.branch_exists(repo, branch)

        peer.stdin.write("done\n")
        peer.stdin.close()
        assert peer.wait(timeout=60) == 0
    finally:
        if peer.poll() is None:
            peer.kill()
            peer.wait()
    # Its owner is gone: the branch is plumbing debris now.
    assert GitWorktreeOps.sweep_orphaned_state(repo) == 1
    assert not GitWorktreeOps.branch_exists(repo, branch)

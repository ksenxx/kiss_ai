# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-26, slice sorcar-git: redundant git spawns and leftover markers.

Four findings of the read-only audit, each pinned end-to-end against a
real repository:

* the orphan sweep ran ``git worktree list --porcelain`` twice per pass
  (once for the checked-out branches, once inside the husk removal);
* a squash merge ran ``git diff --cached --quiet`` twice before its one
  commit and a conflict-resolution merge three times;
* ``ensure_scratch_merge_driver`` rewrote ``.git/config`` on every call
  although the two values never change, and the orphan reclaim installed
  it again although its caller (the worktree setup) just had;
* the preserve-for-review marker file of a branch that was deleted by
  hand was never removed, neither by ``delete_branch`` nor by the sweep.

Spawn counts are observed with a ``git`` shim on ``PATH`` that logs every
argv line before exec'ing the real git (the pattern of
``test_audit1020_git_worktree_b2.py``), so the tests count what the code
actually runs; nothing is mocked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, MergeResult
from kiss.tests.conftest import posix_only


def _run(repo: Path, *args: str) -> str:
    """Run git in *repo*, asserting success, and return its stdout."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc.stdout


def _make_repo(path: Path) -> Path:
    """Create a real git repo with one commit on ``main``."""
    path.mkdir(parents=True)
    _run(path, "init", "-q", "-b", "main")
    _run(path, "config", "user.email", "t@t.com")
    _run(path, "config", "user.name", "T")
    (path / "f.txt").write_text("a\nb\nc\n")
    _run(path, "add", ".")
    _run(path, "commit", "-qm", "initial")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path / "repo")


@pytest.fixture
def git_log(tmp_path: Path) -> Iterator[Path]:
    """Put a logging ``git`` shim first on PATH; yield its argv log file."""
    real_git = shutil.which("git")
    assert real_git
    log = tmp_path / "git.log"
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(f'#!/bin/sh\necho "$@" >> {log}\nexec {real_git} "$@"\n')
    shim.chmod(0o755)
    path_before = os.environ["PATH"]
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{path_before}"
    try:
        yield log
    finally:
        os.environ["PATH"] = path_before


def _spawns(log: Path, *words: str) -> int:
    """Count logged git invocations whose argv contains *words* in a row."""
    needle = " ".join(words)
    if not log.exists():
        return 0
    return sum(needle in line for line in log.read_text().splitlines())


# --- sweep lists worktrees once ---------------------------------------


@posix_only("the git shim is a sh script")
def test_sweep_lists_worktrees_once(repo: Path, git_log: Path) -> None:
    """One ``git worktree list`` serves both the branch guard and the husk scan."""
    _run(repo, "branch", "kiss/wt-dead")
    husk = repo / ".kiss-worktrees" / "kiss_wt-husk"
    husk.mkdir(parents=True)
    git_log.unlink(missing_ok=True)

    assert GitWorktreeOps.sweep_orphaned_state(repo) == 1

    assert not husk.exists()
    assert not GitWorktreeOps.branch_exists(repo, "kiss/wt-dead")
    assert _spawns(git_log, "worktree", "list", "--porcelain") == 1


@posix_only("the git shim is a sh script")
def test_sweep_keeps_registered_worktree_with_single_listing(
    repo: Path, git_log: Path,
) -> None:
    """The one listing still protects a live worktree's branch and directory."""
    wt_dir = repo / ".kiss-worktrees" / "kiss_wt-live"
    _run(repo, "worktree", "add", "-q", "-b", "kiss/wt-live", str(wt_dir))
    git_log.unlink(missing_ok=True)

    assert GitWorktreeOps.sweep_orphaned_state(repo) == 0

    assert wt_dir.is_dir()
    assert GitWorktreeOps.branch_exists(repo, "kiss/wt-live")
    assert _spawns(git_log, "worktree", "list", "--porcelain") == 1


# --- one index check per merge commit ---------------------------------


@posix_only("the git shim is a sh script")
def test_squash_merge_checks_the_index_once(repo: Path, git_log: Path) -> None:
    _run(repo, "checkout", "-qb", "task")
    (repo / "new.txt").write_text("new\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "agent: add new.txt")
    _run(repo, "checkout", "-q", "main")
    git_log.unlink(missing_ok=True)

    assert GitWorktreeOps.squash_merge_branch(repo, "task") == MergeResult.SUCCESS

    assert (repo / "new.txt").read_text() == "new\n"
    assert GitWorktreeOps.status_porcelain(repo) == ""
    assert _spawns(git_log, "diff", "--cached", "--quiet") == 1


@posix_only("the git shim is a sh script")
def test_empty_squash_merge_is_success_without_a_commit(repo: Path, git_log: Path) -> None:
    """A branch with nothing new still reports SUCCESS and commits nothing."""
    _run(repo, "branch", "same")
    head = _run(repo, "rev-parse", "HEAD").strip()
    git_log.unlink(missing_ok=True)

    assert GitWorktreeOps.squash_merge_branch(repo, "same") == MergeResult.SUCCESS

    assert _run(repo, "rev-parse", "HEAD").strip() == head
    assert _spawns(git_log, "diff", "--cached", "--quiet") == 1
    assert _spawns(git_log, "commit", "-F", "-") == 0


@posix_only("the git shim is a sh script")
def test_finish_conflicted_merge_checks_the_index_once(repo: Path, git_log: Path) -> None:
    _run(repo, "checkout", "-qb", "task")
    (repo / "f.txt").write_text("a\nB-agent\nc\n")
    _run(repo, "commit", "-qam", "agent edit")
    _run(repo, "checkout", "-q", "main")
    (repo / "f.txt").write_text("a\nB-user\nc\n")
    _run(repo, "commit", "-qam", "user edit")
    head_before = _run(repo, "rev-parse", "HEAD").strip()
    assert GitWorktreeOps.squash_merge_branch(repo, "task") == MergeResult.CONFLICT
    assert GitWorktreeOps.begin_conflicted_merge(repo, "task", None, "") == ["f.txt"]
    (repo / "f.txt").write_text("a\nB-both\nc\n")
    _run(repo, "add", "f.txt")
    git_log.unlink(missing_ok=True)

    result = GitWorktreeOps.finish_conflicted_merge(
        repo, "task", ["f.txt"], head_before, user_prompt="p", task_result="r",
    )

    assert result == MergeResult.SUCCESS
    assert _run(repo, "show", "HEAD:f.txt") == "a\nB-both\nc\n"
    assert "p" in _run(repo, "log", "-1", "--format=%B")
    assert _spawns(git_log, "diff", "--cached", "--quiet") == 1


def test_commit_staged_still_reports_nothing_staged(repo: Path) -> None:
    """The public ``commit_staged`` keeps its own index check for its callers."""
    head = _run(repo, "rev-parse", "HEAD").strip()
    assert GitWorktreeOps.commit_staged(repo, "nothing") is False
    assert _run(repo, "rev-parse", "HEAD").strip() == head
    (repo / "g.txt").write_text("g\n")
    _run(repo, "add", "g.txt")
    assert GitWorktreeOps.commit_staged(repo, "g") is True
    assert _run(repo, "log", "-1", "--format=%s").strip() == "g"


def test_merge_commit_rejected_by_hook_resets_the_tree(repo: Path) -> None:
    """``_commit_merge``'s failure branch: a rejecting hook yields MERGE_FAILED."""
    _run(repo, "checkout", "-qb", "task")
    (repo / "new.txt").write_text("new\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "agent: add new.txt")
    _run(repo, "checkout", "-q", "main")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    head = _run(repo, "rev-parse", "HEAD").strip()

    assert GitWorktreeOps.squash_merge_branch(repo, "task") == MergeResult.MERGE_FAILED

    assert _run(repo, "rev-parse", "HEAD").strip() == head
    assert not (repo / "new.txt").exists()
    assert GitWorktreeOps.status_porcelain(repo) == ""


# --- scratch merge driver installed once ------------------------------


def test_scratch_merge_driver_is_not_rewritten_when_present(repo: Path) -> None:
    """A second call leaves ``.git/config`` untouched (mtime pinned to the epoch)."""
    config = repo / ".git" / "config"
    GitWorktreeOps.ensure_scratch_merge_driver(repo)
    assert _run(repo, "config", "merge.kiss-scratch.driver").strip() == "cp -f %B %A"
    os.utime(config, ns=(0, 0))

    GitWorktreeOps.ensure_scratch_merge_driver(repo)

    assert config.stat().st_mtime_ns == 0, ".git/config was rewritten"
    assert _run(repo, "config", "merge.kiss-scratch.driver").strip() == "cp -f %B %A"
    attributes = (repo / ".git" / "info" / "attributes").read_text().splitlines()
    assert attributes.count("PROGRESS.md merge=kiss-scratch") == 1


def test_scratch_merge_driver_is_repaired_when_changed(repo: Path) -> None:
    """A driver value somebody edited is written back."""
    GitWorktreeOps.ensure_scratch_merge_driver(repo)
    _run(repo, "config", "merge.kiss-scratch.driver", "true")

    GitWorktreeOps.ensure_scratch_merge_driver(repo)

    assert _run(repo, "config", "merge.kiss-scratch.driver").strip() == "cp -f %B %A"


@posix_only("the git shim is a sh script")
def test_reclaim_does_not_reinstall_the_driver(
    repo: Path, git_log: Path,
) -> None:
    """Two reclaimable orphans: the driver its caller installed is not touched."""
    GitWorktreeOps.ensure_scratch_merge_driver(repo)
    for n in (1, 2):
        wt_dir = repo / ".kiss-worktrees" / f"kiss_wt-{n}"
        branch = f"kiss/wt-{n}"
        _run(repo, "worktree", "add", "-q", "-b", branch, str(wt_dir))
        GitWorktreeOps.save_original_branch(repo, branch, "main")
        (wt_dir / f"orphan{n}.txt").write_text(f"{n}\n")
        _run(wt_dir, "add", ".")
        _run(wt_dir, "commit", "-qm", f"agent {n}")
    git_log.unlink(missing_ok=True)

    assert GitWorktreeOps.reclaim_orphaned_worktrees(repo) == 2

    assert (repo / "orphan1.txt").exists() and (repo / "orphan2.txt").exists()
    assert _spawns(git_log, "config --get merge.kiss-scratch.driver") == 0
    assert _spawns(git_log, "config merge.kiss-scratch") == 0


# --- preserve markers of vanished branches ----------------------------


def _marker(repo: Path, branch: str) -> Path:
    path = GitWorktreeOps._preserve_marker_path(repo, branch)
    assert path is not None
    return path


def test_delete_branch_of_vanished_branch_removes_its_marker(repo: Path) -> None:
    _run(repo, "branch", "kiss/wt-gone")
    assert GitWorktreeOps.save_preserve_marker(repo, "kiss/wt-gone")
    assert _marker(repo, "kiss/wt-gone").exists()
    _run(repo, "branch", "-D", "kiss/wt-gone")

    assert GitWorktreeOps.delete_branch(repo, "kiss/wt-gone") is True

    assert not _marker(repo, "kiss/wt-gone").exists()
    assert not GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-gone")


def test_delete_branch_refused_keeps_the_marker(repo: Path) -> None:
    """The branch of a live worktree cannot be deleted; its marker stays."""
    wt_dir = repo / ".kiss-worktrees" / "kiss_wt-live"
    _run(repo, "worktree", "add", "-q", "-b", "kiss/wt-live", str(wt_dir))
    assert GitWorktreeOps.save_preserve_marker(repo, "kiss/wt-live")

    assert GitWorktreeOps.delete_branch(repo, "kiss/wt-live") is False

    assert _marker(repo, "kiss/wt-live").exists()
    assert GitWorktreeOps.branch_exists(repo, "kiss/wt-live")


def test_sweep_removes_marker_of_hand_deleted_branch(repo: Path) -> None:
    """``git branch -D`` by hand drops the branch and its config, never the marker.

    A successful ref enumeration that lacks the branch retires the marker
    and any config section that outlived the branch (older git keeps it).
    """
    _run(repo, "branch", "kiss/wt-manual")
    GitWorktreeOps.save_original_branch(repo, "kiss/wt-manual", "main")
    assert GitWorktreeOps.save_preserve_marker(repo, "kiss/wt-manual")
    _run(repo, "branch", "-D", "kiss/wt-manual")
    assert _marker(repo, "kiss/wt-manual").exists()
    assert GitWorktreeOps.load_original_branch(repo, "kiss/wt-manual") is None
    _run(repo, "config", "branch.kiss/wt-manual.kiss-original-branch", "main")

    assert GitWorktreeOps.sweep_orphaned_state(repo) == 0

    assert not _marker(repo, "kiss/wt-manual").exists()
    assert not GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-manual")
    assert GitWorktreeOps.load_original_branch(repo, "kiss/wt-manual") is None


def test_sweep_leaves_markers_outside_the_agent_prefix(repo: Path) -> None:
    """Only ``kiss/wt-*`` markers are the sweep's to remove."""
    assert GitWorktreeOps.save_preserve_marker(repo, "feature/x")

    GitWorktreeOps.sweep_orphaned_state(repo)

    assert _marker(repo, "feature/x").exists()


def _preserved_branch(repo: Path, branch: str) -> None:
    """Create *branch* with one unmerged commit, its origin config and its marker."""
    _run(repo, "checkout", "-qb", branch)
    (repo / "work.txt").write_text("unmerged\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "unmerged work")
    _run(repo, "checkout", "-q", "main")
    GitWorktreeOps.save_original_branch(repo, branch, "main")
    assert GitWorktreeOps.save_preserve_marker(repo, branch)


def test_sweep_keeps_marker_while_the_branch_exists(repo: Path) -> None:
    """A preserved branch (unmerged work) survives the sweep with its marker."""
    _preserved_branch(repo, "kiss/wt-kept")

    assert GitWorktreeOps.sweep_orphaned_state(repo) == 0

    assert _marker(repo, "kiss/wt-kept").exists()
    assert GitWorktreeOps.branch_exists(repo, "kiss/wt-kept")


@pytest.fixture
def failing_git(tmp_path: Path) -> Iterator[Path]:
    """Put a ``git`` shim first on PATH that fails the subcommand named in a file.

    Yields the file; write ``"<sub> <arg>"`` (e.g. ``"rev-parse --verify"``)
    to make exactly that invocation exit 124 (``_git``'s timeout code);
    everything else is forwarded to the real git.  ``_git`` runs
    ``git -c core.quotepath=false -C <cwd> <sub> <arg> ...``, so the
    subcommand is argv 5 and 6.
    """
    real_git = shutil.which("git")
    assert real_git
    spec = tmp_path / "fail.txt"
    spec.write_text("")
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/sh\n"
        f'[ "$5 $6" = "$(cat {spec})" ] && exit 124\n'
        f'exec {real_git} "$@"\n'
    )
    shim.chmod(0o755)
    path_before = os.environ["PATH"]
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{path_before}"
    try:
        yield spec
    finally:
        os.environ["PATH"] = path_before


@posix_only("the git shim is a sh script")
@pytest.mark.parametrize(
    "failing_call", ["rev-parse --verify", "for-each-ref --format=%(refname:short)"],
)
def test_sweep_keeps_marker_and_config_when_git_fails(
    repo: Path, failing_git: Path, failing_call: str,
) -> None:
    """A git error or timeout must not be read as "the branch is gone".

    Only a SUCCESSFUL ref enumeration may retire a preserve marker or a
    branch's config section; with the lookup failing (exit 124, a
    timeout) both survive, so the next reclaim still sees the branch
    as preserved instead of merging its unaccepted work.
    """
    _preserved_branch(repo, "kiss/wt-probe")
    failing_git.write_text(failing_call)

    assert GitWorktreeOps.sweep_orphaned_state(repo) == 0

    failing_git.write_text("")
    assert GitWorktreeOps.branch_exists(repo, "kiss/wt-probe")
    assert _marker(repo, "kiss/wt-probe").exists()
    assert GitWorktreeOps.load_preserve_marker(repo, "kiss/wt-probe")
    assert GitWorktreeOps.load_original_branch(repo, "kiss/wt-probe") == "main"


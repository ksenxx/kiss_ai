"""End-to-end tests of the host ``checkpoint`` / ``rollback`` tools.

Every test builds a real git repository in a temporary directory and
drives ``UsefulTools.checkpoint`` / ``UsefulTools.rollback`` (and the
:mod:`kiss.agents.sorcar.checkpoints` functions under them), then checks
the files on disk and that HEAD, the branch and the index are untouched.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import checkpoints
from kiss.agents.sorcar.checkpoints import (
    CheckpointError,
    Checkpoints,
    Snapshot,
    apply_restore,
    plan_restore,
    snapshot,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.utils import rmtree_force
from kiss.tests.conftest import is_root, posix_only
from kiss.tests.server.pause import Pause


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout


def _repo(path: Path) -> Path:
    """A repository with one commit: ``a.py``, ``run.sh`` (executable), ``d/b.txt``."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / ".gitignore").write_text("ignored/\n")
    (path / "a.py").write_text("print('v1')\n")
    (path / "run.sh").write_text("echo hi\n")
    (path / "run.sh").chmod(0o755)
    (path / "d").mkdir()
    (path / "d" / "b.txt").write_text("b1\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


def _state(repo: Path) -> tuple[str, str, str]:
    """HEAD, the branch and the index: what a checkpoint must never change."""
    return (_git(repo, "rev-parse", "HEAD"), _git(repo, "branch", "--show-current"),
            _git(repo, "ls-files", "--stage"))


class _AtFirstEntry:
    """A :class:`Pause` matcher: run *step* on the thread first entering the watched function."""

    def __init__(self, step: Callable[[], object]) -> None:
        self.step: Callable[[], object] | None = step

    def __call__(self, frame_locals: dict[str, Any]) -> bool:
        step, self.step = self.step, None
        if step is not None:
            step()
        return False


def _swap_for_link(directory: Path, outside: Path) -> None:
    """What a concurrent writer does: move *directory* away and leave a link to *outside*."""
    directory.rename(directory.with_name("moved"))
    directory.symlink_to(outside, target_is_directory=True)


class _SwapOtherBranch:
    """A :class:`Pause` matcher on ``Path.rmdir``: as a first directory goes, swap the other branch.

    The branches are ``left`` and ``right``; the one the first directory is in is left alone.
    """

    def __init__(self, outside: Path) -> None:
        self.outside: Path | None = outside

    def __call__(self, frame_locals: dict[str, Any]) -> bool:
        outside, self.outside = self.outside, None
        if outside is not None:
            branch = frame_locals["self"].parent
            other = "left" if branch.name == "right" else "right"
            _swap_for_link(branch.with_name(other), outside)
        return False


def _put_cache(directory: Path) -> None:
    """What a concurrent build does: write an ignored file in *directory*, or in place of a file."""
    if directory.is_file():
        directory.unlink()
    (directory / "ignored").mkdir(parents=True)
    (directory / "ignored" / "cache.bin").write_text("keep\n")


def test_rollback_restores_every_non_ignored_file(tmp_path: Path) -> None:
    """Changed, deleted, created and mode-changed files return; ignored ones stay."""
    repo = _repo(tmp_path / "repo")
    (repo / "notes.txt").write_text("untracked before\n")
    tools = UsefulTools(work_dir=str(repo / "d"))  # any directory of the tree
    before = _state(repo)
    saved = tools.checkpoint("v1")
    assert saved.startswith(f"Saved checkpoint 'v1' of {repo.resolve()} (git tree ")
    assert _state(repo) == before
    (repo / "a.py").write_text("print('v2')\n")
    (repo / "d" / "b.txt").unlink()
    (repo / "notes.txt").write_text("untracked after\n")
    (repo / "run.sh").chmod(0o644)
    (repo / "new" / "deep").mkdir(parents=True)
    (repo / "new" / "deep" / "c.py").write_text("c\n")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "cache.bin").write_text("keep\n")
    # Windows has no executable bit (git init sets core.fileMode=false there),
    # so the chmod of run.sh is no change to roll back.
    rewritten = 3 if os.name == "nt" else 4
    out = tools.rollback("v1")
    assert out.startswith(
        f"Rolled {repo.resolve()} back to checkpoint 'v1': rewrote {rewritten} files "
    ), out
    assert "deleted 1 file (new/deep/c.py)" in out and "'before-rollback'" in out
    assert (repo / "a.py").read_text() == "print('v1')\n"
    assert (repo / "d" / "b.txt").read_text() == "b1\n"
    assert (repo / "notes.txt").read_text() == "untracked before\n"
    assert os.access(repo / "run.sh", os.X_OK)
    assert not (repo / "new").exists()  # the directories it left empty go too
    assert (repo / "ignored" / "cache.bin").read_text() == "keep\n"
    assert _state(repo) == before
    # The replaced state is itself a checkpoint: the rollback can be undone.
    undo = tools.rollback("before-rollback")
    assert f"rewrote {rewritten} files" in undo and "deleted 1 file (d/b.txt)" in undo, undo
    assert (repo / "a.py").read_text() == "print('v2')\n"
    assert (repo / "new" / "deep" / "c.py").read_text() == "c\n"
    # d/ is left empty but kept: it is the work dir
    assert (repo / "d").is_dir() and not list((repo / "d").iterdir())
    # ...and the undo saved the state it replaced in turn.
    assert tools.rollback("before-rollback").startswith("Rolled ")
    assert (repo / "a.py").read_text() == "print('v1')\n"


def test_names_and_unknown_checkpoints_are_errors(tmp_path: Path) -> None:
    """Bad and reserved names are refused; an unknown name lists the known ones."""
    repo = _repo(tmp_path / "repo")
    tools = UsefulTools(work_dir=str(repo))
    for bad in ("", "has space", "x" * 41, "../etc"):
        assert tools.checkpoint(bad).startswith("Error: name must be letters"), bad
    assert tools.checkpoint("before-rollback").startswith(
        "Error: 'before-rollback' is saved by rollback itself")
    assert tools.rollback("v1") == "Error: unknown checkpoint 'v1'. Known: none."
    assert tools.checkpoint("v1").startswith("Saved checkpoint 'v1'")
    assert tools.checkpoint("v1").startswith("Saved checkpoint 'v1'")  # a re-save replaces it
    assert tools.rollback("v2") == "Error: unknown checkpoint 'v2'. Known: v1."
    assert tools.rollback("v1") == (
        "Nothing to roll back: the files already match checkpoint 'v1'.")


def test_checkpoints_need_git_and_stay_with_their_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside git, without git, or after the work dir moved to another tree: errors."""
    plain = tmp_path / "plain"
    plain.mkdir()
    out = UsefulTools(work_dir=str(plain)).checkpoint("v1")
    assert out.startswith("Error: checkpoint 'v1' not saved: fatal: not a git repository"), out
    first, second = _repo(tmp_path / "first"), _repo(tmp_path / "second")
    tools = UsefulTools(work_dir=str(first))
    tools.checkpoint("v1")
    tools.work_dir = str(second)  # what create_worktree does to the live tools
    out = tools.rollback("v1")
    assert out == (f"Error: checkpoint 'v1' holds {first.resolve()}, but the work dir is now "
                   f"in {second.resolve()}; take a new checkpoint there.")
    # A machine without git: the subprocess cannot start.
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    assert tools.checkpoint("v2").startswith(
        "Error: checkpoint 'v2' not saved: git rev-parse failed")
    assert tools.rollback("v1").startswith("Error: rollback to 'v1' failed: git rev-parse failed")


def test_file_and_directory_swaps_and_a_fresh_repository(tmp_path: Path) -> None:
    """A file replaced by a directory (and back) is restored; a repo without an index works."""
    repo = tmp_path / "fresh"
    repo.mkdir()
    _git(repo, "init", "-q")  # no commit, no index file yet
    (repo / "x").write_text("file x\n")
    (repo / "d").mkdir()
    (repo / "d" / "a.txt").write_text("a\n")
    tools = UsefulTools(work_dir=str(repo))
    assert tools.checkpoint("start").startswith("Saved checkpoint 'start'")
    assert not (repo / ".git" / "index").exists()
    (repo / "x").unlink()
    (repo / "x").mkdir()
    (repo / "x" / "inner.txt").write_text("inner\n")
    (repo / "d" / "a.txt").unlink()
    (repo / "d").rmdir()
    (repo / "d").write_text("now a file\n")
    out = tools.rollback("start")
    assert "rewrote 2 files" in out and "deleted 2 files" in out, out
    assert (repo / "x").read_text() == "file x\n"
    assert (repo / "d" / "a.txt").read_text() == "a\n"


@pytest.mark.skipif(is_root(), reason="root may delete files in a read-only directory")
@posix_only("a read-only directory does not stop deleting its files on Windows")
def test_a_file_that_cannot_be_deleted_fails_the_rollback(tmp_path: Path) -> None:
    """The rollback stops with the OS error; the state it replaced is still saved."""
    repo = _repo(tmp_path / "repo")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    (repo / "d" / "new.txt").write_text("new\n")
    (repo / "d").chmod(0o555)
    try:
        out = tools.rollback("v1")
    finally:
        (repo / "d").chmod(0o755)
    assert out.startswith("Error: rollback to 'v1' failed: cannot delete d/new.txt: "), out
    assert (repo / "d" / "new.txt").exists()
    assert tools.rollback("v1").startswith("Rolled ")  # writable again: it goes through
    assert not (repo / "d" / "new.txt").exists()


@pytest.mark.skipif(is_root(), reason="root makes a directory in a read-only one")
@posix_only("a read-only directory still takes new files on Windows")
def test_a_git_directory_that_takes_no_new_directory_still_rolls_back(tmp_path: Path) -> None:
    """With nowhere to put the new files first, the old ones are deleted and git writes them."""
    repo = _repo(tmp_path / "repo")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    (repo / "a.py").write_text("print('v2')\n")
    (repo / ".git").chmod(0o555)
    try:
        out = tools.rollback("v1")
    finally:
        (repo / ".git").chmod(0o755)
    assert "rewrote 1 file (a.py); deleted 0 files" in out, out
    assert (repo / "a.py").read_text() == "print('v1')\n"


def test_a_rollback_never_destroys_what_no_checkpoint_holds(tmp_path: Path) -> None:
    """Ignored files and the work dir in the way of a restored file stop the rollback first."""
    repo = _repo(tmp_path / "repo")
    (repo / "notes.txt").write_text("notes v1\n")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    # a.py becomes a directory that also holds an ignored file; d (the
    # directory of d/b.txt) becomes an ignored file; notes.txt becomes ignored.
    (repo / ".gitignore").write_text("ignored/\n*.log\n/d\nnotes.txt\n")
    (repo / "a.py").unlink()
    (repo / "a.py").mkdir()
    (repo / "a.py" / "x.txt").write_text("x\n")
    (repo / "a.py" / "debug.log").write_text("log\n")
    (repo / "d" / "b.txt").unlink()
    (repo / "d").rmdir()
    (repo / "d").write_text("an ignored file\n")
    (repo / "notes.txt").write_text("notes v2\n")
    out = tools.rollback("v1")
    assert out == (
        "Error: rollback to 'v1' failed: nothing was changed: a.py is now a directory "
        "holding a.py/debug.log, which git ignores; d is now a file git ignores, where "
        "the checkpoint has the directory of d/b.txt; notes.txt is now a file git "
        "ignores. A rollback replaces only files a checkpoint holds: move these away "
        "first."), out
    assert (repo / "a.py" / "x.txt").read_text() == "x\n"
    assert (repo / "a.py" / "debug.log").read_text() == "log\n"
    assert (repo / "d").read_text() == "an ignored file\n"
    assert (repo / "notes.txt").read_text() == "notes v2\n"
    (repo / "a.py" / "debug.log").unlink()
    (repo / "d").unlink()
    (repo / "notes.txt").unlink()
    out = tools.rollback("v1")
    assert out.startswith("Rolled ") and "deleted 1 file (a.py/x.txt)" in out, out
    assert (repo / "a.py").read_text() == "print('v1')\n"
    assert (repo / "d" / "b.txt").read_text() == "b1\n"
    assert (repo / "notes.txt").read_text() == "notes v1\n"
    # The work dir is never removed, even when it holds only files to delete.
    (repo / "a.py").unlink()
    (repo / "a.py").mkdir()
    (repo / "a.py" / "x.txt").write_text("x\n")
    tools.work_dir = str(repo / "a.py")  # the work dir moved down the same tree
    out = tools.rollback("v1")
    assert out == (
        "Error: rollback to 'v1' failed: nothing was changed: a.py is now a directory "
        "holding the work dir. A rollback replaces only files a checkpoint holds: move "
        "these away first."), out
    assert (repo / "a.py" / "x.txt").is_file()
    # Links count as files: an ignored one to a directory, a dangling one.
    tools.work_dir = str(repo)
    (repo / ".gitignore").write_text("ignored/\n*.log\nnotes.txt\n")
    (repo / "a.py" / "x.txt").unlink()
    (repo / "ignored").mkdir()
    (repo / "a.py" / "data.log").symlink_to(repo / "ignored")
    (repo / "notes.txt").unlink()
    (repo / "notes.txt").symlink_to(repo / "nowhere")
    out = tools.rollback("v1")
    assert out == (
        "Error: rollback to 'v1' failed: nothing was changed: a.py is now a directory "
        "holding a.py/data.log, which git ignores; notes.txt is now a file git ignores. "
        "A rollback replaces only files a checkpoint holds: move these away first."), out
    assert (repo / "a.py" / "data.log").is_symlink() and (repo / "notes.txt").is_symlink()


def test_a_directory_in_place_of_a_file_and_an_embedded_repository(tmp_path: Path) -> None:
    """git replaces a directory by the file it was; a nested repository is left alone."""
    repo = _repo(tmp_path / "repo")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    # a.py becomes a directory (with an empty directory in it)
    (repo / "a.py").unlink()
    (repo / "a.py" / "empty").mkdir(parents=True)
    (repo / "a.py" / "x.txt").write_text("x\n")
    out = tools.rollback("v1")
    assert "rewrote 1 file (a.py)" in out and "deleted 1 file (a.py/x.txt)" in out, out
    assert (repo / "a.py").read_text() == "print('v1')\n"
    # A repository embedded after the checkpoint is a gitlink in the
    # snapshot: neither deleted nor reported.
    nested = _repo(tmp_path / "nested")
    clean = _repo(tmp_path / "clean")
    tools = UsefulTools(work_dir=str(clean))
    tools.checkpoint("v1")
    nested.rename(clean / "vendored")
    out = tools.rollback("v1")
    assert "rewrote 0 files; deleted 0 files" in out, out
    assert (clean / "vendored" / "a.py").read_text() == "print('v1')\n"
    # The converse: the checkpoint holds the embedded repository as a
    # gitlink; its files were since vendored plainly (``.git`` removed) and
    # one added.  No snapshot holds them, so the rollback leaves them alone
    # instead of deleting what the gitlink stood for.
    tools.checkpoint("v2")
    rmtree_force(clean / "vendored" / ".git")  # git's object files are read-only (Windows)
    (clean / "vendored" / "new-work.txt").write_text("new\n")
    out = tools.rollback("v2")
    assert "rewrote 0 files; deleted 0 files" in out, out
    assert (clean / "vendored" / "a.py").read_text() == "print('v1')\n"
    assert (clean / "vendored" / "new-work.txt").read_text() == "new\n"
    # A repository with no commit yet (a fresh ``git init``) makes ``git add``
    # refuse the whole tree: it is left out, the rest is saved and restored,
    # an embedded repository that has a commit as the gitlink it always was.
    (clean / "fresh").mkdir()
    _git(clean / "fresh", "init", "-q")
    (clean / "fresh" / "wip.txt").write_text("wip\n")
    _repo(clean / "lib")
    assert tools.checkpoint("v3").startswith("Saved checkpoint 'v3'")
    (clean / "a.py").write_text("print('v3')\n")
    (clean / "lib" / "a.py").write_text("print('lib')\n")
    out = tools.rollback("v3")
    assert "rewrote 1 file (a.py); deleted 0 files" in out, out
    assert (clean / "a.py").read_text() == "print('v1')\n"
    assert (clean / "fresh" / "wip.txt").read_text() == "wip\n"
    assert (clean / "lib" / "a.py").read_text() == "print('lib')\n"


@pytest.mark.skipif(is_root(), reason="root reads every file")
@posix_only("a file's mode does not keep its owner from reading it on Windows")
def test_a_file_git_cannot_read_fails_the_checkpoint(tmp_path: Path) -> None:
    """``git add``'s error is the tool's, with or without a repository that has no commit."""
    repo = _repo(tmp_path / "repo")
    (repo / "secret").write_text("s\n")
    (repo / "secret").chmod(0)
    tools = UsefulTools(work_dir=str(repo))
    for _ in range(2):
        out = tools.checkpoint("v1")
        assert out.startswith("Error: checkpoint 'v1' not saved: ") and "secret" in out, out
        (repo / "fresh").mkdir(exist_ok=True)
        _git(repo / "fresh", "init", "-q")


def test_listing_cuts_long_path_lists(tmp_path: Path) -> None:
    """More than ten paths are summarised; ``plan_restore`` reports them all."""
    repo = _repo(tmp_path / "repo")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    for n in range(12):
        (repo / f"f{n:02d}.txt").write_text(f"{n}\n")
    out = tools.rollback("v1")
    assert "deleted 12 files (f00.txt, f01.txt," in out and "f09.txt and 2 more)" in out, out
    assert not list(repo.glob("f*.txt"))
    # Obstacles are cut the same way: twelve files the checkpoint has, now ignored.
    for n in range(12):
        (repo / f"f{n:02d}.txt").write_text(f"{n}\n")
    tools.checkpoint("twelve")
    (repo / ".gitignore").write_text("f*.txt\n")
    out = tools.rollback("twelve")
    assert out.endswith("f09.txt is now a file git ignores; and 2 more. A rollback replaces "
                        "only files a checkpoint holds: move these away first."), out
    # The library functions are usable directly; a snapshot of a missing tree is a git error.
    current = snapshot(str(repo))
    with pytest.raises(CheckpointError):
        plan_restore(Snapshot(current.root, "0" * 40), current, repo)


@posix_only("Windows file names cannot hold a newline or a carriage return")
def test_file_names_are_kept_byte_for_byte(tmp_path: Path) -> None:
    """A carriage return in a name is not read as a newline: the right file is deleted."""
    repo = _repo(tmp_path / "repo")
    (repo / "victim\n.txt").write_text("keep me\n")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    (repo / "victim\r.txt").write_text("created later\n")
    out = tools.rollback("v1")
    assert "deleted 1 file (victim\r.txt)" in out, out
    assert (repo / "victim\n.txt").read_text() == "keep me\n"
    assert not (repo / "victim\r.txt").exists()


@posix_only("Windows forbids a newline in a directory name")
def test_a_root_ending_in_a_newline_is_not_its_neighbour(tmp_path: Path) -> None:
    """Only git's record terminator is removed from the root: ``project\\n`` is not ``project``."""
    intended, neighbour = _repo(tmp_path / "project\n"), _repo(tmp_path / "project")
    tools = UsefulTools(work_dir=str(intended))
    assert tools.checkpoint("v1").startswith(f"Saved checkpoint 'v1' of {intended.resolve()} (")
    (intended / "a.py").write_text("print('v2')\n")
    (neighbour / "a.py").write_text("print('neighbour')\n")
    assert "rewrote 1 file (a.py)" in tools.rollback("v1")
    assert (intended / "a.py").read_text() == "print('v1')\n"
    assert (neighbour / "a.py").read_text() == "print('neighbour')\n"


def test_the_environment_cannot_redirect_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE`` of the daemon never pick the repository."""
    wanted, other = _repo(tmp_path / "wanted"), _repo(tmp_path / "other")
    tools = UsefulTools(work_dir=str(wanted))
    before = _state(wanted), _state(other)
    with monkeypatch.context() as env:
        env.setenv("GIT_DIR", str(other / ".git"))
        env.setenv("GIT_WORK_TREE", str(other))
        env.setenv("GIT_INDEX_FILE", str(tmp_path / "stray-index"))
        assert tools.checkpoint("v1").startswith(
            f"Saved checkpoint 'v1' of {wanted.resolve()} (")
        (wanted / "a.py").write_text("print('v2')\n")
        (other / "a.py").write_text("print('other')\n")
        assert "rewrote 1 file (a.py)" in tools.rollback("v1")
    assert (wanted / "a.py").read_text() == "print('v1')\n"
    assert (other / "a.py").read_text() == "print('other')\n"
    assert (_state(wanted), _state(other)) == before
    assert not (tmp_path / "stray-index").exists()


def test_files_git_assumes_unchanged_are_still_saved(tmp_path: Path) -> None:
    """assume-unchanged and skip-worktree bits of the real index do not hide changes."""
    repo = _repo(tmp_path / "repo")
    _git(repo, "update-index", "--assume-unchanged", "a.py")
    _git(repo, "update-index", "--skip-worktree", "d/b.txt")
    flags = _git(repo, "ls-files", "-v")
    (repo / "a.py").write_text("print('working')\n")
    (repo / "d" / "b.txt").write_text("b working\n")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("good")
    (repo / "a.py").write_text("print('broken')\n")
    (repo / "d" / "b.txt").write_text("b broken\n")
    out = tools.rollback("good")
    assert "rewrote 2 files (a.py, d/b.txt)" in out, out
    assert (repo / "a.py").read_text() == "print('working')\n"
    assert (repo / "d" / "b.txt").read_text() == "b working\n"
    assert _git(repo, "ls-files", "-v") == flags  # the real index keeps its bits


def test_git_gc_keeps_a_checkpoint_until_it_is_replaced_or_closed(tmp_path: Path) -> None:
    """Each saved tree has a ref of its own, so ``git gc`` in any worktree leaves it alone."""
    main = _repo(tmp_path / "main")
    work = tmp_path / "work"
    _git(main, "worktree", "add", "-q", "-b", "task", str(work))
    before = _state(main), _state(work)
    stop = threading.Event()
    saved = Checkpoints(stop)
    (work / "a.py").write_text("print('draft')\n")  # content no commit holds
    assert saved.save("draft", str(work)).startswith("Saved checkpoint 'draft'")
    (work / "a.py").write_text("print('later')\n")
    _git(main, "gc", "-q", "--prune=now")
    assert saved.roll_back("draft", str(work)).startswith("Rolled ")
    assert (work / "a.py").read_text() == "print('draft')\n"
    # Undoing it replaces 'before-rollback' while restoring from it: the tree
    # restored keeps its ref until the files are back, whatever git collects.
    collect = partial(_git, main, "gc", "-q", "--prune=now")
    with Pause(checkpoints.apply_restore.__code__, _AtFirstEntry(collect)):
        assert saved.roll_back("before-rollback", str(work)).startswith("Rolled ")
    assert (work / "a.py").read_text() == "print('later')\n"
    collect()
    assert saved.roll_back("before-rollback", str(work)).startswith("Rolled ")
    assert (work / "a.py").read_text() == "print('draft')\n"
    # A replaced checkpoint's ref is deleted: one ref per name is left.
    assert saved.save("draft", str(work)).startswith("Saved checkpoint 'draft'")
    refs = partial(_git, main, "for-each-ref", "--format=%(objectname)", "refs/kiss-checkpoints")
    assert sorted(refs().split()) == sorted(kept.tree for kept in saved.saved.values())
    assert (_state(main), _state(work)) == before
    # close() deletes the rest, also those of a worktree that is gone and after the
    # task's Stop, but none of another task; git may then collect the trees again.
    later = saved.saved["before-rollback"].tree
    other = Checkpoints()
    other.save("theirs", str(main))
    _git(main, "worktree", "remove", "--force", str(work))
    stop.set()
    saved.close()
    assert refs().split() == [other.saved["theirs"].tree]
    assert saved.roll_back("draft", str(main)) == "Error: unknown checkpoint 'draft'. Known: none."
    collect()
    gone = subprocess.run(["git", "-C", str(main), "cat-file", "-e", later], capture_output=True)
    assert gone.returncode != 0


def test_the_end_of_a_run_deletes_the_refs_of_its_checkpoints(tmp_path: Path) -> None:
    """A run's checkpoints end with it: no ref keeps a finished task's trees from ``git gc``."""
    repo = _repo(tmp_path / "repo")
    tools = UsefulTools(work_dir=str(repo))
    agent = SorcarAgent("t")
    agent._useful_tools = tools
    agent._begin_task(None)
    assert tools.checkpoint("v1").startswith("Saved checkpoint 'v1'")
    (repo / "a.py").write_text("print('v2')\n")
    assert tools.rollback("v1").startswith("Rolled ")
    refs = partial(_git, repo, "for-each-ref", "refs/kiss-checkpoints")
    assert len(refs().splitlines()) == 2  # 'v1' and 'before-rollback'
    agent._finish_run(unwinding=False)
    assert refs() == ""


def test_the_refs_are_made_where_git_must_be_told_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree of a bare repository under ``safe.bareRepository=explicit`` keeps its refs too."""
    bare, work = tmp_path / "bare.git", tmp_path / "work"
    _git(_repo(tmp_path / "seed"), "clone", "-q", "--bare", ".", str(bare))
    for key, value in (("COUNT", "1"), ("KEY_0", "safe.bareRepository"), ("VALUE_0", "explicit")):
        monkeypatch.setenv(f"GIT_CONFIG_{key}", value)
    _git(tmp_path, "--git-dir", str(bare), "worktree", "add", "-q", "-b", "task", str(work))
    refs = ("--git-dir", str(bare), "for-each-ref", "refs/kiss-checkpoints")
    saved = Checkpoints()
    assert saved.save("v1", str(work)).startswith("Saved checkpoint 'v1'")
    assert len(_git(tmp_path, *refs).splitlines()) == 1
    rmtree_force(work)  # as after commit_and_merge: only the git directory is left
    saved.close()
    assert _git(tmp_path, *refs) == ""
    rmtree_force(bare)
    saved.close()  # the repository gone as well: nothing to delete, and nothing raised


def test_a_ref_git_does_not_make_or_does_not_delete(tmp_path: Path) -> None:
    """A snapshot git makes no ref for is not saved; a ref it cannot delete fails nothing."""
    resource = pytest.importorskip("resource")  # the limit of open files is POSIX's
    repo = _repo(tmp_path / "repo")
    refs = partial(_git, repo, "for-each-ref", "refs/kiss-checkpoints")
    saved = Checkpoints()
    assert saved.save("v1", str(repo)).startswith("Saved checkpoint 'v1'")
    hook = tmp_path / "hooks" / "reference-transaction"  # the repository's veto of a ref
    hook.parent.mkdir()
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    _git(repo, "config", "core.hooksPath", str(hook.parent))
    (repo / "a.py").write_text("print('v2')\n")
    assert saved.save("v1", str(repo)).startswith("Error: checkpoint 'v1' not saved: ")
    hook.unlink()
    assert saved.roll_back("v1", str(repo)).startswith("Rolled ")
    assert (repo / "a.py").read_text() == "print('v1')\n"
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    use_up = partial(resource.setrlimit, resource.RLIMIT_NOFILE, (0, limits[1]))
    try:
        # From the moment the replaced ref is to go, every ``open`` fails (``EMFILE``):
        # no descriptor is left for git's files, the save stands and close() raises nothing.
        with Pause(Checkpoints._release.__code__, _AtFirstEntry(use_up)):
            assert saved.save("v1", str(repo)).startswith("Saved checkpoint 'v1'")
        saved.close()
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)
    assert len(refs().splitlines()) == 3  # both v1 and before-rollback: none could go
    saved.close()
    assert refs() == ""


@posix_only("making a link takes a privilege on Windows")
def test_a_directory_swapped_for_a_link_is_not_followed(tmp_path: Path) -> None:
    """A link put where a deletion's directory was fails the rollback: nothing outside goes."""
    repo = _repo(tmp_path / "repo")
    stray = tmp_path / "outside" / "sub" / "new.txt"  # where a followed link leads
    stray.parent.mkdir(parents=True)
    stray.write_text("outside\n")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    (repo / "sub").mkdir()
    (repo / "sub" / "new.txt").write_text("draft\n")
    swap = partial(_swap_for_link, repo / "sub", stray.parent)
    with Pause(checkpoints._delete.__code__, _AtFirstEntry(swap)):
        out = tools.rollback("v1")
    assert out.startswith("Error: rollback to 'v1' failed: cannot delete sub/new.txt: "), out
    assert stray.read_text() == "outside\n"
    assert (repo / "moved" / "new.txt").read_text() == "draft\n"
    # Native Windows cannot open a directory and deletes by pathname (``_delete`` hands over to
    # ``_delete_by_name`` only there, a line no POSIX run reaches): the link is refused as well.
    with pytest.raises(CheckpointError, match="cannot delete sub/new.txt: .* is a link"):
        checkpoints._delete_by_name(repo, "sub/new.txt", repo)
    assert stray.read_text() == "outside\n"
    # Its removal of empty directories looks into no link either, takes nothing from a
    # directory that holds a file, and ends at a Stop.
    (stray.parent / "empty").mkdir()
    (repo / "moved" / "a" / "b").mkdir(parents=True)
    for path in ("sub", "sub/empty", "moved"):
        checkpoints._remove_empty_directories_by_name(repo, path, None)
    assert (stray.parent / "empty").is_dir() and (repo / "moved" / "a" / "b").is_dir()
    stop = threading.Event()
    stop.set()
    with pytest.raises(CheckpointError, match="^rewriting moved/a was stopped$"):
        checkpoints._remove_empty_directories_by_name(repo, "moved/a", stop)
    checkpoints._remove_empty_directories_by_name(repo, "moved/a", None)
    assert not (repo / "moved" / "a").exists()
    # A link put in once the directories were listed is not looked through either.
    beside, gone = tmp_path / "beside", repo / "gone"
    for leaf in (beside / "leaf", gone / "left" / "leaf", gone / "right" / "leaf"):
        leaf.mkdir(parents=True)
    with Pause(Path.rmdir.__code__, _SwapOtherBranch(beside)):
        checkpoints._remove_empty_directories_by_name(repo, "gone", None)
    assert (beside / "leaf").is_dir() and (gone / "moved" / "leaf").is_dir()
    rmtree_force(gone)
    # The link is a file of the tree now: the next rollback deletes it, not what it names.
    out = tools.rollback("v1")
    assert "deleted 2 files (moved/new.txt, sub)" in out, out
    assert not (repo / "sub").is_symlink() and not (repo / "moved").exists()
    # The working tree itself swapped for a link is not followed either.
    (repo / "sub").mkdir()
    (repo / "sub" / "new.txt").write_text("draft\n")
    swap = partial(_swap_for_link, repo, stray.parent.parent)
    with Pause(checkpoints._delete.__code__, _AtFirstEntry(swap)):
        out = tools.rollback("v1")
    assert out.startswith("Error: rollback to 'v1' failed: cannot delete sub/new.txt: "), out
    assert stray.read_text() == "outside\n"


@pytest.mark.skipif(is_root() or not hasattr(os, "O_PATH"),
                    reason="root reads every directory; without O_PATH one is opened to be read")
def test_a_file_in_a_directory_that_cannot_be_read_is_still_deleted(tmp_path: Path) -> None:
    """A deletion searches and writes the directories on its way: it needs to read none."""
    repo = _repo(tmp_path / "repo")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    (repo / "d" / "sub").mkdir()
    (repo / "d" / "sub" / "new.txt").write_text("new\n")
    _git(repo, "add", "d/sub/new.txt")  # git cannot list such a directory: its index knows the file
    (repo / "d").chmod(0o300)
    try:
        out = tools.rollback("v1")
    finally:
        (repo / "d").chmod(0o755)
    assert "deleted 1 file (d/sub/new.txt)" in out, out
    assert not (repo / "d" / "sub").exists() and (repo / "d" / "b.txt").read_text() == "b1\n"


def test_a_file_gone_since_the_plan_is_no_error(tmp_path: Path) -> None:
    """A deletion whose file, or whole directory, went away meanwhile has nothing left to do."""
    repo = _repo(tmp_path / "repo")
    target = snapshot(str(repo))
    for name in ("gone/deep/x.txt", "left/y.txt"):
        (repo / name).parent.mkdir(parents=True)
        (repo / name).write_text("new\n")
    (repo / "d" / "b.txt").write_text("b2\n")
    rewrite, missing, delete = plan_restore(target, snapshot(str(repo)), repo)
    rmtree_force(repo / "gone")
    (repo / "left" / "y.txt").unlink()
    rmtree_force(repo / "d")
    apply_restore(target, rewrite, missing, delete, repo)
    assert (rewrite, missing, delete) == (["d/b.txt"], [], ["gone/deep/x.txt", "left/y.txt"])
    assert not (repo / "left").exists()  # the directory the file left empty still goes
    # A file to replace whose directory went away is written back, directory and all.
    assert (repo / "d" / "b.txt").read_text() == "b1\n"
    # The pathname deletion of native Windows prunes alike, up to the work dir.
    (repo / "w" / "deep").mkdir(parents=True)
    (repo / "w" / "deep" / "z.txt").write_text("new\n")
    checkpoints._delete_by_name(repo, "w/deep/z.txt", repo / "w")
    assert (repo / "w").is_dir() and not (repo / "w" / "deep").exists()


def test_what_appears_in_a_files_way_after_the_plan_stays(tmp_path: Path) -> None:
    """The restore removes only what a snapshot holds: an ignored file made meanwhile survives."""
    repo = _repo(tmp_path / "repo")
    cache = repo / "a.py" / "ignored" / "cache.bin"
    build = partial(_put_cache, repo / "a.py")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    # a.py is now a directory of one file the rollback deletes, so the plan lets the
    # file go back there; a build then writes an ignored file into the directory.
    (repo / "a.py").unlink()
    (repo / "a.py").mkdir()
    (repo / "a.py" / "x.txt").write_text("x\n")
    with Pause(checkpoints.apply_restore.__code__, _AtFirstEntry(build)):
        out = tools.rollback("v1")
    assert out == "Error: rollback to 'v1' failed: a.py already exists, no checkout", out
    assert cache.read_text() == "keep\n"
    # Without the file the rollback goes through: empty directories are no one's.
    cache.unlink()
    out = tools.rollback("v1")
    assert "rewrote 1 file (a.py); deleted 0 files" in out, out
    assert (repo / "a.py").read_text() == "print('v1')\n"
    # The same for a file the rollback replaces, swapped for a directory meanwhile.
    (repo / "a.py").write_text("print('v2')\n")
    with Pause(checkpoints.apply_restore.__code__, _AtFirstEntry(build)):
        out = tools.rollback("v1")
    assert out.startswith("Error: rollback to 'v1' failed: cannot delete a.py: "), out
    assert cache.read_text() == "keep\n"


@posix_only("making a link takes a privilege on Windows")
def test_a_link_where_the_checkpoint_has_a_directory_is_replaced(tmp_path: Path) -> None:
    """A link the rollback deletes is not looked through for obstacles; an ignored one stops it."""
    repo = _repo(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "b.txt").write_text("outside\n")
    tools = UsefulTools(work_dir=str(repo))
    tools.checkpoint("v1")
    rmtree_force(repo / "d")
    (repo / "d").symlink_to(outside, target_is_directory=True)
    (repo / ".gitignore").write_text("ignored/\n/d\n")  # no snapshot holds an ignored link
    assert tools.rollback("v1") == (
        "Error: rollback to 'v1' failed: nothing was changed: d is now a file git ignores, "
        "where the checkpoint has the directory of d/b.txt. A rollback replaces only files "
        "a checkpoint holds: move these away first.")
    (repo / ".gitignore").write_text("ignored/\n")
    out = tools.rollback("v1")
    assert "rewrote 1 file (d/b.txt); deleted 1 file (d)" in out, out
    assert (repo / "d" / "b.txt").read_text() == "b1\n" and not (repo / "d").is_symlink()
    assert (outside / "b.txt").read_text() == "outside\n"

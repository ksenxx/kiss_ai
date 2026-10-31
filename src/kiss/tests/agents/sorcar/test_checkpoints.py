"""End-to-end tests of the host ``checkpoint`` / ``rollback`` tools.

Every test builds a real git repository in a temporary directory and
drives ``UsefulTools.checkpoint`` / ``UsefulTools.rollback`` (and the
:mod:`kiss.agents.sorcar.checkpoints` functions under them), then checks
the files on disk and that HEAD, the branch and the index are untouched.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.checkpoints import CheckpointError, Snapshot, plan_restore, snapshot
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.conftest import is_root, posix_only


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

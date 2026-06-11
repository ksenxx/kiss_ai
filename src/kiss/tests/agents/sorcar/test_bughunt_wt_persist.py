# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt integration tests for ``git_worktree.copy_dirty_state``.

Each test builds a REAL temporary git repository (via subprocess git)
and a REAL ``git worktree`` — no mocks, patches, or fakes — and
verifies that ``GitWorktreeOps.copy_dirty_state`` mirrors the main
worktree's dirty state into the task worktree correctly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps


def _run(cmd: list[str], cwd: Path) -> None:
    """Run a command in *cwd*, raising on failure."""
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> None:
    """Create a fresh git repo with identity config at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-b", "main"], path)
    _run(["git", "config", "user.email", "test@test.com"], path)
    _run(["git", "config", "user.name", "Test"], path)
    _run(["git", "config", "commit.gpgsign", "false"], path)


def _status_codes(repo: Path) -> set[str]:
    """Return the set of two-char porcelain status codes in *repo*."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line[:2] for line in out.splitlines() if len(line) >= 4}


def test_copy_dirty_state_staged_rename_then_deleted_new_file(
    tmp_path: Path,
) -> None:
    """``RD old -> new``: stale OLD file must not survive in the worktree.

    The user staged a rename (``git mv old.txt new.txt``) and then
    deleted the new file from the working tree.  Net dirty state:
    ``old.txt`` is gone and ``new.txt`` is gone.  The baseline copy in
    the task worktree must reflect that — neither file may remain.

    Bug: ``copy_dirty_state`` only unlinked the rename's old path when
    the NEW path still existed as a file in the main worktree, so the
    stale ``old.txt`` (checked out from HEAD) survived in the worktree.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "old.txt").write_text("content\n")
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "init"], repo)
    _run(["git", "mv", "old.txt", "new.txt"], repo)
    (repo / "new.txt").unlink()
    assert "RD" in _status_codes(repo)

    wt = tmp_path / "wt"
    assert GitWorktreeOps.create(repo, "kiss/wt-bughunt-rd", wt)
    assert (wt / "old.txt").is_file()  # fresh checkout from HEAD

    GitWorktreeOps.copy_dirty_state(repo, wt)

    assert not (wt / "new.txt").exists()
    assert not (wt / "old.txt").exists()


def test_copy_dirty_state_tracked_dir_replaced_by_file(
    tmp_path: Path,
) -> None:
    """Tracked directory replaced by a same-named file must become a file.

    HEAD tracks ``a/x.txt``; the user removed the directory ``a`` and
    created a plain file named ``a``.  After the baseline copy, the
    worktree must contain file ``a`` with the new content — not a
    directory ``a/`` with the file copied INTO it as ``a/a``.

    Bug: ``shutil.copy2(src, dst)`` with ``dst`` being the still-present
    directory copied the file inside the directory instead of replacing
    it.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a").mkdir()
    (repo / "a" / "x.txt").write_text("x\n")
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "init"], repo)
    shutil.rmtree(repo / "a")
    (repo / "a").write_text("now a file\n")

    wt = tmp_path / "wt"
    assert GitWorktreeOps.create(repo, "kiss/wt-bughunt-dirfile", wt)
    assert (wt / "a").is_dir()  # fresh checkout from HEAD

    GitWorktreeOps.copy_dirty_state(repo, wt)

    assert (wt / "a").is_file()
    assert (wt / "a").read_text() == "now a file\n"
    assert not (wt / "a" / "a").exists() if (wt / "a").is_dir() else True


def test_copy_dirty_state_staged_dir_to_file_then_deleted(
    tmp_path: Path,
) -> None:
    """Deleted path whose worktree counterpart is a directory must not crash.

    HEAD tracks ``a/x.txt``.  The user ran ``git rm -r a``, created a
    plain file ``a``, staged it, then deleted it again — porcelain
    status ``AD a``.  The path ``a`` does not exist in the main
    worktree, but the freshly checked-out task worktree has a
    DIRECTORY ``a/``.

    Bug: ``copy_dirty_state`` called ``dst.unlink()`` on the directory,
    raising ``IsADirectoryError``/``PermissionError`` and crashing the
    whole worktree setup.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "a").mkdir()
    (repo / "a" / "x.txt").write_text("x\n")
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "init"], repo)
    _run(["git", "rm", "-r", "a"], repo)
    (repo / "a").write_text("f\n")
    _run(["git", "add", "a"], repo)
    (repo / "a").unlink()
    assert "AD" in _status_codes(repo)

    wt = tmp_path / "wt"
    assert GitWorktreeOps.create(repo, "kiss/wt-bughunt-ad", wt)
    assert (wt / "a").is_dir()  # fresh checkout from HEAD

    GitWorktreeOps.copy_dirty_state(repo, wt)  # must not raise

    assert not (wt / "a").exists()

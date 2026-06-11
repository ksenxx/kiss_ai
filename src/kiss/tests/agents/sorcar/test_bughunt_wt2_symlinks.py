# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt integration tests: symlink handling in copy_dirty_state.

``GitWorktreeOps.copy_dirty_state`` used ``Path.is_file()`` /
``Path.exists()`` / ``Path.is_dir()`` (which all FOLLOW symlinks) and
``shutil.copy2`` (which also follows symlinks).  That mishandled every
dirty entry that is, or collides with, a symbolic link.  These tests
reproduce the concrete failure modes with real git repos (no mocks).
"""

import os
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo(path: Path) -> Path:
    """Create a real git repo with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-b", "main"], path)
    _run(["git", "config", "user.email", "t@t.com"], path)
    _run(["git", "config", "user.name", "T"], path)
    (path / "README.md").write_text("hello\n")
    _run(["git", "add", "-A"], path)
    _run(["git", "commit", "-m", "init"], path)
    return path


def _commit_all(repo: Path, msg: str) -> None:
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", msg], repo)


def _add_worktree(repo: Path, wt_dir: Path, branch: str) -> Path:
    _run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], repo)
    return wt_dir


def test_untracked_broken_symlink_is_copied(tmp_path: Path) -> None:
    """An untracked broken symlink must be mirrored into the worktree.

    Old code: ``src.is_file()`` is False (broken link), ``dst.exists()``
    is False, so the entry was silently dropped.
    """
    repo = _make_repo(tmp_path / "repo")
    wt = _add_worktree(repo, tmp_path / "wt", "br-a")
    os.symlink("missing-target", repo / "link")

    GitWorktreeOps.copy_dirty_state(repo, wt)

    dst = wt / "link"
    assert os.path.lexists(dst), "broken symlink was dropped"
    assert dst.is_symlink()
    assert os.readlink(dst) == "missing-target"


def test_untracked_symlink_copied_as_symlink_not_file(tmp_path: Path) -> None:
    """An untracked symlink to a file must stay a symlink in the worktree.

    Old code: ``shutil.copy2`` follows the link and materializes a
    regular file in the worktree.
    """
    repo = _make_repo(tmp_path / "repo")
    wt = _add_worktree(repo, tmp_path / "wt", "br-b")
    (repo / "data.txt").write_text("payload\n")
    os.symlink("data.txt", repo / "link")

    GitWorktreeOps.copy_dirty_state(repo, wt)

    dst = wt / "link"
    assert os.path.lexists(dst)
    assert dst.is_symlink(), "symlink was copied as a regular file"
    assert os.readlink(dst) == "data.txt"


def test_typechange_symlink_to_file_does_not_corrupt_target(
    tmp_path: Path,
) -> None:
    """Tracked symlink replaced by a regular file (typechange).

    Old code: ``shutil.copy2`` wrote THROUGH the still-symlinked dst in
    the fresh worktree, corrupting the link's target file instead of
    replacing the link with a regular file.
    """
    repo = _make_repo(tmp_path / "repo")
    (repo / "target.txt").write_text("original\n")
    os.symlink("target.txt", repo / "link")
    _commit_all(repo, "add symlink")
    wt = _add_worktree(repo, tmp_path / "wt", "br-c")

    (repo / "link").unlink()
    (repo / "link").write_text("new regular content\n")

    GitWorktreeOps.copy_dirty_state(repo, wt)

    assert (wt / "target.txt").read_text() == "original\n", (
        "copy2 wrote through the symlink and corrupted target.txt"
    )
    dst = wt / "link"
    assert not dst.is_symlink(), "link should have become a regular file"
    assert dst.read_text() == "new regular content\n"


def test_renamed_symlink_to_directory_does_not_crash(tmp_path: Path) -> None:
    """``git mv`` of a tracked symlink pointing at a directory.

    Old code: ``old_dst.is_dir()`` follows the link, so
    ``shutil.rmtree(old_dst)`` raised ``OSError: Cannot call rmtree on
    a symbolic link`` and copy_dirty_state crashed.
    """
    repo = _make_repo(tmp_path / "repo")
    (repo / "d").mkdir()
    (repo / "d" / "inner.txt").write_text("inner\n")
    os.symlink("d", repo / "dlink")
    _commit_all(repo, "add dir symlink")
    wt = _add_worktree(repo, tmp_path / "wt", "br-d")

    _run(["git", "mv", "dlink", "dlink2"], repo)

    GitWorktreeOps.copy_dirty_state(repo, wt)

    assert not os.path.lexists(wt / "dlink"), "old symlink not removed"
    new = wt / "dlink2"
    assert new.is_symlink(), "renamed symlink missing or not a symlink"
    assert os.readlink(new) == "d"
    assert (wt / "d" / "inner.txt").read_text() == "inner\n", (
        "rmtree followed the symlink and deleted the directory contents"
    )


def test_deleted_broken_symlink_is_removed(tmp_path: Path) -> None:
    """A tracked broken symlink deleted in main must be removed in wt.

    Old code: ``dst.exists()`` follows the (broken) link and returns
    False, so the stale symlink was left behind in the worktree.
    """
    repo = _make_repo(tmp_path / "repo")
    os.symlink("missing", repo / "blink")
    _commit_all(repo, "add broken symlink")
    wt = _add_worktree(repo, tmp_path / "wt", "br-e")

    _run(["git", "rm", "blink"], repo)

    GitWorktreeOps.copy_dirty_state(repo, wt)

    assert not os.path.lexists(wt / "blink"), (
        "stale broken symlink left in worktree after deletion"
    )

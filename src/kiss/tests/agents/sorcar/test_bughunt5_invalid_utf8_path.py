# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUGHUNT-5: a non-UTF-8 path in the repo must not crash every git call.

Git paths are byte strings — a repo can legally contain index/HEAD
entries whose names are not valid UTF-8 (e.g. Latin-1 filenames
committed on Linux, where the filesystem stores raw bytes).  Because
``git_worktree._git`` runs every command with
``-c core.quotepath=false`` (bytes > 0x7F emitted verbatim) and
decodes stdout with strict ``encoding="utf-8"``, ANY git invocation
whose output mentions such a path raised ``UnicodeDecodeError``:
``has_uncommitted_changes``, ``copy_dirty_state``, ``stash_if_dirty``
... all crashed, and the error (a ``ValueError``, not the ``OSError``
that ``_try_setup_worktree`` guards against) propagated out of
``WorktreeSorcarAgent.run()`` and killed the whole task.

This was also internally inconsistent: ``_unquote_git_path`` already
decodes quoted paths with ``errors="surrogateescape"``.

The tests build a real repo and inject an invalid-UTF-8 path into the
index with ``git update-index --cacheinfo`` (bytes argv) — no on-disk
file is needed, so this reproduces on macOS too.  No mocks.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

_INVALID_NAME = b"caf\xe9.txt"  # 0xE9 is invalid as UTF-8


def _make_repo(path: Path) -> Path:
    """Create a git repo on branch ``main`` with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    for key, val in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, val],
            capture_output=True,
            check=True,
        )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _stage_invalid_utf8_path(repo: Path) -> None:
    """Stage an index entry whose path is not valid UTF-8 (bytes argv)."""
    sha = (
        subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=b"data\n",
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    result = subprocess.run(
        [
            b"git",
            b"-C",
            str(repo).encode(),
            b"update-index",
            b"--add",
            b"--cacheinfo",
            b"100644," + sha.encode() + b"," + _INVALID_NAME,
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


class TestInvalidUtf8Path:
    """git output containing invalid UTF-8 must not crash worktree ops."""

    def test_has_uncommitted_changes_survives_invalid_utf8(self) -> None:
        """status output with a raw 0xE9 byte must not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            _stage_invalid_utf8_path(repo)
            assert GitWorktreeOps.has_uncommitted_changes(repo) is True

    def test_copy_dirty_state_survives_invalid_utf8(self) -> None:
        """copy_dirty_state must still mirror the decodable dirty files."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            _stage_invalid_utf8_path(repo)
            (repo / "README.md").write_text("# Test\nuser edit\n")
            wt = Path(tmp) / "wt"
            wt.mkdir()
            assert GitWorktreeOps.copy_dirty_state(repo, wt) is True
            assert (wt / "README.md").read_text() == "# Test\nuser edit\n"

    def test_worktree_setup_survives_invalid_utf8(self) -> None:
        """End-to-end: _try_setup_worktree must not die with
        UnicodeDecodeError when the user's repo has a non-UTF-8 staged
        path alongside normal dirty edits."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            _stage_invalid_utf8_path(repo)
            (repo / "README.md").write_text("# Test\nuser edit\n")

            agent = WorktreeSorcarAgent("bh5-invalid-utf8")
            try:
                wt_work_dir = agent._try_setup_worktree(repo, None)
            except UnicodeDecodeError as exc:  # the pre-fix crash
                raise AssertionError(
                    f"_try_setup_worktree crashed on non-UTF-8 path: {exc}"
                ) from exc
            # Either a working worktree or a clean fallback is fine;
            # a crash is not.
            if wt_work_dir is not None:
                assert agent._wt is not None
                assert (
                    wt_work_dir / "README.md"
                ).read_text() == "# Test\nuser edit\n"
                agent.discard()

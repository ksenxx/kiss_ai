# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing bugs found in the vscode helper modules.

No mocks, patches, fakes, or test doubles. All tests use real objects,
real temp files, and real git repositories.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kiss.server.diff_merge import _parse_diff_hunks, _prepare_merge_view
from kiss.server.server import VSCodeServer


def _git(repo: Path, *args: str) -> None:
    """Run a git command in *repo*, asserting success."""
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def _make_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with one initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("# repo\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_autocomplete_survives_binary_active_file(tmp_path: Path) -> None:
    """A non-UTF-8 active file must not crash the autocomplete pipeline.

    Bug: ``_active_file_identifier_matches`` reads the active editor
    file from disk in text mode and catches only ``OSError``.  A binary
    (non-UTF-8) active file raises ``UnicodeDecodeError``, which
    propagates out of ``_complete`` and permanently kills the single
    ``_complete_worker_loop`` thread — autocomplete then stays dead for
    the rest of the daemon's life because ``_ensure_complete_worker``
    sees the (dead) worker as already started.
    """
    binary_file = tmp_path / "image.bin"
    binary_file.write_bytes(b"\xff\xfe\x00\x01\x80binary\x00data\xff")
    server = VSCodeServer()

    # Must return a (possibly empty) match list, never raise.
    matches = server._active_file_identifier_matches("fo", str(binary_file), "")

    assert matches == []


def test_parse_diff_hunks_path_with_space_b_segment(tmp_path: Path) -> None:
    """Diff parsing must handle a path containing the substring ``" b/"``.

    Bug: ``_parse_diff_hunks`` extracts the filename from the
    ``diff --git a/<path> b/<path>`` header with the greedy regex
    ``^diff --git a/.* b/(.*)``.  For a file inside a directory whose
    name ends in ``" b"`` (e.g. ``x b/y.txt``) git emits::

        diff --git a/x b/y.txt b/x b/y.txt

    and the greedy ``.*`` consumes up to the LAST ``" b/"``, yielding
    the bogus filename ``y.txt`` instead of ``x b/y.txt``.
    """
    repo = _make_repo(tmp_path)
    sub = repo / "x b"
    sub.mkdir()
    fname = "x b/y.txt"
    (repo / fname).write_text("line1\n")
    _git(repo, "add", fname)
    _git(repo, "commit", "-m", "add file in space-b dir")

    (repo / fname).write_text("line1\nline2\n")

    hunks = _parse_diff_hunks(str(repo))

    assert fname in hunks
    assert "y.txt" not in hunks


def test_merge_view_includes_file_in_space_b_directory(tmp_path: Path) -> None:
    """The merge view must list the real path of a file under ``"x b/"``.

    End-to-end consequence of the greedy diff-header regex: the merge
    manifest records the misparsed name ``y.txt`` (treated as a deleted
    file, since it does not exist on disk) instead of the real modified
    file ``x b/y.txt``.
    """
    repo = _make_repo(tmp_path)
    (repo / "x b").mkdir()
    fname = "x b/y.txt"
    (repo / fname).write_text("before\n")
    _git(repo, "add", fname)
    _git(repo, "commit", "-m", "add file in space-b dir")

    (repo / fname).write_text("before\nafter\n")
    data_dir = tmp_path / "merge-data"

    result = _prepare_merge_view(str(repo), str(data_dir), {}, set())

    assert result.get("status") == "opened"
    manifest = json.loads((data_dir / "pending-merge.json").read_text())
    names = [f["name"] for f in manifest["files"]]
    assert names == [fname]

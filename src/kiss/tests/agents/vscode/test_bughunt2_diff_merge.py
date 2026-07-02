# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: attribute-forced binary files must not escape merge review.

Reproduces a real bug in ``_prepare_merge_view`` (diff_merge.py): a file
marked binary via ``.gitattributes`` (e.g. ``*.dat binary``) whose content
is plain TEXT on disk (no NUL bytes) produces ``git diff -U0`` output of
the form ``Binary files a/data.dat and b/data.dat differ`` with NO hunk
lines.  ``_parse_diff_hunks`` therefore yields ``{"data.dat": []}``.

In ``_prepare_merge_view`` the first-loop condition was::

    if not hunks and (not fpath.is_file() or _is_binary_file(fpath)):

For this file the condition is False (the file exists and the NUL-byte
sniff in ``_is_binary_file`` finds no NUL), so control fell through to
``_agent_file_hunks(..., post_file_hunks=[])`` which returned ``[]``, the
following ``elif fpath.is_file() and _is_binary_file(fpath)`` was also
False, and the file was COMPLETELY DROPPED from the review.  If it was
the only change, ``_prepare_merge_view`` returned ``{"error": "No
changes"}`` and the agent's modification silently escaped merge review —
it could be neither accepted nor rejected.

The fix: an empty hunk list in ``post_hunks`` can ONLY come from the
``Binary files … differ`` parser branch (mode-only changes never create
a ``post_hunks`` entry at all), so ``if not hunks:`` alone must route the
file to binary whole-file review.

No mocks, patches, fakes, or test doubles. Real git repository, real
merge-view preparation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
    _snapshot_files,
)


def _git(repo: Path, *args: str) -> None:
    """Run a git command in *repo*, asserting success."""
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a repo whose ``.gitattributes`` forces ``*.dat`` binary.

    Returns:
        The repo path and the committed text content of ``data.dat``.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / ".gitattributes").write_text("*.dat binary\n")
    content = "".join(f"record {i}\n" for i in range(1, 11))
    (repo / "data.dat").write_text(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo, content


def test_gitattributes_binary_text_file_is_reviewable(tmp_path: Path) -> None:
    """An edited attr-binary (but text-on-disk) file must appear in review."""
    repo, original = _make_repo(tmp_path)

    # Pre-task capture, exactly as task_runner does it.
    pre_hunks = _parse_diff_hunks(str(repo))
    pre_untracked = _capture_untracked(str(repo))
    pre_hashes = _snapshot_files(
        str(repo), pre_untracked | set(pre_hunks.keys()),
    )

    # Agent action: edit the attribute-binary file with pure text bytes.
    (repo / "data.dat").write_text(
        original.replace("record 5\n", "record 5 changed\n"),
    )

    # Sanity: git really reports this as a hunk-less binary diff.
    diff = subprocess.run(
        ["git", "diff", "-U0", "--no-renames", "HEAD", "--no-color"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert "Binary files " in diff.stdout, diff.stdout
    assert "@@" not in diff.stdout, diff.stdout

    data_dir = tmp_path / "merge-data"
    result = _prepare_merge_view(
        str(repo), str(data_dir), pre_hunks, pre_untracked, pre_hashes,
    )

    # The bug made the only change invisible: {"error": "No changes"}.
    assert result.get("status") == "opened", result

    manifest = json.loads((data_dir / "pending-merge.json").read_text())
    entries = {f["name"]: f for f in manifest["files"]}
    assert "data.dat" in entries, sorted(entries)
    entry = entries["data.dat"]

    # It must be reviewed as a whole-file binary decision, and its base
    # copy must hold the pre-task bytes so rejecting restores them.
    assert entry.get("binary") is True, entry
    assert Path(entry["base"]).read_bytes() == original.encode()
    assert Path(entry["current"]).read_text() != original

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: merge view must survive a git rename of a tracked file.

Reproduces a real bug in ``_parse_diff_hunks``: the ``git diff -U0`` call
did not pass ``--no-renames``, so when the agent renamed a tracked file
(``git mv old.txt new.txt`` plus a small edit) git emitted a single
*rename* diff entry.  The parser keyed the (old-file-relative) hunks
under the NEW file name, after which ``_write_base_copy``'s
``git show HEAD:new.txt`` failed (the path did not exist at HEAD) and an
EMPTY base file was written.  The resulting manifest was internally
inconsistent:

- the hunk referenced base lines that do not exist in the empty base
  copy (``bs + bc`` past the end of the base file), so the merge view
  rendered garbage and rejecting the hunk corrupted the file instead of
  restoring the pre-task content; and
- the deletion of the old path was completely invisible — rejecting
  everything could never bring ``old.txt`` back.

No mocks, patches, fakes, or test doubles. Real git repository, real
merge-view preparation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kiss.server.diff_merge import (
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


def _make_repo(tmp_path: Path) -> Path:
    """Create a git repo whose initial commit contains old.txt."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    content = "".join(f"line{i}\n" for i in range(1, 31))
    (repo / "old.txt").write_text(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def _read_lines(path: str) -> list[str]:
    """Return the lines of *path*, treating a missing file as empty."""
    p = Path(path)
    if not p.is_file():
        return []
    return p.read_text().splitlines(keepends=True)


def _reject_all(entry: dict) -> str:
    """Apply web_server-style 'reject hunk' to every hunk of *entry*.

    Rejecting a hunk replaces the current region ``[cs, cs+cc)`` with the
    base region ``[bs, bs+bc)`` (see ``web_server.py``), so rejecting all
    hunks must reproduce the base (pre-task) content exactly.
    """
    base_lines = _read_lines(entry["base"])
    cur_lines = list(_read_lines(entry["current"]))
    for hunk in sorted(entry["hunks"], key=lambda h: h["cs"], reverse=True):
        cur_lines[hunk["cs"] : hunk["cs"] + hunk["cc"]] = base_lines[
            hunk["bs"] : hunk["bs"] + hunk["bc"]
        ]
    return "".join(cur_lines)


def test_renamed_file_merge_view_is_consistent_and_restorable(
    tmp_path: Path,
) -> None:
    """A git rename + edit must produce a self-consistent merge manifest."""
    repo = _make_repo(tmp_path)
    original = (repo / "old.txt").read_text()

    # Pre-task capture, exactly as task_runner does it.
    pre_hunks = _parse_diff_hunks(str(repo))
    pre_untracked = _capture_untracked(str(repo))
    pre_hashes = _snapshot_files(
        str(repo), pre_untracked | set(pre_hunks.keys()),
    )

    # Agent action: rename the tracked file and make a small edit.
    _git(repo, "mv", "old.txt", "new.txt")
    (repo / "new.txt").write_text(
        original.replace("line5\n", "line5-agent\n"),
    )

    data_dir = tmp_path / "merge-data"
    result = _prepare_merge_view(
        str(repo), str(data_dir), pre_hunks, pre_untracked, pre_hashes,
    )
    assert result.get("status") == "opened", result

    manifest = json.loads((data_dir / "pending-merge.json").read_text())
    entries = {f["name"]: f for f in manifest["files"]}

    # Every hunk must reference base/current line ranges that actually
    # exist — an empty base with hunks pointing past its end is the
    # broken-rename signature.
    for name, entry in entries.items():
        base_lines = _read_lines(entry["base"])
        cur_lines = _read_lines(entry["current"])
        for hunk in entry["hunks"]:
            assert hunk["bs"] + hunk["bc"] <= len(base_lines), (
                f"{name}: hunk {hunk} references base lines past the end "
                f"of its {len(base_lines)}-line base copy"
            )
            assert hunk["cs"] + hunk["cc"] <= len(cur_lines), (
                f"{name}: hunk {hunk} references current lines past the "
                f"end of its {len(cur_lines)}-line current file"
            )

    # The deletion of old.txt must be reviewable: rejecting everything
    # has to be able to restore the original file at its old path.
    assert "old.txt" in entries, (
        "rename made the old path's deletion invisible in the merge view: "
        f"{sorted(entries)}"
    )
    assert _reject_all(entries["old.txt"]) == original

    # And rejecting all of new.txt's hunks must yield its pre-task state
    # (the file did not exist, i.e. empty content), not a corrupted file.
    assert "new.txt" in entries
    assert _reject_all(entries["new.txt"]) == ""

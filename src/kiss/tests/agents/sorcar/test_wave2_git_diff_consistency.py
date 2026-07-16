# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Wave-2 git-diff consistency fixes — end-to-end tests.

Covers three cross-file findings, each against real temporary git
repositories (no mocks, patches, fakes, or test doubles):

* ``GitWorktreeOps._diff_name_only`` historically parsed
  ``git diff --name-only`` with ``stdout.strip().splitlines()``, which
  mangled filenames with leading/trailing spaces (legal on POSIX and
  NOT C-quoted by git).  It now uses NUL-separated ``-z`` output.

* ``GitWorktreeOps.copy_dirty_state`` historically re-implemented
  porcelain-status parsing with ``splitlines()``, which splits inside
  an unquoted filename containing a raw unicode line separator
  (U+2028) when ``core.quotepath=false`` is in force (as it always is
  via the module's ``_git`` wrapper).  It now routes through the
  shared ``_porcelain_entries`` parser (which splits on ``\\n`` only)
  — the same parser backing ``merge_flow._porcelain_paths``.

* The canonical ``User prompt:`` / ``Result:`` commit-message block
  format was spelled independently in ``vscode/helpers.py`` (writer)
  and ``sorcar/git_worktree.py`` (dedup detector).  It is now
  single-sourced via ``USER_PROMPT_HEADING`` / ``TASK_RESULT_HEADING``
  constants; the tests pin the byte-exact writer/detector agreement.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    TASK_RESULT_HEADING,
    USER_PROMPT_HEADING,
    GitWorktreeOps,
    _ensure_task_metadata,
    _porcelain_entries,
)
from kiss.agents.vscode.helpers import _append_task_result, _append_user_prompt


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in *repo* and assert that it succeeds."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def _make_repo(tmp_path: Path) -> Path:
    """Create a real git repo with one initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("# repo\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


# ---------------------------------------------------------------------------
# Finding A: _diff_name_only must not mangle space-adjacent filenames.
# ---------------------------------------------------------------------------


def test_diff_name_only_preserves_leading_and_trailing_spaces(
    tmp_path: Path,
) -> None:
    """A filename with leading AND trailing spaces survives byte-exact.

    Failing before the fix: the whole-output ``strip()`` ate the
    leading spaces of the first listed path and the trailing spaces of
    the last, so the returned name never matched the on-disk file.
    """
    repo = _make_repo(tmp_path)
    filename = "  spacey name  "
    (repo / filename).write_text("before\n")
    _git(repo, "add", "--", filename)
    _git(repo, "commit", "-m", "add space-adjacent file")

    (repo / filename).write_text("unstaged change\n")
    assert GitWorktreeOps._diff_name_only(repo) == [filename]

    _git(repo, "add", "--", filename)
    assert GitWorktreeOps._diff_name_only(repo, "--cached") == [filename]


def test_diff_name_only_multiple_files_with_spaces_and_flags(
    tmp_path: Path,
) -> None:
    """Every entry of a multi-file listing is preserved exactly."""
    repo = _make_repo(tmp_path)
    names = [" lead.txt", "trail.txt ", "plain.txt"]
    for name in names:
        (repo / name).write_text("v1\n")
        _git(repo, "add", "--", name)
    _git(repo, "commit", "-m", "add files")

    for name in names:
        (repo / name).write_text("v2\n")
    assert sorted(GitWorktreeOps._diff_name_only(repo, "--no-renames")) == sorted(
        names
    )


def test_diff_name_only_does_not_unquote_z_output(tmp_path: Path) -> None:
    """``-z`` output is never C-quoted; a quote-wrapped name stays raw.

    A filename that starts AND ends with a double-quote must be
    returned verbatim — running the C-unquoting helper on ``-z``
    output would strip the quotes and corrupt the name.
    """
    repo = _make_repo(tmp_path)
    filename = '"quoted"'
    (repo / filename).write_text("v1\n")
    _git(repo, "add", "--", filename)
    _git(repo, "commit", "-m", "add quoted-name file")

    (repo / filename).write_text("v2\n")
    assert GitWorktreeOps._diff_name_only(repo) == [filename]


def test_diff_name_only_unicode_name(tmp_path: Path) -> None:
    """Non-ASCII names are returned as real paths (parity with old fix)."""
    repo = _make_repo(tmp_path)
    _git(repo, "config", "core.quotePath", "true")
    filename = "café.txt"
    (repo / filename).write_text("before\n")
    _git(repo, "add", "--", filename)
    _git(repo, "commit", "-m", "add unicode file")

    (repo / filename).write_text("after\n")
    assert GitWorktreeOps._diff_name_only(repo) == [filename]


# ---------------------------------------------------------------------------
# Finding B: copy_dirty_state routed through the shared porcelain parser.
# ---------------------------------------------------------------------------


def _make_worktree_checkout(repo: Path, tmp_path: Path) -> Path:
    """Create a real linked git worktree of *repo* at the same HEAD."""
    wt_dir = tmp_path / "wt"
    _git(repo, "worktree", "add", "--detach", str(wt_dir))
    return wt_dir


def test_copy_dirty_state_unicode_line_separator_name(tmp_path: Path) -> None:
    """A dirty file with U+2028 in its name is mirrored into the worktree.

    Failing before the fix: ``copy_dirty_state`` iterated
    ``status.stdout.splitlines()``, which splits on the raw U+2028
    byte sequence that git (running with ``core.quotepath=false``)
    emits verbatim inside the porcelain line, mis-parsing the entry so
    the file was silently dropped from the baseline copy.
    """
    repo = _make_repo(tmp_path)
    wt_dir = _make_worktree_checkout(repo, tmp_path)

    filename = "a\u2028b.txt"
    (repo / filename).write_text("dirty content\n")

    assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
    assert (wt_dir / filename).read_text() == "dirty content\n"


def test_copy_dirty_state_space_adjacent_names_and_rename(
    tmp_path: Path,
) -> None:
    """Space-adjacent dirty files and staged renames mirror exactly."""
    repo = _make_repo(tmp_path)
    old_name = " old name .txt"
    (repo / old_name).write_text("v1\n")
    (repo / "keep.txt").write_text("v1\n")
    _git(repo, "add", "--", old_name, "keep.txt")
    _git(repo, "commit", "-m", "add files")

    wt_dir = _make_worktree_checkout(repo, tmp_path)
    assert (wt_dir / old_name).exists()

    new_name = "  renamed to .txt"
    _git(repo, "mv", "--", old_name, new_name)
    (repo / "keep.txt").write_text("modified\n")
    (repo / " untracked lead.txt").write_text("new\n")

    assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
    assert not (wt_dir / old_name).exists()
    assert (wt_dir / new_name).read_text() == "v1\n"
    assert (wt_dir / "keep.txt").read_text() == "modified\n"
    assert (wt_dir / " untracked lead.txt").read_text() == "new\n"


def test_porcelain_entries_rename_and_line_separator(tmp_path: Path) -> None:
    """The shared parser handles renames and U+2028 names on real output."""
    repo = _make_repo(tmp_path)
    (repo / "src.txt").write_text("v1\n")
    _git(repo, "add", "src.txt")
    _git(repo, "commit", "-m", "add src")
    _git(repo, "mv", "src.txt", "dst.txt")
    (repo / "u\u2028v.txt").write_text("x\n")

    status = subprocess.run(
        ["git", "-c", "core.quotepath=false", "status", "--porcelain", "-uall"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert status.returncode == 0
    entries = _porcelain_entries(status.stdout)
    assert ("R ", "src.txt", "dst.txt") in entries
    assert ("??", None, "u\u2028v.txt") in entries


# ---------------------------------------------------------------------------
# Finding C: single-sourced commit-message block format.
# ---------------------------------------------------------------------------


def test_helpers_blocks_detected_as_duplicate_by_dedup() -> None:
    """Blocks appended by the helpers.py writers must dedup byte-exactly.

    ``_ensure_task_metadata`` (the git_worktree.py auto-commit dedup
    detector) must recognise a message stamped by the helpers.py
    writers as already carrying the current prompt/result blocks and
    return it unchanged — pinning the byte-exact agreement that the
    shared ``USER_PROMPT_HEADING`` / ``TASK_RESULT_HEADING`` constants
    guarantee.
    """
    prompt = "Fix the flaky login test"
    result = "Stabilized the test by awaiting the redirect."
    msg = _append_task_result(
        _append_user_prompt("fix: stabilize login test", prompt), result
    )
    assert _ensure_task_metadata(msg, prompt, result) == msg

    prompt_only = _append_user_prompt("fix: stabilize login test", prompt)
    assert _ensure_task_metadata(prompt_only, prompt, None) == prompt_only

    result_only = _append_task_result("fix: stabilize login test", result)
    assert _ensure_task_metadata(result_only, None, result) == result_only


def test_helpers_blocks_use_shared_heading_constants() -> None:
    """The writers compose blocks exactly as ``HEADING + text``."""
    msg = _append_user_prompt("subject", "  a prompt  ")
    assert msg == f"subject{USER_PROMPT_HEADING}a prompt"
    msg = _append_task_result("subject", "  a result  ")
    assert msg == f"subject{TASK_RESULT_HEADING}a result"


def test_missing_prompt_block_inserted_before_helper_result_block() -> None:
    """A helpers-stamped result-only message gains the prompt block."""
    result = "Done."
    prompt = "Do the thing"
    msg = _append_task_result("chore: thing", result)
    stamped = _ensure_task_metadata(msg, prompt, result)
    assert stamped == (
        f"chore: thing{USER_PROMPT_HEADING}{prompt}{TASK_RESULT_HEADING}{result}"
    )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The staged diff handed to the commit-message LLM is bounded.

Field incident (``~/.kiss/kiss-web-stderr.log``, 2026-09-23 15:11 UTC):
a task had copied 38,081 benchmark result files (38.5 million lines)
into its worktree.  "Auto-commit and merge" ran ``git diff --cached``
over all of it and read the whole patch into one Python string for
the commit-message prompt; the daemon grew to 45 GB and the merge
thread, which held the tab's ``is_merging`` claim, never finished, so
every main-tree task was refused with "A worktree merge is in
progress" until the daemon was restarted.

Everything here runs against real git repositories; the watchdog case
uses a real ``diff.external`` helper that hangs.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import (
    COMMIT_MESSAGE_DIFF_LIMIT_BYTES,
    GitWorktreeOps,
    _git_stdout_head,
)
from kiss.agents.sorcar.sorcar_agent import auto_commit_changes


def _git(cwd: Path, *args: str) -> str:
    """Run git in *cwd* and return its stdout, byte for byte.

    Decoded from bytes rather than read in text mode: the product
    returns git's output verbatim, and on Windows the files written
    here contain ``\\r\\n`` (text-mode writes), so a reference read
    with universal newlines would drop the ``\\r`` the patch contains.
    """
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, check=True,
    ).stdout.decode("utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one commit."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


def _stage_big_file(repo: Path, name: str, lines: int) -> None:
    """Stage a generated file of *lines* distinct lines."""
    (repo / name).write_text(
        "".join(f"result line {i} " + "x" * 40 + "\n" for i in range(lines)),
        encoding="utf-8",
    )
    _git(repo, "add", name)


def test_no_staged_changes(repo: Path) -> None:
    assert GitWorktreeOps.has_staged_changes(repo) is False
    assert GitWorktreeOps.staged_diff(repo) == ""


def test_small_diff_is_returned_whole(repo: Path) -> None:
    (repo / "README.md").write_text("hello\nworld\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    assert GitWorktreeOps.has_staged_changes(repo) is True
    expected = _git(repo, "diff", "--cached").strip()
    assert GitWorktreeOps.staged_diff(repo) == expected
    assert "[Patch truncated" not in expected


def test_huge_diff_is_capped_and_summarised(repo: Path) -> None:
    """A multi-megabyte patch yields a bounded prompt that still names
    every file and the totals."""
    _stage_big_file(repo, "results.txt", 60_000)  # ~3.5 MB
    (repo / "note.txt").write_text("small\n", encoding="utf-8")
    _git(repo, "add", "note.txt")
    full_size = len(_git(repo, "diff", "--cached").encode())
    assert full_size > COMMIT_MESSAGE_DIFF_LIMIT_BYTES

    text = GitWorktreeOps.staged_diff(repo)

    assert len(text.encode()) < 2 * COMMIT_MESSAGE_DIFF_LIMIT_BYTES
    assert text.startswith(
        f"[Patch truncated to its first {COMMIT_MESSAGE_DIFF_LIMIT_BYTES:,} bytes; "
    )
    assert "2 files changed, 60001 insertions(+)" in text
    # The --stat head names both files even though the patch head only
    # reaches the first one.
    assert re.search(r"note\.txt\s+\|\s+1 \+", text)
    assert re.search(r"results\.txt\s+\|\s+60000 \+", text)
    assert "diff --git a/note.txt b/note.txt" in text
    # The patch head ends on a complete line, not mid-way through one.
    assert re.fullmatch(r"\+result line \d+ x{40}", text.rsplit("\n", 1)[-1])
    assert text.count("\n+result line") < 60_000


def test_custom_cap_is_honoured(repo: Path) -> None:
    _stage_big_file(repo, "results.txt", 5_000)
    text = GitWorktreeOps.staged_diff(repo, max_bytes=10_000)
    assert text.startswith("[Patch truncated to its first 10,000 bytes;")
    assert len(text.encode()) < 20_000


def test_pathspecs_restrict_the_diff(repo: Path) -> None:
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    _git(repo, "add", "a.txt", "b.txt")
    text = GitWorktreeOps.staged_diff(repo, pathspecs=["a.txt"])
    assert "diff --git a/a.txt b/a.txt" in text
    assert "b.txt" not in text
    # With a cap, the summary is restricted the same way.
    _stage_big_file(repo, "big.txt", 5_000)
    text = GitWorktreeOps.staged_diff(repo, max_bytes=2_000, pathspecs=["big.txt", "a.txt"])
    assert "2 files changed" in text
    assert "b.txt" not in text


def test_stdout_head_without_newline_keeps_the_bytes(repo: Path) -> None:
    """A cap smaller than the first line cannot cut back to a line
    boundary and returns the raw head."""
    (repo / "README.md").write_text("hello\nworld\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    text, truncated = _git_stdout_head("diff", "--cached", cwd=repo, max_bytes=4)
    assert (text, truncated) == ("diff", True)
    text, truncated = _git_stdout_head("diff", "--cached", cwd=repo, max_bytes=1_000_000)
    assert truncated is False
    assert text.strip() == _git(repo, "diff", "--cached").strip()


def test_stdout_head_watchdog_kills_a_hung_git_and_reports_truncation(
    repo: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A git that stops producing output is killed at the timeout and
    its partial output is flagged incomplete, never returned as whole."""
    (repo / "README.md").write_text("hello\nworld\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    hang = tmp_path / "hang.sh"
    hang.write_text(
        "#!/bin/sh\necho PARTIAL-LINE\necho more\nsleep 30\n", encoding="utf-8", newline="\n",
    )
    hang.chmod(0o755)
    started = time.monotonic()
    # git hands diff.external to its POSIX shell (Git bash on Windows), where
    # a native ``C:\...`` path loses its backslashes: use forward slashes.
    with caplog.at_level(logging.WARNING, logger="kiss.agents.sorcar.git_worktree"):
        text, truncated = _git_stdout_head(
            "-c", f"diff.external={hang.as_posix()}", "diff", "--cached",
            cwd=repo, max_bytes=10_000, timeout=0.5,
        )
    assert time.monotonic() - started < 10
    assert truncated is True
    assert text == "PARTIAL-LINE\nmore"
    assert any("timed out after" in r.getMessage() for r in caplog.records)


def test_has_staged_changes_reports_git_failures(
    repo: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A git failure (rc > 1) is logged and treated as staged so the
    following commit attempt surfaces git's own error."""
    (repo / ".git" / "index").write_bytes(b"DIRC garbage")
    with caplog.at_level(logging.WARNING, logger="kiss.agents.sorcar.git_worktree"):
        assert GitWorktreeOps.has_staged_changes(repo) is True
    assert any(
        "git diff --cached --quiet failed" in r.getMessage() and "rc=128" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_unborn_head_repository(tmp_path: Path) -> None:
    """A repository without any commit compares the index to the empty tree."""
    _git(tmp_path, "init", "-q")
    assert GitWorktreeOps.has_staged_changes(tmp_path) is False
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    _git(tmp_path, "add", "a.txt")
    assert GitWorktreeOps.has_staged_changes(tmp_path) is True
    assert "diff --git a/a.txt b/a.txt" in GitWorktreeOps.staged_diff(tmp_path)


def test_auto_commit_hands_a_bounded_diff_to_the_message_generator(repo: Path) -> None:
    """The auto-commit path (worktree merge, task end) must never build
    the whole patch: the message generator sees the capped text."""
    (repo / "results.txt").write_text(
        "".join(f"line {i}\n" for i in range(200_000)), encoding="utf-8",
    )
    seen: list[str] = []

    def message_fn(commit_dir: Path, user_prompt: str | None, task_result: str | None) -> str:
        seen.append(GitWorktreeOps.staged_diff(commit_dir))
        return "test: capped commit"

    assert auto_commit_changes(repo, None, message_fn) is True
    assert len(seen) == 1
    assert seen[0].startswith("[Patch truncated")
    assert len(seen[0].encode()) < 2 * COMMIT_MESSAGE_DIFF_LIMIT_BYTES
    assert _git(repo, "log", "-1", "--format=%s").strip() == "test: capped commit"
    assert GitWorktreeOps.has_staged_changes(repo) is False
    # Nothing staged: the emptiness check runs without reading a patch.
    assert auto_commit_changes(repo, None, message_fn) is False
    assert len(seen) == 1

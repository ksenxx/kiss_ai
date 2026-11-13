"""Explorer git commands after the 2026-10-26 sweep.

1. ``git_action`` takes the repository's lock, the one every other
   mutation (the worktree merge transaction, auto-commit) takes from
   any checkout of the repository, instead of a lock of the linked
   worktree's own path, which nothing else ever took.
2. ``gitShow`` / ``gitFileAt`` / ``gitFileDiff`` / ``gitCompare`` stream
   git's output and stop at ``GIT_SHOW_MAX_BYTES``: a huge blob or
   patch costs the daemon the cap, not its whole size (it used to be
   read whole, then cut).  The kept prefix is byte-exact and decoded
   as every other git output (``surrogateescape``), so both sides of a
   diff of one unchanged file come back equal whatever its bytes, and
   a git that fails (not one the cap killed) is an error, not content.
3. ``gitCompare`` verifies the target commit, not only the base.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import threading
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from types import CodeType

import pytest

from kiss.agents.sorcar import git_worktree
from kiss.agents.sorcar.git_worktree import repo_lock
from kiss.server import explorer
from kiss.server.explorer import (
    GIT_SHOW_MAX_BYTES,
    git_action,
    git_compare,
    git_file_at,
    git_file_diff,
    git_show,
)
from kiss.tests.conftest import posix_only

#: Holds the flock on the file named on the command line until its stdin has a line.
_HOLD_FLOCK = """\
import fcntl, sys
lock = open(sys.argv[1], "a+")
fcntl.flock(lock, fcntl.LOCK_EX)
print("held", flush=True)
sys.stdin.readline()
"""

_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@x",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@x",
    "HOME": "/nonexistent",
}


def _git(repo: Path, *args: str) -> str:
    """Run git in *repo* and return its stdout, raising on failure."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True, env=_ENV,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    """Stage everything, commit, and return the new HEAD sha."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one small commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    # The tests compare the texts byte for byte: write ``\n`` as is (Windows
    # ``write_text`` would store ``\r\n``).
    (repo / "a.txt").write_text("one\n", newline="\n")
    _commit(repo, "first")
    return repo


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8", "surrogateescape"))


class TestGitActionLock:
    """``git_action`` serialises with the repository's lock, from any of its checkouts."""

    def test_action_in_linked_worktree_waits_for_the_main_worktree_lock(
        self, repo: Path, tmp_path: Path,
    ) -> None:
        sha = _git(repo, "rev-parse", "HEAD")
        linked = tmp_path / "linked"
        _git(repo, "worktree", "add", "-q", str(linked), "-b", "side")
        held = threading.Event()
        release = threading.Event()

        def hold_main_lock() -> None:
            with repo_lock(repo):
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold_main_lock, daemon=True)
        holder.start()
        assert held.wait(10)
        result: dict[str, object] = {}

        def run_action() -> None:
            result.update(git_action(str(linked), "checkoutDetached", sha))

        action = threading.Thread(target=run_action, daemon=True)
        try:
            action.start()
            # Meet the action at its lock acquisition: a thread parked in
            # the lock's (C) acquire has ``git_action``'s own frame as its
            # innermost Python frame, on the ``with repo_lock(`` line, for
            # as long as the holder keeps the lock; anywhere else it is
            # inside a callee (git runs through subprocess) or past the
            # lock and gone.
            lock_line = _source_line(git_action, "with repo_lock(")
            stable = 0
            deadline = time.monotonic() + 10
            while stable < 20 and time.monotonic() < deadline:
                time.sleep(0.01)
                line = _innermost_line(action, git_action.__code__)
                stable = stable + 1 if line == lock_line else 0
            assert stable == 20, "git_action is not waiting for the lock"
            assert result == {}
        finally:
            release.set()
            action.join(10)
            holder.join(10)
        assert result.get("ok") is True, result
        assert _git(linked, "rev-parse", "HEAD") == sha
        assert _git(linked, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"  # detached
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"

    @pytest.mark.parametrize(
        ("layout", "holder", "actor"),
        [
            ("separate", "main", "linked"),
            ("separate", "linked", "main"),
            ("bare", "two", "one"),
            ("separate", "main", "main"),
            ("separate", "main", "main/.kiss-worktrees/kiss_wt-1"),
            pytest.param(
                "symlinked", "repo", "repo/.kiss-worktrees/kiss_wt-1",
                marks=posix_only("the worktrees folder is a symlink"),
            ),
        ],
    )
    def test_action_waits_for_a_transaction_in_any_checkout_of_the_repository(
        self, repo: Path, tmp_path: Path, layout: str, holder: str, actor: str,
    ) -> None:
        """An action waits for a transaction of its repository, whatever its worktree list names.

        The lock was that of the first row of ``git worktree list``, which
        is not the main checkout of every repository: of one made with
        ``--separate-git-dir`` git lists the metadata directory, a bare one
        has none, and a git before 2.36 gave no list to read.  The lock
        then was one no transaction holds, and the action ran in the
        middle of a merge.  A repository has one ``repo_lock``, whichever
        of its checkouts names it, and a transaction holds it around its
        git steps: the action waits for that lock alone.

        Nor is the lock found by a folder's name.  Git names a worktree by
        where its files are: with ``.kiss-worktrees`` a symlink to a folder
        elsewhere, two levels above the task's worktree is the folder that
        one is in, not the task's checkout, and its lock is one the task's
        repository never holds.
        """
        if layout == "separate":
            _git(tmp_path, "init", "-q", "-b", "main", "--separate-git-dir", "meta", "main")
            (tmp_path / "main" / "a.txt").write_text("one\n")
            _commit(tmp_path / "main", "first")
            for branch, where in ("side", "../linked"), ("task", ".kiss-worktrees/kiss_wt-1"):
                _git(tmp_path / "main", "worktree", "add", "-q", where, "-b", branch)
        elif layout == "symlinked":
            kept = tmp_path / "storage" / ".kiss-worktrees"
            kept.mkdir(parents=True)
            (repo / ".kiss-worktrees").symlink_to(kept, target_is_directory=True)
            _git(repo, "worktree", "add", "-q", ".kiss-worktrees/kiss_wt-1", "-b", "task")
        else:
            _git(tmp_path, "clone", "-q", "--bare", str(repo), "bare.git")
            for name in ("one", "two"):
                added = str(tmp_path / name)
                _git(tmp_path / "bare.git", "worktree", "add", "-q", added, "-b", name)
        sha = _git(tmp_path / actor, "rev-parse", "HEAD")
        held = threading.Event()
        release = threading.Event()

        def transaction() -> None:
            with repo_lock(tmp_path / holder):
                held.set()
                release.wait(30)

        result: dict[str, object] = {}

        def run_action() -> None:
            result.update(git_action(str(tmp_path / actor), "createTag", sha, "v1"))

        merging = threading.Thread(target=transaction, daemon=True)
        merging.start()
        assert held.wait(10)
        action = threading.Thread(target=run_action, daemon=True)
        action.start()
        try:
            action.join(1.0)
            assert action.is_alive() and result == {}, "the action ran inside the transaction"
            assert _git(tmp_path / actor, "tag", "--list") == ""
        finally:
            release.set()
            action.join(30)
            merging.join(10)
        assert result.get("ok") is True, result
        assert _git(tmp_path / actor, "tag", "--list") == "v1"

    @posix_only("the other process holds a fcntl.flock")
    def test_a_repository_another_process_keeps_is_reported_busy(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Another process's transaction is waited for too; one that outlasts the wait is an error.

        Not an exception: the reply names the busy repository, and the
        action runs once the other process is done.
        """
        monkeypatch.setattr(git_worktree, "PROCESS_LOCK_TIMEOUT_SECONDS", 0.5)
        sha = _git(repo, "rev-parse", "HEAD")
        peer = subprocess.Popen(
            [sys.executable, "-c", _HOLD_FLOCK, str(repo / ".git" / "kiss-reclaim.lock")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        assert peer.stdin is not None and peer.stdout is not None
        try:
            assert peer.stdout.readline().strip() == "held"
            busy = git_action(str(repo), "createTag", sha, "v1")
            assert "still using this repository" in str(busy.get("error")), busy
            assert _git(repo, "tag", "--list") == ""
        finally:
            peer.stdin.write("\n")
            peer.stdin.close()
            peer.wait(timeout=10)
        assert git_action(str(repo), "createTag", sha, "v1").get("ok") is True


def _innermost_line(thread: threading.Thread, code: CodeType) -> int | None:
    """The line *thread* is on when its innermost Python frame runs *code*, else None."""
    frame = sys._current_frames().get(thread.ident or -1)
    if frame is None or frame.f_code is not code:
        return None
    return frame.f_lineno


def _source_line(function: Callable[..., object], text: str) -> int:
    """The number of the first line of *function* containing *text*."""
    lines, first = inspect.getsourcelines(function)
    return first + next(i for i, line in enumerate(lines) if text in line)


class TestStreamedOutput:
    """Huge git output is cut at the cap while being read, not after."""

    def test_git_file_at_streams_a_huge_blob(self, repo: Path) -> None:
        line = "x" * 1023 + "\n"
        (repo / "big.txt").write_text(line * (32 * 1024))  # 32 MiB
        sha = _commit(repo, "big")
        tracemalloc.start()
        try:
            reply = git_file_at(str(repo), sha, "big.txt")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert reply["truncated"] is True
        # The first GIT_SHOW_MAX_BYTES of the blob, byte for byte.
        assert reply["text"] == (line * (32 * 1024))[:GIT_SHOW_MAX_BYTES]
        # Reading the 32 MiB blob whole took eight times the cap (and
        # as much again for the cut's re-encoding).
        assert peak < 4 * GIT_SHOW_MAX_BYTES, peak

    def test_git_show_and_compare_cut_a_huge_patch(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        line = "y" * 1023 + "\n"
        (repo / "big.txt").write_text(line * (8 * 1024))  # 8 MiB
        sha = _commit(repo, "huge patch")
        shown = git_show(str(repo), sha, "")
        assert shown["subject"] == "huge patch"
        assert shown["truncated"] is True
        assert 0 < _utf8_len(shown["text"]) <= GIT_SHOW_MAX_BYTES
        assert shown["text"].startswith("commit " + sha)
        compared = git_compare(str(repo), base, sha)
        assert compared["truncated"] is True
        assert 0 < _utf8_len(compared["text"]) <= GIT_SHOW_MAX_BYTES
        assert "big.txt" in compared["text"]

    def test_git_file_diff_cuts_both_sides(self, repo: Path) -> None:
        line = "z" * 1023 + "\n"
        (repo / "big.txt").write_text(line * (5 * 1024))  # 5 MiB
        sha = _commit(repo, "big")
        (repo / "big.txt").write_text(line * (6 * 1024))  # 6 MiB, uncommitted
        against_worktree = git_file_diff(str(repo), "", "big.txt", "big.txt")
        assert against_worktree["truncated"] is True
        assert _utf8_len(against_worktree["original"]) <= GIT_SHOW_MAX_BYTES
        assert _utf8_len(against_worktree["modified"]) == GIT_SHOW_MAX_BYTES
        committed = git_file_diff(str(repo), sha, "big.txt", "big.txt")
        assert committed["truncated"] is True and committed["originalMissing"] is True
        assert _utf8_len(committed["modified"]) <= GIT_SHOW_MAX_BYTES

    def test_small_output_is_complete_and_not_truncated(self, repo: Path) -> None:
        (repo / "a.txt").write_text("one\ntwo\n", newline="\n")
        sha = _commit(repo, "second")
        shown = git_show(str(repo), sha, "a.txt")
        assert shown["truncated"] is False and "+two" in shown["text"]
        at = git_file_at(str(repo), sha, "a.txt")
        assert at == {
            "repo": str(repo), "sha": sha, "path": "a.txt", "text": "one\ntwo\n",
            "truncated": False,
        }
        diff = git_file_diff(str(repo), sha, "a.txt", "a.txt")
        assert diff["original"] == "one\n" and diff["modified"] == "one\ntwo\n"
        assert diff["truncated"] is False
        (repo / "a.txt").write_text("one\ntwo\nthree\n", newline="\n")
        worktree = git_file_diff(str(repo), "", "a.txt", "a.txt")
        assert worktree["modified"] == "one\ntwo\nthree\n" and worktree["truncated"] is False


class TestStreamedOutputIsGitsOutput:
    """Streaming keeps git's bytes, its decoding, its byte cap and its failures."""

    def test_non_utf8_file_compares_equal_to_itself(self, repo: Path) -> None:
        (repo / "latin.txt").write_bytes(b"caf\xe9\n")
        sha = _commit(repo, "latin")
        assert git_file_at(str(repo), sha, "latin.txt")["text"] == "caf\udce9\n"
        diff = git_file_diff(str(repo), "", "latin.txt", "latin.txt")
        assert diff["original"] == diff["modified"] == "caf\udce9\n"
        assert "caf\udce9" in git_show(str(repo), sha, "latin.txt")["text"]

    def test_unchanged_huge_file_compares_equal_to_itself(self, repo: Path) -> None:
        content = ("w" * 1023 + "\n") * (5 * 1024)  # 5 MiB
        (repo / "big.txt").write_text(content, newline="\n")
        _commit(repo, "big")
        diff = git_file_diff(str(repo), "", "big.txt", "big.txt")
        assert diff["truncated"] is True
        assert diff["original"] == diff["modified"] == content[:GIT_SHOW_MAX_BYTES]

    def test_cut_text_stays_within_the_cap_whatever_its_bytes(self, repo: Path) -> None:
        # Every byte above 0x7f is one lone surrogate, which decoding
        # with "replace" would have made a three-byte U+FFFD: the cap is
        # on the bytes the text encodes back to.
        (repo / "high.txt").write_bytes(("\xe9" * 1023 + "\n").encode("latin-1") * (5 * 1024))
        sha = _commit(repo, "high")
        at = git_file_at(str(repo), sha, "high.txt")
        assert at["truncated"] is True
        assert _utf8_len(at["text"]) == GIT_SHOW_MAX_BYTES
        assert set(at["text"]) == {"\udce9", "\n"}
        shown = git_show(str(repo), sha, "high.txt")
        assert shown["truncated"] is True
        assert _utf8_len(shown["text"]) == GIT_SHOW_MAX_BYTES

    def test_a_failing_git_is_an_error_not_content(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        (repo / "latin.txt").write_text("plain\n")
        (repo / ".gitattributes").write_text("latin.txt diff=broken\n")
        sha = _commit(repo, "broken textconv")
        _git(repo, "config", "diff.broken.textconv", "missing-test-textconv-command")
        shown = git_show(str(repo), sha, "")
        assert "text" not in shown and "missing-test-textconv-command" in shown["error"]
        compared = git_compare(str(repo), base, sha)
        assert "text" not in compared and "missing-test-textconv-command" in compared["error"]
        # A blob is read without textconv: still served.
        assert git_file_at(str(repo), sha, "latin.txt")["text"] == "plain\n"


class TestExistenceChecks:
    """The checks that ran on the whole-output read now run on their own."""

    def test_missing_and_untracked_paths(self, repo: Path) -> None:
        sha = _git(repo, "rev-parse", "HEAD")
        assert git_file_at(str(repo), sha, "nope.txt")["error"] == (
            f"No nope.txt in commit {sha[:8]}"
        )
        (repo / "new.txt").write_text("fresh\n", newline="\n")
        untracked = git_file_diff(str(repo), "", "new.txt", "new.txt")
        assert untracked["originalMissing"] is True and untracked["modified"] == "fresh\n"
        (repo / "new.txt").unlink()
        gone = git_file_diff(str(repo), "", "new.txt", "new.txt")
        assert gone["originalMissing"] is True and gone["modifiedMissing"] is True

    def test_binary_blob_is_refused(self, repo: Path) -> None:
        (repo / "dot.png").write_bytes(b"\x89PNG\x00\x01\x02")
        sha = _commit(repo, "png")
        assert git_file_at(str(repo), sha, "dot.png")["error"] == (
            "Cannot display binary file: dot.png"
        )
        assert git_file_diff(str(repo), sha, "dot.png", "dot.png")["error"] == (
            "Cannot display binary file: dot.png"
        )

    def test_git_compare_verifies_both_ends(self, repo: Path) -> None:
        sha = _git(repo, "rev-parse", "HEAD")
        unknown = "0123456789abcdef0123456789abcdef01234567"
        assert git_compare(str(repo), sha, unknown) == {"error": f"Unknown commit: {unknown}"}
        assert git_compare(str(repo), "nope", sha) == {"error": "Unknown revision: nope"}
        same = git_compare(str(repo), sha, sha)
        assert same["text"] == "" and same["truncated"] is False

    def test_git_compare_shows_gits_own_diff_despite_an_external_diff_tool(
        self, repo: Path,
    ) -> None:
        """``git diff`` (unlike ``git show``) runs ``diff.external`` unless told not to."""
        base = _git(repo, "rev-parse", "HEAD")
        (repo / "a.txt").write_text("two\n", newline="\n")
        sha = _commit(repo, "second")
        _git(repo, "config", "diff.external", "echo EXTERNAL-TOOL-RAN")
        text = git_compare(str(repo), base, sha)["text"]
        assert "EXTERNAL-TOOL-RAN" not in text
        assert "-one\n+two\n" in text

    def test_a_revision_followed_by_a_newline_is_not_a_revision(self, repo: Path) -> None:
        sha = _git(repo, "rev-parse", "HEAD")
        assert git_show(str(repo), sha + "\n", "") == {"error": f"Not a commit id: {sha}\n"}
        assert git_compare(str(repo), "main\n", sha) == {"error": "Not a revision: main\n"}

    def test_a_path_with_a_nul_byte_is_an_error_reply(self, repo: Path) -> None:
        """``subprocess`` and ``os.open`` raise ``ValueError`` for it, not ``OSError``."""
        sha = _git(repo, "rev-parse", "HEAD")
        assert list(git_show(str(repo), sha, "p\0q")) == ["error"]
        assert list(git_file_at(str(repo), sha, "p\0q")) == ["error"]
        for commit in (sha, ""):
            reply = git_file_diff(str(repo), commit, "p\0q")
            assert reply == {"error": "Not a repository path: p\0q"}

    def test_git_show_of_an_unknown_commit(self, repo: Path) -> None:
        unknown = "0123456789abcdef0123456789abcdef01234567"
        reply = git_show(str(repo), unknown, "")
        assert "error" in reply and unknown in reply["error"]


def test_truncate_still_serves_git_action_output() -> None:
    """``_truncate`` keeps its byte-cap semantics for the hook chatter of ``git_action``."""
    text = explorer._truncate("é" * (GIT_SHOW_MAX_BYTES // 2 + 1))
    assert len(text) == GIT_SHOW_MAX_BYTES // 2 and _utf8_len(text) <= GIT_SHOW_MAX_BYTES

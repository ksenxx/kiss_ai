# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for copy_dirty_state and _do_merge bugs.

Reproduces and guards against three bugs (see tmp/findings-2.md):

* B1 — ``copy_dirty_state`` applied the ``" -> "`` rename split to
  EVERY porcelain status line instead of only rename/copy (R/C)
  entries, so a dirty file literally named ``x -> y`` was never
  copied into the worktree and an unrelated tracked file could be
  unlinked from the worktree.
* B2 — for quoted rename lines the whole tail
  ``"a\\tb.txt" -> "c\\td.txt"`` was unquoted as ONE string before
  splitting on ``" -> "`` and then unquoted again, corrupting both
  paths so renames of files with tabs/quotes were silently not
  mirrored.
* B3 — ``WorktreeSorcarAgent._do_merge`` checked out the original
  branch BEFORE stashing, so a user sitting on a different branch
  with dirty edits got ``CHECKOUT_FAILED`` even though ``merge()``'s
  docstring promises uncommitted changes are stashed first.

All tests use real temporary git repos (subprocess git commands, no
mocks) and are independent of each other.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in *repo* and return the completed process."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _make_repo(path: Path) -> Path:
    """Create a fresh git repo on branch ``main`` with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    _git(path, "config", "user.email", "test@test.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("# Test\n")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    return path


class TestB1ArrowFilenameNotARename:
    """A dirty file literally named ``x -> y`` is NOT a rename entry."""

    def test_untracked_arrow_file_is_copied_and_nothing_unlinked(
        self, tmp_path: Path
    ) -> None:
        """``copy_dirty_state`` mirrors 'x -> y' and keeps tracked x."""
        repo = _make_repo(tmp_path / "repo")
        (repo / "x").write_text("tracked x\n")
        (repo / "y").write_text("tracked y\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add x and y")

        # A perfectly ordinary untracked file whose NAME contains the
        # porcelain rename separator.  Status line: ``?? x -> y``.
        (repo / "x -> y").write_text("dirty arrow file\n")

        wt_dir = tmp_path / "wt"
        assert GitWorktreeOps.create(repo, "kiss/wt-b1", wt_dir)
        try:
            assert (wt_dir / "x").exists()

            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True

            assert (wt_dir / "x -> y").exists(), (
                "dirty file named 'x -> y' must be mirrored into the worktree"
            )
            assert (wt_dir / "x -> y").read_text() == "dirty arrow file\n"
            assert (wt_dir / "x").exists(), (
                "unrelated tracked file 'x' must NOT be unlinked"
            )
            assert (wt_dir / "x").read_text() == "tracked x\n"
            assert (wt_dir / "y").read_text() == "tracked y\n"
        finally:
            GitWorktreeOps.cleanup_partial(repo, "kiss/wt-b1", wt_dir)

    def test_modified_arrow_file_is_copied(self, tmp_path: Path) -> None:
        """A tracked, modified file named 'a -> b' is mirrored verbatim."""
        repo = _make_repo(tmp_path / "repo")
        (repo / "a -> b").write_text("v1\n")
        (repo / "a").write_text("plain a\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add arrow file")
        (repo / "a -> b").write_text("v2 dirty\n")

        wt_dir = tmp_path / "wt"
        assert GitWorktreeOps.create(repo, "kiss/wt-b1b", wt_dir)
        try:
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
            assert (wt_dir / "a -> b").read_text() == "v2 dirty\n"
            assert (wt_dir / "a").exists(), (
                "tracked file 'a' must NOT be unlinked by misparsed rename"
            )
        finally:
            GitWorktreeOps.cleanup_partial(repo, "kiss/wt-b1b", wt_dir)

    def test_real_rename_is_still_mirrored(self, tmp_path: Path) -> None:
        """A genuine staged rename (plain names) is still handled."""
        repo = _make_repo(tmp_path / "repo")
        (repo / "old.txt").write_text("content\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add old.txt")
        _git(repo, "mv", "old.txt", "new.txt")

        wt_dir = tmp_path / "wt"
        assert GitWorktreeOps.create(repo, "kiss/wt-b1c", wt_dir)
        try:
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
            assert (wt_dir / "new.txt").read_text() == "content\n"
            assert not (wt_dir / "old.txt").exists(), (
                "renamed-away file must be removed from the worktree"
            )
        finally:
            GitWorktreeOps.cleanup_partial(repo, "kiss/wt-b1c", wt_dir)


class TestB2QuotedRenameSplit:
    """Quoted rename tails are split first, then unquoted exactly once."""

    def test_rename_with_tabs_both_sides_quoted(self, tmp_path: Path) -> None:
        """git mv 'a\\tb.txt' -> 'c\\td.txt' is mirrored into the worktree."""
        repo = _make_repo(tmp_path / "repo")
        old = "a\tb.txt"
        new = "c\td.txt"
        (repo / old).write_text("tabbed content\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add tabbed file")
        _git(repo, "mv", old, new)

        # Sanity: porcelain emits both sides quoted.
        status = _git(repo, "status", "--porcelain")
        assert '" -> "' in status.stdout

        wt_dir = tmp_path / "wt"
        assert GitWorktreeOps.create(repo, "kiss/wt-b2", wt_dir)
        try:
            assert (wt_dir / old).exists()

            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True, (
                "quoted rename must be detected as dirty state to copy"
            )
            assert (wt_dir / new).exists(), (
                "rename target with tab in name must exist in the worktree"
            )
            assert (wt_dir / new).read_text() == "tabbed content\n"
            assert not (wt_dir / old).exists(), (
                "rename source with tab in name must be removed from worktree"
            )
        finally:
            GitWorktreeOps.cleanup_partial(repo, "kiss/wt-b2", wt_dir)

    def test_rename_only_new_side_quoted(self, tmp_path: Path) -> None:
        """git mv plain.txt -> 'c\\td.txt' (only target quoted) works."""
        repo = _make_repo(tmp_path / "repo")
        new = "c\td2.txt"
        (repo / "plain.txt").write_text("plain content\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add plain file")
        _git(repo, "mv", "plain.txt", new)

        wt_dir = tmp_path / "wt"
        assert GitWorktreeOps.create(repo, "kiss/wt-b2b", wt_dir)
        try:
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
            assert (wt_dir / new).read_text() == "plain content\n"
            assert not (wt_dir / "plain.txt").exists()
        finally:
            GitWorktreeOps.cleanup_partial(repo, "kiss/wt-b2b", wt_dir)

    def test_rename_only_old_side_quoted(self, tmp_path: Path) -> None:
        """git mv 'a\\tb2.txt' -> plain2.txt (only source quoted) works."""
        repo = _make_repo(tmp_path / "repo")
        old = "a\tb2.txt"
        (repo / old).write_text("was tabbed\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add tabbed file")
        _git(repo, "mv", old, "plain2.txt")

        wt_dir = tmp_path / "wt"
        assert GitWorktreeOps.create(repo, "kiss/wt-b2c", wt_dir)
        try:
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir) is True
            assert (wt_dir / "plain2.txt").read_text() == "was tabbed\n"
            assert not (wt_dir / old).exists()
        finally:
            GitWorktreeOps.cleanup_partial(repo, "kiss/wt-b2c", wt_dir)


class TestB3StashBeforeCheckout:
    """_do_merge stashes dirty edits BEFORE checking out the original branch."""

    def test_merge_succeeds_with_dirty_tree_on_other_branch(
        self, tmp_path: Path
    ) -> None:
        """Dirty edits on another branch no longer cause CHECKOUT_FAILED."""
        repo = _make_repo(tmp_path / "repo")
        (repo / "base.txt").write_text("base v1\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add base.txt")

        # Agent worktree branched from main with one agent commit.
        wt_dir = tmp_path / "wt"
        branch = "kiss/wt-b3"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        (wt_dir / "agent_work.txt").write_text("agent output\n")
        _git(wt_dir, "add", ".")
        _git(wt_dir, "commit", "-m", "agent work")

        # User switches to a feature branch, commits a different
        # base.txt, then leaves an UNCOMMITTED edit on top — the exact
        # situation where `git checkout main` refuses unless the dirty
        # state is stashed first.
        _git(repo, "checkout", "-b", "feature")
        (repo / "base.txt").write_text("feature version\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "feature change to base.txt")
        (repo / "base.txt").write_text("dirty uncommitted edit\n")

        # Sanity: plain checkout of main must fail right now.
        checkout = _git(repo, "checkout", "main")
        assert checkout.returncode != 0, (
            "precondition: dirty checkout must fail without a stash"
        )

        agent = WorktreeSorcarAgent("test-b3")
        wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )
        result, stash_warning = agent._do_merge(wt)

        assert result == MergeResult.SUCCESS, (
            f"merge must stash before checkout; got {result} "
            f"(stash_warning={stash_warning!r})"
        )
        assert GitWorktreeOps.current_branch(repo) == "main"
        assert (repo / "agent_work.txt").read_text() == "agent output\n"

        # The user's dirty edit must not be lost: either it was popped
        # back into the working tree, or it is still safe in the stash
        # (with a warning telling the user how to recover it).
        stash_list = _git(repo, "stash", "list").stdout.strip()
        if stash_warning:
            assert stash_list, (
                "when pop fails the dirty edits must remain in git stash"
            )
        else:
            assert (repo / "base.txt").read_text() == "dirty uncommitted edit\n"

    def test_merge_with_dirty_tree_on_original_branch_still_works(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: dirty edits on the original branch itself."""
        repo = _make_repo(tmp_path / "repo")
        (repo / "base.txt").write_text("base v1\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add base.txt")

        wt_dir = tmp_path / "wt"
        branch = "kiss/wt-b3b"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        (wt_dir / "agent_work.txt").write_text("agent output\n")
        _git(wt_dir, "add", ".")
        _git(wt_dir, "commit", "-m", "agent work")

        # Dirty edit while staying on main (no checkout needed).
        (repo / "base.txt").write_text("dirty on main\n")

        agent = WorktreeSorcarAgent("test-b3b")
        wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )
        result, stash_warning = agent._do_merge(wt)

        assert result == MergeResult.SUCCESS
        assert stash_warning == ""
        assert (repo / "agent_work.txt").read_text() == "agent output\n"
        assert (repo / "base.txt").read_text() == "dirty on main\n", (
            "dirty edits on the original branch must be restored after merge"
        )
        assert _git(repo, "stash", "list").stdout.strip() == ""

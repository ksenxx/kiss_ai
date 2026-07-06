# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for git_worktree.py findings B9, B7, B4, B2.

Every test drives a real git repository created under ``tmp_path`` —
no mocks, patches, or fakes.

B9  ``stash_if_dirty`` must return True only when a stash entry was
    actually created.  ``git stash push`` exits 0 with "No local
    changes to save" (creating NO stash) for dirtiness that stash
    cannot capture — e.g. a submodule with untracked content.  The old
    code returned True, so callers later popped an unrelated,
    pre-existing user stash.

B7  ``_append_info_line`` accumulated blank lines (it wrote
    ``"\\n{entry}\\n"`` unconditionally) and its read-check-append
    sequence was unlocked, so concurrent callers appended duplicate
    entries.

B4  ``remove()`` early-returned when the worktree directory was
    already deleted, skipping ``git worktree prune`` — leaving a stale
    ``.git/worktrees`` registration that blocks ``git branch -d/-D``.

B2  ``stash_pop`` fell back to a blind plain ``git stash pop`` even
    when the failed ``--index`` attempt had already modified the
    working tree (partial application with the stash retained),
    risking a double-apply.  The fix only falls back when the failed
    ``--index`` attempt left the tree untouched.  (No observable
    corruption could be reproduced on current git — git's own refusal
    checks stop the second apply — so the B2 tests lock down both the
    fallback-success path and the partial-application path.)
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps

_GIT_ID = (
    "-c",
    "user.email=test@example.com",
    "-c",
    "user.name=Test",
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git in *repo* with a fixed identity, optionally asserting success."""
    result = subprocess.run(
        ["git", *_GIT_ID, "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check:
        assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result


def _init_repo(tmp_path: Path, name: str = "repo") -> Path:
    """Create a git repo with one commit on branch ``main``; return its root."""
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "f.txt").write_text("base\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "c1")
    return repo


def _stash_count(repo: Path) -> int:
    """Return the number of stash entries in *repo*."""
    result = _git(repo, "stash", "list")
    return len([ln for ln in result.stdout.splitlines() if ln.strip()])


def _add_dirty_submodule(tmp_path: Path, repo: Path) -> None:
    """Add a submodule to *repo* and dirty it with untracked content only.

    ``git status --porcelain`` in the superproject then reports
    `` M sub`` while ``git stash push --include-untracked`` is a no-op
    that still exits 0 ("No local changes to save").
    """
    sub = tmp_path / "sub"
    sub.mkdir()
    _git(sub, "init", "-b", "main")
    _git(sub, "commit", "--allow-empty", "-m", "s1")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(sub),
        "sub",
    )
    _git(repo, "commit", "-m", "add submodule")
    (repo / "sub" / "untracked_in_sub.txt").write_text("dirty\n")
    status = _git(repo, "status", "--porcelain").stdout
    assert " M sub" in status, status


class TestB9StashIfDirtyContract:
    """B9: True only iff a stash entry was actually created."""

    def test_noop_stash_push_returns_false(self, tmp_path: Path) -> None:
        """Dirty submodule content: status is dirty but push saves nothing."""
        repo = _init_repo(tmp_path)
        _add_dirty_submodule(tmp_path, repo)
        assert GitWorktreeOps.has_uncommitted_changes(repo)

        assert GitWorktreeOps.stash_if_dirty(repo) is False
        assert _stash_count(repo) == 0

    def test_noop_stash_push_does_not_pop_user_stash(self, tmp_path: Path) -> None:
        """A pre-existing user stash must survive the stash/pop caller flow."""
        repo = _init_repo(tmp_path)
        # The user parked precious work in a stash of their own.
        (repo / "precious.txt").write_text("precious\n")
        _git(repo, "add", "precious.txt")
        _git(repo, "stash", "push", "-m", "user-precious")
        assert _stash_count(repo) == 1

        _add_dirty_submodule(tmp_path, repo)

        # Caller flow used by merge paths: pop only when a stash was made.
        if GitWorktreeOps.stash_if_dirty(repo):
            GitWorktreeOps.stash_pop(repo)

        stash_list = _git(repo, "stash", "list").stdout
        assert "user-precious" in stash_list, (
            "B9: stash_if_dirty claimed a stash was created for a no-op "
            "push, so stash_pop consumed the user's pre-existing stash"
        )

    def test_real_dirty_tree_still_stashes(self, tmp_path: Path) -> None:
        """Regression: genuine dirtiness still creates a stash (True)."""
        repo = _init_repo(tmp_path)
        (repo / "f.txt").write_text("modified\n")
        (repo / "new.txt").write_text("untracked\n")

        assert GitWorktreeOps.stash_if_dirty(repo) is True
        assert _stash_count(repo) == 1
        assert (repo / "f.txt").read_text() == "base\n"
        assert not (repo / "new.txt").exists()

        assert GitWorktreeOps.stash_pop(repo) is True
        assert (repo / "f.txt").read_text() == "modified\n"
        assert (repo / "new.txt").read_text() == "untracked\n"

    def test_clean_tree_returns_false(self, tmp_path: Path) -> None:
        """Regression: a clean tree returns False without pushing."""
        repo = _init_repo(tmp_path)
        assert GitWorktreeOps.stash_if_dirty(repo) is False
        assert _stash_count(repo) == 0


class TestB7AppendInfoLine:
    """B7: no blank-line accumulation; concurrency-safe idempotence."""

    def test_no_blank_lines_accumulate(self, tmp_path: Path) -> None:
        """Appending entries must not insert blank lines."""
        repo = _init_repo(tmp_path)
        GitWorktreeOps._append_info_line(repo, "exclude", ".kiss-worktrees/")
        GitWorktreeOps._append_info_line(repo, "exclude", "other-entry/")

        exclude = repo / ".git" / "info" / "exclude"
        content = exclude.read_text()
        lines = content.splitlines()
        assert "" not in lines, f"blank lines accumulated: {content!r}"
        assert lines.count(".kiss-worktrees/") == 1
        assert lines.count("other-entry/") == 1
        assert content.endswith("\n")

    def test_existing_content_without_trailing_newline(self, tmp_path: Path) -> None:
        """An existing last line without a newline must not be merged into."""
        repo = _init_repo(tmp_path)
        exclude = repo / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text("existing-pattern")  # no trailing newline

        GitWorktreeOps._append_info_line(repo, "exclude", ".kiss-worktrees/")

        lines = exclude.read_text().splitlines()
        assert "existing-pattern" in lines
        assert ".kiss-worktrees/" in lines
        assert "" not in lines

    def test_idempotent_across_calls(self, tmp_path: Path) -> None:
        """Repeated sequential calls keep exactly one copy of the entry."""
        repo = _init_repo(tmp_path)
        for _ in range(3):
            GitWorktreeOps.ensure_excluded(repo)
        exclude = repo / ".git" / "info" / "exclude"
        lines = exclude.read_text().splitlines()
        assert lines.count(".kiss-worktrees/") == 1

    def test_concurrent_appends_write_entry_once(self, tmp_path: Path) -> None:
        """16 threads racing on the same entry must append it exactly once."""
        repo = _init_repo(tmp_path)
        n_threads = 16
        barrier = threading.Barrier(n_threads)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=30)
                GitWorktreeOps._append_info_line(
                    repo, "exclude", ".kiss-worktrees/"
                )
            except BaseException as exc:  # noqa: BLE001 — surface in main thread
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors

        exclude = repo / ".git" / "info" / "exclude"
        lines = exclude.read_text().splitlines()
        assert lines.count(".kiss-worktrees/") == 1, (
            f"B7: racing threads appended duplicates: {lines}"
        )


class TestB4RemoveStaleRegistration:
    """B4: remove() must prune when the worktree dir is already gone."""

    def test_remove_prunes_deleted_worktree_dir(self, tmp_path: Path) -> None:
        """Manually-deleted worktree dir must not block branch deletion."""
        import shutil

        repo = _init_repo(tmp_path)
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-test"
        assert GitWorktreeOps.create(repo, "kiss-task-b4", wt_dir)
        # A crash / manual cleanup deleted the directory behind git's back.
        shutil.rmtree(wt_dir)
        assert not wt_dir.exists()

        GitWorktreeOps.remove(repo, wt_dir)

        assert GitWorktreeOps.delete_branch(repo, "kiss-task-b4") is True, (
            "B4: remove() left a stale .git/worktrees registration, so "
            "git refused to delete the branch"
        )
        assert not GitWorktreeOps.branch_exists(repo, "kiss-task-b4")


class TestB2StashPopFallback:
    """B2: fallback behavior of stash_pop around --index failures."""

    def test_fallback_succeeds_when_index_attempt_left_tree_untouched(
        self, tmp_path: Path
    ) -> None:
        """--index fails pre-merge ("conflicts in index"); plain pop works.

        The stash holds a staged change f=v1; at pop time the index
        already contains f=v1, so ``git apply --cached`` of the stash's
        index diff fails without touching the tree, and the plain pop
        merges the identical content cleanly.
        """
        repo = _init_repo(tmp_path)
        (repo / "f.txt").write_text("v1\n")
        _git(repo, "add", "f.txt")
        _git(repo, "stash", "push", "-m", "kiss-stash")
        (repo / "f.txt").write_text("v1\n")
        _git(repo, "add", "f.txt")

        assert GitWorktreeOps.stash_pop(repo) is True
        assert (repo / "f.txt").read_text() == "v1\n"
        assert _stash_count(repo) == 0

    def test_no_second_apply_after_partial_index_application(
        self, tmp_path: Path
    ) -> None:
        """--index partially applies (untracked file in the way): no re-apply.

        The failed ``--index`` attempt restores no untracked files but
        DOES apply the stash's tracked change to the tree while keeping
        the stash.  ``stash_pop`` must return False and leave the
        tracked change applied exactly once with the stash retained —
        it must not corrupt the tree by re-applying the same stash.
        """
        repo = _init_repo(tmp_path)
        (repo / "u.txt").write_text("stashed-untracked\n")
        (repo / "f.txt").write_text("mod\n")
        _git(repo, "stash", "push", "--include-untracked", "-m", "kiss-stash")
        # An identically-named untracked file now blocks the pop.
        (repo / "u.txt").write_text("in-the-way\n")

        assert GitWorktreeOps.stash_pop(repo) is False

        content = (repo / "f.txt").read_text()
        assert content == "mod\n", (
            f"B2: tracked change applied more than once or conflicted: "
            f"{content!r}"
        )
        assert (repo / "u.txt").read_text() == "in-the-way\n"
        assert _stash_count(repo) == 1, "stash must be retained on failure"

    def test_pop_preserves_index_on_clean_pop(self, tmp_path: Path) -> None:
        """Regression: the happy path still restores staged state."""
        repo = _init_repo(tmp_path)
        (repo / "f.txt").write_text("staged\n")
        _git(repo, "add", "f.txt")
        _git(repo, "stash", "push", "-m", "kiss-stash")

        assert GitWorktreeOps.stash_pop(repo) is True
        staged = _git(repo, "diff", "--cached", "--name-only").stdout
        assert "f.txt" in staged

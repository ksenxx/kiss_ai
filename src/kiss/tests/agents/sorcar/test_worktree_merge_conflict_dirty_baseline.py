# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: merge must not fabricate conflicts from baseline dirty state.

When the user has uncommitted dirty edits at WT-creation time that fall
INSIDE a region the agent later deletes (or modifies) on the worktree
branch, the previous ``cherry-pick --no-commit baseline..branch``
implementation produced a spurious modify/delete conflict on merge.

Mechanism of the spurious conflict:
- ``_do_merge`` stashed the dirty main state before the merge, leaving
  main HEAD == ``baseline^`` (clean).
- ``cherry-pick`` of ``baseline..branch`` performs a 3-way merge per
  commit with **base = each commit's parent**.  For the first commit
  on the worktree branch the parent is ``baseline`` (which captured
  the dirty edits), so cherry-pick saw:

      base   = baseline                  (has dirty edits)
      ours   = main HEAD == baseline^    (no dirty edits)
      theirs = branch tip                (has dirty + agent deletions)

  ``base → ours`` reverts the dirty edits while ``base → theirs``
  deletes the surrounding hunk → modify/delete conflict — even though
  the agent's actual net diff (``baseline..branch``) and main HEAD's
  actual content (``baseline^``) are perfectly compatible.

These tests build the exact scenario with real git in a temp repo and
exercise the real :class:`GitWorktreeOps` API (no mocks, patches, or
test doubles).  Pre-fix they fail with ``MergeResult.CONFLICT``; the
fix in ``squash_merge_from_baseline`` must let them pass.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    _git,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.sorcar.test_git_worktree_locking_and_paths import SaveBeforeGitWrites
from kiss.tests.server.pause import Pause

DIRTY_FILE = "papers/kisssorcar/social/twitter_mimic.md"

ORIGINAL = (
    "# Twitter mimic — drafts\n"
    "\n"
    "## Option A — corporate\n"
    "Sentence A1.\n"
    "Sentence A2 about KISS Sorcar.\n"
    "Sentence A3 wrap-up.\n"
    "\n"
    "## Option B — playful\n"
    "Sentence B1.\n"
    "Sentence B2 with emoji.\n"
    "Sentence B3 close.\n"
    "\n"
    "## Option C — technical\n"
    "Sentence C1 architecture detail.\n"
    "Sentence C2 throughput numbers.\n"
    "Sentence C3 link to repo.\n"
    "\n"
    "## Option D — provocateur\n"
    "Sentence D1 sharp hook.\n"
    "Sentence D2 about prompt runners.\n"
    "Sentence D3 closing punch.\n"
)


def _dirty_edits(text: str) -> str:
    """Apply the user's pre-WT wording tweaks inside Option A AND D."""
    text = text.replace(
        "Sentence A2 about KISS Sorcar.\n",
        "Sentence A2 about KISS Sorcar (user-edited).\n",
    )
    text = text.replace(
        "Sentence D2 about prompt runners.\n",
        "Sentence D2 about prompt runners — user wording.\n",
    )
    return text


def _branch_tip_content() -> str:
    """What the agent leaves after 'keep only Option D' — incl. dirty edit."""
    return (
        "# Twitter mimic — drafts\n"
        "\n"
        "## Option D — provocateur\n"
        "Sentence D1 sharp hook.\n"
        "Sentence D2 about prompt runners — user wording.\n"
        "Sentence D3 closing punch.\n"
    )


def _make_repo(path: Path) -> Path:
    """Create a git repo with the unedited ORIGINAL file committed on main."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True,
        check=True,
    )
    target = path / "papers" / "kisssorcar" / "social"
    target.mkdir(parents=True, exist_ok=True)
    (target / "twitter_mimic.md").write_text(ORIGINAL, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _setup_worktree_with_dirty_baseline(
    repo: Path, branch: str
) -> tuple[Path, str]:
    """Reproduce the production flow: dirty edits captured into baseline."""
    target = repo / "papers" / "kisssorcar" / "social" / "twitter_mimic.md"
    target.write_text(_dirty_edits(ORIGINAL), encoding="utf-8")

    wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")
    assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
    GitWorktreeOps.stage_all(wt_dir)
    GitWorktreeOps.commit_staged(wt_dir, "kiss: baseline from dirty state")
    baseline = GitWorktreeOps.head_sha(wt_dir)
    assert baseline is not None
    GitWorktreeOps.save_baseline_commit(repo, branch, baseline)
    return wt_dir, baseline


def _agent_keeps_only_option_d(wt_dir: Path) -> None:
    """Agent edit: delete Options A, B, C; keep Option D (with dirty edit)."""
    target = wt_dir / "papers" / "kisssorcar" / "social" / "twitter_mimic.md"
    target.write_text(_branch_tip_content(), encoding="utf-8")
    GitWorktreeOps.stage_all(wt_dir)
    GitWorktreeOps.commit_staged(wt_dir, "agent: keep only Option D")


class TestMergeWithDirtyBaselineRegion:
    """Real-git scenario from chat session task 3.

    User had small dirty edits inside Option A AND Option D.  Agent
    deleted Options A, B, C (keeping only Option D, with its dirty
    line preserved).  Pre-fix merge failed with CONFLICT; post-fix
    must SUCCEED with the agent's intent committed and the user's
    surviving dirty edits restored.
    """

    def test_merge_succeeds_without_spurious_conflict(self) -> None:
        """Exact bug repro: cherry-pick must not fabricate a modify/delete
        conflict from baseline-captured dirty edits when the agent
        deletes the surrounding region.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            wt_dir, baseline = _setup_worktree_with_dirty_baseline(
                repo, "kiss/wt-conflict-repro"
            )
            _agent_keeps_only_option_d(wt_dir)

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)

            assert GitWorktreeOps.stash_if_dirty(repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, "kiss/wt-conflict-repro", baseline,
            )
            assert result == MergeResult.SUCCESS, (
                "Cherry-pick fabricated a spurious modify/delete conflict "
                "from the baseline-captured dirty edits.  The agent's net "
                "diff (baseline..branch) and main HEAD content (baseline^) "
                "are semantically compatible."
            )

            committed = _git(
                "show",
                "HEAD:papers/kisssorcar/social/twitter_mimic.md",
                cwd=repo,
            )
            assert committed.returncode == 0
            content = committed.stdout
            assert "Option A" not in content
            assert "Option B" not in content
            assert "Option C" not in content
            assert "Option D" in content

            assert "<<<<<<<" not in content
            assert "=======" not in content
            assert ">>>>>>>" not in content

    def test_dirty_main_with_non_overlapping_edit_preserved(self) -> None:
        """When user dirty edits do NOT overlap any agent-touched hunk,
        the merge succeeds AND the stash pop cleanly restores them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "notes.txt").write_text("user scratch notes\n")

            branch = "kiss/wt-conflict-repro-2"
            wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
            assert GitWorktreeOps.create(repo, branch, wt_dir)
            GitWorktreeOps.save_original_branch(repo, branch, "main")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            GitWorktreeOps.save_baseline_commit(repo, branch, baseline)

            _agent_keeps_only_option_d(wt_dir)

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)

            assert GitWorktreeOps.stash_if_dirty(repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.SUCCESS

            assert GitWorktreeOps.stash_pop(repo), (
                "Dirty edits to a file the agent never touched must be "
                "restored cleanly after merge."
            )

            assert (repo / "notes.txt").read_text() == "user scratch notes\n"
            committed_notes = _git("show", "HEAD:notes.txt", cwd=repo)
            assert committed_notes.returncode != 0
            target = (
                repo / "papers" / "kisssorcar" / "social" / "twitter_mimic.md"
            )
            assert "Option A" not in target.read_text()
            assert "Option D" in target.read_text()


@pytest.fixture
def task() -> Iterator[GitWorktree]:
    """A task worktree off a main tree with a dirty file and two untracked ones."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        (repo / "notes.txt").write_text("user baseline\n")
        (repo / "other.txt").write_text("unrelated\n")
        branch = "kiss/wt-untracked"
        wt_dir, baseline = _setup_worktree_with_dirty_baseline(repo, branch)
        yield GitWorktree(repo, branch, "main", wt_dir, baseline)


def _finish(wt: GitWorktree) -> None:
    """Commit the task's work and drop its worktree, as the end of a task does."""
    assert GitWorktreeOps.commit_all(wt.wt_dir, "agent work")
    GitWorktreeOps.remove(wt.repo_root, wt.wt_dir)


def _install_hook(repo: Path, name: str, script: str) -> None:
    """Make *script* (sh) the *name* hook of *repo*."""
    hook = repo / ".git" / "hooks" / name
    hook.parent.mkdir(exist_ok=True)
    hook.write_text(f"#!/bin/sh\n{script}\n")
    hook.chmod(0o755)


def _merge(wt: GitWorktree) -> tuple[MergeResult, bool, str]:
    """Finish the task and merge it as the agent does: result, resolved by agent, stash warning."""
    _finish(wt)
    result, resolved, stash_warning, _ = SorcarAgent("merge")._do_merge(wt)
    return result, resolved, stash_warning


def _assert_dirty_tree_is_back(repo: Path, also: str = "") -> None:
    """Assert the user's dirty file and unrelated untracked file are back and the stash is empty."""
    assert GitWorktreeOps.status_porcelain(repo) == f"M {DIRTY_FILE}\n{also}?? other.txt"
    assert (repo / DIRTY_FILE).read_text(encoding="utf-8") == _dirty_edits(ORIGINAL)
    assert (repo / "other.txt").read_text() == "unrelated\n"
    assert _git("stash", "list", cwd=repo).stdout == ""


class TestTaskChangesAFileThatWasUntracked:
    """The baseline holds a copy of each file that was untracked when the task began.

    ``HEAD`` lacks it, so the task's change to it is a modify/delete
    conflict no hunk settles, and the stash taken for the merge holds
    the copy the task started from.
    """

    def test_an_edit_merges_and_the_dirty_tree_comes_back(self, task: GitWorktree) -> None:
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")

        assert _merge(task) == (MergeResult.SUCCESS, False, "")

        repo = task.repo_root
        assert _git("show", "HEAD:notes.txt", cwd=repo).stdout == "user baseline\nagent line\n"
        assert (repo / "notes.txt").read_text() == "user baseline\nagent line\n"
        _assert_dirty_tree_is_back(repo)

    def test_a_deletion_is_not_undone_by_the_stash(self, task: GitWorktree) -> None:
        (task.wt_dir / "notes.txt").unlink()

        assert _merge(task) == (MergeResult.SUCCESS, False, "")

        assert not (task.repo_root / "notes.txt").exists()
        _assert_dirty_tree_is_back(task.repo_root)

    def test_a_stash_without_untracked_files_comes_back_whole(self, task: GitWorktree) -> None:
        repo = task.repo_root
        (repo / "notes.txt").unlink()
        (repo / "other.txt").unlink()
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")

        assert _merge(task) == (MergeResult.SUCCESS, False, "")

        assert GitWorktreeOps.status_porcelain(repo) == f"M {DIRTY_FILE}"
        assert (repo / DIRTY_FILE).read_text(encoding="utf-8") == _dirty_edits(ORIGINAL)
        assert _git("stash", "list", cwd=repo).stdout == ""

    def test_a_rename_moves_the_file(self, task: GitWorktree) -> None:
        (task.wt_dir / "notes.txt").rename(task.wt_dir / "renamed.txt")

        assert _merge(task) == (MergeResult.SUCCESS, False, "")

        repo = task.repo_root
        assert not (repo / "notes.txt").exists()
        assert _git("show", "HEAD:renamed.txt", cwd=repo).stdout == "user baseline\n"
        _assert_dirty_tree_is_back(repo)

    def test_a_copy_the_user_changed_since_stays_in_the_stash(self, task: GitWorktree) -> None:
        repo = task.repo_root
        (repo / "notes.txt").write_text("NEW USER WORK\n")
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")

        result, resolved, stash_warning = _merge(task)

        assert (result, resolved) == (MergeResult.SUCCESS, False)
        assert "git stash pop" in stash_warning
        assert _git("show", "stash@{0}^3:notes.txt", cwd=repo).stdout == "NEW USER WORK\n"
        assert (repo / "notes.txt").read_text() == "user baseline\nagent line\n"

    def test_a_copy_the_user_changed_since_outlives_the_deletion(self, task: GitWorktree) -> None:
        repo = task.repo_root
        (repo / "notes.txt").write_text("NEW USER WORK\n")
        (task.wt_dir / "notes.txt").unlink()

        assert _merge(task) == (MergeResult.SUCCESS, False, "")

        assert (repo / "notes.txt").read_text() == "NEW USER WORK\n"
        _assert_dirty_tree_is_back(repo, also="?? notes.txt\n")

    def test_a_dirty_file_the_task_deleted_is_deleted(self, task: GitWorktree) -> None:
        (task.wt_dir / DIRTY_FILE).unlink()

        result, resolved, stash_warning = _merge(task)

        repo = task.repo_root
        assert (result, resolved) == (MergeResult.SUCCESS, False)
        assert _git("show", f"HEAD:{DIRTY_FILE}", cwd=repo).returncode != 0
        # The user's edits to it have no file to go back to: they stay stashed.
        assert "git stash pop" in stash_warning
        assert _git("show", f"stash@{{0}}:{DIRTY_FILE}", cwd=repo).stdout == _dirty_edits(ORIGINAL)

    def test_a_file_directory_clash_stays_a_conflict(self, task: GitWorktree) -> None:
        """Git parks the task's file as ``social~<side>``: no path of the task's to settle."""
        social = (task.wt_dir / DIRTY_FILE).parent
        shutil.rmtree(social)
        social.write_text("a file now\n")

        result, resolved, stash_warning = _merge(task)

        repo = task.repo_root
        assert (result, resolved) == (MergeResult.CONFLICT, False)
        assert "safe in 'git stash'" in stash_warning
        assert GitWorktreeOps.status_porcelain(repo) == ""
        assert GitWorktreeOps.merge_has_content_conflict(repo, task.branch, task.baseline_commit)

    def test_a_commit_on_main_keeps_the_conflict(self, task: GitWorktree) -> None:
        """``HEAD != baseline^``: what main lacks may be a commit of the user's, not dirty state."""
        repo = task.repo_root
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")
        (repo / "committed.txt").write_text("by the user, meanwhile\n")
        _git("add", "committed.txt", cwd=repo)
        _git("commit", "-m", "user work on main", cwd=repo)

        result, resolved, stash_warning = _merge(task)

        assert (result, resolved) == (MergeResult.CONFLICT, False)
        assert "safe in 'git stash'" in stash_warning
        assert GitWorktreeOps.status_porcelain(repo) == ""
        assert GitWorktreeOps.merge_has_content_conflict(repo, task.branch, task.baseline_commit)
        # Taken from its fork point instead (no baseline), the same branch merges cleanly.
        assert not GitWorktreeOps.merge_has_content_conflict(repo, task.branch, None)

    def test_a_path_git_cannot_settle_stays_a_conflict(self, task: GitWorktree) -> None:
        """A git command of the user's holds the index the pick wrote: the merge stays, whole."""
        repo = task.repo_root
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")
        _finish(task)
        _install_hook(repo, "post-index-change", "test -e .git/AUTO_MERGE && : > .git/index.lock")

        result, resolved, stash_warning, _ = SorcarAgent("merge")._do_merge(task)

        assert (result, resolved) == (MergeResult.CONFLICT, False)
        assert "safe in 'git stash'" in stash_warning
        (repo / ".git" / "index.lock").unlink()
        assert GitWorktreeOps.status_porcelain(repo) == "DU notes.txt"
        assert (repo / "notes.txt").read_text() == "user baseline\nagent line\n"

    def test_a_copy_saved_after_the_pick_survives_a_rejected_commit(
        self, task: GitWorktree,
    ) -> None:
        """Taking the merge back resets no file saved since git wrote it: the merge stays."""
        repo = task.repo_root
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")
        _finish(task)
        _install_hook(
            repo, "post-index-change", "test -e .git/AUTO_MERGE && echo saved since > notes.txt",
        )
        _install_hook(repo, "pre-commit", "exit 1")

        result, resolved, stash_warning, _ = SorcarAgent("merge")._do_merge(task)

        assert (result, resolved) == (MergeResult.CONFLICT, False)
        assert (repo / "notes.txt").read_text() == "saved since\n"
        assert "safe in 'git stash'" in stash_warning

    def test_a_rejected_commit_takes_a_settled_merge_back_whole(self, task: GitWorktree) -> None:
        """The file the settling deleted is back as well: the tree is the one the merge found."""
        repo = task.repo_root
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")
        (task.wt_dir / DIRTY_FILE).unlink()
        _finish(task)
        _install_hook(repo, "pre-commit", "exit 1")

        result, resolved, stash_warning, _ = SorcarAgent("merge")._do_merge(task)

        assert (result, resolved) == (MergeResult.MERGE_FAILED, False)
        assert GitWorktreeOps.status_porcelain(repo) == ""
        assert "safe in 'git stash'" in stash_warning

    def test_the_probe_and_the_reapplication_settle_it_too(self, task: GitWorktree) -> None:
        repo = task.repo_root
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")
        _finish(task)
        assert GitWorktreeOps.stash_if_dirty(repo)

        assert not GitWorktreeOps.merge_has_content_conflict(
            repo, task.branch, task.baseline_commit,
        )
        assert GitWorktreeOps.begin_conflicted_merge(repo, task.branch, task.baseline_commit) == []
        assert _git("diff", "--cached", "--name-only", cwd=repo).stdout == "notes.txt\n"

    def test_a_copy_saved_after_the_pick_wrote_it_is_not_overwritten(
        self, task: GitWorktree,
    ) -> None:
        """A smudge filter saves ``notes.txt`` as git writes ``z.txt``, after its own copy."""
        repo = task.repo_root
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")
        (task.wt_dir / "z.txt").write_text("the task adds z\n")
        _finish(task)
        _git("config", "filter.late.smudge", "sh -c 'cat; echo saved since > notes.txt'", cwd=repo)
        (repo / ".git" / "info" / "attributes").write_text("z.txt filter=late\n")

        result, resolved, stash_warning, _ = SorcarAgent("merge")._do_merge(task)

        assert (result, resolved, stash_warning) == (MergeResult.SUCCESS, False, "")
        assert _git("show", "HEAD:notes.txt", cwd=repo).stdout == "user baseline\nagent line\n"
        assert (repo / "notes.txt").read_text() == "saved since\n"
        # The rest of the dirty tree is back around it, and the stash is empty.
        status = GitWorktreeOps.status_porcelain(repo)
        assert status == f"M notes.txt\n M {DIRTY_FILE}\n?? other.txt"
        assert _git("stash", "list", cwd=repo).stdout == ""

    @pytest.mark.parametrize("saved", ["notes.txt", DIRTY_FILE], ids=["edited", "deleted"])
    def test_a_copy_saved_right_before_the_settling_is_not_overwritten(
        self, task: GitWorktree, saved: str,
    ) -> None:
        """After every look the merge takes: the file the task edited or the one it deleted."""
        repo = task.repo_root
        (task.wt_dir / "notes.txt").write_text("user baseline\nagent line\n")
        (task.wt_dir / DIRTY_FILE).unlink()
        _finish(task)

        with Pause(_git.__code__, SaveBeforeGitWrites(repo / saved)):
            result, resolved, _, _ = SorcarAgent("merge")._do_merge(task)

        assert (result, resolved) == (MergeResult.SUCCESS, False)
        assert (repo / saved).read_text().endswith("saved since\n")
        assert _git("show", "HEAD:notes.txt", cwd=repo).stdout == "user baseline\nagent line\n"
        assert _git("show", f"HEAD:{DIRTY_FILE}", cwd=repo).returncode != 0

    @pytest.mark.parametrize(
        "breaks", ['git config user.name ""', "git branch -D kiss/wt-untracked"],
        ids=["commit-tree", "diff"],
    )
    def test_a_copy_that_cannot_be_made_leaves_the_stash_as_it_is(
        self, task: GitWorktree, monkeypatch: pytest.MonkeyPatch, breaks: str,
    ) -> None:
        """A post-commit hook fails git on the way to the copy: nothing of the stash is applied.

        Least of all the file the task deleted, which the whole entry holds.
        """
        repo = task.repo_root
        (task.wt_dir / "notes.txt").unlink()
        (task.wt_dir / "z.txt").write_text("the task adds z\n")  # something to commit
        _finish(task)
        for name in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
            monkeypatch.delenv(name, raising=False)
        hook = repo / ".git" / "hooks" / "post-commit"
        hook.parent.mkdir(exist_ok=True)
        hook.write_text(f"#!/bin/sh\n{breaks}\n")
        hook.chmod(0o755)

        result, resolved, stash_warning, _ = SorcarAgent("merge")._do_merge(task)

        assert (result, resolved) == (MergeResult.SUCCESS, False)
        assert "git stash pop" in stash_warning
        assert not (repo / "notes.txt").exists()
        assert GitWorktreeOps.status_porcelain(repo) == ""
        assert _git("show", "stash@{0}^3:notes.txt", cwd=repo).stdout == "user baseline\n"

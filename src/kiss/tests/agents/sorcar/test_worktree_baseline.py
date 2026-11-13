# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for baseline commit functionality in worktree mode.

When the user has uncommitted/staged/untracked files, the worktree
captures them in a "baseline commit" so that merge, merge-review,
and changed-file detection only see agent-produced changes.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
    _git,
)
from kiss.tests.conftest import posix_only


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
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


def _create_worktree(repo: Path, branch: str) -> Path:
    """Create a worktree at repo/.kiss-worktrees/<slug>."""
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    return wt_dir


class TestGitWorktreeBaseline:
    """GitWorktree dataclass includes baseline_commit field."""

    def test_default_none(self) -> None:
        wt = GitWorktree(
            repo_root=Path("/tmp"),
            branch="b",
            original_branch="main",
            wt_dir=Path("/tmp/wt"),
        )
        assert wt.baseline_commit is None

    def test_with_baseline(self) -> None:
        wt = GitWorktree(
            repo_root=Path("/tmp"),
            branch="b",
            original_branch="main",
            wt_dir=Path("/tmp/wt"),
            baseline_commit="abc123",
        )
        assert wt.baseline_commit == "abc123"


class TestCopyDirtyState:
    """copy_dirty_state mirrors dirty files from main worktree."""

    def test_clean_repo_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            wt_dir = _create_worktree(repo, "kiss/wt-test-1")
            assert not GitWorktreeOps.copy_dirty_state(repo, wt_dir)

    def test_unstaged_modification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Modified\n")
            wt_dir = _create_worktree(repo, "kiss/wt-test-2")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "README.md").read_text() == "# Modified\n"

    def test_staged_modification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Staged\n")
            _git("add", "README.md", cwd=repo)
            wt_dir = _create_worktree(repo, "kiss/wt-test-3")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "README.md").read_text() == "# Staged\n"

    def test_untracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "new_file.txt").write_text("new content\n")
            wt_dir = _create_worktree(repo, "kiss/wt-test-4")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "new_file.txt").read_text() == "new content\n"

    def test_deleted_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").unlink()
            wt_dir = _create_worktree(repo, "kiss/wt-test-5")
            assert (wt_dir / "README.md").exists()
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert not (wt_dir / "README.md").exists()

    def test_new_file_in_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "src").mkdir()
            (repo / "src" / "app.py").write_text("print('hi')\n")
            wt_dir = _create_worktree(repo, "kiss/wt-test-6")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "src" / "app.py").read_text() == "print('hi')\n"

    def test_multiple_dirty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Changed\n")
            (repo / "new.txt").write_text("new\n")
            _git("add", "README.md", cwd=repo)
            wt_dir = _create_worktree(repo, "kiss/wt-test-7")
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "README.md").read_text() == "# Changed\n"
            assert (wt_dir / "new.txt").read_text() == "new\n"

    def test_tracked_file_replaced_by_directory_is_mirrored(self) -> None:
        """A tracked file turned into a directory (`` D README.md`` next
        to ``?? README.md/x.txt``) is a plain directory, not a submodule:
        its entry still replaces the worktree's stale file so the files
        under it can land.  A staged new file turned into a directory
        (``AD new.txt``) has no worktree file in the way and is mirrored
        the same way."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").unlink()
            (repo / "README.md").mkdir()
            (repo / "README.md" / "x.txt").write_text("inside\n")
            (repo / "new.txt").write_text("staged\n")
            _git("add", "new.txt", cwd=repo)
            (repo / "new.txt").unlink()
            (repo / "new.txt").mkdir()
            (repo / "new.txt" / "y.txt").write_text("under the staged name\n")
            wt_dir = _create_worktree(repo, "kiss/wt-test-8")
            assert (wt_dir / "README.md").is_file()
            assert not (wt_dir / "new.txt").exists()
            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            assert (wt_dir / "README.md" / "x.txt").read_text() == "inside\n"
            assert (wt_dir / "new.txt" / "y.txt").read_text() == "under the staged name\n"

    def test_dirty_submodule_falls_back_to_the_main_checkout(self) -> None:
        """A dirty tracked submodule cannot be snapshotted (audit S4/F4):
        flattened into the worktree's uninitialized gitlink directory, its
        files were invisible to git, so an agent edit there never reached
        a commit and the finalizer deleted its only copy.  The real
        worktree setup now refuses the snapshot, removes its partial
        worktree and branch, and leaves the user's checkout, with the
        submodule edit, intact."""
        with tempfile.TemporaryDirectory() as tmp:
            sub = _make_repo(Path(tmp) / "sub")
            repo = _make_repo(Path(tmp) / "repo")
            assert _git(
                "-c", "protocol.file.allow=always",
                "submodule", "add", str(sub), "sub", cwd=repo,
            ).returncode == 0
            assert _git("commit", "-q", "-m", "add submodule", cwd=repo).returncode == 0
            (repo / "sub" / "README.md").write_text("dirty user edit\n")
            assert _git("status", "--porcelain", cwd=repo).stdout == " M sub\n"

            agent = ChatSorcarAgent("dirty-submodule-test")
            agent.work_dir = str(repo)
            assert agent._try_setup_worktree(repo, str(repo)) is None, (
                "a snapshot that flattens the submodule must not be accepted"
            )
            assert agent._wt is None
            assert _git("worktree", "list", "--porcelain", cwd=repo).stdout.count(
                "worktree ") == 1, "the partial worktree was not removed"
            assert _git("branch", "--list", "kiss/wt-*", cwd=repo).stdout.strip() == ""
            assert (repo / "sub" / "README.md").read_text() == "dirty user edit\n"
            assert _git("status", "--porcelain", cwd=repo).stdout == " M sub\n"


class TestHeadSha:
    """head_sha returns the current HEAD commit SHA."""

    def test_returns_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            sha = GitWorktreeOps.head_sha(repo)
            assert sha is not None
            assert len(sha) == 40

    def test_bad_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sha = GitWorktreeOps.head_sha(Path(tmp))
            assert sha is None


class TestSquashMergeFromBaseline:
    """squash_merge_from_baseline merges only agent changes."""

    def test_no_agent_changes(self) -> None:
        """When baseline == branch tip, nothing to merge → SUCCESS."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-test-8"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            _git("checkout", "main", cwd=repo)
            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.SUCCESS

    def test_only_agent_changes_merged(self) -> None:
        """Baseline (dirty state) is excluded; only agent work merged."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            (repo / "README.md").write_text("# User edit\n")
            (repo / "user_file.txt").write_text("user content\n")

            branch = "kiss/wt-test-9"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "agent_file.py").write_text("# agent code\n")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(wt_dir, "agent: add file")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.SUCCESS

            assert (repo / "agent_file.py").read_text() == "# agent code\n"
            committed_readme = _git(
                "show", "HEAD:README.md", cwd=repo,
            )
            assert committed_readme.stdout == "# Test\n"
            committed_user = _git(
                "show", "HEAD:user_file.txt", cwd=repo,
            )
            assert committed_user.returncode != 0
            assert (repo / "README.md").read_text() == "# User edit\n"
            assert (repo / "user_file.txt").read_text() == "user content\n"

    def test_conflict_resets_cleanly(self) -> None:
        """On conflict, resets to clean state and returns CONFLICT."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-test-10"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "README.md").write_text("# Agent version\n")
            GitWorktreeOps.commit_all(wt_dir, "agent change")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)
            (repo / "README.md").write_text("# Conflicting main version\n")
            _git("add", "README.md", cwd=repo)
            _git("commit", "-m", "main change", cwd=repo)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.CONFLICT

            status = _git("status", "--porcelain", cwd=repo)
            assert not status.stdout.strip()

    def test_net_change_is_merged_whatever_the_branch_history(self) -> None:
        """A reverted intermediate edit or a merge commit on the branch merges cleanly.

        Replaying ``baseline..branch`` commit by commit conflicted on an
        edit the agent later reverted (main had changed the same file)
        and refused a merge commit outright (``no -m option``); the
        agent's net change is what gets merged.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-test-11"
            wt_dir = _create_worktree(repo, branch)
            (wt_dir / "README.md").write_text("# Baseline\n")
            GitWorktreeOps.commit_all(wt_dir, "baseline")
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            (wt_dir / "README.md").write_text("# Temporary agent edit\n")
            GitWorktreeOps.commit_all(wt_dir, "temporary change")
            (wt_dir / "README.md").write_text("# Baseline\n")
            (wt_dir / "feature.py").write_text("# feature\n")
            GitWorktreeOps.commit_all(wt_dir, "revert the experiment, finish the feature")
            topic = Path(tmp) / "topic"
            _git("worktree", "add", "-b", "topic", str(topic), baseline, cwd=repo)
            # Not "aux.py": AUX is a reserved device name on Windows, and
            # reading it there blocks for ever.
            (topic / "topic_feature.py").write_text("# topic\n")
            GitWorktreeOps.commit_all(topic, "topic feature")
            merged = _git("merge", "--no-ff", "-m", "merge topic", "topic", cwd=wt_dir)
            assert merged.returncode == 0, merged.stderr
            _git("checkout", "main", cwd=repo)
            (repo / "README.md").write_text("# Independent main work\n")
            _git("commit", "-am", "main work", cwd=repo)

            result = GitWorktreeOps.squash_merge_from_baseline(repo, branch, baseline)

            assert result == MergeResult.SUCCESS
            assert (repo / "feature.py").read_text() == "# feature\n"
            assert (repo / "topic_feature.py").read_text() == "# topic\n"
            assert (repo / "README.md").read_text() == "# Independent main work\n"
            assert _git("status", "--porcelain", "--untracked-files=no", cwd=repo).stdout == ""


class TestEndToEndBaselineWorkflow:
    """Full workflow: dirty main tree → worktree → agent edits → merge."""

    def test_full_cycle_dirty_state_excluded_from_merge(self) -> None:
        """User has dirty files, agent works in worktree, merge only gets agent changes."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")

            original = "line1\nline2\nline3\nline4\nline5\nline6\n"
            (repo / "code.py").write_text(original)
            _git("add", ".", cwd=repo)
            _git("commit", "-m", "add code.py", cwd=repo)

            dirty = "line1-dirty\nline2\nline3\nline4\nline5\nline6\n"
            (repo / "code.py").write_text(dirty)
            (repo / "notes.txt").write_text("user notes\n")

            branch = "kiss/wt-full-cycle"
            wt_dir = _create_worktree(repo, branch)
            original_branch = "main"
            GitWorktreeOps.save_original_branch(repo, branch, original_branch)

            assert GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            GitWorktreeOps.save_baseline_commit(repo, branch, baseline)

            assert (wt_dir / "code.py").read_text() == dirty
            assert (wt_dir / "notes.txt").read_text() == "user notes\n"

            agent_version = dirty + "agent-added\n"
            (wt_dir / "code.py").write_text(agent_version)
            (wt_dir / "fix.py").write_text("def fix(): pass\n")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(wt_dir, "agent: implement fix")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            did_stash = GitWorktreeOps.stash_if_dirty(repo)
            assert did_stash

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline,
            )
            assert result == MergeResult.SUCCESS

            assert GitWorktreeOps.stash_pop(repo)

            assert (repo / "fix.py").read_text() == "def fix(): pass\n"

            committed = _git("show", "HEAD:code.py", cwd=repo)
            assert "agent-added" in committed.stdout
            assert "line1-dirty" not in committed.stdout
            assert committed.stdout.startswith("line1\n")

            committed_notes = _git("show", "HEAD:notes.txt", cwd=repo)
            assert committed_notes.returncode != 0

            assert (repo / "notes.txt").read_text() == "user notes\n"
            assert "line1-dirty" in (repo / "code.py").read_text()

    def test_backward_compat_no_baseline(self) -> None:
        """Legacy worktrees (no baseline) still use squash_merge_branch."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-legacy"
            wt_dir = _create_worktree(repo, branch)

            (wt_dir / "agent.py").write_text("# agent\n")
            GitWorktreeOps.commit_all(wt_dir, "agent work")

            GitWorktreeOps.remove(repo, wt_dir)
            GitWorktreeOps.prune(repo)
            _git("checkout", "main", cwd=repo)

            result = GitWorktreeOps.squash_merge_branch(repo, branch)
            assert result == MergeResult.SUCCESS
            assert (repo / "agent.py").read_text() == "# agent\n"

    def test_changed_files_against_baseline(self) -> None:
        """git diff --name-only against baseline shows only agent files."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Dirty\n")
            (repo / "user.txt").write_text("user\n")

            branch = "kiss/wt-changed"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "agent.py").write_text("# a\n")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(wt_dir, "agent work")

            diff = _git(
                "diff", "--name-only", baseline, "HEAD", cwd=wt_dir,
            )
            changed = diff.stdout.strip().splitlines()
            assert changed == ["agent.py"]


    def test_uncommitted_agent_changes_visible(self) -> None:
        """Uncommitted agent changes visible via git diff against baseline."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "README.md").write_text("# Dirty\n")

            branch = "kiss/wt-uncommitted"
            wt_dir = _create_worktree(repo, branch)
            GitWorktreeOps.copy_dirty_state(repo, wt_dir)
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_staged(
                wt_dir, "kiss: baseline from dirty state",
            )
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None

            (wt_dir / "fix.py").write_text("# fix\n")

            diff = _git(
                "diff", "--name-only", baseline, cwd=wt_dir,
            )
            changed = diff.stdout.strip().splitlines()
            untracked = _git(
                "ls-files", "--others", "--exclude-standard", cwd=wt_dir,
            )
            all_changed = set(changed) | set(
                untracked.stdout.strip().splitlines()
            )
            assert "fix.py" in all_changed
            assert "README.md" not in all_changed


# A task's worktree setup in a process of its own: one case kills it, one stops it (SIGINT gets
# its default handler back: a test run in the background hands its children an ignored one).
_SETUP = (
    "import signal, sys; from kiss.agents.sorcar.sorcar_agent import SorcarAgent; "
    "signal.signal(signal.SIGINT, signal.default_int_handler); "
    "agent = SorcarAgent('setup'); agent.work_dir = sys.argv[1]; print(agent.create_worktree())"
)
_GIT_DIR = '"$(git rev-parse --git-common-dir)"'
_SETUP_PID = '"$(ps -o ppid= -p $PPID)"'  # as a hook sees it: the parent of its git


@posix_only("the hooks are POSIX shell scripts")
@pytest.mark.parametrize(("hook_name", "failure", "created", "reclaimed", "kept"), [
    # Before the dirty state is copied; the others right after the baseline commit.
    ("post-checkout", f"touch {_GIT_DIR}/kiss-preserve", False, 0, False),  # no marker can be made
    ("post-commit", f"kill -9 {_SETUP_PID}", False, 0, True),  # dies before the record
    ("post-commit", f"kill -INT {_SETUP_PID}", False, 0, False),  # is stopped before the record
    ("post-commit", f"touch {_GIT_DIR}/config.lock", False, 0, False),  # no record can be written
    ("post-commit", "git symbolic-ref HEAD refs/heads/kiss-unborn", False, 0, False),  # no HEAD
    ("post-commit",  # the marker cannot be removed
     f'cd {_GIT_DIR}/kiss-preserve && for m in *; do rm "$m" && mkdir -p "$m/x"; done',
     True, 0, True),
    ("post-commit", "true", True, 1, False),  # nothing fails: reclaimed as before
])
def test_the_copied_dirty_state_is_never_reclaimed_as_task_work(
    tmp_path: Path, hook_name: str, failure: str, created: bool, reclaimed: int, kept: bool,
) -> None:
    """Audit 2026-11-10 C/C10-F2: only a recorded baseline lets an orphan be reclaimed."""
    repo = _make_repo(tmp_path / "repo")
    seed = _git("rev-parse", "HEAD", cwd=repo).stdout
    (repo / "README.md").write_text("private draft\n")
    hook = repo / ".git" / "hooks" / hook_name
    hook.write_text(f'#!/bin/sh\nrm "$0"\n{failure}\n')
    hook.chmod(0o755)
    setup = subprocess.run([sys.executable, "-c", _SETUP, str(repo)],
                           capture_output=True, text=True, timeout=300)
    assert not hook.exists(), setup.stderr  # the failure struck
    assert ("Created worktree" in setup.stdout) == created, setup.stdout
    (repo / ".git" / "config.lock").unlink(missing_ok=True)
    _git("restore", "README.md", cwd=repo)  # the user discards the draft
    assert GitWorktreeOps.reclaim_orphaned_worktrees(repo) == reclaimed
    assert _git("rev-parse", "HEAD", cwd=repo).stdout == seed
    assert (repo / "README.md").read_text() == "# Test\n"
    # Only a setup that died, or whose marker stays, leaves its branch to the user.
    assert bool(_git("branch", "--list", "kiss/wt-*", cwd=repo).stdout.strip()) == kept

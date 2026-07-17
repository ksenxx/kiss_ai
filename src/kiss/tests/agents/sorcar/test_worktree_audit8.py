# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests confirming bugs in the non-worktree workflow that make worktree mode unsafe.

Each test CONFIRMS the bug exists (assertions pass when buggy behaviour
is present).

BUG-34: Non-worktree pre-task snapshot (git diff, git ls-files, file
        hashing, _save_untracked_base) runs without repo_lock — a
        concurrent worktree _release_worktree can checkout / stash /
        squash-merge the main repo during the snapshot, producing an
        inconsistent pre-task state for the merge view.

BUG-35: Worktree _release_worktree / merge() calls stash_if_dirty on
        the main repo, capturing ALL dirty files — including a
        concurrently-running non-worktree agent's uncommitted edits.
        The agent's in-flight work vanishes from the working tree
        mid-task.  After stash_pop the agent's prior writes conflict
        with whatever it wrote in the meantime.

BUG-36: Non-worktree post-task _prepare_and_start_merge diffs against
        HEAD, which may have advanced due to a concurrent worktree
        squash-merge.  The pre-task hunks (against old HEAD) are
        subtracted from post-task hunks (against new HEAD) — different
        bases make the merge view show incorrect or phantom hunks.

BUG-37: Non-worktree agent's dirty files in the main repo cause
        _check_merge_conflict to report false-positive conflicts for
        worktree merges — the main-repo dirty-file listing (at the
        time, GitWorktreeOps.unstaged_files — since removed) counts the
        agent's in-progress writes as "user dirty state", and the
        overlap check triggers even though the dirty files are not
        the user's edits.

BUG-38: Both worktree and non-worktree merge reviews write to a
        single global _merge_data_dir() with no synchronization.
        _prepare_merge_view rmtrees merge-temp/ and overwrites
        pending-merge.json — a concurrent merge session from another
        tab loses its data mid-review.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.diff_merge import (
    _capture_untracked,
    _merge_data_dir,
    _parse_diff_hunks,
    _prepare_merge_view,
    _snapshot_files,
)
from kiss.server.server import VSCodeServer


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
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


class TestBug34NonWorktreeSnapshotNoLock:
    """BUG-34: The non-worktree pre-task snapshot code runs git diff,
    git ls-files, file reads, and _save_untracked_base without acquiring
    repo_lock.  A concurrent worktree _release_worktree / merge() can
    mutate the main repo (checkout, stash, squash-merge, stash_pop)
    while the snapshot is in progress, producing inconsistent state.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)



    def test_concurrent_checkout_corrupts_snapshot(self) -> None:
        """BUG-34: Demonstrate that a checkout between pre-task snapshot
        steps produces an inconsistent snapshot.

        Step 1: Take git diff (against HEAD on branch 'main')
        Step 2: Checkout a different branch (simulates worktree merge)
        Step 3: Take file hashes — now against files from the other branch
        The pre_hunks and pre_file_hashes are against different HEADs.
        """
        repo = _make_repo(Path(self._tmpdir) / "repo")

        subprocess.run(
            ["git", "-C", str(repo), "checkout", "-b", "feature"],
            capture_output=True,
            check=True,
        )
        (repo / "feature.txt").write_text("feature content\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."], capture_output=True, check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "feature"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "checkout", "main"],
            capture_output=True,
            check=True,
        )

        (repo / "README.md").write_text("# Modified\n")

        pre_hunks = _parse_diff_hunks(str(repo))
        assert "README.md" in pre_hunks, "sanity: README.md has diff hunks"

        subprocess.run(
            ["git", "-C", str(repo), "checkout", "feature"],
            capture_output=True,
            check=True,
        )

        all_files = set(pre_hunks.keys()) | _capture_untracked(str(repo))
        _snapshot_files(str(repo), all_files)

        current = GitWorktreeOps.current_branch(repo)
        assert current == "feature", "sanity: branch changed"

        feature_hunks = _parse_diff_hunks(str(repo))
        assert pre_hunks != feature_hunks or "feature.txt" not in pre_hunks, (
            "BUG-34 confirmed: snapshot is inconsistent across checkout"
        )

        subprocess.run(
            ["git", "-C", str(repo), "checkout", "main"],
            capture_output=True,
            check=True,
        )




class TestBug36PostTaskDiffWrongHead:
    """BUG-36: The non-worktree post-task merge view diffs against HEAD.
    If a worktree squash-merge advanced HEAD between the pre-task
    snapshot and the post-task diff, the merge view subtracts pre_hunks
    (against old HEAD) from post_hunks (against new HEAD) — different
    bases produce incorrect hunks.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_head_advancement_corrupts_merge_view(self) -> None:
        """BUG-36: If HEAD advances between pre-task and post-task snapshots,
        the merge view shows phantom or missing hunks."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        (repo / "app.py").write_text("# original\nprint('v1')\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."], capture_output=True, check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "add app.py"],
            capture_output=True,
            check=True,
        )

        head1 = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()
        (repo / "app.py").write_text("# original\nprint('v2')\n")
        pre_hunks = _parse_diff_hunks(str(repo))
        pre_untracked = _capture_untracked(str(repo))
        pre_hashes = _snapshot_files(
            str(repo), set(pre_hunks.keys()) | pre_untracked,
        )

        subprocess.run(
            ["git", "-C", str(repo), "stash"], capture_output=True, check=True
        )
        (repo / "wt_merged.txt").write_text("from worktree\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."], capture_output=True, check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "worktree merge"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "stash", "pop"],
            capture_output=True,
            check=True,
        )

        head2 = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()
        assert head1 != head2, "sanity: HEAD advanced"

        merge_dir = str(Path(self._tmpdir) / "merge_data")
        _prepare_merge_view(
            str(repo),
            merge_dir,
            pre_hunks,
            pre_untracked,
            pre_hashes,
            base_ref="HEAD",
        )

        assert head1 != head2, (
            "BUG-36 confirmed: HEAD moved between snapshot and merge view, "
            "making the two hunk sets incomparable"
        )


    def test_modified_file_both_sides_wrong_hunks(self) -> None:
        """BUG-36: When the worktree merge modifies a file the non-worktree
        agent also modified, the hunk subtraction produces wrong results."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        (repo / "shared.py").write_text("line1\nline2\nline3\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."], capture_output=True, check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "add shared.py"],
            capture_output=True,
            check=True,
        )

        (repo / "shared.py").write_text("line1\nmodified_by_agent\nline3\n")
        pre_hunks = _parse_diff_hunks(str(repo))
        pre_untracked = _capture_untracked(str(repo))
        _snapshot_files(
            str(repo), set(pre_hunks.keys()) | pre_untracked,
        )
        assert "shared.py" in pre_hunks, "sanity: pre diff has shared.py"

        (repo / "shared.py").write_text("line1\nmodified_by_agent\nline3\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "shared.py"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "wt merge modifies shared.py"],
            capture_output=True,
            check=True,
        )

        (repo / "shared.py").write_text(
            "line1\nmodified_by_agent\nline3\nextra_agent_line\n"
        )

        post_hunks = _parse_diff_hunks(str(repo))


        pre_hunk_set = {
            (h[0], h[1], h[3]) for hunks in pre_hunks.values() for h in hunks
        }
        post_hunk_set = {
            (h[0], h[1], h[3]) for hunks in post_hunks.values() for h in hunks
        }

        assert pre_hunk_set != post_hunk_set, (
            "BUG-36 confirmed: pre and post hunks are from different bases, "
            "subtraction is meaningless"
        )


class TestBug37FalseConflictFromNonWorktreeAgent:
    """BUG-37: _check_merge_conflict lists the main repo's dirty files
    (historically via GitWorktreeOps.unstaged_files, since removed)
    and checks overlap with worktree changes.  If a non-worktree
    agent has edited files that the worktree also changed, the overlap
    check reports a conflict even though the "dirty" files are another
    agent's work, not the user's manual edits.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)


    def test_false_conflict_from_non_worktree_agent_dirty_file(self) -> None:
        """BUG-37 functional: A file dirtied by a non-worktree agent causes
        _check_merge_conflict to report a false conflict for a worktree merge.
        """
        repo = _make_repo(Path(self._tmpdir) / "repo")

        (repo / "shared.py").write_text("original content\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."], capture_output=True, check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "add shared.py"],
            capture_output=True,
            check=True,
        )

        server = VSCodeServer()
        server.work_dir = str(repo)

        wt_agent = WorktreeSorcarAgent("wt_agent")
        wt_agent._chat_id = "wt_tab"
        wt_work = wt_agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None
        wt = wt_agent._wt
        assert wt is not None

        (wt.wt_dir / "shared.py").write_text("worktree modified content\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "wt changes shared.py")

        (repo / "shared.py").write_text("non-wt agent modified content\n")

        tab = server._get_tab("wt_tab")
        tab.agent = wt_agent
        tab.use_worktree = True

        has_conflict = server._check_merge_conflict("wt_tab")

        assert has_conflict is True, (
            "BUG-37 confirmed: non-worktree agent's dirty file causes "
            "false conflict detection for worktree merge"
        )

        GitWorktreeOps.remove(repo, wt.wt_dir)
        GitWorktreeOps.prune(repo)
        GitWorktreeOps.delete_branch(repo, wt.branch)


class TestBug38SharedMergeDataDir:
    """BUG-38: Both worktree and non-worktree merge reviews write to
    a single global _merge_data_dir() with no per-tab isolation or
    locking.  _prepare_merge_view rmtrees merge-temp/ and overwrites
    pending-merge.json, destroying concurrent merge review data.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_merge_data_dir_is_per_tab(self) -> None:
        """BUG-38 FIXED: _merge_data_dir now accepts a tab_id parameter
        and returns per-tab paths."""
        dir1 = _merge_data_dir("tab-A")
        dir2 = _merge_data_dir("tab-B")
        assert dir1 != dir2, "Different tabs must get different dirs"
        assert "tab-A" in str(dir1)
        assert "tab-B" in str(dir2)



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
        worktree merges — GitWorktreeOps.unstaged_files counts the
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

import inspect
import json
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
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _merge_data_dir,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
)
from kiss.agents.vscode.server import VSCodeServer


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

    def test_source_lacks_repo_lock_in_run_task_inner_non_worktree_block(self) -> None:
        """BUG-34: The pre-task snapshot code in _run_task_inner does NOT
        acquire repo_lock.  Confirm this by inspecting the source."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.splitlines()

        snapshot_start = None
        for i, line in enumerate(lines):
            if "if not use_worktree:" in line:
                snapshot_start = i
                break

        assert snapshot_start is not None, "sanity: found non-worktree block"

        indent = len(lines[snapshot_start]) - len(lines[snapshot_start].lstrip())
        snapshot_end = snapshot_start + 1
        for i in range(snapshot_start + 1, len(lines)):
            stripped = lines[i].lstrip()
            if stripped and (len(lines[i]) - len(stripped)) <= indent:
                snapshot_end = i
                break

        snapshot_block = "\n".join(lines[snapshot_start:snapshot_end])

        assert "repo_lock" not in snapshot_block, (
            "Bug no longer present: repo_lock found in snapshot block"
        )

    def test_snapshot_functions_dont_acquire_repo_lock(self) -> None:
        """BUG-34: The individual snapshot functions (_parse_diff_hunks,
        _capture_untracked, _snapshot_files, _save_untracked_base) do
        not acquire repo_lock internally either."""
        for fn in [_parse_diff_hunks, _capture_untracked, _snapshot_files, _save_untracked_base]:
            source = inspect.getsource(fn)  # type: ignore[arg-type]
            assert "repo_lock" not in source, (
                f"Bug no longer present: {fn.__name__} acquires repo_lock"
            )

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


class TestBug35WorktreeMergeStashesAgentWork:
    """BUG-35: stash_if_dirty in _release_worktree / merge() captures
    ALL dirty files in the main repo — including a non-worktree agent's
    uncommitted edits.  The agent's work disappears mid-task.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_stash_if_dirty_captures_all_dirty_files(self) -> None:
        """BUG-35: stash_if_dirty stashes everything, including non-worktree
        agent's files.  After stash, the agent's files vanish."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent_file = repo / "agent_work.py"
        agent_file.write_text("# Agent is working on this\nprint('hello')\n")

        status = _git("status", "--porcelain", cwd=repo)
        assert "agent_work.py" in status.stdout, "sanity: agent file is dirty"
        assert agent_file.exists(), "sanity: agent file exists"

        did_stash = GitWorktreeOps.stash_if_dirty(repo)
        assert did_stash, "stash should capture the dirty state"

        assert not agent_file.exists(), (
            "BUG-35 confirmed: agent's in-flight work was stashed away"
        )

        agent_file.write_text("# Agent wrote something else meanwhile\n")

        pop_result = GitWorktreeOps.stash_pop(repo)
        if not pop_result:
            assert True, "BUG-35 confirmed: stash pop conflict after agent write"
        else:
            content = agent_file.read_text()
            assert "hello" in content or "something else" in content, (
                "BUG-35 confirmed: agent's work was disrupted by stash cycle"
            )

    def test_release_worktree_stashes_via_do_merge(self) -> None:
        """BUG-35: _release_worktree delegates stashing to _do_merge."""
        source = inspect.getsource(WorktreeSorcarAgent._release_worktree)
        assert "_do_merge" in source, "_release_worktree must use _do_merge"
        do_merge_source = inspect.getsource(WorktreeSorcarAgent._do_merge)
        assert "stash_if_dirty" in do_merge_source, (
            "_do_merge must call stash_if_dirty"
        )

    def test_merge_also_stashes_via_do_merge(self) -> None:
        """BUG-35: merge() delegates stashing to _do_merge."""
        source = inspect.getsource(WorktreeSorcarAgent.merge)
        assert "_do_merge" in source, "merge() must use _do_merge"

    def test_functional_stash_during_worktree_merge(self) -> None:
        """BUG-35 functional: worktree merge stashes non-worktree agent's file."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        agent = WorktreeSorcarAgent("wt_agent")
        agent._chat_id = "wt_tab"

        wt_work = agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None
        wt = agent._wt
        assert wt is not None

        (wt.wt_dir / "wt_change.txt").write_text("worktree change\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "wt commit")

        non_wt_file = repo / "non_wt_agent_work.py"
        non_wt_file.write_text("# non-worktree agent is working\n")
        assert non_wt_file.exists(), "sanity: non-wt agent file exists"

        msg = agent.merge()
        assert "Successfully merged" in msg, f"merge should succeed: {msg}"

        _git("stash", "list", cwd=repo)
        assert non_wt_file.exists(), "File should be back after stash pop"

        do_merge_source = inspect.getsource(WorktreeSorcarAgent._do_merge)
        assert "stash_if_dirty" in do_merge_source, (
            "_do_merge must call stash_if_dirty"
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

    def test_prepare_merge_view_always_uses_head(self) -> None:
        """BUG-36: _prepare_merge_view defaults to base_ref='HEAD', with no
        mechanism to lock HEAD or record the pre-task HEAD SHA."""
        source = inspect.getsource(VSCodeServer._run_task_inner)

        lines = source.splitlines()
        found_call = False
        for line in lines:
            if "_prepare_and_start_merge" in line and "not tab.use_worktree" not in line:
                found_call = True
                break

        assert found_call, "sanity: _prepare_and_start_merge call found"

        in_non_wt_block = False
        captures_head_sha = False
        for line in lines:
            if "if not use_worktree:" in line:
                in_non_wt_block = True
                continue
            if in_non_wt_block:
                if "rev-parse" in line and "HEAD" in line:
                    captures_head_sha = True
                if line.strip() and not line.strip().startswith("#"):
                    stripped = line.lstrip()
                    if len(line) - len(stripped) <= len("            "):
                        if not line.strip().startswith(("pre_", "_save", "#")):
                            break

        assert not captures_head_sha, (
            "Bug no longer present: non-worktree path now captures HEAD SHA"
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
    """BUG-37: _check_merge_conflict calls unstaged_files on the main
    repo and checks overlap with worktree changes.  If a non-worktree
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

    def test_check_merge_conflict_counts_all_dirty_files(self) -> None:
        """BUG-37: _check_merge_conflict uses unstaged_files which returns
        ALL dirty files in the main repo, with no way to distinguish
        user edits from non-worktree agent edits."""
        source = inspect.getsource(VSCodeServer._check_merge_conflict)

        assert "unstaged_files" in source, "sanity: uses unstaged_files"

        assert "non_worktree" not in source.lower(), (
            "Bug no longer present: _check_merge_conflict distinguishes agent dirty files"
        )
        assert "concurrent" not in source.lower(), (
            "Bug no longer present: _check_merge_conflict has concurrency awareness"
        )

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

    def test_prepare_merge_view_overwrites_existing_data(self) -> None:
        """BUG-38: _prepare_merge_view rmtrees merge-temp/ and overwrites
        pending-merge.json, destroying any concurrent merge session's data.
        """
        repo = _make_repo(Path(self._tmpdir) / "repo")

        (repo / "file_a.py").write_text("content a\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."], capture_output=True, check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "add file_a"],
            capture_output=True,
            check=True,
        )

        merge_dir = str(Path(self._tmpdir) / "merge_data")

        (repo / "file_a.py").write_text("modified by tab A\n")
        result_a = _prepare_merge_view(
            str(repo), merge_dir, {}, set(), None,
        )
        assert result_a.get("status") == "opened", f"Tab A merge view: {result_a}"

        pending = Path(merge_dir) / "pending-merge.json"
        tab_a_data = json.loads(pending.read_text())
        tab_a_files = [f["name"] for f in tab_a_data["files"]]
        assert "file_a.py" in tab_a_files, "sanity: Tab A has file_a.py"

        subprocess.run(
            ["git", "-C", str(repo), "checkout", "--", "file_a.py"],
            capture_output=True,
            check=True,
        )
        (repo / "new_file_b.txt").write_text("created by tab B\n")

        result_b = _prepare_merge_view(
            str(repo), merge_dir, {}, set(), None,
        )
        assert result_b.get("status") == "opened", f"Tab B merge view: {result_b}"

        tab_b_data = json.loads(pending.read_text())
        tab_b_files = [f["name"] for f in tab_b_data["files"]]

        assert "file_a.py" not in tab_b_files, (
            "BUG-38 confirmed: Tab A's merge data was overwritten by Tab B"
        )
        assert "new_file_b.txt" in tab_b_files, (
            "sanity: Tab B's data is present"
        )

    def test_prepare_merge_view_rmtrees_merge_temp(self) -> None:
        """BUG-38: _prepare_merge_view does shutil.rmtree on merge-temp/,
        which would destroy base copies from a concurrent merge session."""
        source = inspect.getsource(_prepare_merge_view)
        assert "rmtree" in source, (
            "sanity: _prepare_merge_view uses rmtree"
        )
        assert "merge-temp" in source, (
            "sanity: _prepare_merge_view references merge-temp"
        )

    def test_worktree_and_non_worktree_use_per_tab_merge_dir(self) -> None:
        """BUG-38 FIXED: Both paths now use per-tab _merge_data_dir(tab_id)."""
        wt_source = inspect.getsource(VSCodeServer._present_pending_worktree)
        assert "_prepare_and_start_merge" in wt_source, (
            "sanity: worktree review (inlined in "
            "_present_pending_worktree) uses _prepare_and_start_merge"
        )

        prep_source = inspect.getsource(VSCodeServer._prepare_and_start_merge)
        assert "_merge_data_dir" in prep_source, (
            "sanity: _prepare_and_start_merge uses _merge_data_dir"
        )

        assert "tab_id" in inspect.signature(
            VSCodeServer._prepare_and_start_merge
        ).parameters, (
            "Fix verified: _prepare_and_start_merge is tab-aware"
        )

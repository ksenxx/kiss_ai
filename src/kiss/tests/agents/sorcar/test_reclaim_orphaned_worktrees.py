# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for orphan-worktree reclaim.

Bug reproduced by these tests: when a Sorcar process is killed while a
worktree task is pending, the ``kiss/wt-*`` worktree stays registered
on disk with dirty uncommitted work but no in-memory ``self._wt`` state
survives.  Neither ``_release_worktree`` (per-agent) nor
``sweep_orphaned_state`` (config-debris only) ever merges it, so the
work is silently stranded.  The fix is
:meth:`GitWorktreeOps.reclaim_orphaned_worktrees`, wired into
:meth:`WorktreeSorcarAgent._try_setup_worktree` right before
``sweep_orphaned_state``.

Each test creates a real git repo in a temp dir, plants one or more
``kiss/wt-*`` worktrees with the same on-disk state a killed-mid-task
Sorcar process leaves behind (dirty index / dirty worktree / untracked
files, saved ``branch.<name>.kiss-original`` and optionally
``kiss-baseline`` config), then calls the reclaim path and asserts on
the resulting merge commits, on-disk worktree directory, and remaining
branches.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter


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
    subprocess.run(["git", "init", "-b", "main", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _patch_super_run(
    return_value: str = "success: true\nsummary: test done\n",
) -> Any:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


def _plant_orphan_worktree(
    repo: Path,
    branch: str,
    *,
    original_branch: str = "main",
    with_baseline: bool = False,
    dirty_kind: str = "staged",
) -> Path:
    """Create a registered ``kiss/wt-*`` worktree with dirty state.

    Simulates a Sorcar process killed mid-task: the worktree is
    registered in git, has the ``kiss-original`` / (optionally)
    ``kiss-baseline`` branch config saved, and holds uncommitted work
    of the requested kind.  No in-memory ``_wt`` state is created for
    the caller.

    Args:
        repo: Main repo path.
        branch: Worktree branch name (must start with ``kiss/wt-``).
        original_branch: Original branch to record in config.
        with_baseline: When True, also create a baseline commit
            (empty ``kiss-baseline`` change) and store its SHA so
            :meth:`squash_merge_from_baseline` is exercised.
        dirty_kind: ``"staged"``, ``"unstaged"``, ``"untracked"``, or
            ``"clean"`` — how the worktree's tree ends up.

    Returns:
        The worktree directory path.
    """
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    assert GitWorktreeOps.save_original_branch(repo, branch, original_branch)
    if with_baseline:
        (wt_dir / "baseline_marker.txt").write_text("baseline\n")
        GitWorktreeOps.stage_all(wt_dir)
        assert GitWorktreeOps.commit_all(wt_dir, "kiss: baseline")
        sha = GitWorktreeOps.head_sha(wt_dir)
        assert sha is not None
        assert GitWorktreeOps.save_baseline_commit(repo, branch, sha)
    if dirty_kind == "clean":
        return wt_dir
    if dirty_kind == "staged":
        (wt_dir / "staged_file.txt").write_text("staged content\n")
        GitWorktreeOps.stage_all(wt_dir)
    elif dirty_kind == "unstaged":
        # Modify an already-tracked file so the change is unstaged.
        (wt_dir / "README.md").write_text("# Test\nmodified\n")
    elif dirty_kind == "untracked":
        (wt_dir / "new_untracked.txt").write_text("untracked\n")
    else:
        raise ValueError(f"unknown dirty_kind={dirty_kind}")
    return wt_dir


def _first_line_of_head(repo: Path) -> str:
    result = _git("log", "-1", "--pretty=%s", cwd=repo)
    return result.stdout.strip()


class TestReclaimStagedDirty:
    """Reclaim commits staged edits and merges them into main."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reclaims_staged_worktree(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-aaaa", dirty_kind="staged",
        )
        head_before = GitWorktreeOps.head_sha(self.repo)

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        assert reclaimed == 1
        assert not wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, "kiss/wt-1000-aaaa")
        # Main advanced by exactly one squash-merge commit.
        head_after = GitWorktreeOps.head_sha(self.repo)
        assert head_after != head_before
        # The staged file must have made it onto main.
        assert (self.repo / "staged_file.txt").exists()


class TestReclaimUnstagedAndUntracked:
    """Reclaim commits unstaged edits and merges them into main."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reclaims_unstaged_worktree(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-bbbb", dirty_kind="unstaged",
        )
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1
        assert not wt_dir.exists()
        assert (self.repo / "README.md").read_text() == "# Test\nmodified\n"

    def test_reclaims_untracked_worktree(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-cccc", dirty_kind="untracked",
        )
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1
        assert not wt_dir.exists()
        assert (self.repo / "new_untracked.txt").exists()


class TestReclaimWithBaseline:
    """Reclaim uses squash_merge_from_baseline when a baseline is saved."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reclaims_with_baseline_commit(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo,
            "kiss/wt-1000-dddd",
            with_baseline=True,
            dirty_kind="staged",
        )
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1
        assert not wt_dir.exists()
        # Baseline marker should NOT reach main (baseline captures the
        # user's pre-existing dirty state so only diffs after baseline
        # are merged).  The staged_file added AFTER baseline should.
        assert (self.repo / "staged_file.txt").exists()
        assert not (self.repo / "baseline_marker.txt").exists()


class TestReclaimTwoOrphans:
    """Reclaim merges multiple orphans in one pass (repro of user's report)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_two_orphan_worktrees_are_reclaimed(self) -> None:
        wt1 = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-first", dirty_kind="staged",
        )
        (wt1 / "unique_first.txt").write_text("first\n")
        GitWorktreeOps.stage_all(wt1)
        wt2 = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-second", dirty_kind="staged",
        )
        (wt2 / "unique_second.txt").write_text("second\n")
        GitWorktreeOps.stage_all(wt2)

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        assert reclaimed == 2
        assert not wt1.exists()
        assert not wt2.exists()
        assert (self.repo / "unique_first.txt").exists()
        assert (self.repo / "unique_second.txt").exists()
        assert not GitWorktreeOps.branch_exists(
            self.repo, "kiss/wt-1000-first",
        )
        assert not GitWorktreeOps.branch_exists(
            self.repo, "kiss/wt-1000-second",
        )


class TestReclaimSafetyGuards:
    """Every safety guard preserves the worktree instead of merging it."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_skips_when_original_branch_no_longer_exists(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo,
            "kiss/wt-1000-lost",
            original_branch="ghost-branch",
            dirty_kind="staged",
        )
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 0
        assert wt_dir.exists()

    def test_skips_when_current_branch_differs(self) -> None:
        # Main tree is on main, but the worktree was cut from a
        # different branch that still exists.
        _git("branch", "feature", cwd=self.repo)
        wt_dir = _plant_orphan_worktree(
            self.repo,
            "kiss/wt-1000-wrongbase",
            original_branch="feature",
            dirty_kind="staged",
        )
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 0
        assert wt_dir.exists()

    def test_skips_when_main_tree_is_dirty(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-dirtymain", dirty_kind="staged",
        )
        (self.repo / "user_edit.txt").write_text("user is editing\n")

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        assert reclaimed == 0
        assert wt_dir.exists()
        # User's edit was left untouched.
        assert (self.repo / "user_edit.txt").read_text() == "user is editing\n"

    def test_skips_when_detached_head(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-detached", dirty_kind="staged",
        )
        head = GitWorktreeOps.head_sha(self.repo)
        assert head is not None
        _git("checkout", "--detach", head, cwd=self.repo)

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        assert reclaimed == 0
        assert wt_dir.exists()

    def test_skips_branches_in_exclude_set(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-live", dirty_kind="staged",
        )
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(
            self.repo,
            exclude_branches={"kiss/wt-1000-live"},
        )
        assert reclaimed == 0
        assert wt_dir.exists()

    def test_ignores_non_kiss_worktree_entries(self) -> None:
        # Create a non-kiss worktree; reclaim must not touch it.
        other_dir = Path(self.tmpdir) / "other_wt"
        # ``create`` mints ``other`` as a new branch on ``git worktree
        # add -b`` — we deliberately do NOT pre-create it, since the
        # point of this test is that non-kiss branch names are
        # skipped by the branch-prefix filter.
        assert GitWorktreeOps.create(self.repo, "other", other_dir)
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 0
        assert other_dir.exists()
        assert GitWorktreeOps.branch_exists(self.repo, "other")

    def test_preserves_on_merge_conflict(self) -> None:
        # Baseline captures the user's dirty state.  Then we commit a
        # divergent change on main AFTER worktree creation so the
        # cherry-pick has a real cross-branch conflict.
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-conflict",
            with_baseline=True,
            dirty_kind="clean",
        )
        # Divergent commit inside the worktree.
        (wt_dir / "README.md").write_text("worktree change\n")
        GitWorktreeOps.stage_all(wt_dir)
        assert GitWorktreeOps.commit_all(wt_dir, "wt edit")
        # Divergent commit on main.
        (self.repo / "README.md").write_text("main change\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "main edit", cwd=self.repo)

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        # Merge conflict → preserve.
        assert reclaimed == 0
        assert wt_dir.exists()
        assert GitWorktreeOps.branch_exists(
            self.repo, "kiss/wt-1000-conflict",
        )
        # Main tree returned to its committed state (no conflict
        # markers dangling).
        assert not GitWorktreeOps.has_uncommitted_changes(self.repo)

    def test_preserves_when_precommit_hook_rejects(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-hook", dirty_kind="staged",
        )
        hooks_dir = self.repo / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        assert reclaimed == 0
        assert wt_dir.exists()

    def test_skips_when_wt_dir_gone(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-gone", dirty_kind="staged",
        )
        # Simulate the wt directory being externally deleted while
        # git registration lingers.
        shutil.rmtree(str(wt_dir))
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 0

    def test_skips_when_preserve_marker_set(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-preserve", dirty_kind="staged",
        )
        assert GitWorktreeOps.save_preserve_marker(
            self.repo, "kiss/wt-1000-preserve",
        )

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        assert reclaimed == 0
        assert wt_dir.exists()
        assert GitWorktreeOps.branch_exists(
            self.repo, "kiss/wt-1000-preserve",
        )

    def test_falls_back_to_current_branch_when_original_missing(self) -> None:
        # Create the WT WITHOUT saving kiss-original (reproduces the
        # legacy worktree found in the user's real repo).
        branch = "kiss/wt-1000-legacy"
        wt_dir = self.repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        (wt_dir / "legacy_recovered.txt").write_text("recovered\n")
        GitWorktreeOps.stage_all(wt_dir)

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        assert reclaimed == 1
        assert not wt_dir.exists()
        assert (self.repo / "legacy_recovered.txt").exists()


class TestReclaimCleanWorktree:
    """A clean worktree with a committed diff still gets merged and removed."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_committed_only_worktree_is_reclaimed(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-clean", dirty_kind="clean",
        )
        (wt_dir / "committed_only.txt").write_text("hi\n")
        GitWorktreeOps.stage_all(wt_dir)
        assert GitWorktreeOps.commit_all(wt_dir, "committed on branch")

        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)

        assert reclaimed == 1
        assert not wt_dir.exists()
        assert (self.repo / "committed_only.txt").exists()

    def test_clean_worktree_no_diff_still_removed(self) -> None:
        wt_dir = _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-nodiff", dirty_kind="clean",
        )
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1
        assert not wt_dir.exists()


class TestRegisteredWorktreesHelper:
    """Behaviour of the low-level parser."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_lists_main_and_kiss_worktrees(self) -> None:
        _plant_orphan_worktree(
            self.repo, "kiss/wt-1000-list", dirty_kind="clean",
        )
        pairs = GitWorktreeOps.registered_worktrees(self.repo)
        branches = {b for _, b in pairs}
        assert "kiss/wt-1000-list" in branches
        # Main entry present, whatever the default branch is called.
        assert any(b in ("main", "master") for b in branches)

    def test_ignores_detached_head_entry(self) -> None:
        # Register another worktree in detached-HEAD mode: no
        # ``branch refs/heads/...`` line is emitted for it.
        detached_dir = Path(self.tmpdir) / "detached"
        head = GitWorktreeOps.head_sha(self.repo)
        assert head is not None
        subprocess.run(
            ["git", "-C", str(self.repo), "worktree", "add", "--detach",
             str(detached_dir), head],
            capture_output=True, check=True,
        )
        pairs = GitWorktreeOps.registered_worktrees(self.repo)
        assert detached_dir not in {p for p, _ in pairs}


class TestBranchConfigLoaders:
    """load_original_branch / load_baseline_commit are reciprocal to savers."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_missing_returns_none(self) -> None:
        assert GitWorktreeOps.load_original_branch(
            self.repo, "kiss/wt-nope",
        ) is None
        assert GitWorktreeOps.load_baseline_commit(
            self.repo, "kiss/wt-nope",
        ) is None

    def test_load_returns_saved_values(self) -> None:
        branch = "kiss/wt-1000-cfg"
        assert GitWorktreeOps.save_original_branch(self.repo, branch, "main")
        assert GitWorktreeOps.save_baseline_commit(
            self.repo, branch, "deadbeef",
        )
        assert GitWorktreeOps.load_original_branch(
            self.repo, branch,
        ) == "main"
        assert GitWorktreeOps.load_baseline_commit(
            self.repo, branch,
        ) == "deadbeef"


class TestReclaimWiredIntoWorktreeAgent:
    """End-to-end via WorktreeSorcarAgent: next task adopts orphans."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_next_task_reclaims_orphans(self) -> None:
        # Plant two orphan worktrees from a "dead" prior process.
        wt1 = _plant_orphan_worktree(
            self.repo, "kiss/wt-orphan-a", dirty_kind="staged",
        )
        (wt1 / "orphan_a.txt").write_text("A\n")
        GitWorktreeOps.stage_all(wt1)
        wt2 = _plant_orphan_worktree(
            self.repo, "kiss/wt-orphan-b", dirty_kind="untracked",
        )

        # New agent runs a fresh task: this triggers _try_setup_worktree,
        # which now calls reclaim before creating its own worktree.
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task", work_dir=str(self.repo))

        # Orphans gone.
        assert not wt1.exists()
        assert not wt2.exists()
        assert not GitWorktreeOps.branch_exists(
            self.repo, "kiss/wt-orphan-a",
        )
        assert not GitWorktreeOps.branch_exists(
            self.repo, "kiss/wt-orphan-b",
        )
        # Their file content landed on main.
        # (agent's own worktree is on its own branch and hasn't been
        # released yet, so main must already show the reclaimed files.)
        assert (self.repo / "orphan_a.txt").exists()
        assert (self.repo / "new_untracked.txt").exists()

        agent.discard()

    def test_agent_own_worktree_is_not_reclaimed(self) -> None:
        # First agent creates a worktree and pauses (no release).
        agent1 = WorktreeSorcarAgent("test1")
        agent1.run(prompt_template="task-a", work_dir=str(self.repo))
        own_wt = agent1._wt
        assert own_wt is not None
        # Simulate the server registering the live task (the reclaim
        # exclude set is derived from the task-keyed agent-state
        # registry, reached via the printer's
        # ``live_worktree_branches`` bridge).
        state1 = agent_state.AgentState(
            "reclaim-live-1",
            agent=agent1,
            tab_id="tab1",
            server_owned=True,
            is_task_active=True,
        )
        agent_state.register(state1)

        # Second agent starts a task with agent1 still holding its
        # worktree.  agent1's branch must be excluded from reclaim.
        # The printer is passed exactly the way the production task
        # runner passes it — as a ``run`` kwarg, never by pre-setting
        # ``agent2.printer``.  Pre-setting the attribute used to mask
        # a real bug: on a fresh agent ``self.printer`` is unset until
        # ``super().run()``, so the reclaim inside worktree setup ran
        # with an empty live-branch exclusion set and deleted a live
        # sibling's worktree.
        agent2 = WorktreeSorcarAgent("test2")
        try:
            agent2.run(
                prompt_template="task-b",
                work_dir=str(self.repo),
                printer=JsonPrinter(),
            )

            # agent1's worktree must still exist.
            assert own_wt.wt_dir.exists()
            assert GitWorktreeOps.branch_exists(self.repo, own_wt.branch)
        finally:
            agent_state.unregister("reclaim-live-1", state1)
            agent2.discard()
            agent1.discard()

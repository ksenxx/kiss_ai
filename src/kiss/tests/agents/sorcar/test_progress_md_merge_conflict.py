# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent scratch files must never cause worktree merge conflicts.

Reproduces incidents where merging a task branch back into main failed
with "Merge conflict detected" purely because tracked agent scratch
files drifted: ``PROGRESS.md`` is a per-task agent log that every task
wholesale rewrites, and ``src/kiss/INJECTIONS.md`` can be captured in a
dirty-state baseline and later receive formatting-only task-branch
changes.

The fix installs a repo-local git merge driver (via the untracked
``<git_common_dir>/info/attributes`` + local config, never touching
tracked files) that auto-resolves scratch-file content conflicts by
taking the incoming task branch's version — the newest scratch state
wins.  Because the driver lives in repo plumbing it also fixes the
manual ``git cherry-pick`` / ``git merge`` resolution commands.
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
    MergeResult,
    _git,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

MAIN_LOG = "# Task: bug hunt iteration 7\n" + "- found bug\n" * 50
BRANCH_LOG = "# Task: forensic analysis\n- daemon was killed by SIGTERM\n"
INJECTIONS_HEAD = (
    "## Trick\n\n"
    "Use openrouter/z-ai/glm-5.2 model for all tasks including coding, "
    "bug fixing, and test creation. ALWAYS use gpt-5.5 model (not codex) "
    "and claude-opus-4-7 to carefully and thoroughly review the work done "
    "by openrouter/z-ai/glm-5.2.\n"
)
INJECTIONS_BASELINE = INJECTIONS_HEAD.replace(
    "and claude-opus-4-7 to carefully and thoroughly review",
    "to carefully and thoroughly review",
).replace(
    "by openrouter/z-ai/glm-5.2.\n",
    "by openrouter/z-ai/glm-5.2.  Then use claude-opus-4-7 to carefully "
    "and thoroughly review the work done by openrouter/z-ai/glm-5.2.\n",
)
INJECTIONS_BRANCH = INJECTIONS_BASELINE.replace(".  Then use", ". Then use")


def _make_repo(path: Path) -> Path:
    """Create a git repo with PROGRESS.md and a source file committed."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    for key, val in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, val],
            capture_output=True,
            check=True,
        )
    (path / "PROGRESS.md").write_text("# Task: older task\n- step 1\n")
    (path / "code.py").write_text("x = 1\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _commit_all(cwd: Path, message: str) -> None:
    """Stage and commit everything in *cwd*."""
    _git("add", "-A", cwd=cwd)
    result = _git("commit", "-m", message, cwd=cwd)
    assert result.returncode == 0, result.stderr


def _create_worktree(repo: Path, branch: str) -> Path:
    """Create a worktree at repo/.kiss-worktrees/<slug>."""
    wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    return wt_dir


class TestScratchMergeDriverOps:
    """GitWorktreeOps-level reproduction of the PROGRESS.md conflict."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _diverge(self, wt_dir: Path) -> None:
        """Rewrite PROGRESS.md differently on the branch and on main."""
        (wt_dir / "PROGRESS.md").write_text(BRANCH_LOG)
        (wt_dir / "agent_work.py").write_text("y = 2\n")
        _commit_all(wt_dir, "agent task work")
        (self.repo / "PROGRESS.md").write_text(MAIN_LOG)
        _commit_all(self.repo, "other task merged on main")

    def test_squash_merge_progress_conflict_auto_resolves(self) -> None:
        """Divergent PROGRESS.md rewrites no longer block a squash merge."""
        wt_dir = _create_worktree(self.repo, "kiss/wt-test-a")
        self._diverge(wt_dir)
        GitWorktreeOps.ensure_scratch_merge_driver(self.repo)

        result = GitWorktreeOps.squash_merge_branch(self.repo, "kiss/wt-test-a")

        assert result == MergeResult.SUCCESS
        assert (self.repo / "PROGRESS.md").read_text() == BRANCH_LOG
        assert (self.repo / "agent_work.py").read_text() == "y = 2\n"

    def test_cherry_pick_from_baseline_auto_resolves(self) -> None:
        """Exact incident replay: baseline + cherry-pick path succeeds."""
        # User has a dirty PROGRESS.md when the worktree is created.
        (self.repo / "PROGRESS.md").write_text("# Task: dirty in-flight log\n")
        wt_dir = _create_worktree(self.repo, "kiss/wt-test-b")
        assert GitWorktreeOps.copy_dirty_state(self.repo, wt_dir)
        GitWorktreeOps.stage_all(wt_dir)
        assert GitWorktreeOps.commit_staged(
            wt_dir, "kiss: baseline from dirty state", no_verify=True
        )
        baseline = GitWorktreeOps.head_sha(wt_dir)
        assert baseline is not None
        # Discard main's dirty state (as stash_if_dirty would have).
        _git("checkout", "--", "PROGRESS.md", cwd=self.repo)

        self._diverge(wt_dir)
        GitWorktreeOps.ensure_scratch_merge_driver(self.repo)

        result = GitWorktreeOps.squash_merge_from_baseline(
            self.repo, "kiss/wt-test-b", baseline
        )

        assert result == MergeResult.SUCCESS
        assert (self.repo / "PROGRESS.md").read_text() == BRANCH_LOG
        assert (self.repo / "agent_work.py").read_text() == "y = 2\n"

    def test_real_source_conflict_still_reported(self) -> None:
        """Genuine conflicts in non-scratch files must still be CONFLICT."""
        wt_dir = _create_worktree(self.repo, "kiss/wt-test-c")
        (wt_dir / "code.py").write_text("x = 2  # agent\n")
        _commit_all(wt_dir, "agent edit")
        (self.repo / "code.py").write_text("x = 3  # user\n")
        _commit_all(self.repo, "user edit")
        GitWorktreeOps.ensure_scratch_merge_driver(self.repo)

        result = GitWorktreeOps.squash_merge_branch(self.repo, "kiss/wt-test-c")

        assert result == MergeResult.CONFLICT
        # The failed merge must leave the main tree clean (reset --hard).
        assert not GitWorktreeOps.has_uncommitted_changes(self.repo)
        assert (self.repo / "code.py").read_text() == "x = 3  # user\n"

    def test_driver_installation_is_idempotent(self) -> None:
        """Repeated installation adds exactly one attributes line."""
        GitWorktreeOps.ensure_scratch_merge_driver(self.repo)
        GitWorktreeOps.ensure_scratch_merge_driver(self.repo)

        attrs = self.repo / ".git" / "info" / "attributes"
        lines = attrs.read_text().splitlines()
        assert lines.count("PROGRESS.md merge=kiss-scratch") == 1
        assert lines.count("src/kiss/INJECTIONS.md merge=kiss-scratch") == 1
        driver = _git(
            "config", "merge.kiss-scratch.driver", cwd=self.repo
        ).stdout.strip()
        assert "%A" in driver and "%B" in driver
        # No tracked file may be touched by the installation.
        assert not GitWorktreeOps.has_uncommitted_changes(self.repo)

    def test_manual_cherry_pick_also_auto_resolves(self) -> None:
        """The manual resolution command from the error message works too."""
        wt_dir = _create_worktree(self.repo, "kiss/wt-test-d")
        self._diverge(wt_dir)
        GitWorktreeOps.ensure_scratch_merge_driver(self.repo)

        head = GitWorktreeOps.head_sha(self.repo)
        assert head is not None
        result = _git(
            "cherry-pick",
            "--no-commit",
            f"{head}..kiss/wt-test-d",
            cwd=self.repo,
        )
        assert result.returncode == 0, result.stderr
        assert (self.repo / "PROGRESS.md").read_text() == BRANCH_LOG

    def test_injections_md_prompt_drift_is_scratch_and_auto_resolves(self) -> None:
        """Prompt-injection scratch edits must not block task-branch merge.

        Replays the previous-task incident precisely: the user's dirty
        ``src/kiss/INJECTIONS.md`` was captured as the baseline commit,
        while the agent later made only a whitespace adjustment in the
        same scratch prompt file.  Main meanwhile had the original
        pre-baseline text, so ``git cherry-pick --no-commit
        baseline..branch`` produced a conflict in ``INJECTIONS.md``.

        ``INJECTIONS.md`` has the same per-task scratch semantics as
        ``PROGRESS.md`` and must use the same incoming-branch merge
        driver so a formatting-only prompt-log drift cannot surface as
        a user-visible worktree merge conflict.
        """
        injections = self.repo / "src" / "kiss" / "INJECTIONS.md"
        injections.parent.mkdir(parents=True)
        injections.write_text(INJECTIONS_HEAD)
        _commit_all(self.repo, "add injection scratch prompt")

        injections.write_text(INJECTIONS_BASELINE)
        wt_dir = _create_worktree(self.repo, "kiss/wt-test-injections")
        assert GitWorktreeOps.copy_dirty_state(self.repo, wt_dir)
        GitWorktreeOps.stage_all(wt_dir)
        assert GitWorktreeOps.commit_staged(
            wt_dir, "kiss: baseline from dirty state", no_verify=True
        )
        baseline = GitWorktreeOps.head_sha(wt_dir)
        assert baseline is not None
        _git("checkout", "--", "src/kiss/INJECTIONS.md", cwd=self.repo)

        (wt_dir / "src" / "kiss" / "INJECTIONS.md").write_text(INJECTIONS_BRANCH)
        (wt_dir / "agent_work.py").write_text("y = 2\n")
        _commit_all(wt_dir, "agent task work")
        GitWorktreeOps.ensure_scratch_merge_driver(self.repo)

        result = GitWorktreeOps.squash_merge_from_baseline(
            self.repo, "kiss/wt-test-injections", baseline
        )

        assert result == MergeResult.SUCCESS
        assert injections.read_text() == INJECTIONS_BRANCH
        assert (self.repo / "agent_work.py").read_text() == "y = 2\n"


def _redirect_db(tmpdir: str) -> tuple[Any, Any, Any]:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple[Any, Any, Any]) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _patch_super_run() -> Any:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return "success: true\nsummary: test done\n"

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


class TestAgentMergeWithDivergentProgress:
    """End-to-end: WorktreeSorcarAgent.merge() survives PROGRESS.md drift."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merge_succeeds_despite_divergent_progress_md(self) -> None:
        """The incident scenario through the real agent merge path."""
        agent = WorktreeSorcarAgent("test")
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "PROGRESS.md").write_text(BRANCH_LOG)
        (wt_dir / "agent_work.py").write_text("y = 2\n")
        _commit_all(wt_dir, "agent task work")

        (self.repo / "PROGRESS.md").write_text(MAIN_LOG)
        _commit_all(self.repo, "other task merged on main")

        msg = agent.merge()

        assert "Successfully merged" in msg, msg
        assert (self.repo / "PROGRESS.md").read_text() == BRANCH_LOG
        assert (self.repo / "agent_work.py").read_text() == "y = 2\n"

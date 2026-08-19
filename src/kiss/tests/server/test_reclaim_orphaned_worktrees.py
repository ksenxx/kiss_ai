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
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _git,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.test_reclaim_orphaned_worktrees import (  # noqa: F401
    _make_repo,
    _patch_super_run,
    _redirect_db,
    _restore_db,
    _unpatch_super_run,
)


def _first_line_of_head(repo: Path) -> str:
    result = _git("log", "-1", "--pretty=%s", cwd=repo)
    return result.stdout.strip()


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

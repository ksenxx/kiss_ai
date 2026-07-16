# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: worktree changed-file listing must survive a rename.

Reproduces a real bug in ``_MergeFlowMixin._get_worktree_changed_files``:
the ``git diff --name-only`` invocations did not pass ``--no-renames``,
so when the agent renamed a tracked file inside its worktree
(``git mv old.txt new.txt`` plus an edit), git's rename detection
collapsed the change into a single entry for the NEW path.  The OLD
path — which merging the worktree branch will DELETE from the user's
main tree — was silently missing from ``changedFiles`` (the list shown
to the user on the ``worktree_done`` prompt) and from the file-overlap
sets used by ``_check_merge_conflict``, which could therefore miss a
conflict with dirty main-tree edits to the old path.

No mocks, patches, fakes, or test doubles. Real git repository, real
``WorktreeSorcarAgent`` worktree, real server mixin methods.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.server import VSCodeServer


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


class TestWorktreeRenameChangedFiles(unittest.TestCase):
    """Renamed files must appear (old and new path) in changedFiles."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-rename-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "commit.gpgsign", "false")
        content = "".join(f"line{i}\n" for i in range(1, 31))
        Path(self.repo, "old.txt").write_text(content)
        _git(self.repo, "add", "old.txt")
        _git(self.repo, "commit", "-q", "-m", "initial")

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

        self.tab_id = "t-wt-rename"
        self.tab = self.server._get_tab(self.tab_id)
        self.tab.use_worktree = True
        self.agent = WorktreeSorcarAgent("wt-rename-test")
        self.tab.agent = self.agent
        wt_work_dir = self.agent._try_setup_worktree(Path(self.repo), self.repo)
        assert wt_work_dir is not None
        assert self.agent._wt is not None
        self.wt_dir = str(self.agent._wt.wt_dir)

    def tearDown(self) -> None:
        try:
            if self.agent._wt is not None:
                self.agent.discard()
        except Exception:
            pass
        _RunningAgentState.running_agent_states.pop(self.tab_id, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_renamed_file_lists_old_and_new_paths(self) -> None:
        """git mv + edit in the worktree must report BOTH paths as changed."""
        _git(self.wt_dir, "mv", "old.txt", "new.txt")
        new_path = Path(self.wt_dir, "new.txt")
        new_path.write_text(
            new_path.read_text().replace("line5\n", "line5-agent\n"),
        )

        changed = self.server._get_worktree_changed_files(self.tab_id)
        self.assertIn(
            "new.txt", changed,
            f"new path missing from changed files: {changed}",
        )
        self.assertIn(
            "old.txt", changed,
            "rename hid the old path's deletion from changedFiles: "
            f"{changed}",
        )

    def test_conflict_check_sees_dirty_main_edit_to_renamed_path(self) -> None:
        """A dirty main-tree edit of the old path must flag a conflict."""
        _git(self.wt_dir, "mv", "old.txt", "new.txt")
        new_path = Path(self.wt_dir, "new.txt")
        new_path.write_text(
            new_path.read_text().replace("line5\n", "line5-agent\n"),
        )
        # User meanwhile edits the soon-to-be-deleted old path on main.
        main_old = Path(self.repo, "old.txt")
        main_old.write_text(
            main_old.read_text().replace("line9\n", "line9-user\n"),
        )

        self.assertTrue(
            self.server._check_merge_conflict(self.tab_id),
            "conflict check missed the overlap between the worktree "
            "rename (deletes old.txt) and the user's dirty edit of "
            "old.txt on the main tree",
        )


if __name__ == "__main__":
    unittest.main()

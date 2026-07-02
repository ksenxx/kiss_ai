# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group B): filenames with leading/trailing spaces are
mangled by over-eager whitespace stripping.

Two independent stripping bugs, one per assigned source file:

* ``diff_merge._capture_untracked`` applied ``line.strip()`` to every
  ``git ls-files --others`` output line.  A new untracked file whose
  name ends (or begins) with a space — legal on every POSIX
  filesystem, and NOT C-quoted by git (space is a printable
  character) — was therefore recorded under a mangled name that does
  not exist on disk, so ``_prepare_merge_view`` silently dropped the
  agent-created file from the merge review entirely.

* ``merge_flow._unquoted_name_lines`` called ``output.strip()`` on the
  whole ``git diff --name-only`` output before splitting, eating the
  leading spaces of the FIRST listed path (and trailing spaces of the
  last).  ``_get_worktree_changed_files`` then reported a wrong path
  and ``_check_merge_conflict``'s overlap sets could never match the
  real on-disk name.

No mocks, patches, fakes, or test doubles: real git repositories, a
real ``WorktreeSorcarAgent`` worktree, real server mixin methods.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.diff_merge import _capture_untracked, _prepare_merge_view
from kiss.agents.vscode.server import VSCodeServer


def _git(cwd: str | Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout


class TestUntrackedSpaceNames(unittest.TestCase):
    """Untracked files with space-adjacent names must survive capture."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt8-spc-")
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "t")
        (self.repo / "seed.txt").write_text("seed\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "init")
        self.data_dir = Path(self.tmpdir) / "mergedata"
        self.data_dir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_capture_untracked_preserves_space_names(self) -> None:
        """``_capture_untracked`` must return the exact on-disk names."""
        (self.repo / "trail ").write_text("agent\n")
        (self.repo / " lead.txt").write_text("agent\n")
        captured = _capture_untracked(str(self.repo))
        self.assertIn(
            "trail ", captured,
            f"trailing-space name mangled by capture: {sorted(captured)}",
        )
        self.assertIn(
            " lead.txt", captured,
            f"leading-space name mangled by capture: {sorted(captured)}",
        )

    def test_new_trailing_space_file_appears_in_merge_review(self) -> None:
        """An agent-created ``trail `` file must be reviewable, not
        silently dropped from the merge view."""
        (self.repo / "trail ").write_text("agent line\n")
        result = _prepare_merge_view(
            str(self.repo), str(self.data_dir), {}, set(),
        )
        self.assertEqual(
            result.get("status"), "opened",
            "the new untracked file was dropped from the merge review "
            f"entirely: {result}",
        )
        manifest = json.loads(
            (self.data_dir / "pending-merge.json").read_text(),
        )
        names = [f["name"] for f in manifest["files"]]
        self.assertIn("trail ", names, f"mangled manifest names: {names}")


class TestWorktreeLeadingSpaceChangedFiles(unittest.TestCase):
    """Worktree changed-file listing must keep leading spaces in names."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt8-wtspc-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "t")
        _git(self.repo, "config", "commit.gpgsign", "false")
        Path(self.repo, " lead.txt").write_text("line1\nline2\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "initial")

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

        self.tab_id = "t-wt-space"
        self.tab = self.server._get_tab(self.tab_id)
        self.tab.use_worktree = True
        self.agent = WorktreeSorcarAgent("wt-space-test")
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

    def test_changed_files_keep_leading_space(self) -> None:
        """Editing `` lead.txt`` in the worktree must report the exact
        on-disk path, not a strip()-mangled one."""
        target = Path(self.wt_dir, " lead.txt")
        target.write_text("line1\nline2-agent\n")

        changed = self.server._get_worktree_changed_files(self.tab_id)
        self.assertIn(
            " lead.txt", changed,
            "leading space stripped from the changed-file listing: "
            f"{changed}",
        )

    def test_conflict_check_sees_dirty_main_edit_of_space_name(self) -> None:
        """A dirty main-tree edit of `` lead.txt`` must flag a conflict
        with the worktree's edit of the same file."""
        Path(self.wt_dir, " lead.txt").write_text("line1\nline2-agent\n")
        Path(self.repo, " lead.txt").write_text("line1-user\nline2\n")

        self.assertTrue(
            self.server._check_merge_conflict(self.tab_id),
            "conflict check missed the overlap on the space-named file",
        )


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: git C-quoted paths must be unquoted everywhere.

Even with ``core.quotepath=false``, git C-quotes any path containing a
double-quote, backslash, or control character — in ``diff --git``
headers, ``--name-only`` output, ``ls-files --others`` output, and
``status --porcelain`` output.  Reproduces real bugs:

* ``diff_merge._capture_untracked``: returned the quoted string, which
  does not exist on disk, making the file invisible to callers.
* ``merge_flow._main_dirty_files``: ``.strip('"')`` removed the quotes
  but never unescaped ``\\"`` / ``\\\\``.
* ``merge_flow._get_worktree_changed_files`` and
  ``_check_merge_conflict``: ``--name-only`` parse sites kept the
  quoted form, so changedFiles showed bogus names and the conflict
  file-overlap sets could never intersect the real dirty paths.

No mocks, patches, fakes, or test doubles.  Real git repositories,
real worktrees, real server mixin methods.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.diff_merge import _capture_untracked
from kiss.server.server import VSCodeServer

QUOTED_NAME = 'qu"ote.txt'


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def _make_repo(tmpdir: str) -> str:
    repo = str(Path(tmpdir) / "repo")
    Path(repo).mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


class TestQuotedPathParsing(unittest.TestCase):
    """diff_merge parse sites must return real (unquoted) file names."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-quoted-")
        self.repo = _make_repo(self.tmpdir)
        Path(self.repo, "a.txt").write_text("one\ntwo\nthree\n")
        Path(self.repo, QUOTED_NAME).write_text("alpha\nbravo\ncharlie\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_capture_untracked_unquotes(self) -> None:
        """Untracked files with quotes in the name must be reported as-is."""
        Path(self.repo, 'new"file.txt').write_text("x\n")
        untracked = _capture_untracked(self.repo)
        self.assertIn('new"file.txt', untracked, f"got: {untracked}")

    def test_main_dirty_files_unquotes(self) -> None:
        """_main_dirty_files must unescape git's C-quoted porcelain paths."""
        Path(self.repo, QUOTED_NAME).write_text("alpha\nBRAVO\ncharlie\n")
        server = VSCodeServer()
        server.work_dir = self.repo
        dirty = server._main_dirty_files(self.repo)
        self.assertIn(QUOTED_NAME, dirty, f"got: {dirty}")


class TestWorktreeQuotedPaths(unittest.TestCase):
    """merge_flow worktree queries must survive quoted file names."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-quoted-")
        self.repo = _make_repo(self.tmpdir)
        content = "".join(f"line{i}\n" for i in range(1, 31))
        Path(self.repo, QUOTED_NAME).write_text(content)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

        self.tab_id = "t-wt-quoted"
        self.agent = WorktreeSorcarAgent("wt-quoted-test")
        self.state = agent_state.AgentState(
            "task-wt-quoted",
            agent=self.agent,
            tab_id=self.tab_id,
            server_owned=True,
        )
        self.state.use_worktree = True
        agent_state.register(self.state)
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
        agent_state.unregister(self.state.task_id, self.state)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_changed_files_lists_real_name(self) -> None:
        """changedFiles must contain the real (unquoted) file name."""
        wt_file = Path(self.wt_dir, QUOTED_NAME)
        wt_file.write_text(
            wt_file.read_text().replace("line5\n", "line5-agent\n"),
        )
        changed = self.server._get_worktree_changed_files(self.tab_id)
        self.assertIn(QUOTED_NAME, changed, f"got: {changed}")

    def test_conflict_check_sees_quoted_overlap(self) -> None:
        """A dirty main edit of the same quoted file must flag a conflict."""
        wt_file = Path(self.wt_dir, QUOTED_NAME)
        wt_file.write_text(
            wt_file.read_text().replace("line5\n", "line5-agent\n"),
        )
        main_file = Path(self.repo, QUOTED_NAME)
        main_file.write_text(
            main_file.read_text().replace("line9\n", "line9-user\n"),
        )
        self.assertTrue(
            self.server._check_merge_conflict(self.tab_id),
            "conflict check missed the overlap because the worktree "
            "side kept git's C-quoted name while the dirty-main side "
            "used the real name",
        )


if __name__ == "__main__":
    unittest.main()

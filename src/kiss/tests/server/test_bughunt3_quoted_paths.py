# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: git C-quoted paths must be unquoted everywhere.

Even with ``core.quotepath=false``, git C-quotes any path containing a
double-quote, backslash, or control character — in ``diff --git``
headers, ``--name-only`` output, ``ls-files --others`` output, and
``status --porcelain`` output.  Reproduces a real bug: the merge
flow's changed-file listing kept the quoted form of a changed or an
untracked file, a name that does not exist on disk.

No mocks, patches, fakes, or test doubles.  Real git repositories,
real worktrees, real server mixin methods.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.conftest import posix_only
from kiss.tests.server.worktree_helpers import worktree_changed_files

# With ``core.quotepath=false`` (which the project sets) git C-quotes only
# names containing ``"``, ``\\`` or control characters; none of those is a
# legal NTFS file-name character, so the fixture cannot exist on Windows.
pytestmark = posix_only("git C-quoted file names are impossible on NTFS")

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


class TestWorktreeQuotedPaths(unittest.TestCase):
    """merge_flow worktree queries must survive quoted file names."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-quoted-")
        self.repo = _make_repo(self.tmpdir)
        content = "".join(f"line{i}\n" for i in range(1, 31))
        Path(self.repo, QUOTED_NAME).write_text(content)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")

        self.server = AgentServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

        self.tab_id = "t-wt-quoted"
        self.agent = ChatSorcarAgent("wt-quoted-test")
        self.state = agent_state.AgentState(
            "task-wt-quoted",
            agent=self.agent,
            tab_id=self.tab_id,
            server_owned=True,
        )
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
        """The listing must hold the real (unquoted) names, an untracked file's too."""
        wt_file = Path(self.wt_dir, QUOTED_NAME)
        wt_file.write_text(
            wt_file.read_text().replace("line5\n", "line5-agent\n"),
        )
        Path(self.wt_dir, 'new"file.txt').write_text("x\n")
        changed = worktree_changed_files(self.server, self.tab_id)
        self.assertEqual(changed, ['new"file.txt', QUOTED_NAME])


if __name__ == "__main__":
    unittest.main()

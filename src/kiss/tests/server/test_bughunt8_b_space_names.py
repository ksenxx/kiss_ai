# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group B): filenames with leading/trailing spaces are
mangled by over-eager whitespace stripping.

The merge flow's changed-file listing stripped whitespace from git's
output: ``line.strip()`` on every ``git ls-files --others`` line and
``output.strip()`` on the whole ``git diff --name-only`` output.  A
file whose name begins or ends with a space (legal on every POSIX
filesystem, and NOT C-quoted by git: space is a printable character)
was therefore listed under a name that does not exist on disk.

No mocks, patches, fakes, or test doubles: real git repositories, a
real ``ChatSorcarAgent`` worktree, real server mixin methods.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer
from kiss.tests.conftest import IS_WINDOWS
from kiss.tests.server.worktree_helpers import worktree_changed_files


def _git(cwd: str | Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout


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

        self.server = AgentServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

        self.tab_id = "t-wt-space"
        self.agent = ChatSorcarAgent("wt-space-test")
        self.state = AgentState(
            "task-t-wt-space",
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

    def test_changed_files_keep_leading_space(self) -> None:
        """An edited `` lead.txt`` and new untracked files must be listed
        under their exact on-disk paths, not strip()-mangled ones.

        Win32 silently strips trailing spaces when creating a file
        (``"trail "`` lands on disk as ``trail``), so only the
        leading-space names can be verified on Windows.
        """
        target = Path(self.wt_dir, " lead.txt")
        target.write_text("line1\nline2-agent\n")
        untracked = [" new.txt"] if IS_WINDOWS else [" new.txt", "trail "]
        for name in untracked:
            Path(self.wt_dir, name).write_text("agent\n")

        changed = worktree_changed_files(self.server, self.tab_id)
        self.assertEqual(changed, sorted([" lead.txt", *untracked]))


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group B): autocommit misreports a failed ``git add``.

``_MergeFlowMixin._handle_autocommit_action`` ignored the exit status
of ``git add -A``.  When staging fails — e.g. ``.git/index.lock`` is
held by another git process (editor, GUI client, crashed git) — the
subsequent ``git diff --cached`` is empty, so the handler broadcast
``autocommit_done`` with ``success=True`` and the message
``"Nothing to commit."`` even though the working tree HAS uncommitted
changes and nothing was committed because staging itself failed.  The
user is told their tree is clean when it is not.

No mocks, patches, fakes, or test doubles: a real git repository with
a real ``index.lock`` file, a real ``VSCodeServer``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer


def _git(cwd: str | Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout


class TestAutocommitStagingFailure(unittest.TestCase):
    """A failed ``git add -A`` must be reported as a failure."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt8-lock-")
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "t")
        (self.repo / "f.txt").write_text("one\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "init")

        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]
        self.tab_id = "t-ac-lock"

    def tearDown(self) -> None:
        _RunningAgentState.running_agent_states.pop(self.tab_id, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_index_lock_reports_failure_not_nothing_to_commit(self) -> None:
        """index.lock held by another process: the reply must not claim
        success / a clean tree while dirty changes remain unstaged."""
        (self.repo / "f.txt").write_text("one\ntwo\n")
        # Simulate a concurrent git process holding the index lock.
        (self.repo / ".git" / "index.lock").write_text("")
        try:
            self.server._handle_autocommit_action(
                "commit", self.tab_id, work_dir=str(self.repo),
            )
        finally:
            (self.repo / ".git" / "index.lock").unlink()

        done = [e for e in self.events if e.get("type") == "autocommit_done"]
        self.assertEqual(len(done), 1, self.events)
        self.assertFalse(
            done[0]["success"],
            "staging failed (index.lock) but autocommit_done claimed "
            f"success: {done[0]}",
        )
        self.assertFalse(done[0]["committed"])
        self.assertNotIn(
            "Nothing to commit", done[0]["message"],
            "misleading clean-tree message despite dirty working tree: "
            f"{done[0]}",
        )
        # The user's dirty change is still there, unstaged.
        self.assertIn("two", (self.repo / "f.txt").read_text())


if __name__ == "__main__":
    unittest.main()

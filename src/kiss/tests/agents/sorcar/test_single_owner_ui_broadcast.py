# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for single-owner blocking-UI broadcasts.

All frontend clients are mirror copies of each other: every client
shows the same tabs and their contents.  A blocking UI the daemon
opens for a chat (the post-task auto-commit progress strip, the
worktree merge/discard strip) therefore needs exactly ONE broadcast
copy, stamped with the single owning ``tabId`` — every client renders
that same tab.  There is no per-viewer fan-out and no ``mirrorOf``
stamping of viewer copies.

These tests drive the real backend: a real git repository, the real
``_autocommit_changes`` / ``_present_pending_worktree`` / worktree
code paths, and commands entering through ``_handle_command`` exactly
as the transports deliver them.  Output is observed through
``MemoryPrinter`` — a real :class:`JsonPrinter` subclass that records
every broadcast event (explicit-``tabId`` events are captured exactly
once, verbatim).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.server.merge_flow as merge_flow_module
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter

TASK_ID = "task-77"
OWNER_TAB = "owner-tab"
VIEWER_TAB = "viewer-tab"


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in *cwd* and return the completed process."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    """Create a git repo with one committed file so HEAD exists."""
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "README.md").write_text("# Hello\n\nSome content\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "initial commit")


class TestSingleOwnerUiBroadcast(unittest.TestCase):
    """Blocking UIs broadcast exactly one owner-stamped copy."""

    def setUp(self) -> None:
        """Stand up a real repo, a server and two subscribed tabs."""
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = self.tmpdir
        self.owner = agent_state.AgentState(
            TASK_ID, tab_id=OWNER_TAB, server_owned=True,
        )
        agent_state.register(self.owner)
        # A second subscribed tab must NOT cause a second copy of any
        # blocking-UI event: those are pre-addressed to the owner tab.
        self.printer.subscribe_tab(TASK_ID, OWNER_TAB)
        self.printer.subscribe_tab(TASK_ID, VIEWER_TAB)
        self._orig_gen = merge_flow_module.generate_commit_message_from_diff
        merge_flow_module.generate_commit_message_from_diff = (  # type: ignore[assignment]
            lambda diff_text, user_prompt=None, task_result=None: (
                "chore: auto-commit test"
            )
        )

    def tearDown(self) -> None:
        """Undo the module patch and drop all per-test global state."""
        merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _events_of(self, event_type: str) -> list[dict[str, Any]]:
        """Return every emitted event of *event_type*."""
        return [
            ev for ev in self.printer.emitted if ev.get("type") == event_type
        ]

    def _assert_single_owner_copy(
        self, event_type: str, owner: str = OWNER_TAB,
    ) -> dict[str, Any]:
        """Assert exactly one *event_type* was emitted, stamped *owner*."""
        events = self._events_of(event_type)
        self.assertEqual(
            len(events), 1,
            f"expected exactly ONE {event_type!r} broadcast copy, got "
            f"{len(events)}: {events}",
        )
        self.assertEqual(
            events[0].get("tabId"), owner,
            f"{event_type!r} must be stamped with the owning tabId",
        )
        return events[0]

    def _assert_no_mirror_of(self) -> None:
        """Assert no emitted event carries a ``mirrorOf`` field."""
        mirrored = [ev for ev in self.printer.emitted if "mirrorOf" in ev]
        self.assertEqual(
            mirrored, [],
            "no event may carry a mirrorOf field: viewer copies do not "
            "exist under the mirror-clients model",
        )

    def test_autocommit_done_broadcasts_one_owner_copy(self) -> None:
        """Post-task autocommit commits and emits one ``autocommit_done``."""
        Path(self.tmpdir, "README.md").write_text(
            "# Hello\n\nUpdated content by the agent\n",
        )
        self.server._autocommit_changes(OWNER_TAB, work_dir=self.tmpdir)

        done = self._assert_single_owner_copy("autocommit_done")
        self._assert_no_mirror_of()
        self.assertTrue(done["committed"], done["message"])
        self.assertEqual(done.get("commitMessage"), "chore: auto-commit test")
        status = _git(self.tmpdir, "status", "--porcelain")
        self.assertEqual(status.stdout.strip(), "")

    def test_autocommit_progress_stamped_with_owner_tab(self) -> None:
        """Every ``autocommit_progress`` step carries the owning tabId."""
        Path(self.tmpdir, "README.md").write_text(
            "# Hello\n\nUpdated content by the agent\n",
        )
        self.server._autocommit_changes(OWNER_TAB, work_dir=self.tmpdir)

        progress = self._events_of("autocommit_progress")
        self.assertTrue(progress, "autocommit must report progress steps")
        seen_messages = {ev["message"] for ev in progress}
        self.assertEqual(
            len(progress), len(seen_messages),
            f"each progress step must be broadcast exactly once: {progress}",
        )
        for ev in progress:
            self.assertEqual(
                ev.get("tabId"), OWNER_TAB,
                "autocommit_progress must be stamped with the owning tabId",
            )
        self._assert_no_mirror_of()

    def _make_pending_worktree(self) -> GitWorktree:
        """Create a real worktree with an agent commit, owned by the tab."""
        repo = Path(self.tmpdir)
        branch = "kiss/wt-single-owner-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        self.assertTrue(GitWorktreeOps.create(repo, branch, wt_dir))
        GitWorktreeOps.save_original_branch(repo, branch, "main")
        (wt_dir / "agent.txt").write_text("agent produced this\n")
        _git(str(wt_dir), "add", ".")
        _git(str(wt_dir), "commit", "-q", "-m", "agent")
        wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._wt = wt
        self.owner.agent = cast(WorktreeSorcarAgent, agent)
        self.owner.use_worktree = True
        return wt

    def test_worktree_strip_broadcasts_one_owner_copy(self) -> None:
        """``worktree_done`` and ``worktree_result`` are emitted once."""
        self._make_pending_worktree()
        self.server._present_pending_worktree(OWNER_TAB, discard_if_empty=False)

        done = self._assert_single_owner_copy("worktree_done")
        self.assertEqual(done.get("changedFiles"), ["agent.txt"])
        self._assert_no_mirror_of()

        self.server._handle_command({
            "type": "worktreeAction",
            "action": "discard",
            "tabId": OWNER_TAB,
            "workDir": self.tmpdir,
        })
        result = self._assert_single_owner_copy("worktree_result")
        self.assertTrue(result.get("success"), result)
        self._assert_no_mirror_of()


if __name__ == "__main__":
    unittest.main()

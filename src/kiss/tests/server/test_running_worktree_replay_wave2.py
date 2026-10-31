# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A tab attached to a running worktree task tracks that task's worktree.

A running task broadcasts its ``worktree_created`` once, stamped with
the tabs watching it then, and never records or persists it.  A session
replay of the task's tab first drops the tab's tracked worktree
(``WebPrinter.cleanup_tab``), and nothing re-presented it while the task
ran: remote ``checkPaths`` / ``openFile`` resolved the transcript's
relative paths against the main checkout, and ``gitStatus`` reported no
task worktree, until the task's closing ``worktree_done``.  A tab that
started viewing the running task later never learned the worktree at
all.  Attaching a tab to a running task now re-sends the task's current
worktree to it.

Everything is real: a git repository, a worktree created by a
``ChatSorcarAgent``, the daemon's ``WebPrinter`` and the server's
``resumeSession`` command.  The running task is a registered agent
state marked active: an agent holding a worktree needs no model call.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer
from kiss.server.web_server import WebPrinter
from kiss.tests.server.conftest import (
    init_repo,
    redirect_persistence,
    restore_persistence,
)


class TestRunningWorktreeReplay(unittest.TestCase):
    """Every attach to a running worktree task re-sends its worktree."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-running-wt-replay-")
        self.saved = redirect_persistence(self.tmpdir)
        self.repo = str(Path(self.tmpdir) / "repo")
        init_repo(self.repo)
        self.printer = WebPrinter()
        self.server = AgentServer(self.printer)
        self.server.work_dir = self.repo
        task_id, self.chat_id = th._add_task("change things in a worktree")
        self.agent = ChatSorcarAgent("agent")
        self.state = AgentState(
            str(task_id), agent=self.agent, chat_id=self.chat_id,
            tab_id="tab-w", is_task_active=True,
        )
        agent_state.register(self.state)

    def tearDown(self) -> None:
        self.state.is_task_active = False
        agent_state.unregister(self.state.task_id, self.state)
        if self.agent._wt_pending:
            self.agent.discard()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_worktree(self) -> str:
        """Isolate the agent in a worktree of the repo; return its work dir."""
        self.agent._enter_work_dir(self.repo)
        message = self.agent.create_worktree()
        self.assertTrue(self.agent._wt_pending, message)
        return str(self.agent._wt_work_dir)

    def _resume(self, tab_id: str, **extra: object) -> None:
        self.server._handle_command({
            "type": "resumeSession", "chatId": self.chat_id, "tabId": tab_id, **extra,
        })

    def test_replay_restores_the_running_tasks_worktree(self) -> None:
        wt_work_dir = self._create_worktree()
        # The tab tracked the worktree from the live event; the replay
        # drops that entry before re-subscribing the tab.
        self.printer.broadcast({
            "type": "worktree_created", "worktreeDir": wt_work_dir, "tabId": "tab-w",
        })
        self._resume("tab-w")
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-w"), wt_work_dir)

    def test_conn_scoped_replay_restores_the_running_tasks_worktree(self) -> None:
        wt_work_dir = self._create_worktree()
        self._resume("tab-w", replayConnId="conn-1")
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-w"), wt_work_dir)

    def test_deferred_viewer_tab_learns_the_running_tasks_worktree(self) -> None:
        wt_work_dir = self._create_worktree()
        self._resume("tab-v", deferred=True, replayConnId="conn-1")
        self.assertIn("tab-v", self.printer._fanout_targets(self.state.task_id))
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-v"), wt_work_dir)

    def test_replay_after_the_merge_tracks_no_worktree(self) -> None:
        wt_work_dir = self._create_worktree()
        self._resume("tab-w")
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-w"), wt_work_dir)
        self.agent.discard()
        self._resume("tab-w")
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-w"), "")

    def test_running_task_without_an_agent_tracks_no_worktree(self) -> None:
        self.state.agent = None
        self._resume("tab-w")
        self.assertIn("tab-w", self.printer._fanout_targets(self.state.task_id))
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-w"), "")


if __name__ == "__main__":
    unittest.main()

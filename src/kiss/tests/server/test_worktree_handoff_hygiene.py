# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Worktree paths and hand-offs: fewer dead first turns (P10 of the 2026-10-24 audit).

A continuation's first message names the active worktree (unmerged)
and the files written so far; a merge blocked by an unstashable main
checkout names the blocking files; a successful merge copies the
worktree's ``tmp/PROGRESS.md`` to the main checkout.
"""

from __future__ import annotations

import os
import unittest

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.conftest import IS_WINDOWS
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome


class TestHandoffHygiene(unittest.TestCase):
    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-worktree-handoff-")
        self.repo = self.home.repo
        self.agent = ChatSorcarAgent("t")
        self.agent.work_dir = str(self.repo)
        self.agent._run_tool_names = frozenset({"create_worktree"})

    def tearDown(self) -> None:
        if self.agent._wt_pending:
            self.agent.discard()
        self.home.cleanup()

    def test_continuation_state_names_the_worktree_and_the_written_files(self) -> None:
        self.assertEqual(self.agent._continuation_state(), "")
        self.agent.create_worktree()
        (wt,) = self.agent.worktrees
        tools = UsefulTools(work_dir=str(wt.wt_dir))
        tools.Write(str(wt.wt_dir / "new.txt"), "x\n")
        self.agent._useful_tools = tools
        state = self.agent._continuation_state()
        self.assertIn("# Current state", state)
        self.assertIn(
            f"Active worktree (unmerged; call commit_and_merge when done): {wt.wt_dir}", state
        )
        self.assertIn(f"on branch {wt.branch} of {wt.repo_root}", state)
        self.assertIn(f"Files written so far: {wt.wt_dir / 'new.txt'}", state)
        # A later session's tools instance adds to, not replaces, the list.
        later = UsefulTools(work_dir=str(wt.wt_dir))
        later.Write(str(wt.wt_dir / "other.txt"), "y\n")
        self.agent._useful_tools = later
        state = self.agent._continuation_state()
        self.assertIn("new.txt", state)
        self.assertIn("other.txt", state)

    def test_merge_copies_the_progress_log_to_the_main_checkout(self) -> None:
        self.agent.create_worktree()
        (wt,) = self.agent.worktrees
        (wt.wt_dir / "a.txt").write_text("A\n")
        (wt.wt_dir / "tmp").mkdir(exist_ok=True)
        (wt.wt_dir / "tmp" / "PROGRESS.md").write_text("# Progress\n- did A\n")
        msg = self.agent.merge()
        self.assertIn(f"Successfully merged branch '{wt.branch}'", msg)
        self.assertIn("Copied tmp/PROGRESS.md to", msg)
        self.assertEqual((self.repo / "tmp" / "PROGRESS.md").read_text(), "# Progress\n- did A\n")
        self.assertEqual((self.repo / "a.txt").read_text(), "A\n")

    @pytest.mark.skipif(IS_WINDOWS or os.geteuid() == 0, reason="needs an unreadable file")
    def test_unstashable_main_checkout_names_the_blocking_files(self) -> None:
        self.agent.create_worktree()
        (wt,) = self.agent.worktrees
        (wt.wt_dir / "a.txt").write_text("A\n")
        # An untracked, unreadable file makes ``git stash push
        # --include-untracked`` fail on the main checkout.
        blocker = self.repo / "secret.bin"
        blocker.write_bytes(b"x")
        blocker.chmod(0)
        try:
            msg = self.agent.merge()
        finally:
            blocker.chmod(0o644)
        self.assertIn("could not be stashed", msg)
        self.assertIn(f"'{wt.branch}' is committed and safe", msg)
        self.assertIn("Blocking files (git status --porcelain of the main checkout):", msg)
        self.assertIn("?? secret.bin", msg)
        self.assertIn("retry commit_and_merge", msg)
        self.assertEqual([w.branch for w in self.agent.worktrees], [wt.branch])

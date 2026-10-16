# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A pending worktree must survive the tab's next run.

R09-1.  A task whose agent called ``create_worktree`` but never
``commit_and_merge`` leaves its ``kiss/wt-*`` branch pending and the
chat page shows the Merge / Discard bar.  If the user clicks neither
button and submits the next task on the same tab, the server must not
lose the worktree: before the new run starts the pending worktree is
parked on its branch (``_retire_previous_worktree``: the work is
committed, the directory is removed, a preserve marker is written) and
the branch is never squash-merged into the user's branch behind their
back — not by the retire, and not by the orphan-reclaim sweep a later
``create_worktree`` in the repo runs.

Everything here is real: a real git repository, a real linked worktree
created by the agent's own ``create_worktree`` tool, real ``run``
commands dispatched through ``AgentServer._run_task`` on real worker
threads, and the real reclaim sweep.  The model is a local stand-in
endpoint scripting each task's tool calls.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import kiss_wt_branches, run_git
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    OfflineFastModel,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)


class TestPendingWorktreeSurvivesNextRun(unittest.TestCase):
    """The next run must never silently orphan a pending worktree."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-wt-orphan-")
        self.repo = str(self.home.repo)
        self.home.write_config(max_budget=5.0, use_web_browser=False)
        self.offline = OfflineFastModel()
        self.offline.__enter__()
        self.standin = StandInModelServer(self._respond)
        self.printer = CapturePrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = self.repo
        # Per task prompt: how many model calls have been answered.
        self.calls: dict[str, int] = {}

    def tearDown(self) -> None:
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        self.standin.stop()
        self.offline.__exit__(None, None, None)
        self.home.cleanup()

    # -- helpers ---------------------------------------------------

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """Script one task: ``create_worktree``, write the file, finish.

        A prompt ``write <name>`` makes the agent create a worktree
        and write ``<name>`` into it; the worktree is never merged, so
        the first ``finish`` is refused by the worktree gate and the
        second one ends the task.  A prompt ``nothing`` finishes at
        once, leaving the tree untouched.
        """
        text = request_text(request)
        # The chat history carries earlier prompts: the latest decides.
        marker = text.rsplit("TASK:", 1)[1].split("\n", 1)[0].strip()
        n = self.calls.get(marker, 0)
        self.calls[marker] = n + 1
        if marker == "nothing":
            return finish_response("nothing to do")
        filename = marker.removeprefix("write ")
        if n == 0:
            return tool_call_response("create_worktree", {})
        if n == 1:
            return tool_call_response("Bash", {
                "command": f"printf 'agent output\\n' > {filename}",
                "description": f"write {filename}",
            })
        return finish_response(f"wrote {filename}")

    def _run(self, tab_id: str, prompt: str) -> None:
        """Dispatch a real ``run`` command and wait for it to finish."""
        before = len(self._status_ends(tab_id))
        self.server._run_task({
            "type": "run",
            "prompt": f"TASK: {prompt}",
            "workDir": self.repo,
            "tabId": tab_id,
            "model": STANDIN_MODEL,
            "modelConfig": self.standin.model_config,
            "isParallel": False,
            "useWebTools": False,
            "classifyTasks": False,
            "maxBudget": 5.0,
        })
        deadline = time.time() + 60
        while time.time() < deadline:
            if len(self._status_ends(tab_id)) > before:
                return
            time.sleep(0.01)
        raise AssertionError(f"task on tab {tab_id!r} never finished")

    def _status_ends(self, tab_id: str) -> list[dict[str, Any]]:
        """Recorded ``status running=False`` events for *tab_id*."""
        return [
            ev
            for ev in self.printer.events_of_type("status")
            if ev.get("running") is False and ev.get("tabId") == tab_id
        ]

    def _pending_branch(self, tab_id: str) -> str:
        """The ``kiss/wt-*`` branch the tab's agent currently owns."""
        state = agent_state.find_by_tab(tab_id)
        assert state is not None, f"no state for tab {tab_id!r}"
        agent = state.agent
        assert agent is not None and agent._wt_pending, (
            "the first run must leave a pending worktree; "
            f"agent={agent!r}"
        )
        return str(agent._wt_branch)

    def _pending_worktree_dir(self, tab_id: str) -> Path:
        """The worktree directory the tab's agent currently owns."""
        state = agent_state.find_by_tab(tab_id)
        assert state is not None and state.agent is not None
        wt_dir = state.agent._wt_dir
        assert wt_dir is not None
        return wt_dir

    def _branch_files(self, branch: str) -> list[str]:
        """Paths committed at the tip of *branch*."""
        return run_git(
            self.repo, "show", "--name-only", "--pretty=format:", branch,
        ).stdout.split()

    # -- tests -----------------------------------------------------

    def test_first_run_leaves_the_worktree_pending(self) -> None:
        """The task's work stays in its unmerged worktree, never on main."""
        tab_id = "tab-pending"
        self._run(tab_id, "write declined.txt")

        branch = self._pending_branch(tab_id)
        wt_dir = self._pending_worktree_dir(tab_id)
        assert branch in kiss_wt_branches(self.repo)
        assert Path(wt_dir, "declined.txt").read_text() == "agent output\n"
        assert not Path(self.repo, "declined.txt").exists()
        assert run_git(self.repo, "status", "--porcelain").stdout == ""
        created = self.printer.events_of_type("worktree_created")
        assert [e.get("branch") for e in created] == [branch]
        bars = self.printer.events_of_type("worktree_done")
        assert [b.get("branch") for b in bars] == [branch], (
            "the Merge / Discard bar was not presented for the pending worktree"
        )
        assert bars[0].get("changedFiles") == ["declined.txt"]

    def test_next_run_parks_the_pending_worktree_on_its_branch(self) -> None:
        """The declined worktree is committed to its branch, not merged.

        "Parked" is the preserve-for-review contract: the work is
        committed onto the ``kiss/wt-*`` branch, the worktree directory
        is removed and pruned, the branch carries a preserve marker so
        no reclaim sweep publishes it, and the user is told where the
        work went.  What must never happen is the third state —
        directory and branch still there with nobody holding them —
        or the work landing on the user's branch.
        """
        tab_id = "tab-orphan"
        self._run(tab_id, "write declined.txt")
        branch = self._pending_branch(tab_id)
        wt_dir = self._pending_worktree_dir(tab_id)
        head_before = run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

        self._run(tab_id, "nothing")

        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        assert state.agent is None or not state.agent._wt_pending, (
            "the retire left the previous worktree pending on the agent"
        )
        assert not wt_dir.exists(), (
            f"worktree directory {wt_dir} was left on disk after the retire"
        )
        registered = [
            b for _d, b in GitWorktreeOps.registered_worktrees(Path(self.repo))
        ]
        assert branch not in registered, (
            f"branch {branch!r} is still a registered worktree with no owner"
        )
        assert branch in kiss_wt_branches(self.repo), (
            f"branch {branch!r} was deleted; the declined work is lost"
        )
        assert "declined.txt" in self._branch_files(branch), (
            f"branch {branch!r} survived but no longer carries the "
            f"declined work: {self._branch_files(branch)}"
        )
        assert GitWorktreeOps.load_preserve_marker(Path(self.repo), branch), (
            f"branch {branch!r} has no preserve marker; a reclaim sweep "
            "could publish it"
        )
        assert run_git(self.repo, "rev-parse", "HEAD").stdout.strip() == (
            head_before
        ), "the retire merged the declined work into the user's branch"
        assert not Path(self.repo, "declined.txt").exists()
        warnings = [
            e.get("message", "") for e in self.printer.events_of_type("warning")
        ]
        assert any(branch in w and "not merged" in w for w in warnings), (
            f"the user was not told where the work went: {warnings}"
        )

    def test_declined_work_is_not_published_by_the_next_worktree_task(
        self,
    ) -> None:
        """The parked branch must not reach the user's branch via reclaim."""
        tab_id = "tab-publish"
        self._run(tab_id, "write declined.txt")
        branch = self._pending_branch(tab_id)

        self._run(tab_id, "nothing")
        assert run_git(self.repo, "status", "--porcelain").stdout == "", (
            "the main tree must be clean, otherwise the reclaim sweep "
            "declines for that reason and the regression is masked"
        )

        self._run("tab-other", "write third.txt")

        tracked = run_git(self.repo, "ls-files").stdout.split()
        assert "declined.txt" not in tracked, (
            f"work the user declined to merge (branch {branch!r}) was "
            "auto-published onto their branch by the orphan reclaim sweep"
        )
        assert not Path(self.repo, "declined.txt").exists(), (
            "the declined worktree's file reached the main working tree"
        )
        assert branch in kiss_wt_branches(self.repo), (
            f"the reclaim sweep deleted the parked branch {branch!r}"
        )

    def test_next_worktree_run_still_owns_a_worktree(self) -> None:
        """A second ``create_worktree`` on the tab gets a fresh branch.

        The carried-over worktree is retired before the second run
        starts (parked on its own branch, directory removed) and the
        tab ends up owning exactly the new run's worktree; the retired
        one is never left registered without an owner.
        """
        tab_id = "tab-chain"
        self._run(tab_id, "write first.txt")
        first_branch = self._pending_branch(tab_id)
        first_dir = self._pending_worktree_dir(tab_id)

        self._run(tab_id, "write second.txt")
        second_branch = self._pending_branch(tab_id)
        second_dir = self._pending_worktree_dir(tab_id)

        assert second_branch != first_branch, (
            "the second worktree run reused the first run's branch"
        )
        assert not first_dir.exists()
        registered = [
            b for _d, b in GitWorktreeOps.registered_worktrees(Path(self.repo))
        ]
        assert first_branch not in registered, (
            f"the retired branch {first_branch!r} is still a registered worktree"
        )
        assert second_branch in registered, (
            "the tab must own the new run's worktree"
        )
        assert "first.txt" in self._branch_files(first_branch)
        assert GitWorktreeOps.load_preserve_marker(
            Path(self.repo), first_branch,
        )
        assert Path(second_dir, "second.txt").exists()
        assert not Path(second_dir, "first.txt").exists(), (
            "the new worktree inherited the retired run's work"
        )

    def test_changed_files_probe_ignores_unknown_tabs(self) -> None:
        """A tab with no state has no worktree changes to report."""
        assert self.server._get_worktree_changed_files("no-such-tab") == []


if __name__ == "__main__":
    unittest.main()

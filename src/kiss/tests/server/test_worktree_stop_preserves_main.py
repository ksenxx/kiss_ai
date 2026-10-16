# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: user-stopped worktree task must not silently overwrite main.

User-reported bug ("the lost slides bug"):

    The user ran a task that was asked to update a binary ``.pptx``
    file in place.  The agent had created a worktree and already
    written partial bytes to the ``.pptx`` inside it when the user
    clicked "Stop".  Later, when the user closed the chat tab (or VS
    Code reloaded), the partial pptx was silently squash-merged into
    the main branch — destroying the user's previous, complete deck.

Root cause (historical):

    On tab close, ``_teardown_tab_resources`` released the worktree by
    auto-committing the partial work and silently squash-merging it
    into the original branch.  No user confirmation.

Fix:

    A worktree the agent never ``commit_and_merge``d is pending at task
    end, and a tab close retires it by committing the partial work onto
    the worktree branch but NOT merging — the branch survives in ``git
    branch`` so the user can recover via ``git checkout <branch>``
    later, and the main branch is untouched.

This is an end-to-end test: a real ``run`` command goes through
``AgentServer._handle_command`` against a real on-disk git repo, the
model is a local stand-in endpoint whose script calls the agent's own
``create_worktree`` tool, writes partial pptx bytes with a real Bash
call and then stops the task through the server's ``_stop_task`` (the
user clicking Stop).  Then ``server._close_tab`` is called to simulate
the user closing the chat tab.  Assertions are made on the git repo's
on-disk state (branch list, file contents on main vs on the wt branch).
No mocks or test doubles.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from typing import Any

from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import run_git
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    OfflineFastModel,
    StandInModelServer,
    finish_response,
    tool_call_response,
)

_GOOD_PPTX_BYTES: bytes = b"GOOD-DECK\x00" + os.urandom(8192)
_PARTIAL_PREFIX: bytes = b"PARTIAL\x00"
_PARTIAL_RANDOM_LEN = 4096


def _seed_pptx(repo: str) -> None:
    """Commit the binary "good" ``slides.pptx`` on the scratch repo.

    Mirrors the user's pre-Stop state: the branch has the "good" pptx,
    and the workspace is clean.
    """
    Path(repo, "slides.pptx").write_bytes(_GOOD_PPTX_BYTES)
    run_git(repo, "add", "slides.pptx")
    run_git(repo, "commit", "-q", "-m", "seed: add good slides.pptx")


def _list_kiss_wt_branches(repo: str) -> list[str]:
    """Return all ``kiss/wt-*`` branches still present in *repo*."""
    result = run_git(repo, "branch", "--list", "kiss/wt-*")
    return [
        line.strip().lstrip("+* ").strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _file_bytes_on_branch(repo: str, branch: str, path: str) -> bytes:
    """Read *path* from *branch*'s tip without checking it out."""
    result = subprocess.run(
        ["git", "-C", repo, "show", f"{branch}:{path}"],
        capture_output=True, check=False,
    )
    return result.stdout



class _WorktreeStopBase(unittest.TestCase):
    """Fresh git repo + isolated KISS home + stand-in model per test."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-wt-stop-test-")
        self.repo = str(self.home.repo)
        _seed_pptx(self.repo)
        self.home.write_config(max_budget=5.0, use_web_browser=False)
        self.offline = OfflineFastModel()
        self.offline.__enter__()
        self.standin = StandInModelServer(self._respond)
        self.printer = CapturePrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = self.repo
        self.tab_id = "tab-stop-test"
        self.model_calls = 0

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

    def _types(self) -> list[str]:
        with self.printer._capture_lock:
            return [e.get("type", "") for e in self.printer.captured]

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """Script the agent: worktree, partial pptx write, then Stop.

        The third model call arrives after the Bash tool has written
        the partial bytes; the user clicks Stop at that moment, so the
        answer the model gives is never acted on.
        """
        self.model_calls += 1
        if self.model_calls == 1:
            return tool_call_response("create_worktree", {})
        if self.model_calls == 2:
            return tool_call_response("Bash", {
                "command": (
                    "{ printf 'PARTIAL\\0'; head -c "
                    f"{_PARTIAL_RANDOM_LEN} /dev/urandom; }} > slides.pptx"
                ),
                "description": "write partial slides.pptx",
            })
        self.server._stop_task(self.tab_id)
        return finish_response("the deck is updated")



class TestUserStopPreservesMainBranch(_WorktreeStopBase):
    """The "lost slides" bug: user-Stop must not silently overwrite main.

    When the user clicks Stop on a task that is working in a worktree
    and later closes the chat tab, the main branch's binary file must
    NOT have been silently overwritten by the agent's partial work,
    and the partial work must remain recoverable on the ``kiss/wt-*``
    branch.
    """

    def test_close_tab_after_stop_does_not_overwrite_main_pptx(
        self,
    ) -> None:
        """End-to-end repro of the "lost slides" bug.

        Sequence:
            1. Real git repo with committed good ``slides.pptx``.
            2. A real ``run`` command is driven through
               ``_handle_command``; the scripted model calls
               ``create_worktree``, writes partial bytes to
               ``slides.pptx`` with Bash, and the user clicks Stop
               (``_stop_task``) while the model answers next; the
               worker thread is joined.
            3. ``server._close_tab(tab_id)`` simulates the user
               closing the chat tab afterward — the point where the
               buggy code silently squash-merged the partial work.

        After the fix:
            - Main branch's ``slides.pptx`` MUST still equal the
              original "good" bytes (the user's prior work is intact).
            - A ``kiss/wt-*`` branch MUST still exist with the
              partial bytes at its tip, so the user can recover the
              in-flight work via ``git checkout <branch>`` if they
              want it.
        """
        tab_id = self.tab_id
        self.server._handle_command({
            "type": "run",
            "prompt": "update the slides",
            "workDir": self.repo,
            "tabId": tab_id,
            "model": STANDIN_MODEL,
            "modelConfig": self.standin.model_config,
            "isParallel": False,
            "useWebTools": False,
            "classifyTasks": False,
            "maxBudget": 5.0,
        })
        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        thread = state.task_thread
        assert thread is not None
        thread.join(timeout=60)
        assert not thread.is_alive(), "the task never finished"

        assert "task_stopped" in self._types() or (
            "task_interrupted" in self._types()
        ), (
            f"Expected task_stopped / task_interrupted; got: {self._types()}"
        )
        assert "worktree_created" in self._types(), self._types()
        assert state.agent is not None and state.agent._wt_pending, (
            "the stopped task must leave its worktree pending, not merged"
        )

        assert Path(self.repo, "slides.pptx").read_bytes() == _GOOD_PPTX_BYTES

        self.server._close_tab(tab_id)

        main_pptx_now = Path(self.repo, "slides.pptx").read_bytes()
        assert main_pptx_now == _GOOD_PPTX_BYTES, (
            "REGRESSION (lost-slides bug): closing a chat tab after "
            "Stop silently overwrote the main branch's slides.pptx "
            "with the agent's partial work.  Expected the prior "
            "'good' bytes to remain on main; got "
            f"{len(main_pptx_now)} bytes "
            f"(partial: {main_pptx_now.startswith(_PARTIAL_PREFIX)})."
        )
        assert run_git(self.repo, "status", "--porcelain").stdout == "", (
            "closing the tab left the main working tree dirty"
        )

        branches = _list_kiss_wt_branches(self.repo)
        assert len(branches) == 1, (
            "Expected exactly one kiss/wt-* branch preserved for "
            "manual recovery; got: "
            f"{branches}.  Events: {self._types()}"
        )
        branch = branches[0]
        partial_on_branch = _file_bytes_on_branch(
            self.repo, branch, "slides.pptx",
        )
        expected_len = len(_PARTIAL_PREFIX) + _PARTIAL_RANDOM_LEN
        assert (
            partial_on_branch.startswith(_PARTIAL_PREFIX)
            and len(partial_on_branch) == expected_len
        ), (
            f"Partial work on branch '{branch}' was lost.  Expected "
            f"the agent's partial bytes at its tip, but found "
            f"different content ({len(partial_on_branch)} bytes vs "
            f"{expected_len} expected)."
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

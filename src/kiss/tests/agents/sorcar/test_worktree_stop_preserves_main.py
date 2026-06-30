# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: user-stopped worktree task must not silently overwrite main.

User-reported bug ("the lost slides bug"):

    The user ran a worktree task with ``autoCommit=True`` that was
    asked to update a binary ``.pptx`` file in place.  Midway through
    the task the user clicked "Stop".  The agent had already written
    partial bytes to the ``.pptx`` inside the worktree.  Later, when
    the user closed the chat tab (or VS Code reloaded), the partial
    pptx was silently squash-merged into the main branch — destroying
    the user's previous, complete deck.

Root cause:

    On ``Stop`` with ``autoCommit=True``, ``task_runner._run_task_inner``
    overrides ``effective_auto_commit = False`` and routes through
    ``_present_pending_worktree(try_merge_review=True,
    discard_if_empty=False)``.  For binary files
    (``total_hunks == 0``) ``_start_merge_session`` returns False, so
    the merge-review never opens.  The partial work sits uncommitted
    in the worktree.

    Later, on tab close, ``_teardown_tab_resources`` calls
    ``WorktreeSorcarAgent._release_worktree()`` which runs
    ``_finalize_worktree`` (auto-commits the partial work) followed by
    ``_do_merge`` (silent squash-merge into the original branch).  No
    user confirmation.

Fix:

    When a worktree task ends in failure / user-stop, mark the agent
    as ``pending_review``.  In ``_teardown_tab_resources`` (tab close),
    if the agent is ``pending_review`` AND has a pending worktree,
    commit the partial work onto the worktree branch but DO NOT merge
    — the branch survives in ``git branch`` so the user can recover
    via ``git checkout <branch>`` later, and the main branch is
    untouched.

This is an end-to-end test: it runs a real ``VSCodeServer._run_task_inner``
against a real on-disk git repo, with a deterministic stub agent that
writes partial pptx bytes then raises ``KeyboardInterrupt`` to simulate
the user clicking Stop.  Then ``server._close_tab`` is called to simulate
the user closing the chat tab.  Assertions are made on the git repo's
on-disk state (branch list, file contents on main vs on the wt branch).
No mocks or test doubles.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Binary content for the user's pre-existing "good" pptx, large enough
# that a partial overwrite is detectable on a byte-for-byte compare and
# unlikely to match the agent's "partial" bytes by chance.
_GOOD_PPTX_BYTES: bytes = b"GOOD-DECK\x00" + os.urandom(8192)
_PARTIAL_PPTX_BYTES: bytes = b"PARTIAL\x00" + os.urandom(4096)


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo_with_pptx(repo: str) -> None:
    """Create a fresh git repo with a committed binary ``slides.pptx``.

    Mirrors the user's pre-Stop state: main branch has the "good"
    pptx, and the workspace is clean.
    """
    _run_git(repo, "init", "-q", "-b", "main")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "slides.pptx").write_bytes(_GOOD_PPTX_BYTES)
    _run_git(repo, "add", "slides.pptx")
    _run_git(repo, "commit", "-q", "-m", "seed: add good slides.pptx")


def _list_kiss_wt_branches(repo: str) -> list[str]:
    """Return all ``kiss/wt-*`` branches still present in *repo*."""
    result = _run_git(repo, "branch", "--list", "kiss/wt-*")
    return [
        line.strip().lstrip("* ").strip()
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


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _WorktreeStopBase(unittest.TestCase):
    """Fresh git repo + isolated persistence DB per test."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-stop-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo_with_pptx(self.repo)

        # Redirect persistence to a temp DB.
        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict] = []

        def capture(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # Save the parent class' ``run`` so the test can patch and
        # restore it without leaking state across tests.
        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run

        # Clean up any pending worktree branches the test left behind.
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState
        for tab in list(_RunningAgentState.running_agent_states.values()):
            if tab.agent is not None and tab.agent._wt_pending:
                try:
                    tab.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        _RunningAgentState.running_agent_states.clear()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _types(self) -> list[str]:
        return [e.get("type", "") for e in self.events]


def _stub_run_partial_then_stop(filename: str, partial: bytes) -> Any:
    """Replace ``ChatSorcarAgent``'s parent ``run`` with a "Stop" stub.

    The stub writes *partial* bytes to *filename* inside the per-task
    ``work_dir`` (simulating mid-task agent work on a binary file)
    then raises ``KeyboardInterrupt`` (simulating the user clicking
    Stop in the VS Code webview).

    Returns the original ``run`` so the caller can restore it in
    ``tearDown``.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def stub_run(self_agent: object, **kwargs: object) -> str:
        work_dir = kwargs.get("work_dir")
        if isinstance(work_dir, str) and work_dir:
            (Path(work_dir) / filename).write_bytes(partial)
        raise KeyboardInterrupt("Stopped by user (simulated)")

    parent_class.run = stub_run
    return original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUserStopPreservesMainBranch(_WorktreeStopBase):
    """The "lost slides" bug: user-Stop must not silently overwrite main.

    When the user clicks Stop on a worktree task with
    ``autoCommit=True`` and later closes the chat tab, the main
    branch's binary file must NOT have been silently overwritten by
    the agent's partial work, and the partial work must remain
    recoverable on the ``kiss/wt-*`` branch.
    """

    def test_close_tab_after_stop_does_not_overwrite_main_pptx(
        self,
    ) -> None:
        """End-to-end repro of the "lost slides" bug.

        Sequence:
            1. Real git repo with committed good ``slides.pptx``.
            2. Agent stub writes partial bytes to ``slides.pptx`` then
               raises ``KeyboardInterrupt`` (= user clicked Stop).
            3. ``_run_task_inner`` is driven with ``useWorktree=True``
               and ``autoCommit=True`` — matching the user's exact
               configuration in the lost-slides incident.
            4. ``server._close_tab(tab_id)`` simulates the user
               closing the chat tab afterward.  At this point the
               merge view is still open so the close is *deferred*
               (``frontend_closed=True``, real teardown waits for
               ``is_merging`` to drop).
            5. ``server._cmd_merge_action({"action": "all-done", ...})``
               simulates the WebSocket close path that fires when
               the webview disappears — production behaviour from
               ``web_server.py`` is to send ``all-done`` on socket
               drop, "treating the close as 'accept the remaining'"
               (see the comment around line 3386).  This is the
               line that triggers the silent squash-merge in the
               buggy code.

        After the fix:
            - Main branch's ``slides.pptx`` MUST still equal the
              original "good" bytes (the user's prior work is intact).
            - A ``kiss/wt-*`` branch MUST still exist with the
              partial bytes at its tip, so the user can recover the
              in-flight work via ``git checkout <branch>`` if they
              want it.
        """
        self._original_run = _stub_run_partial_then_stop(
            "slides.pptx", _PARTIAL_PPTX_BYTES,
        )

        tab_id = "tab-stop-test"
        self.server._run_task_inner({
            "prompt": "update the slides",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": True,
            "autoCommit": True,
            "model": "",
        })

        # Sanity: the stop path was actually taken.
        assert "task_stopped" in self._types() or (
            "task_interrupted" in self._types()
        ), (
            f"Expected task_stopped / task_interrupted; got: {self._types()}"
        )

        # Sanity: the post-stop main branch still has the user's
        # original good pptx (the partial work is in the worktree
        # only, not yet merged).
        assert Path(self.repo, "slides.pptx").read_bytes() == _GOOD_PPTX_BYTES

        # User closes the chat tab while the merge view is still
        # open: the close is deferred (frontend_closed=True).
        self.server._close_tab(tab_id)

        # WebSocket-close path now sends ``all-done`` to drain the
        # deferred close.  In the buggy code this triggers the
        # silent squash-merge inside ``_teardown_tab_resources``
        # → ``_release_worktree`` → ``_do_merge``.
        self.server._cmd_merge_action({
            "action": "all-done",
            "tabId": tab_id,
            "workDir": self.repo,
        })

        # Assertion #1: main's good pptx is untouched.
        main_pptx_now = Path(self.repo, "slides.pptx").read_bytes()
        assert main_pptx_now == _GOOD_PPTX_BYTES, (
            "REGRESSION (lost-slides bug): closing a chat tab after "
            "Stop silently overwrote the main branch's slides.pptx "
            "with the agent's partial work.  Expected the prior "
            "'good' bytes to remain on main; got "
            f"{len(main_pptx_now)} bytes "
            f"(matches PARTIAL: {main_pptx_now == _PARTIAL_PPTX_BYTES})."
        )

        # Assertion #2: the partial work is preserved on a kiss/wt-*
        # branch so the user can recover it manually.
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
        assert partial_on_branch == _PARTIAL_PPTX_BYTES, (
            f"Partial work on branch '{branch}' was lost.  Expected "
            f"the agent's partial bytes at its tip, but found "
            f"different content ({len(partial_on_branch)} bytes vs "
            f"{len(_PARTIAL_PPTX_BYTES)} expected)."
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A pending worktree must survive the tab's next non-worktree run.

R09-1.  With **Auto commit** off and **Use worktree** on, a finished
task leaves its ``kiss/wt-*`` branch pending and the webview shows the
Merge / Discard bar.  If the user does not click either button, unticks
**Use worktree** and submits the next task on the same tab, the server
used to lose the worktree entirely:

* ``_run_task_inner`` set ``state.use_worktree = False`` for the new
  run, so the release guard — gated on that same flag — never ran; and
* ``_run_task``'s ``finally`` dropped ``state.agent`` using the same
  flag, discarding the only in-memory handle to the worktree
  directory, the branch and its ``branch.<name>.*`` config.

Nothing was left to retry the cleanup, and because neither
``_release_worktree`` nor ``_preserve_pending_worktree_for_review``
ran, no preserve marker was written either — so the next worktree task
in the repo let ``GitWorktreeOps.reclaim_orphaned_worktrees``
auto-commit and squash-merge into the user's branch the very work the
user had declined to merge.

Everything here is real: a real git repository, a real linked worktree
created by ``WorktreeSorcarAgent``, real ``run`` commands dispatched
through ``VSCodeServer._handle_command`` on real worker threads, and
the real reclaim sweep.  Only the LLM call is replaced by a
deterministic stub that writes a file, exactly as the sibling
auto-commit suites do.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


def _kiss_wt_branches(repo: str) -> list[str]:
    """Names of the ``kiss/wt-*`` branches present in *repo*."""
    out = _run_git(repo, "branch", "--list", "kiss/wt-*").stdout
    return [
        line.strip().lstrip("*+ ").strip()
        for line in out.splitlines()
        if line.strip()
    ]


class TestPendingWorktreeSurvivesNonWorktreeRun(unittest.TestCase):
    """The next run must never silently orphan a pending worktree."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-orphan-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

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

        self.printer = MemoryPrinter()
        self.server = VSCodeServer(self.printer)
        self.server.work_dir = self.repo

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        agent_state.agent_states.clear()
        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- helpers ---------------------------------------------------

    def _stub_llm(self, filename: str | None) -> None:
        """Make the agent write *filename* into its work dir and succeed.

        ``None`` writes nothing, which is how the non-worktree runs
        below keep the main working tree clean.
        """

        def stub_run(_agent: object, **kwargs: object) -> str:
            work_dir = kwargs.get("work_dir")
            if filename and isinstance(work_dir, str) and work_dir:
                Path(work_dir, filename).write_text("agent output\n")
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def _run(
        self,
        tab_id: str,
        *,
        use_worktree: bool,
        filename: str | None,
    ) -> None:
        """Dispatch a real ``run`` command and wait for it to finish.

        The non-worktree runs pass ``filename=None`` so the main
        working tree stays clean: the post-task commit of a dirty main
        tree generates its message with a live LLM call, and these
        tests must not make one.  Leaving the main tree clean is also
        what the orphan-reclaim sweep requires before it will consider
        a worktree at all, so the regression tail below tests the
        sweep rather than its dirty-tree guard.
        """
        self._stub_llm(filename)
        before = len(self._status_ends(tab_id))
        self.server._handle_command({
            "type": "run",
            "prompt": f"write {filename}",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": use_worktree,
            "autoCommit": False,
            "model": "",
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
            for ev in list(self.printer.emitted)
            if ev.get("type") == "status"
            and ev.get("running") is False
            and ev.get("tabId") == tab_id
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

    # -- tests -----------------------------------------------------

    def test_non_worktree_run_does_not_orphan_the_pending_worktree(
        self,
    ) -> None:
        """The declined worktree keeps an owner, or is disposed of.

        "Disposed of" is the release-without-merging contract: the
        work is committed onto the ``kiss/wt-*`` branch, the worktree
        directory is removed and pruned, and the user is told where
        the work went.  What must never happen is the third state —
        directory and branch still there with nobody left holding
        them.
        """
        tab_id = "tab-orphan"
        self._run(tab_id, use_worktree=True, filename="declined.txt")
        branch = self._pending_branch(tab_id)
        wt_dir = self._pending_worktree_dir(tab_id)
        assert branch in _kiss_wt_branches(self.repo)

        self._run(tab_id, use_worktree=False, filename=None)

        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        if state.agent is not None and state.agent._wt_pending:
            return  # still owned: a later run can still dispose of it

        if wt_dir.exists():
            # These runs carry ``autoCommit: False``, so the release
            # is not allowed to commit: it preserves the worktree for
            # manual review and records a durable marker so a later
            # reclaim sweep never publishes it.  That is an owner —
            # the marker — not the orphan this test forbids.
            assert GitWorktreeOps.load_preserve_marker(
                Path(self.repo), branch,
            ), (
                f"worktree directory {wt_dir} was left on disk without "
                "a preserve marker; a reclaim sweep could publish it"
            )
            assert Path(wt_dir, "declined.txt").exists(), (
                "the preserved worktree no longer carries the declined work"
            )
            return
        registered = [b for _d, b in GitWorktreeOps.registered_worktrees(
            Path(self.repo),
        )]
        assert branch not in registered, (
            f"branch {branch!r} is still a registered worktree with no "
            "owner"
        )
        if branch not in _kiss_wt_branches(self.repo):
            return  # nothing was lost: the worktree held no work

        committed = _run_git(
            self.repo, "show", "--name-only", "--pretty=format:", branch,
        ).stdout.split()
        assert "declined.txt" in committed, (
            f"branch {branch!r} survived but no longer carries the "
            f"declined work: {committed}"
        )

    def test_declined_work_is_not_published_by_the_next_worktree_task(
        self,
    ) -> None:
        """The orphan must not reach the user's branch via reclaim."""
        tab_id = "tab-publish"
        self._run(tab_id, use_worktree=True, filename="declined.txt")
        branch = self._pending_branch(tab_id)

        self._run(tab_id, use_worktree=False, filename=None)
        assert _run_git(self.repo, "status", "--porcelain").stdout == "", (
            "the main tree must be clean, otherwise the reclaim sweep "
            "declines for that reason and the regression is masked"
        )

        self._run("tab-other", use_worktree=True, filename="third.txt")

        tracked = _run_git(self.repo, "ls-files").stdout.split()
        assert "declined.txt" not in tracked, (
            f"work the user declined to merge (branch {branch!r}) was "
            "auto-published onto their branch by the orphan reclaim sweep"
        )
        assert not Path(self.repo, "declined.txt").exists(), (
            "the declined worktree's file reached the main working tree"
        )

    def test_next_worktree_run_still_owns_a_worktree(self) -> None:
        """The worktree→worktree flow is untouched by the fix.

        With the main tree idle the carried-over worktree is not
        released here: the next run's own ``_try_setup_worktree``
        retires it as it always did, and the tab ends up owning the
        new run's worktree.  How that retirement disposes of the work
        is the agent's own auto-commit policy — with ``autoCommit:
        False`` on the wire it preserves the directory for manual
        review — so what is asserted here is only that the tab keeps
        exactly one live worktree and the retired one is never left
        unowned.
        """
        tab_id = "tab-chain"
        self._run(tab_id, use_worktree=True, filename="first.txt")
        first_branch = self._pending_branch(tab_id)
        first_dir = self._pending_worktree_dir(tab_id)

        self._run(tab_id, use_worktree=True, filename="second.txt")
        second_branch = self._pending_branch(tab_id)

        assert second_branch != first_branch, (
            "the second worktree run reused the first run's branch"
        )
        registered = [
            branch
            for _dir, branch in GitWorktreeOps.registered_worktrees(
                Path(self.repo),
            )
        ]
        if first_dir.exists():
            # ``autoCommit: False`` forbids the retirement from
            # committing, so it preserves the directory for manual
            # review instead of removing it, and marks the branch so
            # no later sweep publishes the work.  A preserved worktree
            # stays registered with git precisely because it is still
            # on disk.
            assert GitWorktreeOps.load_preserve_marker(
                Path(self.repo), first_branch,
            ), (
                f"the retired worktree directory {first_dir} was left "
                "behind without a preserve marker"
            )
        else:
            assert first_branch not in registered, (
                f"the retired branch {first_branch!r} is still a "
                "registered worktree"
            )
        assert second_branch in registered, (
            "the tab must own the new run's worktree"
        )

    def test_changed_files_probe_ignores_unknown_tabs(self) -> None:
        """A tab with no state has no worktree changes to report."""
        assert self.server._get_worktree_changed_files("no-such-tab") == []

    def test_pending_worktree_kept_when_the_run_is_refused(self) -> None:
        """A run refused before it starts must not drop the owner.

        A non-worktree run is rejected outright while another tab is
        merging a worktree.  That early return happens after the new
        run's ``useWorktree=False`` has been adopted, so the ownership
        test in ``_run_task``'s ``finally`` must not key off it.
        """
        tab_id = "tab-refused"
        self._run(tab_id, use_worktree=True, filename="declined.txt")
        branch = self._pending_branch(tab_id)

        merging = agent_state.AgentState(
            "merging-task", tab_id="tab-merging", server_owned=True,
        )
        merging.use_worktree = True
        merging.is_merging = True
        agent_state.register(merging)
        try:
            self._run(tab_id, use_worktree=False, filename=None)
        finally:
            agent_state.unregister(merging.task_id, merging)

        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        assert state.agent is not None and state.agent._wt_pending, (
            f"the refused run dropped the owner of branch {branch!r}, "
            "leaking the worktree directory and branch forever"
        )
        assert branch in _kiss_wt_branches(self.repo)


if __name__ == "__main__":
    unittest.main()

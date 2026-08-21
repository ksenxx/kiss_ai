# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the area-D redundancy/race-condition fixes.

Covers (real :class:`VSCodeServer`, real :class:`AgentState` registry,
real threads, real on-disk git repos and worktrees — no mocks, patches
or fakes):

- D-RC1: ``_task_accepts_input`` must treat a created-but-not-yet-
  started worker thread as live (``Thread.is_alive()`` is False before
  ``start()``), so an ``appendUserMessage`` arriving in the
  ``_cmd_run`` startup window is queued instead of dropped.
- D-RC2: ``TabRegistry.open_tab`` decides opened/exists/full
  atomically under the registry lock; ``_cmd_open_tab`` no longer
  re-probes ``has_tab`` unlocked, so a concurrent ``closeTab`` can
  never turn a benign re-announce into a spurious "Tab limit reached"
  rejection.
- D-RC3: ``_present_pending_worktree`` performs the worktree-occupancy
  check and the ``is_merging`` claim in ONE ``_state_lock`` critical
  section, so the task-admission gate (which registers occupancy under
  the same lock) can never slip a task into the worktree between the
  check and the claim — the window in which the empty-branch
  auto-discard used to delete the worktree under the just-started
  task.
- D-R1: ``_cmd_set_work_dir`` and ``_cmd_save_config`` share one
  work-dir update implementation (``_apply_new_work_dir``).

D-RC3 determinism note: the pre-fix bug lived in the gap BETWEEN two
adjacent ``with self._state_lock`` blocks.  ``merge_flow.py`` has no
``KISS_RACE_DELAY`` hook (the repo's ``_race_delay`` pattern lives in
``git_worktree.py`` / ``persistence.py`` only), and there is no way to
pause a thread inside that former gap without instrumenting production
code with a test-only hook or substituting a fake for
``_any_non_wt_running`` — both of which are test doubles this suite
must not use.  The tests below instead verify the invariant the merged
critical section guarantees under real concurrency (the worktree is
never deleted under an admitted occupant, for any interleaving) plus
each side of the interlock deterministically.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.commands import _task_accepts_input
from kiss.server.server import VSCodeServer
from kiss.server.tab_registry import _MAX_TABS, OpenTabOutcome, TabRegistry
from kiss.server.task_runner import _wt_merge_on_repo

from ._memory_printer import MemoryPrinter


def _noop() -> None:
    """Body for worker threads that must exist but do nothing."""


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in *repo*, capturing output."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=False,
    )


def _make_repo(path: Path) -> Path:
    """Create a real git repository with one commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    _run_git(path, "config", "user.email", "t@t.com")
    _run_git(path, "config", "user.name", "T")
    (path / "README.md").write_text("# Test\n")
    _run_git(path, "add", ".")
    _run_git(path, "commit", "-m", "initial")
    return path


def _setup_wt_tab(
    repo: Path, tab_id: str,
) -> tuple[AgentState, WorktreeSorcarAgent, Path]:
    """Create a real worktree agent and register its tab's agent state."""
    wt_agent = WorktreeSorcarAgent("wt")
    wt_agent._chat_id = tab_id
    wt_work = wt_agent._try_setup_worktree(repo, str(repo))
    assert wt_work is not None
    state = AgentState(
        f"{tab_id}-key",
        agent=wt_agent,
        tab_id=tab_id,
        server_owned=True,
    )
    state.use_worktree = True
    agent_state.register(state)
    return state, wt_agent, Path(wt_work)


class TestDRC1TaskAcceptsInputStartupWindow:
    """A created-but-unstarted worker thread must count as a live task."""

    def test_created_but_unstarted_thread_accepts_input(self) -> None:
        """The exact S3-05 window: ``_cmd_run`` installs ``task_thread``
        and broadcasts BEFORE ``thread.start()``; ``is_alive()`` is
        False then, but the task is real and must accept input."""
        state = AgentState("rr-d-rc1-a", tab_id="rr-d-rc1-a-tab")
        state.task_thread = threading.Thread(target=_noop, daemon=True)
        assert state.task_thread.is_alive() is False, (
            "precondition: an unstarted thread reports is_alive()=False"
        )
        with agent_state.STATE_LOCK:
            assert _task_accepts_input(state) is True, (
                "a follow-up typed during the run-startup window was "
                "treated as addressed to a dead task and dropped"
            )

    def test_none_finished_and_active_states(self) -> None:
        """Branch coverage: None, finished-thread, and active states."""
        with agent_state.STATE_LOCK:
            assert _task_accepts_input(None) is False

        finished = AgentState("rr-d-rc1-b", tab_id="rr-d-rc1-b-tab")
        thread = threading.Thread(target=_noop, daemon=True)
        thread.start()
        thread.join()
        finished.task_thread = thread
        with agent_state.STATE_LOCK:
            assert _task_accepts_input(finished) is False, (
                "a started-and-finished worker thread is a dead task"
            )

        active = AgentState("rr-d-rc1-c", tab_id="rr-d-rc1-c-tab")
        active.is_task_active = True
        with agent_state.STATE_LOCK:
            assert _task_accepts_input(active) is True

    def test_append_user_message_queued_during_startup_window(self) -> None:
        """End-to-end: ``appendUserMessage`` from another connection in
        the startup window lands in ``pending_user_messages``."""
        server = VSCodeServer(printer=MemoryPrinter())
        state = AgentState(
            "rr-d-rc1-e2e", tab_id="rr-d-rc1-e2e-tab", server_owned=True,
        )
        state.task_thread = threading.Thread(target=_noop, daemon=True)
        agent_state.register(state)
        try:
            server._cmd_append_user_message({
                "tabId": "rr-d-rc1-e2e-tab",
                "prompt": "follow-up during startup",
            })
            assert state.pending_user_messages == [
                "follow-up during startup",
            ], "the startup-window follow-up was dropped or misrouted"
        finally:
            agent_state.unregister(state.task_id, state)


class TestDRC2OpenTabTriState:
    """``open_tab`` decides opened/exists/full atomically."""

    def test_tristate_outcomes_and_truthiness(self, tmp_path: Path) -> None:
        reg = TabRegistry(tmp_path / "tabs.json")
        opened = reg.open_tab("t1", "one")
        assert opened is OpenTabOutcome.OPENED
        assert bool(opened) is True, (
            "OPENED must stay truthy for legacy boolean callers"
        )
        exists = reg.open_tab("t1", "one again")
        assert exists is OpenTabOutcome.EXISTS
        assert bool(exists) is False
        assert reg.open_tab("") is OpenTabOutcome.EXISTS, (
            "a blank id opens nothing but must not be reported as FULL"
        )
        for i in range(_MAX_TABS - 1):
            assert reg.open_tab(f"fill-{i}") is OpenTabOutcome.OPENED
        full = reg.open_tab("overflow")
        assert full is OpenTabOutcome.FULL
        assert bool(full) is False
        # The registry really is unchanged by the refusal.
        assert not reg.has_tab("overflow")

    def test_full_registry_rejects_unknown_tab(self, tmp_path: Path) -> None:
        """The FULL outcome still produces ``openTabRejected``."""
        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        server.tab_registry = TabRegistry(tmp_path / "tabs.json")
        for i in range(_MAX_TABS):
            assert server.tab_registry.open_tab(f"pre-{i}")
        printer.emitted.clear()
        server._cmd_open_tab({"tabId": "brand-new", "connId": "c1"})
        rejected = [
            e for e in printer.emitted if e.get("type") == "openTabRejected"
        ]
        assert len(rejected) == 1
        assert rejected[0]["tabId"] == "brand-new"

    def test_reannounce_never_rejected_despite_concurrent_close(
        self, tmp_path: Path,
    ) -> None:
        """D-RC2 end-to-end: with the registry AT its cap and a client
        re-announcing a tab that exists, a concurrent ``closeTab`` of
        that same tab must never surface as ``openTabRejected`` — the
        outcome is decided atomically, so every interleaving yields
        EXISTS (still registered) or OPENED (close freed a slot)."""
        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        server.tab_registry = TabRegistry(tmp_path / "tabs.json")
        for i in range(_MAX_TABS - 1):
            assert server.tab_registry.open_tab(f"cap-{i}")
        assert server.tab_registry.open_tab("racy-tab")

        for _ in range(30):
            printer.emitted.clear()
            closer = threading.Thread(
                target=server.tab_registry.close_tab, args=("racy-tab",),
            )
            closer.start()
            server._cmd_open_tab({"tabId": "racy-tab", "connId": "c1"})
            closer.join()
            rejected = [
                e for e in printer.emitted
                if e.get("type") == "openTabRejected"
            ]
            assert rejected == [], (
                "a concurrent closeTab turned a re-announce of an "
                "existing tab into a spurious 'Tab limit reached'"
            )
            # Re-arm: make sure the tab is registered again for the
            # next round (the open may have raced ahead of the close).
            server.tab_registry.close_tab("racy-tab")
            assert server.tab_registry.open_tab("racy-tab")


class TestDRC3AtomicOccupancyCheckAndClaim:
    """Occupancy check + ``is_merging`` claim form one critical section."""

    def test_occupied_worktree_is_not_discarded(self, tmp_path: Path) -> None:
        """An empty pending worktree with a non-wt task running INSIDE
        it must be left alone by the session-resume presentation."""
        repo = _make_repo(tmp_path / "repo")
        server = VSCodeServer(printer=MemoryPrinter())
        server.work_dir = str(repo)
        state, wt_agent, wt_dir = _setup_wt_tab(repo, "rr-d-rc3-occ")
        occupant = AgentState(
            "rr-d-rc3-occ-task", tab_id="rr-d-rc3-occ-task-tab",
            server_owned=True,
        )
        occupant.is_running_non_wt = True
        occupant.non_wt_repo_root = wt_dir.resolve()
        agent_state.register(occupant)
        try:
            server._present_pending_worktree("rr-d-rc3-occ")
            assert wt_dir.exists(), (
                "the auto-discard deleted a worktree a running task "
                "occupies"
            )
            assert state.is_merging is False, (
                "the presentation kept its is_merging claim after "
                "declining to discard"
            )
            branches = _run_git(repo, "branch", "--list", "kiss/wt-*")
            assert branches.stdout.strip(), (
                "the pending worktree branch was deleted despite the "
                "occupying task"
            )
        finally:
            agent_state.unregister(occupant.task_id, occupant)
            agent_state.unregister(state.task_id, state)
            wt_agent.discard(rescue_ignored=False)

    def test_unoccupied_empty_worktree_is_discarded(
        self, tmp_path: Path,
    ) -> None:
        """Control: with no occupant the empty branch is auto-discarded
        and the claim is released afterwards."""
        repo = _make_repo(tmp_path / "repo")
        server = VSCodeServer(printer=MemoryPrinter())
        server.work_dir = str(repo)
        state, _wt_agent, wt_dir = _setup_wt_tab(repo, "rr-d-rc3-free")
        try:
            server._present_pending_worktree("rr-d-rc3-free")
            assert not wt_dir.exists(), (
                "an empty, unoccupied pending worktree must be "
                "auto-discarded on presentation"
            )
            assert state.is_merging is False, (
                "the presentation must release is_merging when done"
            )
        finally:
            agent_state.unregister(state.task_id, state)

    def test_admission_gate_refuses_worktree_while_claimed(
        self, tmp_path: Path,
    ) -> None:
        """The interlock the merged critical section relies on: once
        ``is_merging`` is claimed, the real task-admission predicate
        refuses a task whose work_dir resolves to the worktree."""
        repo = _make_repo(tmp_path / "repo")
        state, wt_agent, wt_dir = _setup_wt_tab(repo, "rr-d-rc3-gate")
        try:
            wt_repo = GitWorktreeOps.discover_repo(wt_dir)
            assert wt_repo is not None
            with agent_state.STATE_LOCK:
                assert _wt_merge_on_repo(state, wt_repo) is False, (
                    "precondition: no claim, the gate admits the task"
                )
                state.is_merging = True
                assert _wt_merge_on_repo(state, wt_repo) is True, (
                    "with is_merging claimed, a task starting inside "
                    "the pending worktree must be refused"
                )
                state.is_merging = False
        finally:
            agent_state.unregister(state.task_id, state)
            wt_agent.discard(rescue_ignored=False)

    def test_concurrent_admission_never_occupies_deleted_worktree(
        self, tmp_path: Path,
    ) -> None:
        """Invariant under real concurrency: whatever the interleaving
        of the presentation and the admission gate, a task admitted
        INTO the worktree implies the worktree was not (and will not
        be) deleted by the presentation's auto-discard."""
        repo = _make_repo(tmp_path / "repo")
        server = VSCodeServer(printer=MemoryPrinter())
        server.work_dir = str(repo)
        state, wt_agent, wt_dir = _setup_wt_tab(repo, "rr-d-rc3-race")
        occupant = AgentState(
            "rr-d-rc3-race-task", tab_id="rr-d-rc3-race-task-tab",
            server_owned=True,
        )
        admitted_inside_wt = False
        barrier = threading.Barrier(2)

        def _admission_gate() -> None:
            """Mirror ``_run_task_inner``'s non-worktree admission:
            discover the repo, check ``_wt_merge_on_repo`` and register
            occupancy — all under ``_state_lock``."""
            nonlocal admitted_inside_wt
            barrier.wait()
            task_repo = GitWorktreeOps.discover_repo(wt_dir)
            with server._state_lock:
                if task_repo is None:
                    return  # worktree already gone: nothing to occupy
                if any(
                    _wt_merge_on_repo(t, task_repo)
                    for t in agent_state.agent_states.values()
                ):
                    return  # refused: a merge/discard owns the repo
                occupant.is_running_non_wt = True
                occupant.non_wt_repo_root = task_repo.resolve()
                agent_state.register(occupant)
                if task_repo.resolve() == wt_dir.resolve():
                    admitted_inside_wt = True

        gate_thread = threading.Thread(target=_admission_gate)
        gate_thread.start()
        try:
            barrier.wait()
            server._present_pending_worktree("rr-d-rc3-race")
            gate_thread.join()
            if admitted_inside_wt:
                assert wt_dir.exists(), (
                    "the presentation's auto-discard deleted the "
                    "worktree under a task the admission gate had "
                    "admitted into it"
                )
        finally:
            gate_thread.join()
            agent_state.unregister(occupant.task_id, occupant)
            agent_state.unregister(state.task_id, state)
            if wt_agent._wt_pending:
                # Clear the synthetic occupancy first or the discard
                # is deferred as "occupied".
                occupant.is_running_non_wt = False
                wt_agent.discard(rescue_ignored=False)


class TestDR1SharedWorkDirUpdate:
    """Both work-dir update paths share ``_apply_new_work_dir``."""

    def test_set_work_dir_updates_state(self, tmp_path: Path) -> None:
        server = VSCodeServer(printer=MemoryPrinter())
        # ``work_dir`` is an optional printer attribute the update
        # mirrors onto when present (hasattr-guarded in production).
        setattr(server.printer, "work_dir", "")
        server._file_cache = {"stale": ["a.py"]}
        new_dir = str(tmp_path / "workspace-a")
        server._cmd_set_work_dir({"workDir": new_dir, "connId": "c1"})
        assert server.work_dir == new_dir
        assert server._file_cache == {}, (
            "changing the work dir must invalidate the file cache"
        )
        assert getattr(server.printer, "work_dir") == new_dir

        # Unchanged dir: the cache survives.
        server._file_cache = {"warm": ["b.py"]}
        server._cmd_set_work_dir({"workDir": new_dir, "connId": "c1"})
        assert server._file_cache == {"warm": ["b.py"]}, (
            "re-announcing the same work dir must not blow the cache"
        )

    def test_save_config_updates_state_identically(
        self, tmp_path: Path,
    ) -> None:
        server = VSCodeServer(printer=MemoryPrinter())
        setattr(server.printer, "work_dir", "")
        server._file_cache = {"stale": ["a.py"]}
        new_dir = str(tmp_path / "workspace-b")
        server._cmd_save_config({
            "config": {"work_dir": new_dir},
            "apiKeys": {},
            "connId": "c1",
        })
        assert server.work_dir == new_dir
        assert server._file_cache == {}
        assert getattr(server.printer, "work_dir") == new_dir

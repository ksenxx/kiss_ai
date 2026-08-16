# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave2-Fixer-7 findings (real repos, no mocks).

F1  ``_MergeFlowMixin._present_pending_worktree`` must claim the tab
    (``state.is_merging = True``) before auto-discarding an empty
    worktree — ``discard()`` runs ``git checkout`` in the MAIN
    repository, so a non-wt task starting in the TOCTOU window would
    race the checkout.  The discard itself is never skipped: an empty
    worktree changes no files and the checkout is a no-op onto the
    branch the tree is already on.
F13 ``diff_merge._scan_files`` must enforce its 5000-entry cap for
    directory entries too, not only in the files loop.
F20 ``vscode_config.source_shell_env`` must not import a forged API key
    from a multi-line environment-variable value (line-based ``env``
    parsing); it must use NUL-separated ``env -0`` records.

All tests use real git repos / real directories / a real shell in
``tmp_path`` and call the production functions directly.  No mocks,
patches, or fakes — recorders are real subclasses in the pattern of
``test_fixer8_merge_config_bugs.py``.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.diff_merge import _scan_files
from kiss.server.json_printer import JsonPrinter
from kiss.server.merge_flow import _MergeFlowMixin


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c", "user.email=test@test",
            "-c", "user.name=test",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _make_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run_git(repo, "init")
    (repo / "a.txt").write_text("hello\n")
    _run_git(repo, "add", "a.txt")
    _run_git(repo, "commit", "-m", "initial")


def _register_tab_state(tab_id: str) -> AgentState:
    """Register a server-owned :class:`AgentState` for *tab_id* and return it."""
    state = AgentState(f"task-{tab_id}", tab_id=tab_id, server_owned=True)
    agent_state.register(state)
    return state


class _RecordingPrinter(JsonPrinter):
    """Real JsonPrinter subclass recording broadcast events in a list."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* in memory instead of persisting it."""
        with self._events_lock:
            self.events.append(event)


class _Host(_MergeFlowMixin):
    """Concrete merge-flow host with the server state the mixin expects.

    Implements the same ``_any_non_wt_running`` / ``_dispose_if_closed``
    contracts as ``VSCodeServer`` against the real task-keyed
    ``kiss.server.agent_state`` registry.
    """

    def __init__(self, work_dir: str, printer: JsonPrinter | None = None) -> None:
        self.work_dir = work_dir
        self._state_lock = threading.RLock()
        self.printer = printer or _RecordingPrinter()

    def _any_non_wt_running(self, repo_root: Path | None = None) -> bool:
        """True if any state runs a non-worktree task (conservative semantics).

        The harness ignores *repo_root*: every simulated non-worktree
        task in these tests runs on the same main tree as the merge.
        """
        return any(st.is_running_non_wt for st in agent_state.snapshot())

    def _dispose_if_closed(self, tab_id: str) -> None:
        """Mirror the server: unregister only closed, fully-idle states."""
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            if state is not None and state.frontend_closed and not (
                state.is_task_active or state.is_merging
            ):
                agent_state.unregister(state.task_id, state)


class _MergingFlagRecordingAgent(WorktreeSorcarAgent):
    """Real agent subclass recording ``state.is_merging`` at ``discard()`` time.

    The F1 contract is that the tab is CLAIMED (``is_merging`` set,
    atomically with observing the worktree free) before ``discard()``
    runs its main-repo ``git checkout``; this recorder observes the
    flag exactly when the production discard begins.
    """

    def __init__(self, name: str, tab_id: str) -> None:
        super().__init__(name)
        self._test_tab_id = tab_id
        self.observed_merging_during_discard: bool | None = None

    def discard(self) -> str:
        """Record the owning state's ``is_merging`` flag, then really discard."""
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(self._test_tab_id)
            assert state is not None
            self.observed_merging_during_discard = state.is_merging
        return super().discard()


class TestScanFilesCapCoversDirectories:
    def test_directory_heavy_tree_respects_cap(self, tmp_path: Path) -> None:
        """A tree dominated by directories must not exceed 5000 entries."""
        wd = tmp_path / "ws"
        wd.mkdir()
        (wd / "only.txt").write_text("x")
        for i in range(5500):
            (wd / f"d{i:04d}").mkdir()

        paths = _scan_files(str(wd))

        assert len(paths) <= 5000

    def test_small_tree_lists_files_and_dirs(self, tmp_path: Path) -> None:
        wd = tmp_path / "ws"
        (wd / "sub").mkdir(parents=True)
        (wd / "f.txt").write_text("x")
        (wd / "sub" / "g.txt").write_text("x")

        paths = _scan_files(str(wd))

        assert "f.txt" in paths
        assert "sub/" in paths
        assert "sub/g.txt" in paths

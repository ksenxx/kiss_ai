# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave2-Fixer-7 findings (real repos, no mocks).

F1  ``_MergeFlowMixin._present_pending_worktree`` must atomically claim
    the main tree (``tab.is_merging = True``) together with the
    "no non-worktree task running" check before auto-discarding an
    empty worktree — ``discard()`` runs ``git checkout`` in the MAIN
    repository, so a non-wt task starting in the TOCTOU window would
    race the checkout.
F3  ``_MergeFlowMixin._finish_merge`` must keep ``tab.is_merging``
    claimed until the pending-worktree presentation and the autocommit
    dirty-file scan are done — clearing it first lets a task start on
    the tab mid-scan, so the prompt could list the NEW task's in-flight
    files as the finished merge's ``changedFiles``.
F13 ``diff_merge._scan_files`` must enforce its 5000-entry cap for
    directory entries too, not only in the files loop.
F18 Newly created EMPTY files must be visible in the merge review
    (previously they produced no hunks and no binary flag, so the
    review never showed them).
F20 ``vscode_config.source_shell_env`` must not import a forged API key
    from a multi-line environment-variable value (line-based ``env``
    parsing); it must use NUL-separated ``env -0`` records.

All tests use real git repos / real directories / a real shell in
``tmp_path`` and call the production functions directly.  No mocks,
patches, or fakes — recorders are real subclasses in the pattern of
``test_fixer8_merge_config_bugs.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.vscode_config import source_shell_env
from kiss.server.diff_merge import _prepare_merge_view, _scan_files
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

    Implements the same ``_get_tab`` / ``_any_non_wt_running``
    contracts as ``VSCodeServer`` (get-or-create under ``_state_lock``;
    "any tab has ``is_running_non_wt``"), against the real
    ``_RunningAgentState`` registry.
    """

    def __init__(self, work_dir: str, printer: JsonPrinter | None = None) -> None:
        self.work_dir = work_dir
        self._state_lock = threading.RLock()
        self.printer = printer or _RecordingPrinter()

    def _get_tab(self, tab_id: str) -> _RunningAgentState:
        """Get or create per-tab state (mirrors ``VSCodeServer._get_tab``)."""
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None:
                tab = _RunningAgentState(tab_id, "test-model")
                _RunningAgentState.running_agent_states[tab_id] = tab
            return tab

    def _any_non_wt_running(self) -> bool:
        """True if any tab runs a non-worktree task (real server semantics)."""
        return any(
            t.is_running_non_wt
            for t in _RunningAgentState.running_agent_states.values()
        )

    def _dispose_if_closed(self, tab_id: str) -> None:
        """Mirror the server: pop only closed, fully-idle tabs (no-op here)."""
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is not None and tab.frontend_closed and not (
                tab.is_task_active or tab.is_merging
            ):
                _RunningAgentState.running_agent_states.pop(tab_id, None)


class _MergingFlagRecordingAgent(WorktreeSorcarAgent):
    """Real agent subclass recording ``tab.is_merging`` at ``discard()`` time.

    The F1 contract is that the main tree is CLAIMED (``is_merging``
    set, atomically with the non-wt-running check) before ``discard()``
    runs its main-repo ``git checkout``; this recorder observes the
    flag exactly when the production discard begins.
    """

    def __init__(self, name: str, host: _Host, tab_id: str) -> None:
        super().__init__(name)
        self._test_host = host
        self._test_tab_id = tab_id
        self.observed_merging_during_discard: bool | None = None

    def discard(self) -> str:
        """Record the owning tab's ``is_merging`` flag, then really discard."""
        host = self._test_host
        with host._state_lock:
            tab = _RunningAgentState.running_agent_states[self._test_tab_id]
            self.observed_merging_during_discard = tab.is_merging
        return super().discard()


class _AutocommitFlagPrinter(_RecordingPrinter):
    """Recorder that snapshots ``tab.is_merging`` at broadcast time.

    Used for F3: the autocommit dirty-file scan (whose result is the
    ``autocommit_prompt`` broadcast) must run while the merge session
    still holds ``is_merging`` — otherwise a task can start on the tab
    mid-scan.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tab: _RunningAgentState | None = None
        self.merging_at_prompt: bool | None = None
        self.merging_at_merge_ended: bool | None = None

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the event plus the tab's live ``is_merging`` flag."""
        if self.tab is not None:
            if event.get("type") == "autocommit_prompt":
                self.merging_at_prompt = self.tab.is_merging
            elif event.get("type") == "merge_ended":
                self.merging_at_merge_ended = self.tab.is_merging
        super().broadcast(event)


# ---------------------------------------------------------------------------
# F1 — empty-worktree auto-discard must claim the main tree atomically
# ---------------------------------------------------------------------------


class TestEmptyWorktreeDiscardClaimsMainTree:
    def test_discard_runs_with_is_merging_claimed(self, tmp_path: Path) -> None:
        """``discard()`` must see ``tab.is_merging`` True (tree claimed)."""
        repo = tmp_path / "repo"
        _make_repo(repo)
        tab_id = "w2f7-f1-tab"
        host = _Host(str(repo))
        agent = _MergingFlagRecordingAgent("wave2-f1", host, tab_id)
        try:
            assert agent._try_setup_worktree(repo, str(repo)) is not None
            assert agent._wt_pending
            tab = host._get_tab(tab_id)
            tab.use_worktree = True
            tab.agent = agent

            host._present_pending_worktree(tab_id, try_merge_review=True)

            # The empty worktree was really discarded.
            assert agent.observed_merging_during_discard is not None, (
                "empty worktree was not auto-discarded"
            )
            assert not agent._wt_pending
            # The main tree was claimed for the duration of the discard.
            assert agent.observed_merging_during_discard is True, (
                "discard() ran without claiming tab.is_merging — a "
                "non-wt task could start and race the main-repo checkout"
            )
            # The claim is released afterwards.
            assert tab.is_merging is False
        finally:
            _RunningAgentState.running_agent_states.pop(tab_id, None)

    def test_discard_skipped_while_non_wt_task_running(
        self, tmp_path: Path,
    ) -> None:
        """No discard (no main-repo checkout) while a non-wt task runs."""
        repo = tmp_path / "repo"
        _make_repo(repo)
        tab_id = "w2f7-f1b-tab"
        other_id = "w2f7-f1b-other"
        host = _Host(str(repo))
        agent = _MergingFlagRecordingAgent("wave2-f1b", host, tab_id)
        try:
            assert agent._try_setup_worktree(repo, str(repo)) is not None
            tab = host._get_tab(tab_id)
            tab.use_worktree = True
            tab.agent = agent
            other = host._get_tab(other_id)
            other.is_running_non_wt = True

            host._present_pending_worktree(tab_id, try_merge_review=True)

            assert agent.observed_merging_during_discard is None
            assert agent._wt_pending
            assert tab.is_merging is False
        finally:
            if agent._wt_pending:
                agent.discard()
            for tid in (tab_id, other_id):
                _RunningAgentState.running_agent_states.pop(tid, None)


# ---------------------------------------------------------------------------
# F3 — _finish_merge keeps is_merging until the autocommit scan is done
# ---------------------------------------------------------------------------


class TestFinishMergeHoldsClaimThroughCleanup:
    def test_autocommit_prompt_scanned_while_still_merging(
        self, tmp_path: Path,
    ) -> None:
        """The dirty-file scan/broadcast happens under the merge claim."""
        repo = tmp_path / "repo"
        _make_repo(repo)
        (repo / "dirty.txt").write_text("uncommitted\n")
        tab_id = "w2f7-f3-tab"
        printer = _AutocommitFlagPrinter()
        host = _Host(str(repo), printer)
        try:
            tab = host._get_tab(tab_id)
            tab.use_worktree = False
            tab.is_merging = True  # a merge review session is live
            printer.tab = tab

            host._finish_merge(tab_id, work_dir=str(repo))

            types = [e.get("type") for e in printer.events]
            assert "merge_ended" in types
            assert "autocommit_prompt" in types
            prompt = next(
                e for e in printer.events
                if e.get("type") == "autocommit_prompt"
            )
            assert "dirty.txt" in prompt["changedFiles"]
            assert printer.merging_at_prompt is True, (
                "is_merging was cleared before the autocommit dirty scan "
                "— a task starting in that window races the scan"
            )
            # The claim is released once _finish_merge returns.
            assert tab.is_merging is False
        finally:
            _RunningAgentState.running_agent_states.pop(tab_id, None)

    def test_clean_tree_ends_merge_without_prompt(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo)
        tab_id = "w2f7-f3b-tab"
        printer = _AutocommitFlagPrinter()
        host = _Host(str(repo), printer)
        try:
            tab = host._get_tab(tab_id)
            tab.is_merging = True
            printer.tab = tab

            host._finish_merge(tab_id, work_dir=str(repo))

            types = [e.get("type") for e in printer.events]
            assert "merge_ended" in types
            assert "autocommit_prompt" not in types
            assert tab.is_merging is False
        finally:
            _RunningAgentState.running_agent_states.pop(tab_id, None)

    def test_missing_tab_id_is_noop(self, tmp_path: Path) -> None:
        host = _Host(str(tmp_path))
        host._finish_merge("", work_dir=str(tmp_path))
        assert isinstance(host.printer, _RecordingPrinter)
        assert host.printer.events == []


# ---------------------------------------------------------------------------
# F13 — _scan_files 5000-entry cap applies to directory entries too
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# F18 — newly created empty files are visible in the merge review
# ---------------------------------------------------------------------------


class TestEmptyNewFileVisibleInMergeReview:
    def test_empty_new_file_gets_whole_file_entry(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo)
        (repo / "pkg").mkdir()
        (repo / "pkg" / "__init__.py").write_bytes(b"")
        (repo / ".gitkeep").write_bytes(b"")
        data_dir = tmp_path / "data"

        result = _prepare_merge_view(str(repo), str(data_dir), {}, set(), None)

        assert result.get("status") == "opened", (
            f"empty new files invisible in merge review: {result}"
        )
        manifest = json.loads((data_dir / "pending-merge.json").read_text())
        by_name = {f["name"]: f for f in manifest["files"]}
        for fname in ("pkg/__init__.py", ".gitkeep"):
            assert fname in by_name, f"{fname} missing from merge review"
            entry = by_name[fname]
            # Whole-file (binary-style) single decision entry.
            assert entry.get("binary") is True
            assert entry["hunks"] == [{"bs": 0, "bc": 0, "cs": 0, "cc": 0}]

    def test_empty_new_file_alongside_text_change(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo)
        (repo / "a.txt").write_text("hello\nworld\n")
        (repo / "empty.marker").write_bytes(b"")
        data_dir = tmp_path / "data"

        result = _prepare_merge_view(str(repo), str(data_dir), {}, set(), None)

        assert result.get("status") == "opened"
        manifest = json.loads((data_dir / "pending-merge.json").read_text())
        names = {f["name"] for f in manifest["files"]}
        assert "a.txt" in names
        assert "empty.marker" in names

    def test_nonempty_new_text_file_still_line_reviewed(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo)
        (repo / "new.py").write_text("x = 1\ny = 2\n")
        data_dir = tmp_path / "data"

        result = _prepare_merge_view(str(repo), str(data_dir), {}, set(), None)

        assert result.get("status") == "opened"
        manifest = json.loads((data_dir / "pending-merge.json").read_text())
        by_name = {f["name"]: f for f in manifest["files"]}
        assert by_name["new.py"]["hunks"] == [
            {"bs": 0, "bc": 0, "cs": 0, "cc": 2},
        ]
        assert not by_name["new.py"].get("binary")


# ---------------------------------------------------------------------------
# F20 — source_shell_env must not import forged keys from multi-line values
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("/bin/bash").exists(), reason="requires /bin/bash",
)
class TestSourceShellEnvMultilineValues:
    def test_forged_key_inside_multiline_value_not_imported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A value containing ``\\nOPENAI_API_KEY=...`` must not be imported."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bashrc").write_text(
            'export INNOCENT="first line\n'
            'OPENAI_API_KEY=forged-by-multiline-value"\n'
            "export TOGETHER_API_KEY=real-together-key\n"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        monkeypatch.delenv("INNOCENT", raising=False)

        source_shell_env()

        # Real single-line key from the RC is imported…
        assert os.environ.get("TOGETHER_API_KEY") == "real-together-key"
        # …but the forged key embedded in another variable's value is not.
        assert os.environ.get("OPENAI_API_KEY") != "forged-by-multiline-value"

    def test_multiline_api_key_value_preserved_fully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A legitimate multi-line key value must not be truncated."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bashrc").write_text(
            'export OPENROUTER_API_KEY="part1\npart2"\n',
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        source_shell_env()

        assert os.environ.get("OPENROUTER_API_KEY") == "part1\npart2"

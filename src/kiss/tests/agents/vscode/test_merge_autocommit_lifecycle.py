# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the full non-worktree task lifecycle.

Reproduces the bug where the merge/diff interface was not launched and
auto-commit buttons were not shown after a task that modifies files.

Exercises the complete flow:
  pre-snapshot → agent modifies file → _prepare_and_start_merge
  → merge_data/merge_started events → mergeAction all-done
  → merge_ended/autocommit_prompt events → autocommit_action
  → autocommit_done event.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import kiss.agents.vscode.merge_flow as _merge_flow_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.task_runner import _TaskRunnerMixin


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    """Create a git repo with one committed file so HEAD exists."""
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "README.md").write_text("# Hello\n\nSome content\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "initial commit")


def _make_server(work_dir: str) -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with captured events."""
    server = VSCodeServer()
    server.work_dir = work_dir
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _event_types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _find_event(events: list[dict], type_: str) -> dict:
    for e in events:
        if e["type"] == type_:
            return e
    raise AssertionError(f"No event of type {type_!r}: {_event_types(events)}")


class _LifecycleHarness(unittest.TestCase):
    """Shared setUp/tearDown for lifecycle tests."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)
        self.server, self.events = _make_server(self.tmpdir)
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff

        def fake_gen(diff_text: str, user_prompt: str | None = None) -> str:
            return "chore: auto-commit test"

        _merge_flow_module.generate_commit_message_from_diff = fake_gen  # type: ignore[assignment]

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMergeLaunchedAfterFileModification(_LifecycleHarness):
    """After a non-worktree task modifies a tracked file, the merge/diff
    interface must be launched (merge_data + merge_started events)."""

    def test_merge_events_after_modification(self) -> None:
        """Modifying README.md triggers merge_data and merge_started."""
        tab_id = "test-tab-1"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        Path(self.tmpdir, "README.md").write_text(
            "# Hello\n\nUpdated content by the agent\n"
        )

        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )

        assert started, (
            "Merge session should have started but _prepare_and_start_merge "
            f"returned False. Events: {_event_types(self.events)}"
        )

        types = _event_types(self.events)
        assert "merge_data" in types, f"merge_data not found in {types}"
        assert "merge_started" in types, f"merge_started not found in {types}"

        md_event = _find_event(self.events, "merge_data")
        assert md_event["tabId"] == tab_id
        assert "data" in md_event
        assert "files" in md_event["data"]
        assert any(
            f["name"] == "README.md" for f in md_event["data"]["files"]
        ), f"README.md not in merge data files: {md_event['data']['files']}"

        ms_event = _find_event(self.events, "merge_started")
        assert ms_event["tabId"] == tab_id

        assert tab.is_merging is True

    def test_merge_events_after_new_file(self) -> None:
        """Adding a new untracked file triggers merge_data."""
        tab_id = "test-tab-2"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        Path(self.tmpdir, "new_file.txt").write_text("brand new file\n")

        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )

        assert started
        md_event = _find_event(self.events, "merge_data")
        assert any(
            f["name"] == "new_file.txt" for f in md_event["data"]["files"]
        )


class TestAutocommitPromptAfterMergeReview(_LifecycleHarness):
    """After completing merge review (all-done), the autocommit_prompt event
    must be broadcast for non-worktree tabs with dirty files."""

    def test_autocommit_prompt_after_merge_all_done(self) -> None:
        """Full flow: modify file → merge → all-done → autocommit_prompt."""
        tab_id = "test-tab-3"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        Path(self.tmpdir, "README.md").write_text(
            "# Updated README\n\nNew TOC\n"
        )

        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )
        assert started

        self.events.clear()
        self.server._handle_command(
            {"type": "mergeAction", "action": "all-done", "tabId": tab_id}
        )

        types = _event_types(self.events)
        assert "merge_ended" in types, f"merge_ended not found in {types}"
        assert "autocommit_prompt" in types, (
            f"autocommit_prompt not found in {types}. "
            "The auto-commit buttons should appear after merge review."
        )

        ac_event = _find_event(self.events, "autocommit_prompt")
        assert ac_event["tabId"] == tab_id
        assert "README.md" in ac_event["changedFiles"]

        assert tab.is_merging is False


class TestAutocommitActionAfterPrompt(_LifecycleHarness):
    """After the autocommit_prompt, the user can click 'Auto commit' or
    'Do nothing'. Both should work correctly."""

    def _setup_merge_complete(self, tab_id: str) -> None:
        """Run the full lifecycle up to autocommit_prompt."""
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )
        Path(self.tmpdir, "README.md").write_text("# Changed\n")
        self.server._prepare_and_start_merge(
            self.tmpdir, pre_hunks, pre_untracked, pre_file_hashes,
            base_ref=pre_head_sha or "HEAD", tab_id=tab_id,
        )
        self.events.clear()
        self.server._finish_merge(tab_id)

    def test_commit_action_creates_commit(self) -> None:
        """'Auto commit' stages and commits all changes."""
        tab_id = "test-tab-commit"
        self._setup_merge_complete(tab_id)

        assert "autocommit_prompt" in _event_types(self.events)

        self.events.clear()
        self.server._handle_command(
            {"type": "autocommitAction", "action": "commit", "tabId": tab_id}
        )

        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is True
        assert done["committed"] is True
        assert done["tabId"] == tab_id

        status = _git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert status == "", f"Expected clean working tree, got: {status}"

    def test_skip_action_leaves_files_dirty(self) -> None:
        """'Do nothing' leaves the working tree unchanged."""
        tab_id = "test-tab-skip"
        self._setup_merge_complete(tab_id)

        before_head = _git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        self.events.clear()
        self.server._handle_command(
            {"type": "autocommitAction", "action": "skip", "tabId": tab_id}
        )

        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is True
        assert done["committed"] is False

        after_head = _git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        assert before_head == after_head

        status = _git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert "README.md" in status


class TestMergeNotLaunchedWhenNoChanges(_LifecycleHarness):
    """When the agent doesn't modify any files, no merge interface appears."""

    def test_no_merge_when_no_changes(self) -> None:
        tab_id = "test-tab-noop"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )

        assert not started
        assert "merge_data" not in _event_types(self.events)
        assert "merge_started" not in _event_types(self.events)


class TestMergeLaunchedAfterAgentCommit(_LifecycleHarness):
    """When the agent commits changes, the merge view should still detect
    them by diffing against the pre-task HEAD SHA."""

    def test_merge_after_agent_commit(self) -> None:
        """Agent modifies, stages, and commits — merge should still appear."""
        tab_id = "test-tab-commit-detect"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        Path(self.tmpdir, "README.md").write_text("# Agent committed\n")
        _git(self.tmpdir, "add", "README.md")
        _git(self.tmpdir, "commit", "-m", "agent commit")

        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )

        assert started, (
            "Merge should detect changes even when agent committed them. "
            f"Events: {_event_types(self.events)}"
        )

        md_event = _find_event(self.events, "merge_data")
        assert any(
            f["name"] == "README.md" for f in md_event["data"]["files"]
        )


class TestMergeWithPreExistingDirtyFiles(_LifecycleHarness):
    """When the working tree already has dirty files before the task,
    the merge view should only show the agent's changes, not the
    pre-existing dirty state."""

    def test_pre_dirty_files_filtered(self) -> None:
        """Pre-existing modifications are filtered from the merge view."""
        Path(self.tmpdir, "README.md").write_text("# Pre-existing dirty\n")

        tab_id = "test-tab-predirty"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        Path(self.tmpdir, "new_agent_file.py").write_text("print('hello')\n")

        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )

        assert started
        md_event = _find_event(self.events, "merge_data")
        file_names = [f["name"] for f in md_event["data"]["files"]]
        assert "new_agent_file.py" in file_names
        assert "README.md" not in file_names, (
            "Pre-existing dirty README.md should be filtered from merge view"
        )


class TestAutocommitPromptNotShownForCleanWorkTree(_LifecycleHarness):
    """After merge review, if the working tree is clean (e.g., all
    changes were rejected), no autocommit_prompt should appear."""

    def test_no_prompt_when_clean_after_merge(self) -> None:
        tab_id = "test-tab-clean"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False
        tab.is_merging = True

        self.server._finish_merge(tab_id)

        types = _event_types(self.events)
        assert "merge_ended" in types
        assert "autocommit_prompt" not in types


class TestFinishMergeTabMissing(_LifecycleHarness):
    """BUG REPRODUCTION: when _finish_merge is called on a process that
    doesn't have the tab in _running_agent_states (e.g., merge-action was routed
    to the service process after the task process was disposed), the
    autocommit_prompt must still be emitted for dirty non-worktree tabs.

    This reproduces the bug where merge buttons and autocommit were not
    shown because the tab was missing from _running_agent_states.
    """

    def test_autocommit_prompt_when_tab_not_in_states(self) -> None:
        """_finish_merge must send autocommit_prompt even when the tab
        is not pre-existing in _running_agent_states."""
        tab_id = "missing-tab-id"
        assert tab_id not in _RunningAgentState.running_agent_states

        Path(self.tmpdir, "README.md").write_text("# Modified\n")

        self.server._finish_merge(tab_id)

        types = _event_types(self.events)
        assert "merge_ended" in types, f"merge_ended not found: {types}"
        assert "autocommit_prompt" in types, (
            "autocommit_prompt should be sent even when the tab is missing "
            f"from _running_agent_states. Got: {types}"
        )

    def test_merge_ended_still_sent_when_tab_missing(self) -> None:
        """merge_ended must always be broadcast, even when tab is missing."""
        tab_id = "missing-tab-2"
        assert tab_id not in _RunningAgentState.running_agent_states

        self.server._finish_merge(tab_id)

        types = _event_types(self.events)
        assert "merge_ended" in types


class TestMergeStartSessionEventContent(_LifecycleHarness):
    """Verify that _start_merge_session emits properly structured events
    with tabId and data fields so the TypeScript side can process them."""

    def test_merge_data_has_required_fields(self) -> None:
        """merge_data event must have tabId, data.files, and hunk_count."""
        tab_id = "test-tab-fields"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        Path(self.tmpdir, "README.md").write_text("# Changed for fields test\n")

        self.server._prepare_and_start_merge(
            self.tmpdir, pre_hunks, pre_untracked, pre_file_hashes,
            base_ref=pre_head_sha or "HEAD", tab_id=tab_id,
        )

        md_event = _find_event(self.events, "merge_data")

        assert "tabId" in md_event, "merge_data must have tabId"
        assert md_event["tabId"] == tab_id
        assert "data" in md_event, "merge_data must have data"
        assert "files" in md_event["data"], "merge_data.data must have files"
        assert "hunk_count" in md_event, "merge_data must have hunk_count"
        assert md_event["hunk_count"] > 0

        for f in md_event["data"]["files"]:
            assert "name" in f, f"file entry missing 'name': {f}"
            assert "base" in f, f"file entry missing 'base': {f}"
            assert "current" in f, f"file entry missing 'current': {f}"
            assert "hunks" in f, f"file entry missing 'hunks': {f}"
            assert Path(f["base"]).is_absolute(), f"base path not absolute: {f['base']}"
            assert Path(f["current"]).is_absolute(), f"current path not absolute: {f['current']}"
            assert Path(f["base"]).exists(), f"base path doesn't exist: {f['base']}"
            assert Path(f["current"]).exists(), f"current path doesn't exist: {f['current']}"


class TestAutocommitPromptForBinaryOnlyChanges(_LifecycleHarness):
    """Binary-only modifications (e.g. rebuilt PDFs) are now included in
    the merge view with a dummy hunk so the user can accept/reject them.
    The merge session starts normally and autocommit_prompt is shown
    after the merge review completes."""

    def test_binary_only_triggers_merge_session(self) -> None:
        """Modifying only a binary file starts a merge review session
        so the user can accept or reject the binary change."""
        tab_id = "test-tab-binary"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        # Commit a binary file so it's tracked
        pdf_path = Path(self.tmpdir, "output.pdf")
        pdf_path.write_bytes(b"%PDF-1.4 original content\x00\xff\xd8")
        _git(self.tmpdir, "add", "output.pdf")
        _git(self.tmpdir, "commit", "-q", "-m", "add binary pdf")

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        # Agent rebuilds the PDF (binary-only change)
        pdf_path.write_bytes(b"%PDF-1.4 rebuilt content\x00\xff\xd9\xab\xcd")

        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )

        # Binary files are now included in the merge view
        assert started, (
            "Binary-only changes should start a merge session "
            f"(with binary flag). Events: {_event_types(self.events)}"
        )

        types = _event_types(self.events)
        assert "merge_data" in types
        assert "merge_started" in types

        # The merge data should contain the binary file with binary flag
        md_event = _find_event(self.events, "merge_data")
        files = md_event["data"]["files"]
        assert len(files) == 1
        assert files[0]["name"] == "output.pdf"
        assert files[0].get("binary") is True

    def test_binary_plus_text_goes_through_merge(self) -> None:
        """When both binary and text files change, the text changes get
        a merge review and autocommit_prompt comes via _finish_merge."""
        tab_id = "test-tab-mixed"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        pdf_path = Path(self.tmpdir, "output.pdf")
        pdf_path.write_bytes(b"%PDF-1.4 original\x00\xff")
        _git(self.tmpdir, "add", "output.pdf")
        _git(self.tmpdir, "commit", "-q", "-m", "add pdf")

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(
                self.tmpdir, repo, tab_id,
            )
        )

        # Agent modifies both a text file and a binary file
        Path(self.tmpdir, "README.md").write_text("# Updated by agent\n")
        pdf_path.write_bytes(b"%PDF-1.4 rebuilt\x00\xff\xab")

        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )

        # Text hunks exist → merge session starts
        assert started, (
            "Mixed binary+text changes should start a merge session "
            f"for the text hunks. Events: {_event_types(self.events)}"
        )

        # After merge review, autocommit_prompt includes both files
        self.events.clear()
        self.server._finish_merge(tab_id)

        types = _event_types(self.events)
        assert "autocommit_prompt" in types
        ac_event = _find_event(self.events, "autocommit_prompt")
        assert "README.md" in ac_event["changedFiles"]
        assert "output.pdf" in ac_event["changedFiles"]


class TestAutocommitUsesCommandWorkDir(unittest.TestCase):
    """The autocommit button must act on the work_dir carried by the
    command (the tab's own folder), not the daemon-wide ``self.work_dir``.

    Reproduces the reported bug: the shared ``kiss-web`` daemon was
    launched from (or synced to) a non-git folder, so clicking the
    auto-commit button reported ``"Not a git repository."`` even though
    the tab's workspace was a real git repository.  The frontend now
    stamps the tab's ``workDir`` on the ``autocommitAction`` message and
    the backend prefers it over ``self.work_dir``.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        # The daemon-wide work_dir points at a *non-git* folder.
        self.nongit = tempfile.mkdtemp()
        if _git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff

        def fake_gen(diff_text: str, user_prompt: str | None = None) -> str:
            return "chore: auto-commit test"

        _merge_flow_module.generate_commit_message_from_diff = fake_gen  # type: ignore[assignment]

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)

    def test_commit_uses_command_work_dir(self) -> None:
        """A dirty git repo passed via ``workDir`` commits successfully
        even though ``self.work_dir`` is a non-git folder."""
        tab_id = "t-wd"
        self.server._get_tab(tab_id)
        Path(self.repo, "new.txt").write_text("hello\n")

        self.server._handle_command({
            "type": "autocommitAction",
            "action": "commit",
            "tabId": tab_id,
            "workDir": self.repo,
        })

        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is True, done
        assert done["committed"] is True, done
        status = _git(self.repo, "status", "--porcelain").stdout.strip()
        assert status == "", f"Expected clean tree, got: {status}"

    def test_commit_without_work_dir_still_fails_on_nongit(self) -> None:
        """Without a ``workDir`` the handler falls back to the non-git
        ``self.work_dir`` and reports the original failure."""
        tab_id = "t-nowd"
        self.server._get_tab(tab_id)
        self.server._handle_command({
            "type": "autocommitAction",
            "action": "commit",
            "tabId": tab_id,
        })
        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is False, done
        assert done["message"] == "Not a git repository.", done


class TestCommitMessageUsesCommandWorkDir(unittest.TestCase):
    """``generateCommitMessage`` must act on the work_dir carried by the
    command (the tab's own folder), not the daemon-wide ``self.work_dir``.

    Reproduces the reported bug: the shared ``kiss-web`` daemon was
    launched from (or synced to) a non-git folder, so requesting a
    commit message reported ``"Not a git repository."`` even though the
    tab's workspace was a real git repository.  The frontend now stamps
    the tab's ``workDir`` on the ``generateCommitMessage`` command and
    the backend prefers it over ``self.work_dir``.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        # The daemon-wide work_dir points at a *non-git* folder.
        self.nongit = tempfile.mkdtemp()
        if _git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)

    def tearDown(self) -> None:
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)

    def test_no_git_error_with_command_work_dir(self) -> None:
        """A git repo passed via ``workDir`` is recognised even though
        ``self.work_dir`` is a non-git folder.

        With no staged changes the handler proceeds past the repo check
        and reports the (non-misleading) "no staged changes" error
        instead of "Not a git repository.".
        """
        tab_id = "t-cm"
        self.server._generate_commit_message(tab_id, work_dir=self.repo)
        msg = _find_event(self.events, "commitMessage")
        assert msg.get("error") != "Not a git repository.", msg
        assert "No staged changes" in msg.get("error", ""), msg

    def test_git_error_without_work_dir_on_nongit(self) -> None:
        """Without a ``workDir`` the handler falls back to the non-git
        ``self.work_dir`` and reports "Not a git repository."."""
        tab_id = "t-cm2"
        self.server._generate_commit_message(tab_id)
        msg = _find_event(self.events, "commitMessage")
        assert msg.get("error") == "Not a git repository.", msg

    def test_command_dispatch_forwards_work_dir(self) -> None:
        """``generateCommitMessage`` command forwards ``workDir`` so the
        repo check runs against the tab's folder, not ``self.work_dir``."""
        import time

        self.server._handle_command({
            "type": "generateCommitMessage",
            "tabId": "t-cm3",
            "workDir": self.repo,
        })
        # The handler runs in a daemon thread; wait briefly for the event.
        for _ in range(50):
            if any(e["type"] == "commitMessage" for e in self.events):
                break
            time.sleep(0.05)
        msg = _find_event(self.events, "commitMessage")
        assert msg.get("error") != "Not a git repository.", msg


class TestMainDirtyFilesUsesCommandWorkDir(unittest.TestCase):
    """``_main_dirty_files`` must scan the work_dir passed by the caller
    (the tab's own folder), not the daemon-wide ``self.work_dir``.

    Mirrors the autocommit / commit-message fixes: the shared
    ``kiss-web`` daemon may run from a non-git folder, so the post-merge
    dirty-file scan must use the tab's repository to detect changes.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        self.nongit = tempfile.mkdtemp()
        if _git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)

    def tearDown(self) -> None:
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)

    def test_scan_uses_command_work_dir(self) -> None:
        """A dirty file in the command repo is reported even though
        ``self.work_dir`` is a non-git folder."""
        Path(self.repo, "new.txt").write_text("hello\n")
        changed = self.server._main_dirty_files(self.repo)
        assert "new.txt" in changed, changed

    def test_scan_without_work_dir_returns_empty_on_nongit(self) -> None:
        """Without a work_dir the scan falls back to the non-git
        ``self.work_dir`` and reports no changes."""
        Path(self.repo, "new.txt").write_text("hello\n")
        changed = self.server._main_dirty_files()
        assert changed == [], changed


class TestFinishMergeUsesCommandWorkDir(unittest.TestCase):
    """``_finish_merge`` / ``_broadcast_autocommit_prompt`` must run the
    post-merge dirty-file scan against the work_dir carried by the
    command (the tab's own folder), not the daemon-wide ``self.work_dir``.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        self.nongit = tempfile.mkdtemp()
        if _git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)

    def tearDown(self) -> None:
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)

    def test_broadcast_prompt_uses_command_work_dir(self) -> None:
        """A dirty command repo yields an ``autocommit_prompt`` even
        though ``self.work_dir`` is a non-git folder."""
        tab_id = "t-fm"
        self.server._get_tab(tab_id)
        Path(self.repo, "new.txt").write_text("hello\n")
        self.server._broadcast_autocommit_prompt(tab_id, self.repo)
        prompt = _find_event(self.events, "autocommit_prompt")
        assert "new.txt" in prompt["changedFiles"], prompt

    def test_broadcast_prompt_without_work_dir_no_event_on_nongit(self) -> None:
        """Without a work_dir the scan falls back to the non-git
        ``self.work_dir`` and no prompt is emitted."""
        tab_id = "t-fm2"
        self.server._get_tab(tab_id)
        Path(self.repo, "new.txt").write_text("hello\n")
        self.server._broadcast_autocommit_prompt(tab_id)
        assert not any(
            e["type"] == "autocommit_prompt" for e in self.events
        ), _event_types(self.events)

    def test_finish_merge_forwards_work_dir(self) -> None:
        """``_finish_merge`` forwards ``work_dir`` so the dirty-file scan
        runs against the tab's repo, emitting an ``autocommit_prompt``."""
        tab_id = "t-fm3"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False
        Path(self.repo, "new.txt").write_text("hello\n")
        self.server._finish_merge(tab_id, work_dir=self.repo)
        _find_event(self.events, "merge_ended")
        prompt = _find_event(self.events, "autocommit_prompt")
        assert "new.txt" in prompt["changedFiles"], prompt

    def test_finish_merge_without_work_dir_no_prompt_on_nongit(self) -> None:
        """Without a ``work_dir`` ``_finish_merge`` falls back to the
        non-git ``self.work_dir`` and emits no ``autocommit_prompt``."""
        tab_id = "t-fm4"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False
        Path(self.repo, "new.txt").write_text("hello\n")
        self.server._finish_merge(tab_id)
        _find_event(self.events, "merge_ended")
        assert not any(
            e["type"] == "autocommit_prompt" for e in self.events
        ), _event_types(self.events)


class TestMergeActionDispatchForwardsWorkDir(unittest.TestCase):
    """The ``mergeAction`` ``all-done`` command must forward its
    ``workDir`` to ``_finish_merge`` so the post-merge dirty-file scan
    runs against the tab's repo, not the daemon-wide ``self.work_dir``.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        self.nongit = tempfile.mkdtemp()
        if _git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)

    def tearDown(self) -> None:
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)

    def test_all_done_command_forwards_work_dir(self) -> None:
        """An ``all-done`` command carrying ``workDir`` triggers an
        ``autocommit_prompt`` for the tab's dirty repo."""
        tab_id = "t-md"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False
        Path(self.repo, "new.txt").write_text("hello\n")
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": tab_id,
            "workDir": self.repo,
        })
        prompt = _find_event(self.events, "autocommit_prompt")
        assert "new.txt" in prompt["changedFiles"], prompt


class TestStartMergeSessionStampsWorkDir(_LifecycleHarness):
    """``_start_merge_session`` must stamp ``work_dir`` into the
    ``merge_data`` payload so the shared web daemon can echo it back on
    the ``all-done`` command (web all-done autocommit path)."""

    def test_merge_data_carries_work_dir(self) -> None:
        """The broadcast ``merge_data`` event's ``data`` includes the
        repository ``work_dir`` passed to ``_prepare_and_start_merge``."""
        tab_id = "test-tab-wd"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False

        from kiss.agents.sorcar.git_worktree import GitWorktreeOps

        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_file_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(self.tmpdir, repo, tab_id)
        )
        Path(self.tmpdir, "README.md").write_text(
            "# Hello\n\nUpdated content by the agent\n"
        )
        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_file_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=tab_id,
        )
        assert started, _event_types(self.events)
        merge_data = _find_event(self.events, "merge_data")
        assert merge_data["data"].get("work_dir") == self.tmpdir, merge_data


if __name__ == "__main__":
    unittest.main()

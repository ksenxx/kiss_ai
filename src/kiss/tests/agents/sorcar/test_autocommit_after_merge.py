"""Integration tests for the non-worktree auto-commit prompt feature.

After the user resolves all merge-diff hunks in the non-worktree mode,
the server must inspect the main working tree.  When uncommitted or
untracked changes remain, it broadcasts an ``autocommit_prompt`` event
so the webview can render "Auto commit" / "Do nothing" buttons.  When
the user clicks "Auto commit", the server stages everything, generates
a commit message and commits to the current branch.  When the user
clicks "Do nothing", the server leaves the working tree untouched.

These tests drive :class:`VSCodeServer` with real ``git`` state — no
mocks, no test doubles.  The LLM call for commit-message generation is
replaced with a deterministic override of the module-level
``generate_commit_message_from_diff`` function in ``merge_flow``,
since this is a module-level extension point, not a mock / patch of
dependency internals.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import kiss.agents.vscode.merge_flow as _merge_flow_module
from kiss.agents.vscode.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    """Create a git repo with one committed file so HEAD exists."""
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


class _ServerHarness(unittest.TestCase):
    """Shared setUp/tearDown — real git repo + event capture."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff

        def capture(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def _event(self, type_: str) -> dict:
        for e in self.events:
            if e["type"] == type_:
                return e
        raise AssertionError(f"No event of type {type_!r}: {self._types()}")


class TestFinishMergeEmitsAutocommitPrompt(_ServerHarness):
    """``_finish_merge`` broadcasts ``autocommit_prompt`` for non-worktree
    tabs when the main working tree has dirty state after hunk review."""

    def test_prompt_with_unstaged_modification(self) -> None:
        """Modifying a tracked file fires autocommit_prompt with the file."""
        tab = self.server._get_tab("t1")
        tab.use_worktree = False
        tab.is_merging = True
        Path(self.tmpdir, "seed.txt").write_text("modified\n")
        self.server._finish_merge("t1")

        evt = self._event("autocommit_prompt")
        assert evt["tabId"] == "t1"
        assert "seed.txt" in evt["changedFiles"]

    def test_prompt_with_untracked_file(self) -> None:
        """Untracked files trigger the prompt too."""
        tab = self.server._get_tab("t2")
        tab.use_worktree = False
        tab.is_merging = True
        Path(self.tmpdir, "new.txt").write_text("hi\n")
        self.server._finish_merge("t2")

        evt = self._event("autocommit_prompt")
        assert evt["tabId"] == "t2"
        assert "new.txt" in evt["changedFiles"]

    def test_no_prompt_when_clean(self) -> None:
        """No autocommit_prompt when working tree is clean."""
        tab = self.server._get_tab("t3")
        tab.use_worktree = False
        tab.is_merging = True
        self.server._finish_merge("t3")

        assert "autocommit_prompt" not in self._types()

    def test_no_prompt_for_worktree_tab(self) -> None:
        """Worktree tabs keep their own merge/discard flow; the
        non-worktree prompt must not fire for them."""
        tab = self.server._get_tab("t4")
        tab.use_worktree = True
        tab.is_merging = True
        Path(self.tmpdir, "seed.txt").write_text("x\n")
        self.server._finish_merge("t4")

        assert "autocommit_prompt" not in self._types()

    def test_no_prompt_when_tab_id_none(self) -> None:
        """When tab_id is None (global clear), no per-tab prompt is sent."""
        Path(self.tmpdir, "x.txt").write_text("x\n")
        self.server._finish_merge(None)  # type: ignore[arg-type]

        assert "autocommit_prompt" not in self._types()

    def test_no_prompt_when_not_a_git_repo(self) -> None:
        """A non-git work_dir must not crash or broadcast the prompt."""
        non_git = tempfile.mkdtemp()
        try:
            self.server.work_dir = non_git
            tab = self.server._get_tab("t5")
            tab.use_worktree = False
            tab.is_merging = True
            Path(non_git, "loose.txt").write_text("x\n")
            self.server._finish_merge("t5")
            assert "autocommit_prompt" not in self._types()
        finally:
            shutil.rmtree(non_git, ignore_errors=True)

    def test_merge_ended_still_broadcast(self) -> None:
        """The existing merge_ended event is still sent before the prompt."""
        tab = self.server._get_tab("t6")
        tab.use_worktree = False
        tab.is_merging = True
        Path(self.tmpdir, "foo.txt").write_text("foo\n")
        self.server._finish_merge("t6")

        types = self._types()
        assert "merge_ended" in types
        assert "autocommit_prompt" in types
        assert types.index("merge_ended") < types.index("autocommit_prompt")


class TestAutocommitActionSkip(_ServerHarness):
    """Clicking "Do nothing" leaves the working tree untouched and
    broadcasts ``autocommit_done``."""

    def test_skip_leaves_working_tree_dirty(self) -> None:
        Path(self.tmpdir, "seed.txt").write_text("modified\n")
        Path(self.tmpdir, "new.txt").write_text("new\n")
        self.server._get_tab("t1").use_worktree = False

        self.server._handle_autocommit_action("skip", "t1")

        status = _run_git(self.tmpdir, "status", "--porcelain").stdout
        assert "seed.txt" in status
        assert "new.txt" in status

        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["tabId"] == "t1"
        assert evt["committed"] is False

    def test_skip_does_not_create_commit(self) -> None:
        Path(self.tmpdir, "seed.txt").write_text("modified\n")
        self.server._get_tab("t1").use_worktree = False
        before = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()

        self.server._handle_autocommit_action("skip", "t1")

        after = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        assert before == after


class TestAutocommitActionCommit(_ServerHarness):
    """Clicking "Auto commit" stages everything (including untracked),
    generates a commit message, and commits to the current branch."""

    def setUp(self) -> None:
        super().setUp()
        self._messages: list[str] = []

        def fake_compose(diff_text: str) -> str:
            self._messages.append(diff_text)
            return "feat: deterministic test commit"

        _merge_flow_module.generate_commit_message_from_diff = fake_compose  # type: ignore[assignment]

    def test_commit_stages_and_commits_tracked_and_untracked(self) -> None:
        Path(self.tmpdir, "seed.txt").write_text("updated seed\n")
        Path(self.tmpdir, "new.txt").write_text("brand new\n")
        self.server._get_tab("t1").use_worktree = False

        before = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        self.server._handle_autocommit_action("commit", "t1")
        after = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()

        assert before != after
        status = _run_git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert status == ""
        log = _run_git(self.tmpdir, "log", "-1", "--pretty=%s").stdout.strip()
        assert log == "feat: deterministic test commit"
        show = _run_git(
            self.tmpdir, "show", "--name-only", "--pretty=", "HEAD",
        ).stdout
        assert "seed.txt" in show
        assert "new.txt" in show
        assert len(self._messages) == 1
        assert "seed.txt" in self._messages[0] or "new.txt" in self._messages[0]

        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["committed"] is True
        assert evt["tabId"] == "t1"

    def test_commit_with_only_untracked(self) -> None:
        """Commit handles the case with only untracked files."""
        Path(self.tmpdir, "only_new.txt").write_text("hi\n")
        self.server._get_tab("t1").use_worktree = False

        self.server._handle_autocommit_action("commit", "t1")

        status = _run_git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert status == ""
        show = _run_git(
            self.tmpdir, "show", "--name-only", "--pretty=", "HEAD",
        ).stdout
        assert "only_new.txt" in show
        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["committed"] is True

    def test_commit_when_nothing_to_commit(self) -> None:
        """If there's nothing to commit (race), broadcast success with
        ``committed: False`` instead of failing."""
        self.server._get_tab("t1").use_worktree = False
        before = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()

        self.server._handle_autocommit_action("commit", "t1")

        after = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        assert before == after
        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["committed"] is False
        assert self._messages == []

    def test_commit_in_non_git_dir_reports_failure(self) -> None:
        non_git = tempfile.mkdtemp()
        try:
            self.server.work_dir = non_git
            self.server._get_tab("t1").use_worktree = False
            self.server._handle_autocommit_action("commit", "t1")
            evt = self._event("autocommit_done")
            assert evt["success"] is False
            assert evt["committed"] is False
        finally:
            shutil.rmtree(non_git, ignore_errors=True)


class TestAutocommitUnknownAction(_ServerHarness):
    """Unknown actions are reported as failure — no silent success."""

    def test_unknown_action(self) -> None:
        self.server._handle_autocommit_action("bogus", "t1")
        evt = self._event("autocommit_done")
        assert evt["success"] is False
        assert evt["committed"] is False


class TestAutocommitCommandRouting(_ServerHarness):
    """The ``autocommitAction`` command type is routed through
    ``_handle_command``."""

    def test_routed_skip(self) -> None:
        Path(self.tmpdir, "seed.txt").write_text("x\n")
        self.server._handle_command(
            {"type": "autocommitAction", "action": "skip", "tabId": "t1"},
        )
        evt = self._event("autocommit_done")
        assert evt["committed"] is False
        assert evt["success"] is True

    def test_routed_commit(self) -> None:
        Path(self.tmpdir, "seed.txt").write_text("x\n")

        def fake_compose(diff_text: str) -> str:
            return "chore: test"

        _merge_flow_module.generate_commit_message_from_diff = fake_compose  # type: ignore[assignment]
        self.server._handle_command(
            {"type": "autocommitAction", "action": "commit", "tabId": "t1"},
        )
        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["committed"] is True

    def test_unknown_cmd_still_rejected(self) -> None:
        """Safety check: other unknown command types still broadcast
        the generic unknown-command error."""
        self.server._handle_command({"type": "nosuchcmd"})
        evt = self._event("error")
        assert "Unknown command" in evt["text"]


class TestAutocommitPromptRoundtrip(_ServerHarness):
    """End-to-end: finish merge review → prompt event → user clicks
    "Auto commit" → repo committed; or → user clicks "Do nothing" →
    repo untouched."""

    def test_full_auto_commit_flow(self) -> None:
        tab = self.server._get_tab("t1")
        tab.use_worktree = False
        tab.is_merging = True
        Path(self.tmpdir, "seed.txt").write_text("delta\n")
        Path(self.tmpdir, "extra.txt").write_text("extra\n")

        self.server._finish_merge("t1")
        prompt = self._event("autocommit_prompt")
        assert set(prompt["changedFiles"]) >= {"seed.txt", "extra.txt"}

        def fake_compose(_diff: str) -> str:
            return "chore: auto"

        _merge_flow_module.generate_commit_message_from_diff = fake_compose  # type: ignore[assignment]
        self.server._handle_command(
            {"type": "autocommitAction", "action": "commit", "tabId": "t1"},
        )
        done = self._event("autocommit_done")
        assert done["success"] is True
        assert done["committed"] is True

        assert _run_git(
            self.tmpdir, "status", "--porcelain",
        ).stdout.strip() == ""

    def test_full_do_nothing_flow(self) -> None:
        tab = self.server._get_tab("t1")
        tab.use_worktree = False
        tab.is_merging = True
        Path(self.tmpdir, "seed.txt").write_text("untouched dirty\n")

        self.server._finish_merge("t1")
        self._event("autocommit_prompt")
        before = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()

        self.server._handle_command(
            {"type": "autocommitAction", "action": "skip", "tabId": "t1"},
        )

        after = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        assert before == after
        assert "seed.txt" in _run_git(
            self.tmpdir, "status", "--porcelain",
        ).stdout


class TestAutocommitTypesContract(unittest.TestCase):
    """The frontend type-definitions file must advertise the new
    message and command types so the TS compiler picks them up."""

    types_ts: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.types_ts = (
            base / "vscode" / "src" / "types.ts"
        ).read_text()

    def test_autocommit_prompt_event_declared(self) -> None:
        assert "autocommit_prompt" in self.types_ts

    def test_autocommit_done_event_declared(self) -> None:
        assert "autocommit_done" in self.types_ts

    def test_autocommit_action_command_declared(self) -> None:
        assert "autocommitAction" in self.types_ts


class TestMainJsRendersAutocommitButtons(unittest.TestCase):
    """``main.js`` must render "Auto commit" and "Do nothing" buttons
    in the input textarea when an ``autocommit_prompt`` event is
    received."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()

    def test_handles_autocommit_prompt_event(self) -> None:
        assert "autocommit_prompt" in self.js

    def test_has_auto_commit_button_label(self) -> None:
        assert "Auto commit" in self.js

    def test_has_do_nothing_button_label(self) -> None:
        assert "Do nothing" in self.js

    def test_sends_autocommit_action_commit(self) -> None:
        assert "autocommitAction" in self.js
        assert "'commit'" in self.js or '"commit"' in self.js
        assert "'skip'" in self.js or '"skip"' in self.js


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

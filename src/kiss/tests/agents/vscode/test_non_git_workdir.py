# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: every VS Code extension command works in a
non-git working directory without raising and without emitting
spurious ``error`` events.

The VS Code extension is meant to be usable in any folder — including
folders that are not inside a git repository.  Git-specific
functionality (worktree, autocommit, commit-message) must fail
gracefully with informative messages instead of crashing or producing
misleading output.

These tests exercise every command type registered in
``_CommandsMixin._HANDLERS`` against a plain ``tempfile.mkdtemp()``
work_dir that has no ``.git`` folder.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _is_inside_git_repo(path: str) -> bool:
    """True when *path* is recognised by git as part of a repository."""
    res = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=path, capture_output=True, text=True,
    )
    return res.returncode == 0 and res.stdout.strip() == "true"


def _register_tab_state(
    tab_id: str,
    *,
    agent: WorktreeSorcarAgent | None = None,
) -> agent_state.AgentState:
    """Register an idle server-owned state for *tab_id* and return it."""
    state = agent_state.AgentState(
        uuid.uuid4().hex,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
    )
    agent_state.register(state)
    return state


class _NonGitHarness(unittest.TestCase):
    """Common setUp: a non-git temp work_dir + event capture."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        if _is_inside_git_repo(self.tmpdir):
            self.skipTest(
                f"tempdir {self.tmpdir} is inside a git repo; "
                "cannot test non-git behavior here",
            )
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        orig = self.server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with self._lock:
                self.events.append(dict(e))
            orig(e)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _types(self) -> list[str]:
        with self._lock:
            return [e["type"] for e in self.events]

    def _events_of(self, type_: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self.events if e["type"] == type_]


class TestNonGitCommandsDoNotCrash(_NonGitHarness):
    """Every command type must dispatch without raising in a non-git dir."""

    def test_get_models(self) -> None:
        self.server._handle_command({"type": "getModels"})
        assert self._events_of("models")

    def test_get_history(self) -> None:
        self.server._handle_command({"type": "getHistory"})
        assert self._events_of("history")

    def test_get_input_history(self) -> None:
        self.server._handle_command({"type": "getInputHistory"})
        assert self._events_of("inputHistory")

    def test_get_files(self) -> None:
        import time

        Path(self.tmpdir, "alpha.txt").write_text("a\n")
        Path(self.tmpdir, "beta.py").write_text("b\n")
        self.server._handle_command({"type": "getFiles", "prefix": ""})
        for _ in range(50):
            evt = self._events_of("files")
            if evt and evt[-1].get("files"):
                break
            time.sleep(0.05)
        evt = self._events_of("files")
        assert evt
        names = {f["text"].lstrip("./") for f in evt[-1]["files"]}
        assert "alpha.txt" in names
        assert "beta.py" in names

    def test_record_file_usage(self) -> None:
        self.server._handle_command(
            {"type": "recordFileUsage", "path": "foo.txt"},
        )

    def test_select_model(self) -> None:
        from kiss.core.models.model_info import get_default_model

        m = get_default_model()
        self.server._handle_command(
            {"type": "selectModel", "tabId": "t-sel", "model": m},
        )
        assert self.server._tab_models["t-sel"] == m

    def test_new_chat(self) -> None:
        self.server._handle_command({"type": "newChat", "tabId": "t-nc"})
        assert any(e.get("type") == "showWelcome" for e in self.events)

    def test_close_tab_clean(self) -> None:
        state = _register_tab_state("t-close")
        self.server._handle_command({"type": "closeTab", "tabId": "t-close"})
        assert agent_state.get(state.task_id) is None

    def test_user_answer_no_queue(self) -> None:
        self.server._handle_command(
            {"type": "userAnswer", "tabId": "t-noq", "answer": "hi"},
        )

    def test_resume_session_unknown_chat(self) -> None:
        self.server._handle_command(
            {"type": "resumeSession", "chatId": "no-such", "tabId": "t-rs"},
        )

    def test_get_adjacent_task_no_history(self) -> None:
        self.server._handle_command(
            {"type": "getAdjacentTask", "tabId": "t-adj",
             "taskId": None, "direction": "prev"},
        )
        evt = self._events_of("adjacent_task_events")
        assert evt and evt[-1]["task"] == ""

    def test_delete_task_is_unknown_command(self) -> None:
        self.server._handle_command(
            {"type": "deleteTask", "taskId": 999_999_999},
        )
        assert self._events_of("error"), (
            "deleteTask was removed and must be reported as unknown"
        )

    def test_get_config(self) -> None:
        self.server._handle_command({"type": "getConfig"})
        assert self._events_of("configData")

    def test_unknown_command(self) -> None:
        self.server._handle_command({"type": "doesNotExist"})
        errs = self._events_of("error")
        assert any("Unknown command" in e.get("text", "") for e in errs)

    def test_merge_action_is_unknown_command(self) -> None:
        """The interactive merge review was removed: ``mergeAction`` is
        an unknown command and must not touch the tab's merge flag."""
        state = _register_tab_state("t-merge")
        state.is_merging = True
        self.server._handle_command(
            {"type": "mergeAction", "action": "all-done", "tabId": "t-merge"},
        )
        errs = self._events_of("error")
        assert any("Unknown command" in e.get("text", "") for e in errs)
        assert state.is_merging is True


class TestNonGitWorktreeActions(_NonGitHarness):
    """Worktree actions must report failures, never crash."""

    def test_worktree_action_when_not_enabled(self) -> None:
        self.server._handle_command(
            {"type": "worktreeAction", "action": "merge", "tabId": "t-wt"},
        )
        evt = self._events_of("worktree_result")
        assert evt
        assert evt[-1]["success"] is False
        assert "Worktree mode is not enabled" in evt[-1]["message"]

    def test_worktree_action_discard_not_enabled(self) -> None:
        self.server._handle_command(
            {"type": "worktreeAction", "action": "discard", "tabId": "t-wt2"},
        )
        evt = self._events_of("worktree_result")
        assert evt and evt[-1]["success"] is False

    def test_worktree_action_unknown(self) -> None:
        state = _register_tab_state(
            "t-wt3", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        state.use_worktree = True
        self.server._handle_command(
            {"type": "worktreeAction", "action": "frobnicate", "tabId": "t-wt3"},
        )
        evt = self._events_of("worktree_result")
        assert evt and evt[-1]["success"] is False


class TestNonGitAutocommit(_NonGitHarness):
    """The post-task autocommit path must not crash in non-git."""

    def test_autocommit_reports_not_a_git_repository(self) -> None:
        """``_autocommit_changes`` in a non-git dir reports a graceful
        failure through ``autocommit_done`` instead of crashing."""
        _register_tab_state("t2").use_worktree = False
        Path(self.tmpdir, "loose.txt").write_text("x\n")
        self.server._autocommit_changes("t2", work_dir=self.tmpdir)
        evt = self._events_of("autocommit_done")
        assert evt
        assert evt[-1]["success"] is False
        assert evt[-1]["committed"] is False
        assert evt[-1]["message"] == "Not a git repository."


class TestNonGitGenerateCommitMessage(_NonGitHarness):
    """generateCommitMessage in non-git must broadcast a commitMessage
    event with an informative error — never crash."""

    def test_generate_commit_message(self) -> None:
        self.server._handle_command(
            {"type": "generateCommitMessage", "tabId": "t-gen"},
        )
        import time as _t
        for _ in range(50):
            if self._events_of("commitMessage"):
                break
            _t.sleep(0.02)
        evt = self._events_of("commitMessage")
        assert evt, "expected a commitMessage event"
        last = evt[-1]
        assert last.get("message") == ""
        assert last.get("error") == "Not a git repository."


class TestNonGitRunTask(_NonGitHarness):
    """Driving _run_task in a non-git dir must complete cleanly:
    no merge view, no error events, status broadcasts both ways."""

    def _patch_agent_run(
        self, tab_id: str,
    ) -> tuple[dict[str, Any], WorktreeSorcarAgent]:
        """Register a tab state whose agent.run is a no-op that
        simulates a clean agent invocation creating a single new file."""
        called: dict[str, Any] = {"called": False, "kwargs": None}
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        state = _register_tab_state(tab_id, agent=agent)
        state.stop_event = threading.Event()
        state.user_answer_queue = queue.Queue()

        def fake_run(**kwargs: Any) -> str:
            called["called"] = True
            called["kwargs"] = kwargs
            Path(self.tmpdir, "agent_output.txt").write_text("hi\n")
            agent.total_tokens_used = 10
            agent.budget_used = 0.001
            agent.step_count = 1
            agent._last_task_id = None
            printer = kwargs.get("printer", self.server.printer)
            printer.print(
                "success: true\nsummary: ok",
                type="result",
                total_tokens=10,
                cost="$0.0010",
                step_count=1,
            )
            return "success: true\nsummary: ok"

        agent.run = fake_run  # type: ignore[assignment]
        return called, agent

    def test_run_task_non_worktree(self) -> None:
        from kiss.core import config as config_module
        keys = config_module.DEFAULT_CONFIG
        saved = keys.ANTHROPIC_API_KEY
        try:
            keys.ANTHROPIC_API_KEY = "test-key"
            tab_id = "t-run"
            called, _agent = self._patch_agent_run(tab_id)

            self.server._run_task({
                "type": "run",
                "prompt": "create a file",
                "tabId": tab_id,
                "workDir": self.tmpdir,
                "useWorktree": False,
            })

            assert called["called"], "agent.run() must be called"
            statuses = [
                e for e in self.events if e.get("type") == "status"
            ]
            assert any(s.get("running") is True for s in statuses)
            assert any(s.get("running") is False for s in statuses)
            errors = self._events_of("error")
            assert not errors, f"unexpected error events: {errors}"
            assert "merge_data" not in self._types()
            assert "merge_started" not in self._types()
            assert Path(self.tmpdir, "agent_output.txt").is_file()
        finally:
            keys.ANTHROPIC_API_KEY = saved

    def test_run_task_with_use_worktree_falls_back(self) -> None:
        """In a non-git dir, useWorktree=True must fall back to direct
        execution (per WorktreeSorcarAgent.run) and complete cleanly
        with no merge_started event and no errors."""
        from kiss.core import config as config_module
        keys = config_module.DEFAULT_CONFIG
        saved = keys.ANTHROPIC_API_KEY
        try:
            keys.ANTHROPIC_API_KEY = "test-key"
            tab_id = "t-wt-fallback"
            called, agent = self._patch_agent_run(tab_id)

            self.server._run_task({
                "type": "run",
                "prompt": "do work",
                "tabId": tab_id,
                "workDir": self.tmpdir,
                "useWorktree": True,
            })

            assert called["called"]
            errors = self._events_of("error")
            assert not errors, f"unexpected error events: {errors}"
            assert "merge_started" not in self._types()
            assert agent._wt_branch is None
            assert agent._wt_pending is False
        finally:
            keys.ANTHROPIC_API_KEY = saved


if __name__ == "__main__":
    unittest.main()

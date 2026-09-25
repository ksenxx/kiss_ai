# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the merge SEA resolving an auto-commit merge conflict.

Two layers are exercised with real git repositories and no test
doubles:

* :func:`kiss.server.merge_conflict_resolver.resolve_merge_conflict`
  through :meth:`WorktreeSorcarAgent.merge`, with the merge agent
  replaced by plain Python runners that resolve, ignore, or fail —
  the runner is an injection point of the function under test, not a
  patch — and the message / branch / stash outcomes of each.
* The whole auto-commit run through the real task runner: a stand-in
  OpenAI endpoint scripts the task agent (edit a line in the
  worktree), the test commits a competing edit on the user's branch
  meanwhile, and the same endpoint then scripts the merge SEA (resolve
  and ``git add``).  The task's persisted tokens / cost / steps must
  include the merge agent's, and the tab must receive the new totals.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas import merge_sea
from kiss.agents.sorcar import persistence, sea_commands
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, MergeResult, _git
from kiss.agents.sorcar.persistence import _add_task, _add_task_usage
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.merge_conflict_resolver import resolve_merge_conflict
from kiss.server.merge_flow import _MergeFlowMixin
from kiss.server.server import VSCodeServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    run_git,
    tool_call_response,
)
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials

RESOLVED = "a\nB-both\nc\n"


def _usage_rows() -> list[dict[str, Any]]:
    """Return every ``task_history`` row of the isolated DB with its usage."""
    cursor = persistence._get_db().execute(
        "SELECT id, parent_task_id, task, tokens, cost, steps FROM task_history"
    )
    keys = ("id", "parent_task_id", "task", "tokens", "cost", "steps")
    return [dict(zip(keys, row, strict=True)) for row in cursor.fetchall()]


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    for key, value in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(["git", "-C", str(path), "config", key, value], check=True)
    (path / "f.txt").write_text("a\nb\nc\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)
    return path


def _resolving_runner(parent_agent: Any, prompt: str, repo: Path) -> None:
    """Stand-in for the merge agent: resolve ``f.txt`` and stage it."""
    assert "Incoming task branch (theirs)" in prompt
    (repo / "f.txt").write_text(RESOLVED)
    _git("add", "f.txt", cwd=repo)


def _idle_runner(parent_agent: Any, prompt: str, repo: Path) -> None:
    """Stand-in for a merge agent that gives up without touching anything."""


def _raising_runner(parent_agent: Any, prompt: str, repo: Path) -> None:
    raise RuntimeError("model endpoint down")


def _interrupted_runner(parent_agent: Any, prompt: str, repo: Path) -> None:
    raise KeyboardInterrupt


def _stash_dropping_interrupted_runner(parent_agent: Any, prompt: str, repo: Path) -> None:
    """A runner that destroys the user's stash entry, then is stopped."""
    assert _git("stash", "drop", cwd=repo).returncode == 0
    raise KeyboardInterrupt


def _rogue_committing_runner(parent_agent: Any, prompt: str, repo: Path) -> None:
    """A runner that ignores the rules: drops the merge and commits junk."""
    _git("reset", "--hard", "HEAD", cwd=repo)
    (repo / "f.txt").write_text("junk\n")
    _git("commit", "-qam", "rogue commit", cwd=repo)


class TestResolveMergeConflictThroughMerge:
    """``WorktreeSorcarAgent.merge(conflict_resolver=...)`` with real git."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.agent = WorktreeSorcarAgent("test")
        assert self.agent._try_setup_worktree(self.repo, None) is not None
        wt = self.agent._wt
        assert wt is not None
        self.wt = wt
        (wt.wt_dir / "f.txt").write_text("a\nB-agent\nc\n")
        assert GitWorktreeOps.commit_all(wt.wt_dir, "agent work")
        (self.repo / "f.txt").write_text("a\nB-user\nc\n")
        _git("commit", "-qam", "user work", cwd=self.repo)
        self.agent._last_user_prompt = "change line b"
        self.agent._last_result_summary = "changed line b"

    def teardown_method(self) -> None:
        if self.agent._wt is not None:
            try:
                self.agent.discard()
            except Exception:
                pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _resolver(self, runner: Any) -> Any:
        def resolver(agent: Any, wt: Any, user_prompt: Any, task_result: Any) -> MergeResult:
            return resolve_merge_conflict(
                agent, wt, user_prompt, task_result, run_agent=runner,
            )
        return resolver

    def _branch_exists(self) -> bool:
        return _git("rev-parse", "--verify", self.wt.branch, cwd=self.repo).returncode == 0

    def test_without_resolver_conflict_is_reported_for_manual_fix(self) -> None:
        msg = self.agent.merge()
        assert "Merge conflict detected" in msg
        assert self._branch_exists()
        assert self.agent._last_merge_resolved_by_agent is False

    def test_resolving_runner_completes_the_merge(self) -> None:
        msg = self.agent.merge(conflict_resolver=self._resolver(_resolving_runner))

        assert msg.startswith("Successfully merged")
        assert "The merge agent resolved the conflicts." in msg
        assert self.agent._wt is None
        assert not self._branch_exists()
        assert (self.repo / "f.txt").read_text() == RESOLVED
        assert GitWorktreeOps.status_porcelain(self.repo) == ""
        message = _git("log", "-1", "--format=%B", cwd=self.repo).stdout
        assert "change line b" in message and "changed line b" in message

    def test_resolving_runner_restores_the_users_stash(self) -> None:
        (self.repo / "notes.txt").write_text("uncommitted user notes\n")

        msg = self.agent.merge(conflict_resolver=self._resolver(_resolving_runner))

        assert msg.startswith("Successfully merged")
        assert (self.repo / "notes.txt").read_text() == "uncommitted user notes\n"
        assert (self.repo / "f.txt").read_text() == RESOLVED

    def test_idle_runner_keeps_branch_and_clean_tree(self) -> None:
        head = _git("rev-parse", "HEAD", cwd=self.repo).stdout.strip()

        msg = self.agent.merge(conflict_resolver=self._resolver(_idle_runner))

        assert "Merge conflict detected" in msg
        assert self._branch_exists()
        assert self.agent._wt is not None
        assert _git("rev-parse", "HEAD", cwd=self.repo).stdout.strip() == head
        assert GitWorktreeOps.status_porcelain(self.repo) == ""
        assert (self.repo / "f.txt").read_text() == "a\nB-user\nc\n"
        assert self.agent._last_merge_resolved_by_agent is False

    def test_raising_runner_is_a_plain_conflict(self) -> None:
        msg = self.agent.merge(conflict_resolver=self._resolver(_raising_runner))

        assert "Merge conflict detected" in msg
        assert self._branch_exists()
        assert GitWorktreeOps.status_porcelain(self.repo) == ""

    def test_interrupted_runner_propagates_after_restoring_the_tree(self) -> None:
        # The user's uncommitted edit is stashed for the merge; a stop
        # during the resolution must still give it back.
        (self.repo / "notes.txt").write_text("uncommitted user notes\n")

        with pytest.raises(KeyboardInterrupt):
            self.agent.merge(conflict_resolver=self._resolver(_interrupted_runner))

        assert self._branch_exists()
        assert (self.repo / "notes.txt").read_text() == "uncommitted user notes\n"
        assert GitWorktreeOps.status_porcelain(self.repo).strip() == "?? notes.txt"
        assert (self.repo / "f.txt").read_text() == "a\nB-user\nc\n"
        assert _git("stash", "list", cwd=self.repo).stdout == ""

    def test_interrupt_with_unrestorable_stash_records_a_warning(self) -> None:
        (self.repo / "notes.txt").write_text("uncommitted user notes\n")

        with pytest.raises(KeyboardInterrupt):
            self.agent.merge(
                conflict_resolver=self._resolver(_stash_dropping_interrupted_runner),
            )

        assert self._branch_exists()
        assert not (self.repo / "notes.txt").exists()
        warning = self.agent._stash_pop_warning or ""
        assert "git stash pop" in warning and "interrupted merge" in warning

    def test_rogue_committing_runner_is_undone_and_branch_kept(self) -> None:
        head = _git("rev-parse", "HEAD", cwd=self.repo).stdout.strip()

        msg = self.agent.merge(conflict_resolver=self._resolver(_rogue_committing_runner))

        assert "Merge conflict detected" in msg
        assert self._branch_exists()
        assert _git("rev-parse", "HEAD", cwd=self.repo).stdout.strip() == head
        assert (self.repo / "f.txt").read_text() == "a\nB-user\nc\n"
        assert GitWorktreeOps.status_porcelain(self.repo) == ""
        assert _git("show", f"{self.wt.branch}:f.txt", cwd=self.repo).stdout == "a\nB-agent\nc\n"

    def test_unapplicable_branch_skips_the_runner(self) -> None:
        calls: list[str] = []

        def runner(parent_agent: Any, prompt: str, repo: Path) -> None:
            calls.append(prompt)

        broken = dataclasses.replace(self.wt, baseline_commit="0" * 40)
        result = resolve_merge_conflict(self.agent, broken, None, None, run_agent=runner)

        assert result == MergeResult.CONFLICT
        assert calls == []


class _Flow(_MergeFlowMixin):
    """The merge-flow mixin with the one attribute its usage method reads."""

    def __init__(self) -> None:
        self.capture = CapturePrinter()
        self.printer = self.capture


class TestPersistMergeAgentUsage:
    def setup_method(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-merge-usage-")

    def teardown_method(self) -> None:
        self.home.cleanup()

    def test_add_task_usage_updates_row_and_returns_totals(self) -> None:
        task_id, _ = _add_task("some task", "")
        from kiss.agents.sorcar.persistence import _save_task_extra

        _save_task_extra({"tokens": 100, "cost": 0.5, "steps": 3}, task_id=task_id)

        assert _add_task_usage(task_id, 40, 0.25, 2) == (140, 0.75, 5)
        assert _add_task_usage("no-such-row", 1, 1.0, 1) is None
        (row,) = [r for r in _usage_rows() if r["id"] == task_id]
        assert (row["tokens"], row["cost"], row["steps"]) == (140, 0.75, 5)

    def test_delta_is_charged_to_the_task_and_announced(self) -> None:
        from kiss.agents.sorcar.persistence import _save_task_extra
        from kiss.agents.sorcar.sorcar_agent import _agent_usage, _attribute_sub_usage

        task_id, _ = _add_task("task that conflicted", "")
        _save_task_extra({"tokens": 100, "cost": 0.5, "steps": 3}, task_id=task_id)
        agent = WorktreeSorcarAgent("test")
        agent._last_task_id = task_id
        flow = _Flow()
        before = _agent_usage(agent)
        _attribute_sub_usage(agent, 0.25, 40, 2)

        flow._persist_merge_agent_usage(agent, before, "tab-7")

        (row,) = [r for r in _usage_rows() if r["id"] == task_id]
        assert (row["tokens"], row["cost"], row["steps"]) == (140, 0.75, 5)
        (usage,) = flow.capture.events_of_type("usage_info")
        assert usage["tabId"] == "tab-7"
        assert (usage["total_tokens"], usage["cost"], usage["total_steps"]) == (
            140, "$0.7500", 5,
        )
        assert flow.capture.events_of_type("tasks_updated")

    def test_no_spend_and_no_row_are_silent(self) -> None:
        from kiss.agents.sorcar.sorcar_agent import _agent_usage, _attribute_sub_usage

        agent = WorktreeSorcarAgent("test")
        flow = _Flow()
        flow._persist_merge_agent_usage(agent, _agent_usage(agent), "tab")
        # Spend without a persisted row (an agent that never ran).
        before = _agent_usage(agent)
        _attribute_sub_usage(agent, 0.1, 5, 1)
        flow._persist_merge_agent_usage(agent, before, "tab")
        # Spend charged to a row that does not exist.
        agent._last_task_id = "missing-row"
        flow._persist_merge_agent_usage(agent, before, "tab")
        assert flow.capture.events_of_type("usage_info") == []


class _StoppedDuringMergeAgent(WorktreeSorcarAgent):
    """A worktree agent whose merge agent spends, then is stopped by the user."""

    repo_root: Path

    @property
    def _wt_pending(self) -> bool:
        return True

    @property
    def _repo_root(self) -> Path | None:
        return self.repo_root

    def merge(self, conflict_resolver: Any = None) -> str:
        from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

        assert conflict_resolver is not None
        _attribute_sub_usage(self, 0.25, 40, 2)
        raise KeyboardInterrupt


class TestMergeUsagePersistsWhenTheMergeIsStopped:
    """``_handle_worktree_action`` charges the spend even when the merge unwinds."""

    def setup_method(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-merge-stop-")
        self.printer = CapturePrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.home.repo)

    def teardown_method(self) -> None:
        self.home.cleanup()

    def test_delta_reaches_the_row_and_the_tab(self) -> None:
        from kiss.agents.sorcar.persistence import _save_task_extra

        task_id, _ = _add_task("stopped task", "")
        _save_task_extra({"tokens": 100, "cost": 0.5, "steps": 3}, task_id=task_id)
        agent = _StoppedDuringMergeAgent("test")
        agent.repo_root = self.home.repo
        agent._last_task_id = task_id
        tab_id = "stop-tab"
        with self.server._state_lock:
            state = agent_state.AgentState(
                task_id, tab_id=tab_id, server_owned=True, agent=agent,
            )
            state.use_worktree = True
            agent_state.register(state)

        with pytest.raises(KeyboardInterrupt):
            self.server._handle_worktree_action(
                "merge", tab_id, internal=True, resolve_conflicts=True,
            )

        (row,) = [r for r in _usage_rows() if r["id"] == task_id]
        assert (row["tokens"], row["cost"], row["steps"]) == (140, 0.75, 5)
        (usage,) = self.printer.events_of_type("usage_info")
        assert usage["total_tokens"] == 140 and usage["tabId"] == tab_id


class TestMergeSea:
    def test_getters_follow_the_sea_contract(self) -> None:
        assert merge_sea.system_prompt() == merge_sea.SYSTEM_PROMPT
        assert merge_sea.is_parallel() is False
        assert merge_sea.use_web_tools() is False
        assert merge_sea.use_memory() is False
        assert merge_sea.use_worktree() is False
        assert merge_sea.auto_commit() is False
        assert merge_sea.max_budget() == merge_sea.MAX_BUDGET_USD

    def test_prompt_lists_files_and_task(self) -> None:
        # The prompt names the repo in the OS's native form (``/r`` on
        # POSIX, ``\r`` on Windows), as the agent's tools expect it.
        repo = Path("/r")
        prompt = merge_sea.build_prompt(
            repo, "kiss/wt-x", "main", ["a.py", "b.md"], "  do X  ",
        )
        assert f"Repository: {repo}\n" in prompt
        assert "(ours, HEAD): main" in prompt
        assert "(theirs): kiss/wt-x" in prompt
        assert "- a.py\n- b.md" in prompt
        assert "<task>\ndo X\n</task>" in prompt
        assert "<task>" not in merge_sea.build_prompt(Path("/r"), "b", "main", ["a.py"])

    def test_merge_is_a_registered_slash_command(self) -> None:
        sea_commands._reset_for_tests()
        try:
            sea_commands.refresh_registry()
            path = sea_commands.get_command("merge")
            assert path is not None and path.name == "merge_sea.py"
            assert path.parent.name == "seas"
            rewritten = sea_commands.rewrite_prompt_if_command("/merge finish the merge")
            assert rewritten is not None and str(path) in rewritten[0]
        finally:
            sea_commands._reset_for_tests()


#: Marks the task prompt so the stand-in endpoint recognises the task agent.
_TASK_MARKER = "MERGE-SEA-E2E-TASK"
#: Text only the merge SEA's prompt carries.
_MERGE_MARKER = "Incoming task branch (theirs)"


class TestAutoCommitMergeConflictEndToEnd(unittest.TestCase):
    """A real run whose auto-merge conflicts, fixed by the merge SEA."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-merge-sea-e2e-")
        self.repo = self.home.repo
        (self.repo / "f.txt").write_text("a\nb\nc\n")
        run_git(self.repo, "add", "f.txt")
        run_git(self.repo, "commit", "-qm", "add f")
        self.home.write_config(
            auto_commit_mode=True, is_worktree=True, max_budget=5.0, use_web_browser=False,
        )
        self.credentials = _FakeCredentials()
        self.calls: list[str] = []
        self._calls_lock = threading.Lock()
        self.standin = StandInModelServer(self._respond)
        self.printer = CapturePrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.repo)

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            states = list(agent_state.agent_states.values())
        for state in states:
            agent = state.agent
            if agent is not None and getattr(agent, "_wt", None) is not None:
                try:
                    agent.discard()
                except Exception:
                    pass
        self.standin.stop()
        self.credentials.restore()
        self.home.cleanup()

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """Script both agents from the one endpoint.

        Task agent: edit line ``b`` in the worktree, then — while
        finishing — the test commits a competing edit of the same line
        on the user's branch so the auto-merge must conflict.  Merge
        agent: write the resolution and stage it, then finish.
        """
        text = request_text(request)
        with self._calls_lock:
            self.calls.append(text)
            merge_calls = sum(1 for t in self.calls if _MERGE_MARKER in t)
            task_calls = sum(1 for t in self.calls if _TASK_MARKER in t)
        if _MERGE_MARKER in text:
            if merge_calls == 1:
                return tool_call_response(
                    "Bash",
                    {
                        "command": "printf 'a\\nB-both\\nc\\n' > f.txt && git add f.txt",
                        "description": "resolve and stage f.txt",
                    },
                )
            return finish_response("conflict resolved")
        if task_calls == 1:
            return tool_call_response(
                "Bash",
                {
                    "command": "printf 'a\\nB-agent\\nc\\n' > f.txt",
                    "description": "edit line b",
                },
            )
        (self.repo / "f.txt").write_text("a\nB-user\nc\n")
        run_git(self.repo, "commit", "-qam", "user edits line b meanwhile")
        return finish_response("task done")

    def test_merge_agent_completes_the_merge_and_is_charged_to_the_task(self) -> None:
        self.server._run_task(
            {
                "type": "run",
                "tabId": "e2e-tab",
                "prompt": f"{_TASK_MARKER}: change line b of f.txt",
                "model": STANDIN_MODEL,
                "workDir": str(self.repo),
                "useWorktree": True,
                "useParallel": False,
                "autoCommit": True,
                "webTools": False,
                "maxBudget": 5.0,
                "modelConfig": self.standin.model_config,
            },
        )

        (wt_result,) = self.printer.events_of_type("worktree_result")
        assert wt_result["success"] is True, wt_result
        assert "The merge agent resolved the conflicts." in wt_result["message"]
        assert (self.repo / "f.txt").read_text() == "a\nB-both\nc\n"
        assert run_git(self.repo, "branch", "--list", "kiss/wt-*").stdout.strip() == ""
        assert GitWorktreeOps.status_porcelain(self.repo) == ""
        assert any(_MERGE_MARKER in t for t in self.calls), "the merge SEA never ran"

        rows = {r["id"]: r for r in _usage_rows()}
        (task_row,) = [
            r for r in rows.values()
            if _TASK_MARKER in str(r["task"]) and not r["parent_task_id"]
        ]
        (merge_row,) = [
            r for r in rows.values() if r.get("parent_task_id") == task_row["id"]
        ]
        assert merge_row["tokens"] > 0 and merge_row["steps"] > 0
        result_events = [
            e for e in self.printer.events_of_type("result")
            if e.get("taskId") == task_row["id"] or "taskId" not in e
        ]
        own = result_events[0]
        assert task_row["tokens"] == own["total_tokens"] + merge_row["tokens"]
        assert task_row["steps"] == own["step_count"] + merge_row["steps"]
        assert task_row["cost"] == pytest.approx(
            float(own["cost"].lstrip("$")) + merge_row["cost"], abs=1e-4,
        )
        usage_events = [
            e for e in self.printer.events_of_type("usage_info")
            if e.get("total_tokens") == task_row["tokens"]
        ]
        assert usage_events, self.printer.events_of_type("usage_info")
        assert usage_events[-1]["total_steps"] == task_row["steps"]
        assert any(
            e.get("type") == "new_tab" and e.get("task_id") == merge_row["id"]
            for e in self.printer.captured
        )

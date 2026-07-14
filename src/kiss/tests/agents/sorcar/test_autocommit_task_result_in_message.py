# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for embedding the task RESULT in auto-commit messages.

The auto-generated commit message must have BOTH the task description
(the user's prompt) AND the task's result summary appended before the
commit is created, so the commit is fully traceable: what was asked,
and what the agent reported it did.

These tests exercise the real
:func:`~kiss.agents.vscode.helpers.generate_commit_message_from_diff`,
:func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes`, and
:meth:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent._auto_commit_worktree`
paths against on-disk git repositories.  The LLM call inside the
helper is forced through its ``except Exception`` fallback by
patching :class:`~kiss.core.kiss_agent.KISSAgent` to raise — so the
tests assert the *appending* behaviour without needing any external
model.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import auto_commit_changes
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.helpers import (
    _append_task_result,
    generate_commit_message_from_diff,
)


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True,
    )
    return path


def _head_message(repo: Path) -> str:
    """Return HEAD's full commit message (subject + body)."""
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%B", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.rstrip()


class _LLMUnavailable:
    """Force the LLM call inside ``generate_commit_message_from_diff``
    through its ``except Exception`` fallback so tests don't need a
    real model.
    """

    def __enter__(self) -> _LLMUnavailable:
        import kiss.core.kiss_agent as kiss_agent_mod

        self._orig = kiss_agent_mod.KISSAgent

        class _RaisingAgent:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                pass

            def run(self, *_a: Any, **_kw: Any) -> str:
                raise RuntimeError("no LLM in test")

        kiss_agent_mod.KISSAgent = _RaisingAgent  # type: ignore[misc, assignment]
        return self

    def __exit__(self, *_exc: Any) -> None:
        import kiss.core.kiss_agent as kiss_agent_mod

        kiss_agent_mod.KISSAgent = self._orig  # type: ignore[misc]


class TestAppendTaskResultHelper:
    """Pure-function tests for ``_append_task_result``."""

    def test_appends_under_result_heading(self) -> None:
        msg = _append_task_result("subject", "Fixed the bug in foo.py")
        assert msg == "subject\n\nResult:\nFixed the bug in foo.py"

    def test_trims_whitespace_around_result(self) -> None:
        msg = _append_task_result("subject", "   done\n  ")
        assert msg == "subject\n\nResult:\ndone"

    def test_empty_result_returns_message_unchanged(self) -> None:
        assert _append_task_result("subject", "") == "subject"
        assert _append_task_result("subject", "   \n  ") == "subject"

    def test_multiline_result_preserved(self) -> None:
        result = "did X\n- step 1\n- step 2"
        msg = _append_task_result("subject", result)
        assert msg == f"subject\n\nResult:\n{result}"


class TestGenerateCommitMessageIncludesTaskResult:
    """``generate_commit_message_from_diff`` includes the task result
    in its output whenever the result is supplied — across all three
    branches (empty diff, LLM-failure-fallback, and combined with the
    user prompt).
    """

    def test_empty_diff_with_result_appends_result(self) -> None:
        msg = generate_commit_message_from_diff(
            "", task_result="Added CLI flag --foo",
        )
        assert msg == (
            "kiss: auto-commit agent work\n\n"
            "Result:\nAdded CLI flag --foo"
        )

    def test_empty_diff_with_prompt_and_result_appends_both(self) -> None:
        msg = generate_commit_message_from_diff(
            "",
            user_prompt="add a CLI flag",
            task_result="Added CLI flag --foo",
        )
        assert msg == (
            "kiss: auto-commit agent work\n\n"
            "User prompt:\nadd a CLI flag\n\n"
            "Result:\nAdded CLI flag --foo"
        )

    def test_llm_failure_with_prompt_and_result_appends_both(self) -> None:
        with _LLMUnavailable():
            msg = generate_commit_message_from_diff(
                "diff --git a/f b/f\n@@\n+x\n",
                user_prompt="refactor module Y",
                task_result="Refactored Y into Z",
            )
        assert msg.startswith("kiss: auto-commit agent work")
        assert "User prompt:\nrefactor module Y" in msg
        assert msg.endswith("Result:\nRefactored Y into Z")

    def test_llm_failure_without_result_has_no_result_block(self) -> None:
        with _LLMUnavailable():
            msg = generate_commit_message_from_diff(
                "diff --git a/f b/f\n@@\n+x\n",
                user_prompt="refactor module Y",
            )
        assert "Result:" not in msg


class TestAutoCommitChangesFallbackIncludesResult:
    """``auto_commit_changes``'s message_fn-failure fallback path also
    appends the task result (mirroring the user-prompt handling).
    """

    def test_fallback_message_includes_prompt_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "work.txt").write_text("agent output\n")

            def raising_message_fn(
                commit_dir: Path,
                user_prompt: str | None,
                task_result: str | None,
            ) -> str:
                raise RuntimeError("simulated LLM outage")

            committed = auto_commit_changes(
                repo,
                user_prompt="do the thing",
                message_fn=raising_message_fn,
                task_result="the thing is done",
            )
            assert committed is True
            msg = _head_message(repo)
            assert "User prompt:\ndo the thing" in msg
            assert msg.endswith("Result:\nthe thing is done")

    def test_message_fn_receives_task_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "work.txt").write_text("agent output\n")
            seen: dict[str, str | None] = {}

            def capturing_message_fn(
                commit_dir: Path,
                user_prompt: str | None,
                task_result: str | None,
            ) -> str:
                seen["user_prompt"] = user_prompt
                seen["task_result"] = task_result
                return "test: captured"

            committed = auto_commit_changes(
                repo,
                user_prompt="the prompt",
                message_fn=capturing_message_fn,
                task_result="the result",
            )
            assert committed is True
            assert seen == {
                "user_prompt": "the prompt",
                "task_result": "the result",
            }


class TestWorktreeAutoCommitIncludesTaskResult:
    """End-to-end: ``WorktreeSorcarAgent._auto_commit_worktree``
    commits with a message that includes BOTH the task description
    (``self._last_user_prompt``) and the task result
    (``self._last_result_summary``).
    """

    def _make_worktree(
        self, tmp: str, branch: str,
    ) -> tuple[Path, Path]:
        repo = _make_repo(Path(tmp) / "repo")
        slug = branch.replace("/", "_")
        wt_dir = repo / ".kiss-worktrees" / slug
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        subprocess.run(
            ["git", "-C", str(wt_dir), "config", "user.email", "t@t.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(wt_dir), "config", "user.name", "T"],
            check=True,
        )
        return repo, wt_dir

    def test_task_description_and_result_in_commit_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, wt_dir = self._make_worktree(tmp, "kiss/wt-test-result")
            (wt_dir / "new.txt").write_text("hello\n")

            agent = WorktreeSorcarAgent("test")
            agent._wt = GitWorktree(
                repo_root=repo,
                branch="kiss/wt-test-result",
                original_branch="main",
                wt_dir=wt_dir,
                baseline_commit=None,
            )
            agent._last_user_prompt = "implement feature Z"
            agent._last_result_summary = "Implemented feature Z in z.py"

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            msg = _head_message(wt_dir)
            assert "User prompt:\nimplement feature Z" in msg
            assert msg.endswith("Result:\nImplemented feature Z in z.py")

    def test_no_result_means_no_result_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, wt_dir = self._make_worktree(tmp, "kiss/wt-test-nores")
            (wt_dir / "new.txt").write_text("hi\n")

            agent = WorktreeSorcarAgent("test")
            agent._wt = GitWorktree(
                repo_root=repo,
                branch="kiss/wt-test-nores",
                original_branch="main",
                wt_dir=wt_dir,
                baseline_commit=None,
            )
            agent._last_user_prompt = "implement feature Q"
            # _last_result_summary remains ""
            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            msg = _head_message(wt_dir)
            assert "Result:" not in msg


class TestChatRunRecordsResultSummary:
    """``ChatSorcarAgent.run`` stashes the task's result summary onto
    the agent so downstream auto-commit hooks can append it to the
    commit message.
    """

    def test_run_sets_last_result_summary(self) -> None:
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        orig_run = SorcarAgent.run

        def _fake_run(self: SorcarAgent, *_a: Any, **_kw: Any) -> str:
            return "success: true\nsummary: implemented the feature\n"

        SorcarAgent.run = _fake_run  # type: ignore[method-assign, assignment]
        try:
            agent = ChatSorcarAgent("t")
            agent.run(prompt_template="do thing X", _skip_persistence=True)
        finally:
            SorcarAgent.run = orig_run  # type: ignore[method-assign]
        assert agent._last_result_summary == "implemented the feature"

    def test_run_resets_result_summary_between_runs(self) -> None:
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        orig_run = SorcarAgent.run
        calls: list[int] = []

        def _fake_run(self: SorcarAgent, *_a: Any, **_kw: Any) -> str:
            calls.append(1)
            if len(calls) == 1:
                return "success: true\nsummary: first result\n"
            raise RuntimeError("second run fails")

        SorcarAgent.run = _fake_run  # type: ignore[method-assign, assignment]
        try:
            agent = ChatSorcarAgent("t")
            agent.run(prompt_template="task 1", _skip_persistence=True)
            assert agent._last_result_summary == "first result"
            try:
                agent.run(prompt_template="task 2", _skip_persistence=True)
            except RuntimeError:
                pass
            # The stale "first result" must NOT survive into run 2.
            assert agent._last_result_summary == "Task failed"
        finally:
            SorcarAgent.run = orig_run  # type: ignore[method-assign]

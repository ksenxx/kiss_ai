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
:func:`~kiss.server.helpers.generate_commit_message_from_diff`,
:func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes`, and
:meth:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent._auto_commit_worktree`
paths against on-disk git repositories.  The LLM call inside the
helper is forced through its ``except Exception`` fallback by
patching :class:`~kiss.core.kiss_agent.KISSAgent` to raise — so the
tests assert the *appending* behaviour without needing any external
model.
"""

from __future__ import annotations

from kiss.agents.sorcar.commit_message import _append_task_result
from kiss.server.helpers import generate_commit_message_from_diff
from kiss.tests.agents.sorcar.test_autocommit_task_result_in_message import (  # noqa: F401
    _LLMUnavailable,
)


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

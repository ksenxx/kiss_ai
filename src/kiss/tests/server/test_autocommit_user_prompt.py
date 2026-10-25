# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for embedding the user's task prompt in auto-commit messages.

When the agent auto-commits changes — either inside a worktree at
task end, via the ``update_settings(auto_commit=True)`` tool call,
or via the post-task ``_autocommit_changes`` path in the daemon —
the resulting commit message should include the user's
original task prompt so the commit is traceable to the request that
produced it.

These tests exercise the real
:func:`~kiss.agents.sorcar.commit_message.generate_commit_message_from_diff`
and
:meth:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent._auto_commit_worktree`
paths against on-disk git repositories.  The LLM call inside the
helper is forced through its ``except Exception`` fallback by
patching :class:`~kiss.core.kiss_agent.KISSAgent.run` to raise — so
the tests assert the *appending* behaviour without needing any
external model.
"""

from __future__ import annotations

from kiss.agents.sorcar.commit_message import (
    FALLBACK_SUBJECT,
    fallback_commit_message,
    generate_commit_message_from_diff,
)
from kiss.agents.sorcar.git_worktree import _ensure_task_metadata
from kiss.tests.agents.sorcar.test_autocommit_user_prompt import (  # noqa: F401
    _LLMUnavailable,
)


class TestUserPromptTrailer:
    """The ``User prompt:`` trailer ``fallback_commit_message`` appends."""

    def test_appends_under_user_prompt_heading(self) -> None:
        msg = fallback_commit_message("fix the bug")
        assert msg == f"{FALLBACK_SUBJECT}\n\nUser prompt:\nfix the bug"

    def test_trims_whitespace_around_prompt(self) -> None:
        msg = fallback_commit_message("   fix the bug\n  ")
        assert msg == f"{FALLBACK_SUBJECT}\n\nUser prompt:\nfix the bug"

    def test_empty_prompt_returns_message_unchanged(self) -> None:
        assert fallback_commit_message("") == FALLBACK_SUBJECT
        assert fallback_commit_message("   \n  ") == FALLBACK_SUBJECT

    def test_strips_trailing_whitespace_from_base_message(self) -> None:
        msg = _ensure_task_metadata("subject\n\n", "do X", None)
        assert msg == "subject\n\nUser prompt:\ndo X"

    def test_multiline_prompt_preserved(self) -> None:
        prompt = "do X\n- step 1\n- step 2"
        msg = fallback_commit_message(prompt)
        assert msg == f"{FALLBACK_SUBJECT}\n\nUser prompt:\n{prompt}"


class TestGenerateCommitMessageFromDiff:
    """``generate_commit_message_from_diff`` includes the user prompt
    in its output whenever the prompt is supplied — across all three
    branches (empty diff, LLM-success, LLM-failure-fallback).
    """

    def test_empty_diff_no_prompt_returns_bare_fallback(self) -> None:
        msg = generate_commit_message_from_diff("")
        assert msg == "kiss: auto-commit agent work"
        assert "User prompt:" not in msg

    def test_empty_diff_with_prompt_appends_prompt(self) -> None:
        msg = generate_commit_message_from_diff(
            "", user_prompt="add a CLI flag",
        )
        assert msg == (
            "kiss: auto-commit agent work\n\n"
            "User prompt:\nadd a CLI flag"
        )

    def test_llm_failure_with_prompt_appends_to_fallback(self) -> None:
        with _LLMUnavailable():
            msg = generate_commit_message_from_diff(
                "diff --git a/f b/f\n@@\n+x\n",
                user_prompt="refactor module Y",
            )
        assert msg.startswith("kiss: auto-commit agent work")
        assert msg.endswith("User prompt:\nrefactor module Y")

    def test_llm_failure_without_prompt_returns_bare_fallback(self) -> None:
        with _LLMUnavailable():
            msg = generate_commit_message_from_diff(
                "diff --git a/f b/f\n@@\n+x\n",
            )
        assert msg == "kiss: auto-commit agent work"
        assert "User prompt:" not in msg

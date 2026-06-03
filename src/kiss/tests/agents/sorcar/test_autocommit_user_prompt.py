# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for embedding the user's task prompt in auto-commit messages.

When the agent auto-commits changes — either inside a worktree at
task end, via the ``update_settings(auto_commit=True)`` tool call,
or via the post-task ``autocommit_prompt`` handler in the VS Code
server — the resulting commit message should include the user's
original task prompt so the commit is traceable to the request that
produced it.

These tests exercise the real
:func:`~kiss.agents.vscode.helpers.generate_commit_message_from_diff`
and
:meth:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent._auto_commit_worktree`
paths against on-disk git repositories.  The LLM call inside the
helper is forced through its ``except Exception`` fallback by
patching :class:`~kiss.core.kiss_agent.KISSAgent.run` to raise — so
the tests assert the *appending* behaviour without needing any
external model.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.helpers import (
    _append_user_prompt,
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


class TestAppendUserPromptHelper:
    """Pure-function tests for ``_append_user_prompt``."""

    def test_appends_under_user_prompt_heading(self) -> None:
        msg = _append_user_prompt("subject", "fix the bug")
        assert msg == "subject\n\nUser prompt:\nfix the bug"

    def test_trims_whitespace_around_prompt(self) -> None:
        msg = _append_user_prompt("subject", "   fix the bug\n  ")
        assert msg == "subject\n\nUser prompt:\nfix the bug"

    def test_empty_prompt_returns_message_unchanged(self) -> None:
        assert _append_user_prompt("subject", "") == "subject"
        assert _append_user_prompt("subject", "   \n  ") == "subject"

    def test_strips_trailing_whitespace_from_base_message(self) -> None:
        msg = _append_user_prompt("subject\n\n", "do X")
        # Trailing blank line on base is stripped before separator.
        assert msg == "subject\n\nUser prompt:\ndo X"

    def test_multiline_prompt_preserved(self) -> None:
        prompt = "do X\n- step 1\n- step 2"
        msg = _append_user_prompt("subject", prompt)
        assert msg == f"subject\n\nUser prompt:\n{prompt}"


class _LLMUnavailable:
    """Force the LLM call inside ``generate_commit_message_from_diff``
    through its ``except Exception`` fallback so tests don't need a
    real model.

    Patches :class:`kiss.core.kiss_agent.KISSAgent` to a class whose
    ``run`` raises.  Reverts on exit.
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


class TestWorktreeAutoCommitIncludesUserPrompt:
    """End-to-end: ``WorktreeSorcarAgent._auto_commit_worktree``
    commits with a message that includes ``self._last_user_prompt``.
    """

    def test_user_prompt_appended_to_worktree_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-test-userprompt"
            slug = branch.replace("/", "_")
            wt_dir = repo / ".kiss-worktrees" / slug
            assert GitWorktreeOps.create(repo, branch, wt_dir)
            subprocess.run(
                ["git", "-C", str(wt_dir),
                 "config", "user.email", "t@t.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(wt_dir), "config", "user.name", "T"],
                check=True,
            )
            # Agent makes a change in the worktree.
            (wt_dir / "new.txt").write_text("hello\n")

            agent = WorktreeSorcarAgent("test")
            agent._wt = GitWorktree(
                repo_root=repo,
                branch=branch,
                original_branch="main",
                wt_dir=wt_dir,
                baseline_commit=None,
            )
            agent._last_user_prompt = "implement feature Z"

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            msg = _head_message(wt_dir)
            assert msg.endswith("User prompt:\nimplement feature Z")

    def test_no_prompt_means_no_user_prompt_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-test-noprompt"
            slug = branch.replace("/", "_")
            wt_dir = repo / ".kiss-worktrees" / slug
            assert GitWorktreeOps.create(repo, branch, wt_dir)
            subprocess.run(
                ["git", "-C", str(wt_dir),
                 "config", "user.email", "t@t.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(wt_dir), "config", "user.name", "T"],
                check=True,
            )
            (wt_dir / "new.txt").write_text("hi\n")

            agent = WorktreeSorcarAgent("test")
            agent._wt = GitWorktree(
                repo_root=repo,
                branch=branch,
                original_branch="main",
                wt_dir=wt_dir,
                baseline_commit=None,
            )
            # _last_user_prompt remains ""
            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            msg = _head_message(wt_dir)
            assert "User prompt:" not in msg


class TestChatRunRecordsUserPrompt:
    """``ChatSorcarAgent.run`` stashes the raw user prompt onto the
    agent so downstream auto-commit hooks can read it.
    """

    def test_run_sets_last_user_prompt_before_dispatch(self) -> None:
        # We don't need to actually drive the LLM — we just need to
        # observe that ``_last_user_prompt`` is set BEFORE the
        # underlying ``super().run`` is invoked.  Patch
        # ``ChatSorcarAgent``'s parent ``run`` to capture state.
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        captured: dict[str, str] = {}
        orig_run = SorcarAgent.run

        def _fake_run(self: SorcarAgent, *_a: Any, **_kw: Any) -> str:
            captured["last_user_prompt"] = (
                getattr(self, "_last_user_prompt", "<missing>")
            )
            return "summary: ok\n"

        SorcarAgent.run = _fake_run  # type: ignore[method-assign, assignment]
        try:
            agent = ChatSorcarAgent("t")
            agent.run(prompt_template="do thing X", _skip_persistence=True)
        finally:
            SorcarAgent.run = orig_run  # type: ignore[method-assign]
        assert captured["last_user_prompt"] == "do thing X"
        assert agent._last_user_prompt == "do thing X"

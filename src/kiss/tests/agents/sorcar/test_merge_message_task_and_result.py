# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: squash-merge commit message must carry the task and result.

Reproduces the production issue observed on commit ``dd563a7c``: the
agent committed its own work in the worktree with a hand-written
``git commit`` (so the post-task auto-commit was a no-op — nothing left
to commit), then the squash-merge into the user's branch reused the
branch HEAD message verbatim.  Because the agent's own message contains
neither a ``User prompt:`` nor a ``Result:`` block, the final commit on
the user's branch lost the task description and the task result.

The fix threads ``user_prompt`` / ``task_result`` into the squash-merge
message builders so the merge commit ALWAYS carries both blocks —
regardless of whether the branch HEAD message came from the framework's
auto-commit (which already appends them) or from the agent's own manual
commit (which does not).  Duplicate blocks must not be appended when
the HEAD message already carries them.

These tests drive the real :mod:`kiss.agents.sorcar.git_worktree`
operations and the real :class:`WorktreeSorcarAgent` merge path against
actual on-disk git repositories (no mocks).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    MergeResult,
    _git,
)

PROMPT = "make the history panel occupy 1/4th of the browser screen"
RESULT = "History panel now occupies 1/4 of the screen; 43 tests green."


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


def _create_worktree(repo: Path, branch: str) -> Path:
    """Create a worktree at repo/.kiss-worktrees/<slug>."""
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
    return wt_dir


def _head_message(repo: Path) -> str:
    """Return the full commit message of HEAD in *repo*."""
    result = _git("log", "-1", "--format=%B", "HEAD", cwd=repo)
    return result.stdout.rstrip()


def _agent_manual_commit(wt_dir: Path, filename: str = "feature.txt") -> str:
    """Simulate the agent committing its own work with its own message.

    Returns the hand-written commit message (which carries NEITHER a
    ``User prompt:`` nor a ``Result:`` block — exactly the production
    scenario of commit dd563a7c).
    """
    (wt_dir / filename).write_text("hello\n")
    _git("add", filename, cwd=wt_dir)
    agent_msg = (
        "feat(remote webapp): 1/4-screen default history panel\n"
        "\n"
        "- docked sidebar defaults to clamp(220px, 25vw, 600px)\n"
        "- failing-first jsdom e2e suite (9 tests)"
    )
    commit = _git("commit", "-m", agent_msg, cwd=wt_dir)
    assert commit.returncode == 0, commit.stderr
    return agent_msg


class TestMergeMessageBaselinePath:
    """squash_merge_from_baseline must append the task and result."""

    def test_manual_agent_commit_gets_prompt_and_result(self) -> None:
        """PRODUCTION REPRO: agent hand-commits (no blocks in HEAD
        message) → the merge commit on main must still carry BOTH the
        ``User prompt:`` and ``Result:`` blocks.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-e2e-1"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            agent_msg = _agent_manual_commit(wt_dir)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo,
                branch,
                baseline,
                user_prompt=PROMPT,
                task_result=RESULT,
            )
            assert result == MergeResult.SUCCESS
            msg = _head_message(repo)
            assert msg.startswith(agent_msg)
            assert f"User prompt:\n{PROMPT}" in msg
            assert f"Result:\n{RESULT}" in msg

    def test_no_duplicate_blocks_when_head_already_has_them(self) -> None:
        """When the branch HEAD message ALREADY carries the blocks
        (framework auto-commit made the last branch commit), the merge
        must not append them a second time.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-e2e-2"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            (wt_dir / "f.txt").write_text("f\n")
            _git("add", "f.txt", cwd=wt_dir)
            autocommit_msg = (
                "feat: add f\n"
                "\n"
                f"User prompt:\n{PROMPT}\n"
                "\n"
                f"Result:\n{RESULT}"
            )
            _git("commit", "-m", autocommit_msg, cwd=wt_dir)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo,
                branch,
                baseline,
                user_prompt=PROMPT,
                task_result=RESULT,
            )
            assert result == MergeResult.SUCCESS
            msg = _head_message(repo)
            assert msg.count("User prompt:") == 1
            assert msg.count("Result:") == 1

    def test_without_prompt_and_result_behaves_as_before(self) -> None:
        """Backward compatibility: omitting the new arguments keeps
        the verbatim branch-HEAD-message behavior.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-e2e-3"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            agent_msg = _agent_manual_commit(wt_dir)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline
            )
            assert result == MergeResult.SUCCESS
            assert _head_message(repo) == agent_msg


class TestMergeMessageLegacyPath:
    """squash_merge_branch (no baseline) must append the blocks too."""

    def test_manual_agent_commit_gets_prompt_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-e2e-legacy"
            wt_dir = _create_worktree(repo, branch)
            agent_msg = _agent_manual_commit(wt_dir, "x.txt")

            result = GitWorktreeOps.squash_merge_branch(
                repo,
                branch,
                user_prompt=PROMPT,
                task_result=RESULT,
            )
            assert result == MergeResult.SUCCESS
            msg = _head_message(repo)
            assert msg.startswith(agent_msg)
            assert f"User prompt:\n{PROMPT}" in msg
            assert f"Result:\n{RESULT}" in msg


class TestMergeCommitMessageHelper:
    """_merge_commit_message appends prompt/result with dedup."""

    def test_appends_both_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-helper-1"
            wt_dir = _create_worktree(repo, branch)
            (wt_dir / "z.txt").write_text("z\n")
            _git("add", "z.txt", cwd=wt_dir)
            _git("commit", "-m", "helper commit", cwd=wt_dir)

            msg = GitWorktreeOps._merge_commit_message(
                repo, branch, user_prompt=PROMPT, task_result=RESULT,
            )
            assert msg == (
                "helper commit"
                f"\n\nUser prompt:\n{PROMPT}"
                f"\n\nResult:\n{RESULT}"
            )

    def test_fallback_message_also_gets_blocks(self) -> None:
        """Even the synthetic ``kiss: merged from <branch>`` fallback
        (unreadable branch HEAD) must carry the task and result.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            msg = GitWorktreeOps._merge_commit_message(
                repo,
                "no/such-branch",
                user_prompt=PROMPT,
                task_result=RESULT,
            )
            assert msg.startswith("kiss: merged from no/such-branch")
            assert f"User prompt:\n{PROMPT}" in msg
            assert f"Result:\n{RESULT}" in msg

    def test_dedup_is_per_block(self) -> None:
        """A HEAD message carrying only the ``User prompt:`` block
        still gets the missing ``Result:`` block appended (and vice
        versa) — dedup is per block, not all-or-nothing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-helper-2"
            wt_dir = _create_worktree(repo, branch)
            (wt_dir / "w.txt").write_text("w\n")
            _git("add", "w.txt", cwd=wt_dir)
            _git(
                "commit",
                "-m",
                f"subject\n\nUser prompt:\n{PROMPT}",
                cwd=wt_dir,
            )

            msg = GitWorktreeOps._merge_commit_message(
                repo, branch, user_prompt=PROMPT, task_result=RESULT,
            )
            assert msg == (
                f"subject\n\nUser prompt:\n{PROMPT}\n\nResult:\n{RESULT}"
            )

    def test_result_only_suffix_gets_prompt_inserted(self) -> None:
        """A HEAD message ending with only the current ``Result:``
        block gets the missing prompt block inserted BEFORE it,
        preserving canonical prompt-then-result order.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-helper-3"
            wt_dir = _create_worktree(repo, branch)
            (wt_dir / "v.txt").write_text("v\n")
            _git("add", "v.txt", cwd=wt_dir)
            _git(
                "commit",
                "-m",
                f"subject\n\nResult:\n{RESULT}",
                cwd=wt_dir,
            )

            msg = GitWorktreeOps._merge_commit_message(
                repo, branch, user_prompt=PROMPT, task_result=RESULT,
            )
            assert msg == (
                f"subject\n\nUser prompt:\n{PROMPT}\n\nResult:\n{RESULT}"
            )

    def test_incidental_heading_text_does_not_suppress(self) -> None:
        """A hand-written commit body that merely MENTIONS
        ``User prompt:`` or ``Result:`` (or carries STALE blocks from
        a previous task) must not suppress stamping the CURRENT
        values — dedup is by exact current-value suffix, not by
        heading substring (gpt-5.6-sol review HIGH finding).
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-helper-4"
            wt_dir = _create_worktree(repo, branch)
            (wt_dir / "u.txt").write_text("u\n")
            _git("add", "u.txt", cwd=wt_dir)
            stale = (
                "fix: mention Result: in prose\n"
                "\n"
                "User prompt:\nOLD TASK\n"
                "\n"
                "Result:\nOLD RESULT"
            )
            _git("commit", "-m", stale, cwd=wt_dir)

            msg = GitWorktreeOps._merge_commit_message(
                repo, branch, user_prompt=PROMPT, task_result=RESULT,
            )
            assert msg == (
                f"{stale}\n\nUser prompt:\n{PROMPT}\n\nResult:\n{RESULT}"
            )

    def test_prompt_containing_result_heading(self) -> None:
        """A prompt whose own text contains ``Result:`` must not
        suppress appending the actual result block.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-helper-5"
            wt_dir = _create_worktree(repo, branch)
            (wt_dir / "t.txt").write_text("t\n")
            _git("add", "t.txt", cwd=wt_dir)
            _git("commit", "-m", "plain subject", cwd=wt_dir)

            tricky_prompt = "Fix why the Result: block is missing"
            msg = GitWorktreeOps._merge_commit_message(
                repo,
                branch,
                user_prompt=tricky_prompt,
                task_result=RESULT,
            )
            assert msg == (
                "plain subject"
                f"\n\nUser prompt:\n{tricky_prompt}"
                f"\n\nResult:\n{RESULT}"
            )


class TestAgentMergeCarriesPromptAndResult:
    """End-to-end through WorktreeSorcarAgent.merge(): the merge commit
    on the original branch carries the recorded task prompt and result
    even when the agent committed its own work manually.
    """

    def _make_agent(self, repo: Path, branch: str, wt_dir: Path):  # noqa: ANN202
        from kiss.agents.sorcar.git_worktree import GitWorktree
        from kiss.agents.sorcar.worktree_sorcar_agent import (
            WorktreeSorcarAgent,
        )

        agent = WorktreeSorcarAgent("wt-merge-msg-test")
        baseline = GitWorktreeOps.head_sha(wt_dir)
        agent._wt = GitWorktree(
            repo_root=repo,
            wt_dir=wt_dir,
            branch=branch,
            original_branch="main",
            baseline_commit=baseline,
        )
        return agent

    def test_merge_appends_prompt_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-agent-1"
            wt_dir = _create_worktree(repo, branch)
            agent = self._make_agent(repo, branch, wt_dir)
            agent_msg = _agent_manual_commit(wt_dir)
            agent._last_user_prompt = PROMPT
            agent._last_result_summary = RESULT
            # PRODUCTION REPRO precondition: the worktree is clean
            # after the agent's manual commit, so the framework
            # auto-commit is a NO-OP (this is exactly why the merge
            # message must append the blocks itself).
            head_before = GitWorktreeOps.head_sha(wt_dir)
            assert agent._auto_commit_worktree() is False
            assert GitWorktreeOps.head_sha(wt_dir) == head_before

            out = agent.merge()
            assert "Successfully merged" in out
            msg = _head_message(repo)
            assert msg == (
                f"{agent_msg}\n\nUser prompt:\n{PROMPT}\n\nResult:\n{RESULT}"
            )

    def test_merge_no_baseline_appends_prompt_and_result(self) -> None:
        """Normal clean-repo path (baseline_commit=None → the
        squash_merge_branch route): the merge commit must carry both
        blocks too.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-agent-3"
            wt_dir = _create_worktree(repo, branch)
            from kiss.agents.sorcar.git_worktree import GitWorktree

            agent = self._make_agent(repo, branch, wt_dir)
            agent._wt = GitWorktree(
                repo_root=repo,
                wt_dir=wt_dir,
                branch=branch,
                original_branch="main",
                baseline_commit=None,
            )
            agent_msg = _agent_manual_commit(wt_dir)
            agent._last_user_prompt = PROMPT
            agent._last_result_summary = RESULT

            out = agent.merge()
            assert "Successfully merged" in out
            msg = _head_message(repo)
            assert msg == (
                f"{agent_msg}\n\nUser prompt:\n{PROMPT}\n\nResult:\n{RESULT}"
            )

    def test_merge_without_recorded_state_unchanged(self) -> None:
        """No recorded prompt/result → merge message is the branch
        HEAD message verbatim (backward compatible).
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-msg-agent-2"
            wt_dir = _create_worktree(repo, branch)
            agent = self._make_agent(repo, branch, wt_dir)
            agent_msg = _agent_manual_commit(wt_dir)

            out = agent.merge()
            assert "Successfully merged" in out
            assert _head_message(repo) == agent_msg

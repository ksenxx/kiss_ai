# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression test for the misleading auto-commit warning.

Production evidence for the underlying race lives in
:mod:`kiss.tests.agents.sorcar.test_autocommit_race_new_file`.  That
test pins down the leaf-level
:func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes` contract.

This file pins down the *user-facing* contract one level up: when a
file appears during the LLM commit-message call,
:meth:`WorktreeSorcarAgent._finalize_worktree` must NOT log the
misleading "pre-commit hook may have rejected" warning.  Before the
fix, that wording was the only signal in the kiss-web log — and
because no pre-commit hook actually existed in the repo, the wrong
hint sent debugging down a multi-hour rabbit hole.

The test uses a real ``git init`` repo + real ``git worktree add`` so
there are no mocks of git itself; only ``_generate_commit_message``
is monkeypatched (it is the LLM-backed seam that does the slow work
inside which the race fires in production).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar import worktree_sorcar_agent as wta_mod
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo_with_worktree(tmp: Path) -> tuple[Path, Path, str]:
    """Create a git repo with one commit and a worktree on a task branch.

    Returns ``(repo_root, wt_dir, branch_name)``.
    """
    repo = tmp / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "T"],
        check=True,
    )
    (repo / "README.md").write_text("# test\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "initial"],
        check=True,
    )
    branch = "kiss/wt-test-finalize"
    wt_dir = tmp / "wt"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", branch, str(wt_dir)],
        check=True,
    )
    return repo, wt_dir, branch


def test_finalize_worktree_no_spurious_hook_warning(
    caplog: object,
    monkeypatch: object,
) -> None:
    """Race-during-LLM must finalize cleanly with no misleading warning.

    Reproduces the production scenario: an agent has just produced a
    real change (``agent_report.txt``); the LLM commit-message call
    is slow; while it is in flight, a second file
    (``late_arriver.txt``) materializes in the worktree
    (PROGRESS.md rewrite, ``open`` of the report touching DS_Store,
    editor side-channel, …).  The fix must commit BOTH files and
    finalize without logging the misleading "pre-commit hook may
    have rejected" warning that previously dominated the log.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo, wt_dir, branch = _make_repo_with_worktree(Path(tmp))

        # Agent's actual output: a real change present before the
        # auto-commit kicks off.
        (wt_dir / "agent_report.txt").write_text("agent's output\n")

        def racing_message_fn(commit_dir: Path, user_prompt: str | None) -> str:
            """Drop a new file while the (simulated) LLM is in flight.

            Mirrors the production race: ``stage_all`` has already
            run before this fn is called, so the new file is NOT in
            the staged snapshot the old code would have committed.
            """
            (commit_dir / "late_arriver.txt").write_text("race write\n")
            return "test: agent work"

        # The seam: ``auto_commit_changes`` looks up
        # ``worktree_sorcar_agent._generate_commit_message`` at call
        # time via the module's globals (re-exported for exactly this
        # purpose — see the import comment in
        # ``worktree_sorcar_agent``).
        monkeypatch.setattr(  # type: ignore[attr-defined]
            wta_mod, "_generate_commit_message", racing_message_fn,
        )

        agent = WorktreeSorcarAgent("test")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        with caplog.at_level(  # type: ignore[attr-defined]
            logging.WARNING, logger="kiss.agents.sorcar.worktree_sorcar_agent",
        ):
            ok = agent._finalize_worktree()

        assert ok is True, (
            "_finalize_worktree should succeed: the race-write must "
            "have been committed by the re-stage in auto_commit_changes "
            "(or by the late-arriver retry inside _finalize_worktree)."
        )

        # The whole point of this test: no misleading hook hint.
        for record in caplog.records:  # type: ignore[attr-defined]
            assert "pre-commit hook may have rejected" not in record.message, (
                "_finalize_worktree emitted the misleading "
                "'pre-commit hook may have rejected' warning for a "
                "plain race-with-concurrent-write case; this is the "
                "exact misleading-log regression the fix is meant to "
                "prevent."
            )

        # Worktree directory must have been removed on success.
        assert not wt_dir.exists()

        # Both files must be in HEAD of the task branch.
        ls = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "--name-only",
             "-r", branch],
            capture_output=True, text=True, check=True,
        )
        tracked = set(ls.stdout.splitlines())
        assert "agent_report.txt" in tracked
        assert "late_arriver.txt" in tracked


def test_finalize_worktree_warning_includes_porcelain_when_truly_stuck(
    caplog: object,
    monkeypatch: object,
) -> None:
    """When commit cannot succeed, warning must include the leftover files.

    Simulates a worker that keeps producing new files no matter how
    many times we stage — the only failure mode that should still
    trigger the "preserving" warning.  The improved warning must
    include the raw ``git status --porcelain`` output so an
    operator can see WHICH file(s) are stuck without sshing in.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo, wt_dir, branch = _make_repo_with_worktree(Path(tmp))
        (wt_dir / "agent_report.txt").write_text("agent's output\n")

        # Install a real pre-commit hook in the worktree's shared git
        # dir that rejects every commit.  This is the only reliable
        # way to force a "truly stuck" state without racing the
        # filesystem.
        # Worktrees share ``.git/hooks`` with the main repo.
        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        def msg_fn(commit_dir: Path, user_prompt: str | None) -> str:
            return "test: agent work"

        monkeypatch.setattr(  # type: ignore[attr-defined]
            wta_mod, "_generate_commit_message", msg_fn,
        )

        agent = WorktreeSorcarAgent("test")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        with caplog.at_level(  # type: ignore[attr-defined]
            logging.WARNING, logger="kiss.agents.sorcar.worktree_sorcar_agent",
        ):
            ok = agent._finalize_worktree()

        assert ok is False
        assert wt_dir.exists(), "Stuck worktree must be preserved."

        joined = "\n".join(
            r.getMessage() for r in caplog.records  # type: ignore[attr-defined]
        )
        assert "agent_report.txt" in joined, (
            "Preserve-worktree warning must include the leftover "
            "files (git status --porcelain) so the operator can "
            "diagnose without sshing in."
        )
        # Confirm the broader phrasing also landed (race vs hook vs
        # commit failure are all named as candidates).
        assert "concurrent write" in joined or "pre-commit hook" in joined
        # Sanity: status_porcelain output is in the log.
        leftover = GitWorktreeOps.status_porcelain(wt_dir)
        assert leftover  # non-empty

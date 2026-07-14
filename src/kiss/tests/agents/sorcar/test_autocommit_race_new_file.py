# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test for the auto-commit race captured on Jun 26 2026.

Production log evidence (``~/.kiss/kiss-web-stderr.log``)::

    2026-06-26 07:23:14,148 WARNING kiss.agents.sorcar.worktree_sorcar_agent:
        Worktree has uncommitted changes after auto-commit
        (pre-commit hook may have rejected); preserving:
        /Users/ksen/work/kiss/.kiss-worktrees/kiss_wt-1782483430-cb03445c

The worktree had no pre-commit hooks installed at all (only the git
sample hooks ship by default).  The actual cause is a window inside
:func:`~kiss.agents.sorcar.sorcar_agent.auto_commit_changes`:

1. ``GitWorktreeOps.stage_all`` runs ``git add -A`` over everything
   that exists *right now*.
2. ``message_fn`` issues an LLM call that takes several seconds.
3. While the LLM call is in flight, an unrelated process touches the
   worktree (the agent's PROGRESS.md auto-write, a ``open`` of the
   final report, ``.DS_Store`` materializing under TextEdit, …) and
   creates or modifies a tracked / trackable file.
4. ``GitWorktreeOps.commit_staged`` commits only the snapshot from
   step 1; the file from step 3 stays uncommitted.
5. ``_finalize_worktree`` checks ``has_uncommitted_changes`` right
   after, sees the leftover, logs the misleading "pre-commit hook may
   have rejected" warning, and aborts the auto-merge — leaving the
   worktree branch in limbo until the *next* task starts and the
   release pass succeeds.

This test reproduces the race deterministically by writing a new
tracked file from inside ``message_fn`` itself (which stands in for
the racing process), and asserts that ``auto_commit_changes`` ends
with no uncommitted leftover.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import auto_commit_changes


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
    (path / "README.md").write_text("# test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True,
    )
    return path


def test_auto_commit_picks_up_file_appearing_during_message_generation(
) -> None:
    """A file created while ``message_fn`` runs must still be committed.

    The old implementation staged once, then called the (slow) LLM
    message generator, then committed without re-staging — leaving any
    file that materialized during the LLM call uncommitted and
    triggering the spurious "pre-commit hook may have rejected"
    warning in ``_finalize_worktree``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        # Pre-existing change the agent made before the auto-commit.
        (repo / "agent_report.txt").write_text("agent's output\n")

        def racing_message_fn(
            commit_dir: Path,
            user_prompt: str | None,
            task_result: str | None = None,
        ) -> str:
            """Stand-in for the LLM call that races with the worktree.

            Drops a new tracked file into the worktree *after*
            ``stage_all`` has already run, mimicking PROGRESS.md /
            ``open`` / DS_Store activity observed in production.
            """
            (commit_dir / "late_arriver.txt").write_text(
                "appeared mid-LLM-call\n",
            )
            return "test: auto-commit"

        committed = auto_commit_changes(
            repo, user_prompt=None, message_fn=racing_message_fn,
        )

        assert committed is True
        # The regression: ``has_uncommitted_changes`` must be False;
        # before the fix, ``late_arriver.txt`` was left untracked.
        assert not GitWorktreeOps.has_uncommitted_changes(repo), (
            "auto_commit_changes left files uncommitted after a write "
            "during message generation — _finalize_worktree would log "
            "the spurious 'pre-commit hook may have rejected' warning "
            "and abort the auto-merge."
        )
        # Both the original AND the racing file must be in HEAD.
        ls = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "--name-only",
             "-r", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        tracked = set(ls.stdout.splitlines())
        assert "agent_report.txt" in tracked
        assert "late_arriver.txt" in tracked


def test_auto_commit_picks_up_file_appearing_when_message_fn_raises(
) -> None:
    """The retry path through the fallback message also must re-stage.

    When ``message_fn`` raises (LLM unreachable, timeout, …), the
    fallback message is used.  The fix must apply on this branch too,
    because in production the LLM call is exactly the operation most
    likely to be slow / fail and thereby widen the race window.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        (repo / "agent_report.txt").write_text("agent's output\n")

        def raising_message_fn(
            commit_dir: Path,
            user_prompt: str | None,
            task_result: str | None = None,
        ) -> str:
            (commit_dir / "late_arriver.txt").write_text(
                "appeared while LLM was failing\n",
            )
            raise RuntimeError("simulated LLM outage")

        committed = auto_commit_changes(
            repo,
            user_prompt="reproduce REFLEX bug",
            message_fn=raising_message_fn,
        )

        assert committed is True
        assert not GitWorktreeOps.has_uncommitted_changes(repo)
        ls = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "--name-only",
             "-r", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        tracked = set(ls.stdout.splitlines())
        assert "agent_report.txt" in tracked
        assert "late_arriver.txt" in tracked
        # User-prompt traceability is preserved on the fallback path.
        head_msg = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%B", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout
        assert "User prompt:\nreproduce REFLEX bug" in head_msg

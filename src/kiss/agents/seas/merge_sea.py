# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Merge agent — resolves git merge conflicts left by a squash merge.

Two ways to run it:

1. **Auto-commit mode.**  When the post-task auto-merge of a worktree
   branch conflicts, :mod:`kiss.server.merge_conflict_resolver`
   re-applies the branch onto the original branch with the conflict
   markers left in place and runs this SEA in-process, as a sub-agent
   of the task whose merge failed, so the agent's cost, tokens and
   steps are added to that task.  The agent only edits and stages the
   conflicted files; the resolver verifies the tree and commits.
2. **The ``/merge`` slash command.**  Typed into an idle tab, the
   command dispatches this file as a Sorcar Extension Agent through
   ``run_agent`` on the tab's working directory (no worktree, no
   auto-commit), with the user's text as the task — for example
   ``/merge finish the conflicted merge of kiss/wt-foo into main``.

Module-level getters (``system_prompt()``, ``is_parallel()``, ...) follow
the SEA contract in :mod:`kiss.server.agent_file`.
"""

from __future__ import annotations

from pathlib import Path

from kiss.core.brand import PRODUCT_NAME

MAX_BUDGET_USD = 5.0
"""Spending cap of one conflict-resolution run."""

SYSTEM_PROMPT = f"""\
You are the {PRODUCT_NAME} merge agent. Your only job is to finish a git merge
that stopped on conflicts.

Rules you MUST follow:

1. Work only inside the repository named in the task, on its current
   branch. Never check out another branch, never create branches, never
   run `git stash`, `git reset`, `git checkout -- <file>`, `git restore`,
   `git cherry-pick --abort`, `git merge --abort`, or anything else that
   discards the in-progress merge.
2. For every conflicted file, read the whole file, understand both sides
   of each `<<<<<<< ... ======= ... >>>>>>>` block, and write a resolution
   that keeps the intent of BOTH sides: the current branch's version
   (HEAD, "ours") carries the user's own work; the incoming version
   ("theirs") carries the changes of the task branch. Do not simply pick
   one side unless the two sides are alternative edits of the same thing,
   in which case prefer the incoming task-branch version.
3. Remove every conflict marker. Then stage each resolved file with
   `git add <file>` (or `git rm <file>` when the resolution is deletion).
4. Do NOT commit unless the task explicitly asks you to. Do not edit files
   that are not conflicted, do not create new files, and do not run
   formatters, linters or tests over the repository.
5. Before finishing, run `git diff --name-only --diff-filter=U` and make
   sure it prints nothing, and grep the resolved files for leftover
   `<<<<<<<` / `>>>>>>>` markers.
6. Finish with a short summary listing each file and how it was resolved.
   If a file cannot be resolved with confidence, say so explicitly in the
   summary instead of guessing, and leave that file conflicted.
"""


def system_prompt() -> str:
    """Return the merge agent's base system prompt (replaces ``SYSTEM.md``)."""
    return SYSTEM_PROMPT


def is_parallel() -> bool:
    """Never fan out: the resolution of one merge is a single sequential job."""
    return False


def use_web_tools() -> bool:
    """Never enable browser tools: everything needed is in the repository."""
    return False


def use_memory() -> bool:
    """Never load persistent memory tools: the job is local to one merge."""
    return False


def use_worktree() -> bool:
    """Run directly on the checkout that holds the conflicted merge."""
    return False


def auto_commit() -> bool:
    """Never auto-commit: the caller decides what to commit."""
    return False


def max_budget() -> float:
    """Return the USD cap of one conflict-resolution run."""
    return MAX_BUDGET_USD


def build_prompt(
    repo: Path,
    branch: str,
    original_branch: str,
    conflicted_files: list[str],
    task_prompt: str | None = None,
) -> str:
    """Return the task text for one in-process conflict-resolution run.

    Args:
        repo: The repository whose checkout holds the conflicted merge.
        branch: The task branch being merged in (the "theirs" side).
        original_branch: The branch checked out in *repo* (the "ours"
            side).
        conflicted_files: Repository-relative paths still unmerged.
        task_prompt: The prompt of the task that produced *branch*, so
            the agent knows what the incoming changes were meant to
            do, or ``None``.

    Returns:
        The prompt text.
    """
    files = "\n".join(f"- {path}" for path in conflicted_files)
    lines = [
        f"Repository: {repo}",
        f"Checked-out branch (ours, HEAD): {original_branch}",
        f"Incoming task branch (theirs): {branch}",
        "",
        "A squash merge of the task branch into the checked-out branch stopped "
        "with conflicts in these files:",
        files,
        "",
        "Resolve every conflict, remove all conflict markers, and stage the "
        "resolved files with `git add`. Do not commit.",
    ]
    if task_prompt:
        lines += [
            "",
            "The task branch was produced by an agent working on this task:",
            "<task>",
            task_prompt.strip(),
            "</task>",
        ]
    return "\n".join(lines)

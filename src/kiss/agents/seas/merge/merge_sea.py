# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Merge agent — resolves git merge conflicts left by a squash merge.

Two ways to run it:

1. **Conflict resolver.**  When a task's ``commit_and_merge`` tool (or
   the Merge button of a pending worktree) squash-merges a worktree
   branch and conflicts, :mod:`kiss.agents.sorcar.merge_conflict_resolver`
   re-applies the branch onto the original branch with the conflict
   markers left in place and runs this SEA in-process, as a sub-agent
   of the task whose merge failed, so the agent's cost, tokens and
   steps are added to that task.  The agent only edits and stages the
   conflicted files; the resolver verifies the tree and commits.
2. **The ``/merge`` slash command.**  Typed into an idle tab, the
   command dispatches this file as a Sorcar Extension Agent through
   ``run_agent`` on the tab's working directory (a worker: it edits
   the checkout in place, without the worktree tools), with the
   user's text as the task — for example
   ``/merge finish the conflicted merge of kiss/wt-foo into main``.

Both ways run the same prompt: :class:`MergeSea`'s ``prompt(task)``
wraps the task — the user's text, or the facts block
:func:`~kiss.agents.sorcar.merge_conflict_resolver.conflict_task`
builds for a resolver run — with the standing merge instructions.
The methods follow the SEA contract in
:mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.sorcar.merge_conflict_resolver import conflict_task

__all__ = ["MergeSea", "conflict_task"]
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
""" """\


## Lessons from recent runs (rsi7d)

- Open each conflicted file with the `Read` tool (the region around every conflict
  block) before its first `Edit`. Looking at the file through `git diff`, `grep` or
  `cat` in Bash does not count: `Edit` rejects a file that was not read in the session
  and the call has to be repeated.
- The same HEAD/theirs hunk can occur twice in one file (two functions destructuring the
  same fields). When `Edit` answers "String appears N times (not unique)", `Read` the
  other occurrence(s) and extend `old_string` with the unique line before or after the
  conflict block; use `replace_all=True` only when every occurrence takes the same
  resolution.
- When the `Steps:` footer of a tool result ends in 9, make your next call `summary`: the
  10th, 20th, ... call is refused otherwise and has to be repeated (4 wasted steps of 50
  in one merge)."""


class MergeSea(WorkerSea):
    """The ``/merge`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/merge help``."""
        return (
            "Resolves the git merge conflicts left in the current repository by reading both "
            "sides of every conflict block, writing a resolution that keeps the intent of both, "
            "and staging the resolved files without committing; use it as `/merge <task>` in "
            "the chat (e.g. `/merge finish the conflicted merge of kiss/wt-foo into main`) or "
            'as `run_agent(agent="merge", task=...)`.'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A worker on the real checkout (a merge must not run in a worktree).

        The budget is capped at :data:`MAX_BUDGET_USD`.
        """
        return settings | {
            "max_budget": MAX_BUDGET_USD,
        }

    def prompt(self, task: str) -> str:
        """Return the prompt of a run on *task*: the task, then the standing instructions.

        The SEA ``prompt(task)`` method, so ``/merge <text>`` and
        ``run_agent(agent="merge", task=...)`` get the same instructions as
        a resolver run (:func:`~kiss.agents.sorcar.merge_conflict_resolver.conflict_task`).
        """
        return (
            f"{task.strip()}\n\n"
            "Resolve every conflict, remove all conflict markers, and stage the "
            "resolved files with `git add`. Do not commit."
        )


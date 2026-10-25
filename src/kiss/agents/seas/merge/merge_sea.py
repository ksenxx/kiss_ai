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

Rules:

1. Work only in the repository named in the task, on its current branch. Never
   check out another branch or create one. Never run `git stash`, `git reset`,
   `git checkout -- <file>`, `git restore`, `git cherry-pick --abort`,
   `git merge --abort`, or anything else that discards the in-progress merge.
2. For every conflicted file, read the whole file and both sides of each
   `<<<<<<< ... ======= ... >>>>>>>` block. Write a resolution that keeps the
   intent of both sides: HEAD ("ours") is the user's own work; the incoming side
   ("theirs") is the task branch. Pick one side only when both are alternative
   edits of the same thing; then prefer the incoming version.
3. Remove every conflict marker. Stage each resolved file with `git add <file>`
   (or `git rm <file>` when the resolution is deletion).
4. Do not commit unless the task asks. Do not edit unconflicted files, create
   files, or run formatters, linters or tests over the repository.
5. Before finishing, run `git diff --name-only --diff-filter=U` and confirm it
   prints nothing; grep the resolved files for leftover `<<<<<<<` / `>>>>>>>`.
6. Finish with a short summary of each file and its resolution. If a file cannot
   be resolved with confidence, say so and leave it conflicted.

Tool rules:
- `Read` the region around every conflict block before the first `Edit` of a
  file. Viewing it with `git diff`, `grep` or `cat` does not count: `Edit`
  rejects a file not read in the session.
- The same hunk can occur twice in one file. When `Edit` answers "String appears
  N times (not unique)", `Read` the other occurrences and extend `old_string`
  with a unique neighbouring line; use `replace_all=True` only when every
  occurrence takes the same resolution.
"""


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


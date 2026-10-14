# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sync agent — fetches origin/main, merges it into the current branch and pushes.

Typed into a tab as ``/sync`` (or ``/sync upstream/dev`` to merge another
remote branch), the command runs this file as a Sorcar Extension Agent
on the tab's working directory: a worker that works on the checkout in
place, with the ``git_sync`` tool (:func:`kiss.agents.seas.sync.git_sync.sync`)
doing the fetch, merge and push in one deterministic call and the shell
and file tools only for resolving a merge that stops on conflicts.  It
replaces the "pull origin/main, merge, push" prompts that used to be
typed by hand.  The methods follow the SEA contract in
:mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.sync.git_sync import GitSyncError, sync

DEFAULT_REF = "origin/main"
"""The remote branch ``/sync`` merges when the message names none."""

ADD_TO_PROMPT = (
    "Sync the work dir checkout: fetch the remote branch named above "
    f"({DEFAULT_REF} when none is named), merge it into the current branch and push."
)
"""Appended to every task: a bare ``/sync`` runs on this text alone, and a caller that
prefixes its own text (the cron preamble) keeps it."""

MAX_BUDGET_USD = 2.0
"""Spending cap of one sync run; a conflict-free sync is two tool calls."""

SYSTEM_PROMPT = """\
You are the git sync agent. You update the checkout in the work dir from a remote
branch and push the result. Nothing else.

1. Call `git_sync` once with `repo` = the work dir named below and `ref` = the
   `<remote>/<branch>` the message names (default `origin/main`). It fetches, merges,
   pushes and returns a report. Never run `git fetch`, `git merge`, `git pull` or
   `git push` yourself.
2. If the report says the merge stopped on conflicts: read each conflicted file,
   resolve every `<<<<<<< ... ======= ... >>>>>>>` block keeping the intent of both
   sides, remove the markers, `git add` the files, `git commit --no-edit`, then call
   `git_sync` again to push. Never run `git merge --abort`, `git reset`,
   `git checkout --`, `git restore`, `git stash` or `git push --force`.
3. If the report says anything else failed (not a checkout, detached HEAD, a refused
   push), stop and report it. Do not work around it.
4. Call `finish` with a short HTML summary: branch, what was merged (fast-forward,
   merge commit or nothing), what was pushed, and any warning.
"""
"""The whole base system prompt of the sync agent (replaces ``SYSTEM.md``)."""


def git_sync(repo: str, ref: str = DEFAULT_REF) -> str:
    """Fetch *ref*, merge it into the current branch of the checkout at *repo* and push.

    Args:
        repo: Absolute path of the git checkout (the run's work dir).
        ref: The remote branch to merge, as ``<remote>/<branch>``.

    Returns:
        A report of what was fetched, merged and pushed, or of the step
        that failed (conflicts are left in progress for you to resolve).
    """
    try:
        return sync(repo, ref)
    except GitSyncError as exc:
        return str(exc)


class SyncSea(WorkerSea):
    """The ``/sync`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/sync help``."""
        return (
            "Fetches origin/main, merges it into the current branch of the tab's working "
            "directory and pushes, resolving merge conflicts when they stop the merge; use "
            "`/sync` in the chat (`/sync <remote>/<branch>` to merge another remote branch) "
            'or run_agent(agent="sync", task="origin/main").'
        )

    def prompt(self, task: str) -> str:
        """Return the message followed by :data:`ADD_TO_PROMPT` (alone for a bare ``/sync``)."""
        return f"{task.strip()}\n\n{ADD_TO_PROMPT}".strip()

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A worker with the shell and file tools on the checkout, capped at :data:`MAX_BUDGET_USD`.

        ``tool_profile`` is ``locked``: the agent needs Bash to commit a
        resolved merge and the file tools to resolve it, and nothing else.
        """
        return settings | {
            "tool_profile": "shell+edit",
            "max_budget": MAX_BUDGET_USD,
            "locked": ["tool_profile"],
        }

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Return *tools* plus :func:`git_sync`."""
        return [*tools, git_sync]

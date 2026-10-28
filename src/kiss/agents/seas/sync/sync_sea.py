# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sync agent — commits the checkout's changes, merges origin/main into its branch, pushes.

Typed into a tab as ``/sync`` (or ``/sync upstream/dev`` to merge another
remote branch), the command runs this file as a Sorcar Extension Agent
on the tab's working directory: a worker that works on the checkout in
place, with the ``git_sync`` tool (:func:`kiss.agents.seas.sync.git_sync.sync`)
doing the commit of whatever is uncommitted in the working tree, the
fetch, merge and push in one deterministic call and the shell and file
tools only for resolving a merge that stops on conflicts.  It replaces
the "commit, pull origin/main, merge, push" prompts that used to be
typed by hand.

When machines are registered (``/machine add``), the same ``/sync``
then runs on each of them, one after the other, as a sub-task
(``run_agent(agent="sync", machine=<name>)``) on that machine's
checkout of the same repository (committing the changes left
uncommitted there too), and finally syncs this checkout once more so
it gets what the machines pushed.  The sub-task names the
repository by its remote URL (``origin/main of https://... this
machine only``), never by this machine's path: the checkout lives
wherever the machine keeps it (its daemon's working directory, or
another directory it has opened; see
:func:`kiss.agents.seas.sync.git_sync.locate_checkout`), and a path of
this machine forced on it used to fail the sub-task before it started
(``Permission denied: '/Users'`` on a Linux box).  The sub-task's
message ends in :data:`THIS_MACHINE_ONLY`, which keeps it from fanning
out again to the machines registered there; ``/sync this machine
only`` is the user's way of asking for the local sync alone.  The
methods follow the SEA contract in :mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.sync.git_sync import GitSyncError, sync
from kiss.agents.sorcar.machines import load_machines

DEFAULT_REF = "origin/main"
"""The remote branch ``/sync`` merges when the message names none."""

THIS_MACHINE_ONLY = "this machine only"
"""A message containing these words syncs this checkout alone, whatever machines
are registered: the sub-task sent to a machine ends in them, so it never fans out
to the machines registered there."""

ADD_TO_PROMPT = (
    "Sync the work dir checkout: commit its uncommitted changes on the current branch, "
    f"fetch the remote branch named above ({DEFAULT_REF} when none is named), merge it "
    "into the current branch and push."
)
"""Appended to every task: a bare ``/sync`` runs on this text alone, and a caller that
prefixes its own text (the cron preamble) keeps it."""

MACHINES_PROMPT = "Then sync the registered machines, one at a time, in this order: "
"""Followed by the registered machine names, appended to a task that does not contain
:data:`THIS_MACHINE_ONLY` when at least one machine is registered."""

MAX_BUDGET_USD = 2.0
"""Spending cap per checkout (a conflict-free sync is two tool calls): a run's cap is
this times the number of registered machines plus one, and each machine's sub-task is
sent exactly this much, whatever that machine has registered."""

SYSTEM_PROMPT = f"""\
You are the git sync agent. You commit what is uncommitted in the checkout in the work
dir, update it from a remote branch and push the result, here and on the registered
machines the message lists. Nothing else.

1. Call `git_sync` once with `repo` = the work dir named below, `ref` = the
   `<remote>/<branch>` the message names (default `origin/main`) and `url` = the
   repository URL the message names after that ref with the word "of" (as in
   "origin/main of https://github.com/me/kiss.git"; leave `url` empty when the message
   names none). It commits the uncommitted changes of the working tree on the current
   branch (edited, deleted and new files that are not ignored), fetches, merges, pushes
   and returns a report whose first lines name the checkout it synced and the remote's
   URL. Never run `git add`, `git commit`, `git fetch`, `git merge`, `git pull` or
   `git push` yourself, except to finish a conflicted merge as step 2 says.
2. If the report says the merge stopped on conflicts: read each conflicted file,
   resolve every `<<<<<<< ... ======= ... >>>>>>>` block keeping the intent of both
   sides, remove the markers, `git add` the files, `git commit --no-edit`, then call
   `git_sync` again to push. Do this in the checkout the report names (when it is not
   the work dir, the files are under it and git runs there: `git -C <checkout> ...`).
   Never run `git merge --abort`, `git reset`, `git checkout --`, `git restore`,
   `git stash` or `git push --force`.
3. If the report says anything else failed (not a checkout, detached HEAD, a commit
   that failed, a refused push), stop and report it. Do not work around it.
4. If the message lists registered machines, sync each of them in the order given,
   one at a time: call `run_agent(agent="sync", task="<ref> of <the remote's URL from
   the git_sync report> {THIS_MACHINE_ONLY}", machine="<name>",
   max_budget="{MAX_BUDGET_USD:g}")` and wait for its result before the next machine
   (never `run_parallel`, never `wait="false"`). Pass no `options` and no `work_dir`:
   the machine keeps its checkout of that repository wherever it likes (a path of this
   machine means nothing there), and the sub-task finds it from the URL. The budget is
   one checkout's share, whatever the machine has registered. A machine that cannot be
   reached or whose sync fails is reported, with the reason its result gives, not
   retried; go on to the next one. When the last machine is done, call `git_sync` once
   more here so this checkout gets what the machines pushed.
5. Call `finish` with a short HTML summary: branch, what was committed (how many files,
   or nothing), what was merged (fast-forward, merge commit or nothing), what was
   pushed, any warning, and one line per machine saying what its sync did or why it
   failed.
"""
"""The whole base system prompt of the sync agent (replaces ``SYSTEM.md``)."""


def git_sync(repo: str, ref: str = DEFAULT_REF, url: str = "") -> str:
    """Commit the changes at *repo*, fetch *ref*, merge it into the current branch and push.

    Args:
        repo: Absolute path of the git checkout (the run's work dir).
        ref: The remote branch to merge, as ``<remote>/<branch>``.
        url: The URL of the repository to sync, when the message names
            one (``origin/main of <url>``). When *repo* is not a clone of
            it, the clone this machine's daemon has worked in is synced
            instead and the report says so; empty skips the check.

    Returns:
        A report of the checkout and remote URL synced, what was committed
        (the uncommitted changes of the working tree, when there were
        any), fetched, merged and pushed, or of the step that failed
        (conflicts are left in progress for you to resolve).
    """
    try:
        return sync(repo, ref, url)
    except GitSyncError as exc:
        return str(exc)


def machine_names(task: str) -> list[str]:
    """Return the registered machines *task* fans out to, in registration order.

    Empty when none is registered or *task* contains :data:`THIS_MACHINE_ONLY`.
    """
    if THIS_MACHINE_ONLY in task:
        return []
    return [machine.name for machine in load_machines()]


class SyncSea(WorkerSea):
    """The ``/sync`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/sync help``."""
        return (
            "Commits the uncommitted changes of the tab's working directory on its current "
            "branch, fetches origin/main, merges it into that branch "
            "and pushes, resolving merge conflicts when they stop the merge, then "
            "does the same on every registered machine (`/machine list`), one at a time, on "
            "that machine's checkout of the same repository (found by its remote URL), and "
            "syncs here once more; use `/sync` in "
            "the chat (`/sync <remote>/<branch>` to merge another remote branch, `/sync this "
            'machine only` to skip the machines) or run_agent(agent="sync", task="origin/main").'
        )

    def prompt(self, task: str) -> str:
        """Return the message, :data:`ADD_TO_PROMPT` and the machines to sync next.

        A bare ``/sync`` runs on :data:`ADD_TO_PROMPT` alone; the machine
        list (:func:`machine_names`) follows when there is one.
        """
        names = machine_names(task)
        machines = f"\n\n{MACHINES_PROMPT}{', '.join(names)}." if names else ""
        return f"{task.strip()}\n\n{ADD_TO_PROMPT}{machines}".strip()

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A worker with the shell, file and sub-agent tools, :data:`MAX_BUDGET_USD` per checkout.

        ``tool_profile`` is ``locked``: the agent needs Bash to commit a
        resolved merge, the file tools to resolve it, ``run_agent`` to
        sync the registered machines, and nothing else.  The budget
        covers this checkout and one per registered machine (a machine's
        sub-task spend counts in this run's).
        """
        return settings | {
            "tool_profile": "shell+edit+agents",
            "max_budget": MAX_BUDGET_USD * (1 + len(load_machines())),
            "locked": ["tool_profile"],
        }

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Return *tools* plus :func:`git_sync`."""
        return [*tools, git_sync]

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resolve a conflicted worktree merge with the merge SEA.

In auto-commit mode :meth:`kiss.server.merge_flow._MergeFlowMixin._handle_worktree_action`
passes :func:`resolve_merge_conflict` to
:meth:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent.merge`
as its ``conflict_resolver``; ``_do_merge`` calls it after the squash
merge returned :attr:`~kiss.agents.sorcar.git_worktree.MergeResult.CONFLICT`.
The branch is re-applied with the conflict markers left in the tree,
:mod:`kiss.agents.seas.merge_sea` runs in-process as a sub-agent of the
task whose merge failed (so the run shows up as a nested tab and its
spend is attributed to that task through
:func:`~kiss.agents.sorcar.sorcar_agent._attribute_sub_usage`), and the
resolved, staged result is verified and committed here.  Any failure
restores the clean tree and reports CONFLICT again, so the existing
manual-resolution message and branch preservation apply unchanged.

This module lives in the server layer because ``kiss.agents.sorcar``
must not import ``kiss.agents.seas`` (see the layering invariants).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.seas import merge_sea
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps, MergeResult

logger = logging.getLogger(__name__)

MergeAgentRunner = Callable[[Any, str, Path], None]
"""``(parent_agent, prompt, repo)``: run the merge agent on *repo*."""


def run_merge_sea(parent_agent: Any, prompt: str, repo: Path) -> None:
    """Run the merge SEA in-process as a sub-agent of *parent_agent*.

    The child is a :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`
    stamped like a ``run_parallel`` child (``_tab_id`` /
    ``_subagent_info``), so the frontend opens it as a nested tab of the
    parent's tab and its history row nests under the parent's.  It uses
    the parent's model unless the SEA defines ``model()``, the SEA's
    system prompt as its base prompt, and the SEA's budget cap.  Its
    spend is attributed to *parent_agent* whatever way the run ends.

    Args:
        parent_agent: The agent of the task whose merge conflicted.
        prompt: The task text (see :func:`merge_sea.build_prompt`).
        repo: The repository root holding the conflicted merge.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
    from kiss.agents.sorcar.sorcar_agent import (
        _attribute_sub_usage,
        _broadcast_subagent_done,
        _live_agent_usage,
        _persisted_task_id,
    )

    printer = getattr(parent_agent, "printer", None)
    parent_task_id = _persisted_task_id(parent_agent)
    parent_tab_id = str(getattr(parent_agent, "_tab_id", "") or "")
    sub_tab_id = f"task-{parent_task_id or parent_tab_id}__merge"
    model_getter = getattr(merge_sea, "model", None)
    model_name = str(
        model_getter() if callable(model_getter) else parent_agent.model_name
    )
    agent = ChatSorcarAgent("Merge conflict resolver")
    agent._tab_id = sub_tab_id
    agent._subagent_info = {
        "parent_task_id": parent_task_id,
        "parent_tab_id": parent_tab_id,
        "reviewer": False,
    }
    try:
        agent.run(
            prompt_template=prompt,
            model_name=model_name,
            work_dir=str(repo),
            printer=printer,
            is_parallel=merge_sea.is_parallel(),
            max_budget=merge_sea.max_budget(),
            model_config=(
                getattr(parent_agent, "model_config", None)
                if model_name == parent_agent.model_name else None
            ),
            base_system_prompt=merge_sea.system_prompt(),
            web_tools=merge_sea.use_web_tools(),
            use_memory=merge_sea.use_memory(),
        )
    finally:
        budget, tokens, steps = _live_agent_usage(agent)
        _attribute_sub_usage(parent_agent, budget, tokens, steps)
        if printer is not None:
            # Every tab watching the merge agent's task, plus its own
            # synthetic id, exactly like a ``run_parallel`` child.
            viewer_ids: list[str] = []
            fanout = getattr(printer, "_fanout_targets", None)
            sub_task_id = _persisted_task_id(agent)
            found = fanout(sub_task_id) if callable(fanout) and sub_task_id else None
            if isinstance(found, list):
                viewer_ids = [v for v in found if v]
            if sub_tab_id not in viewer_ids:
                viewer_ids.append(sub_tab_id)
            _broadcast_subagent_done(printer, viewer_ids, model_name)


def resolve_merge_conflict(
    parent_agent: Any,
    wt: GitWorktree,
    user_prompt: str | None,
    task_result: str | None,
    run_agent: MergeAgentRunner = run_merge_sea,
) -> MergeResult:
    """Re-apply a conflicted branch, resolve it with the merge SEA, commit.

    Implements the ``conflict_resolver`` callback of
    :meth:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent.merge`
    (the first four parameters are its contract).  Must be called with
    the repository checked out on ``wt.original_branch`` and clean (the
    state ``_do_merge`` is in after a CONFLICT), under the same locks.

    Args:
        parent_agent: The agent of the task whose merge conflicted; the
            merge agent's spend is attributed to it.
        wt: The worktree whose branch is being merged.
        user_prompt: The task prompt, for the agent and the commit message.
        task_result: The task result summary, for the commit message.
        run_agent: Runs the merge agent (``(parent_agent, prompt, repo)``);
            :func:`run_merge_sea` by default.

    Returns:
        :attr:`MergeResult.SUCCESS` when the branch is merged and
        committed, :attr:`MergeResult.MERGE_FAILED` when the resolved
        merge could not be committed (tree reset), otherwise
        :attr:`MergeResult.CONFLICT` with the tree restored to clean.
    """
    repo = wt.repo_root
    before = GitWorktreeOps.status_porcelain(repo)
    head_before = GitWorktreeOps.head_sha(repo) or ""
    conflicted = GitWorktreeOps.begin_conflicted_merge(repo, wt.branch, wt.baseline_commit)
    if conflicted is None:
        return MergeResult.CONFLICT
    try:
        if conflicted:
            prompt = merge_sea.build_prompt(
                repo, wt.branch, wt.original_branch or "", conflicted, user_prompt,
            )
            try:
                run_agent(parent_agent, prompt, repo)
            except Exception:
                logger.warning(
                    "merge agent failed for branch %s", wt.branch, exc_info=True,
                )
        result = GitWorktreeOps.finish_conflicted_merge(
            repo, wt.branch, conflicted, head_before,
            user_prompt=user_prompt, task_result=task_result,
        )
    except BaseException:
        GitWorktreeOps.abort_conflicted_merge(repo, before, head_before)
        raise
    if result == MergeResult.CONFLICT:
        logger.warning(
            "merge agent left branch %s unresolved; restoring clean tree", wt.branch,
        )
        GitWorktreeOps.abort_conflicted_merge(repo, before, head_before)
    return result

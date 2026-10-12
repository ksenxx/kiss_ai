# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resolve a conflicted worktree merge with the merge SEA.

:meth:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent.commit_and_merge`
(the agent's tool) and the server's Merge button pass
:func:`resolve_merge_conflict` to
:meth:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent.merge`
as the ``conflict_resolver``; ``_do_merge`` calls it after the squash
merge returned :attr:`~kiss.agents.sorcar.git_worktree.MergeResult.CONFLICT`.
The branch is re-applied with the conflict markers left in the tree,
the ``merge`` SEA (``src/kiss/agents/seas/merge/merge_sea.py``) runs
in-process as a sub-agent of the task whose merge failed (so the run
shows up as a nested tab and its spend is attributed to that task
through :func:`~kiss.agents.sorcar.sorcar_agent._attribute_sub_usage`),
and the resolved, staged result is verified and committed here.  Any
failure restores the clean tree and reports CONFLICT again, so the
manual-resolution message and branch preservation apply unchanged.

The SEA is loaded by its registered command name (``merge``) through
:func:`~kiss.agents.sorcar.agent_dispatch.resolve_agent`, never
imported: ``kiss.agents.sorcar`` must not import ``kiss.agents.seas``
(see the layering invariants).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps, MergeResult

logger = logging.getLogger(__name__)

MergeAgentRunner = Callable[[Any, str, Path], None]
"""``(parent_agent, prompt, repo)``: run the merge agent on *repo*."""


def conflict_task(
    repo: Path,
    branch: str,
    original_branch: str,
    conflicted_files: list[str],
    task_prompt: str | None = None,
) -> str:
    """Return the task text of one conflict-resolution run of the merge SEA.

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
        The task text.
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
        prompt: The task text (:func:`conflict_task`), shaped by the
            merge SEA's ``prompt``.
        repo: The repository root holding the conflicted merge.

    Raises:
        RuntimeError: When the ``merge`` SEA cannot be resolved.
    """
    from kiss.agents.sorcar.agent_dispatch import dispatch_epoch, resolve_agent
    from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
    from kiss.agents.sorcar.sea_commands import evaluate_sea, load_sea
    from kiss.agents.sorcar.sorcar_agent import (
        _attribute_sub_usage,
        _live_agent_usage,
        _notify_subagent_done,
        _persisted_task_id,
        subagent_parent_tab_id_of,
    )

    resolved = resolve_agent("merge", str(repo))
    if isinstance(resolved, str):
        raise RuntimeError(resolved)
    printer = getattr(parent_agent, "printer", None)
    parent_task_id = _persisted_task_id(parent_agent)
    # The tab the webviews show the parent under — for a sub-agent
    # parent its ``{parent}__sub_{task}`` tab, never its synthetic id.
    parent_tab_id = subagent_parent_tab_id_of(parent_agent)
    sub_tab_id = f"task-{parent_task_id or parent_tab_id}__merge"
    merge = evaluate_sea([load_sea(Path(resolved[0]))], prompt)
    merge_settings = merge.settings
    model_name = str(merge_settings.get("model") or parent_agent.model_name)
    agent = ChatSorcarAgent("Merge conflict resolver")
    agent._tab_id = sub_tab_id
    # A side channel like the task-update child: its work is the
    # resolved merge in the parent's repo, so its finished tab is not
    # worth reopening.  The daemon replays a finished side channel as
    # ``subagentDone`` instead of ``openSubagentTab`` (see
    # ``server._is_side_channel_row``).
    agent._subagent_info = {
        "parent_task_id": parent_task_id,
        "parent_tab_id": parent_tab_id,
        "reviewer": False,
        "side_channel": True,
    }
    # Bind the attribution to the parent's usage epoch of NOW: an
    # interactive merge can finish after the parent reset for its
    # next task, whose budget must not absorb this one's spend.
    epoch = dispatch_epoch(parent_agent)
    try:
        agent.run(
            prompt_template=merge.prompt,
            model_name=model_name,
            work_dir=str(repo),
            printer=printer,
            max_budget=merge_settings["max_budget"],
            model_config=(
                getattr(parent_agent, "model_config", None)
                if model_name == parent_agent.model_name else None
            ),
            system_prompt_hook=merge.system_prompt_hook,
            tools_hook=merge.tools_hook,
            llm_call_hook=merge.llm_call_hook,
            tool_call_hook=merge.tool_call_hook,
            web_tools=merge_settings["use_web_tools"],
            use_memory=merge_settings["use_memory"],
        )
    finally:
        budget, tokens, steps = _live_agent_usage(agent)
        _attribute_sub_usage(parent_agent, budget, tokens, steps, epoch=epoch)
        if printer is not None:
            _notify_subagent_done(
                printer, _persisted_task_id(agent), sub_tab_id, model_name,
            )


def resolve_merge_conflict(
    parent_agent: Any,
    wt: GitWorktree,
    user_prompt: str | None,
    task_result: str | None,
    run_agent: MergeAgentRunner = run_merge_sea,
) -> MergeResult:
    """Re-apply a conflicted branch, resolve it with the merge SEA, commit.

    Implements the ``conflict_resolver`` callback of
    :meth:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent.merge`
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
            prompt = conflict_task(
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

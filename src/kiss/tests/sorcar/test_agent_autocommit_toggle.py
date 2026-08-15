# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Auto-commit toggle must reach the worktree agent.

``WorktreeSorcarAgent.auto_commit_enabled`` used to be a hard-coded
``True`` that nothing ever assigned, guarding four branches whose
messages blamed a ``--no-auto-commit`` flag that does not exist
anywhere in the repository.  The switch the user actually operates
lives in ``config.json`` (``auto_commit_mode``) and travels
``vscode_config`` → ``AgentState`` → ``task_runner``, and was never
handed to the agent — so a follow-up prompt silently committed and
squash-merged the previous task into the user's branch even with
Auto-commit OFF.

Every test here uses a real git repository, a real worktree, a real
temp history DB under an isolated ``KISS_HOME`` whose real
``config.json`` carries the setting, and a local stand-in model
server.  Nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.tests.sorcar.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    run_git,
    tool_call_response,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-f2-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _writer_responder(filename: str, content: str) -> Any:
    """Script an agent that writes one file into its work dir, then finishes.

    The path is resolved against the agent's own ``work_dir`` (the
    worktree), which is what makes the write land inside the worktree.
    """
    state: dict[str, int] = {"calls": 0}

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        state["calls"] += 1
        if state["calls"] == 1:
            return tool_call_response(
                "Write", {"file_path": filename, "content": content},
            )
        return finish_response(f"wrote {filename}")

    return responder


def _main_branch_log(env: IsolatedKissHome) -> str:
    """Return the subject line of every commit on the original branch."""
    return run_git(env.repo, "log", "--format=%s").stdout


def _run_worktree_task(
    env: IsolatedKissHome,
    agent: WorktreeSorcarAgent,
    prompt: str,
    filename: str,
    **kwargs: Any,
) -> str:
    """Run one real worktree task that writes *filename*."""
    server = StandInModelServer(_writer_responder(filename, f"{prompt}\n"))
    try:
        return agent.run(
            prompt_template=prompt,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            **kwargs,
        )
    finally:
        server.stop()


def test_followup_prompt_does_not_merge_when_autocommit_is_off(
    env: IsolatedKissHome,
) -> None:
    """Auto-commit OFF must survive a follow-up prompt on the same tab.

    Task 1 succeeds and its worktree stays pending (the "Auto-commit
    and merge / Discard" bar).  Typing a second prompt is not a
    decision about task 1's work, so retiring the old worktree must
    NOT commit and squash-merge it into the user's branch.
    """
    env.write_config(auto_commit_mode=False)
    agent = WorktreeSorcarAgent("f2-off")

    _run_worktree_task(env, agent, "TASK ONE", "task-one.txt")
    assert agent._wt is not None, "task 1 did not run in a worktree"
    first_branch = agent._wt.branch

    _run_worktree_task(env, agent, "TASK TWO", "task-two.txt")

    log = _main_branch_log(env)
    assert "TASK ONE" not in log, (
        "with Auto-commit OFF, a follow-up prompt silently committed and "
        f"merged the previous task into the user's branch:\n{log}"
    )
    assert not (env.repo / "task-one.txt").exists(), (
        "task 1's file was published into the user's working tree"
    )
    branches = run_git(env.repo, "branch", "--list", first_branch).stdout
    assert first_branch in branches, (
        "task 1's branch was destroyed, so its work is unrecoverable"
    )


def test_followup_prompt_merges_when_autocommit_is_on(
    env: IsolatedKissHome,
) -> None:
    """The default (Auto-commit ON) behaviour is unchanged."""
    env.write_config(auto_commit_mode=True)
    agent = WorktreeSorcarAgent("f2-on")

    _run_worktree_task(env, agent, "TASK ONE", "task-one.txt")
    _run_worktree_task(env, agent, "TASK TWO", "task-two.txt")

    assert (env.repo / "task-one.txt").exists(), (
        "with Auto-commit ON the previous task must still be merged"
    )


def test_run_kwarg_overrides_the_persisted_setting(
    env: IsolatedKissHome,
) -> None:
    """An explicit ``auto_commit`` kwarg wins over ``config.json``.

    This is the surface the server's task runner uses to hand the
    per-task toggle (``AgentState.auto_commit_mode``) to the agent.
    """
    env.write_config(auto_commit_mode=True)
    agent = WorktreeSorcarAgent("f2-kwarg")

    result = _run_worktree_task(
        env, agent, "TASK ONE", "task-one.txt", auto_commit=False,
    )
    assert "success: true" in result, result
    assert agent._wt is not None
    assert (agent._wt.wt_dir / "task-one.txt").exists(), (
        "task 1 never wrote its file, so the merge check below would "
        "pass vacuously"
    )

    _run_worktree_task(env, agent, "TASK TWO", "task-two.txt", auto_commit=False)

    assert not (env.repo / "task-one.txt").exists(), (
        "auto_commit=False was ignored in favour of the persisted setting"
    )


def test_explicit_merge_still_commits_when_autocommit_is_off(
    env: IsolatedKissHome,
) -> None:
    """The "Auto-commit and merge" button must keep working.

    Auto-commit OFF only disables the AUTOMATIC paths.  Clicking merge
    IS the user's explicit instruction to commit and merge, so
    :meth:`WorktreeSorcarAgent.merge` must not refuse.
    """
    env.write_config(auto_commit_mode=False)
    agent = WorktreeSorcarAgent("f2-merge")

    _run_worktree_task(env, agent, "TASK ONE", "task-one.txt")
    message = agent.merge()

    assert "Successfully merged" in message, message
    assert (env.repo / "task-one.txt").exists(), (
        "the explicitly merged work never reached the user's branch"
    )

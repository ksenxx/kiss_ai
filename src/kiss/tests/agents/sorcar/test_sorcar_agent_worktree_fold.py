# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The worktree tools are ``SorcarAgent``'s; ``ChatSorcarAgent`` keeps the old order.

The former ``WorktreeSorcarAgent`` wrapped ``ChatSorcarAgent.run`` from
the outside: it retired a worktree pending from the previous task and
classified the task BEFORE the chat layer replaced the previous task's
prompt / result and prepended the chat history.  With the worktree
code folded into ``SorcarAgent``, ``ChatSorcarAgent.run`` has to keep
that order itself (``_begin_task`` and the raw-task classification run
before its own state changes); these tests pin it against real
behavior:

* a worktree the previous task left pending is committed to its
  kept-for-review branch with THAT task's prompt in the commit
  message, not the new task's;
* the classifier is asked about the task as submitted, not about the
  conversation (the chat history, which ``build_chat_prompt`` prepends
  to the prompt the model sees).

Real agents, a real scratch git repo, a real stand-in classifier
endpoint, real history rows; the run is stopped after classification
and tool construction by a ``tools_hook`` that raises (the channel a
SEA's ``tools()`` uses).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task, _save_task_result
from kiss.tests.agents.sorcar.test_task_classifier import (
    _DEV_TASK,
    _CountingClassifierEndpoint,
    env,  # noqa: F401 — pytest fixture
)
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
)

_PREVIOUS_MARKER = "PREVIOUS-TASK-RESULT-MARKER"


class _StopAfterSetup:
    """A SEA-style ``tools`` hook that raises: the run ends once its toolset is built."""

    def __call__(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        raise RuntimeError("stop-after-setup")


def _run(agent: ChatSorcarAgent, task: str, repo: Path, **kwargs: Any) -> str:
    """Run *task* on *agent* in *repo* up to its toolset and return the failure result."""
    result = agent.run(
        prompt_template=task,
        model_name=STANDIN_MODEL,
        work_dir=str(repo),
        printer=CapturePrinter(),
        tools_hook=_StopAfterSetup(),
        **kwargs,
    )
    assert yaml.safe_load(result)["success"] is False, result
    return result


def _classifier_prompt(request: dict[str, Any]) -> str:
    """The text of every message of a stand-in chat-completions *request*."""
    return json.dumps(request.get("messages", []))


def test_retired_worktree_is_committed_with_the_previous_task_prompt(
    env: IsolatedKissHome,  # noqa: F811 — pytest fixture
) -> None:
    standin = _CountingClassifierEndpoint(is_simple=False)
    agent = ChatSorcarAgent("fold-retire")
    try:
        _run(
            agent, "first task: add the greeting file", env.repo,
            model_config=standin.model_config,
        )
        # What the model would do next: isolate the repository and edit
        # it there, then end without ``commit_and_merge``.
        assert agent.create_worktree().startswith("Created worktree")
        wt = agent._wt
        assert wt is not None
        (wt.wt_dir / "greeting.txt").write_text("hello\n")
        branch = wt.branch

        _run(
            agent, "second task: unrelated follow-up", env.repo,
            model_config=standin.model_config,
        )
    finally:
        standin.stop()

    # The second run retired the first task's worktree: committed to its
    # own branch, kept for review, never merged.
    assert agent.worktrees == []
    assert not (env.repo / "greeting.txt").exists()
    message = subprocess.run(
        ["git", "log", "-1", "--format=%B", branch],
        cwd=env.repo, capture_output=True, text=True, check=True,
    ).stdout
    assert "first task: add the greeting file" in message
    assert "second task" not in message


def test_classifier_sees_the_submitted_task_not_the_chat_history(
    env: IsolatedKissHome,  # noqa: F811 — pytest fixture
) -> None:
    previous_id, chat_id = _add_task("earlier task in this chat")
    _save_task_result(f"{_PREVIOUS_MARKER} " * 200, task_id=previous_id)
    standin = _CountingClassifierEndpoint(is_simple=False)
    agent = ChatSorcarAgent("fold-classify")
    agent.resume_chat_by_id(chat_id)
    try:
        _run(agent, _DEV_TASK, env.repo, model_config=standin.model_config)
    finally:
        standin.stop()

    # The prompt the chat layer hands the model carries the history;
    # the classifier's question does not.
    assert _PREVIOUS_MARKER in agent.build_chat_prompt(_DEV_TASK)
    (request,) = standin.requests
    asked = _classifier_prompt(request)
    assert _DEV_TASK in asked
    assert _PREVIOUS_MARKER not in asked
    assert Path(env.repo, ".kiss-worktrees").exists() is False

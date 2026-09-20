# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test for slash-command persistence.

The runner rewrites a ``/xxx text`` prompt to a long ``run_agent``
directive before handing it to the LLM.  Every user-visible surface
(the ``task_history.task`` column, the frequent-tasks table, the
chat's ``last_user_prompt``) MUST still show the raw ``/xxx text``
the user typed — not the internal directive.  This is achieved via
the ``_history_prompt`` kwarg on :meth:`ChatSorcarAgent.run` (added
by the SEA feature), which decouples the LLM prompt from the
persistence prompt.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    history_rows,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """Isolated KISS_HOME + history DB + scratch repo."""
    isolated = IsolatedKissHome("kiss-sea-hist-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def test_history_prompt_kwarg_persists_raw_slash_prompt(
    env: IsolatedKissHome,
) -> None:
    """``_history_prompt`` overrides every persistence + display write.

    The LLM sees the ``run_agent`` directive in ``prompt_template``;
    the history row, ``last_user_prompt`` and frequent-tasks entry
    all see the raw ``/slack post hi`` the user typed.
    """
    raw_prompt = "/slack post hi to #general"
    llm_directive = (
        "Call run_agent immediately with agent=/abs/slack_sea.py, "
        "task=post hi to #general"
    )

    def _responder(request):  # type: ignore[no-untyped-def]
        return finish_response("slack posted")

    server = StandInModelServer(_responder)
    try:
        agent = ChatSorcarAgent("sea-chat")
        agent.run(
            prompt_template=llm_directive,
            _history_prompt=raw_prompt,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
        )
    finally:
        server.stop()

    # The task-history row's ``task`` column MUST hold the raw prompt.
    rows = history_rows()
    assert len(rows) == 1
    assert rows[0]["task"] == raw_prompt, (
        f"task column expected {raw_prompt!r} got {rows[0]['task']!r}"
    )
    # And the agent's own last_user_prompt attribute (used by
    # replay/history rebind) mirrors it.
    assert agent._last_user_prompt == raw_prompt


def test_default_persistence_unchanged_without_history_prompt(
    env: IsolatedKissHome,
) -> None:
    """Runs that do NOT pass ``_history_prompt`` behave exactly as before.

    Regression guard: the raw ``prompt_template`` is what gets
    persisted when the new kwarg is absent, so every non-slash run
    keeps its historical behaviour.
    """
    def _responder(request):  # type: ignore[no-untyped-def]
        return finish_response("done")

    server = StandInModelServer(_responder)
    try:
        agent = ChatSorcarAgent("sea-chat-plain")
        agent.run(
            prompt_template="plain user task",
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
        )
    finally:
        server.stop()
    rows = history_rows()
    assert len(rows) == 1
    assert rows[0]["task"] == "plain user task"
    assert agent._last_user_prompt == "plain user task"

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``llm_call_hook`` boundary survives a model hand-off.

``KISSAgent._llm_hook_conversation_index`` is the length of the
conversation when the last LLM call was made: the messages after it (the
tool results of the step) are what the next call's ``llm_call_hook``
sees.  A hand-off to a model on another endpoint converts Responses-API
items to Chat-Completions messages (a ``reasoning`` item is dropped,
``function_call`` items merge into the assistant message), so the
conversation shrinks and an unchanged index points past the tool
results: the hook of the first call on the new model saw nothing, so a
SEA counting or rewriting messages missed the switching step.  Seen
live on 2026-11-01 (gpt-4o -> claude-haiku-4-5 through ``set_model``).

The boundary now moves with the hand-off (:func:`hand_off_conversation`
returns where it lands) through :meth:`KISSAgent.switch_model`, which
the fallback swap and ``set_model`` share.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model import hand_off_conversation
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

_OPENAI = "https://api.openai.com/v1"
_READ = "The secret word is QUARTZ-PELICAN"
_SWITCHED = "Model changed from gpt-5.5 to gemini."
_TOOL_RESULTS = [
    {"role": "tool", "tool_call_id": "call_1", "content": _READ},
    {"role": "tool", "tool_call_id": "call_2", "content": _SWITCHED},
]


def _responses_tool_turn() -> list[dict[str, Any]]:
    """The items a Responses-API model holds after a tool-calling turn and its results."""
    return [
        {"role": "user", "content": [{"type": "input_text", "text": "Read, then switch."}]},
        {"id": "rs_1", "type": "reasoning", "summary": []},
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Calling the tools.", "annotations": []}],
        },
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "Read",
            "arguments": '{"file_path": "secret.txt"}',
            "status": "completed",
        },
        {
            "id": "fc_2",
            "type": "function_call",
            "call_id": "call_2",
            "name": "set_model",
            "arguments": '{"model_name": "gemini"}',
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": _READ},
        {"type": "function_call_output", "call_id": "call_2", "output": _SWITCHED},
    ]


# The agent advanced the boundary past the assistant's items when the
# turn returned, before the tool results were appended.
_BOUNDARY = 5


def _old_model() -> OpenAICompatibleModel2:
    old = OpenAICompatibleModel2("gpt-5.5", base_url=_OPENAI, api_key="k")
    old.conversation = _responses_tool_turn()
    return old


def test_the_boundary_lands_after_the_converted_prefix() -> None:
    old = _old_model()
    new = GeminiModel("gemini-2.5-flash", api_key="k")

    landed = hand_off_conversation(old, new, _BOUNDARY)

    assert new.conversation[landed:] == _TOOL_RESULTS
    assert new.conversation[landed - 1]["role"] == "assistant"
    assert [tc["id"] for tc in new.conversation[landed - 1]["tool_calls"]] == ["call_1", "call_2"]


def test_a_verbatim_hand_off_keeps_the_boundary() -> None:
    old = _old_model()
    new = OpenAICompatibleModel2("gpt-5.5-mini", base_url=_OPENAI, api_key="k")

    assert hand_off_conversation(old, new, _BOUNDARY) == _BOUNDARY
    assert new.conversation == old.conversation


class _HookReachedError(Exception):
    """Raised by the hook: the step reached its LLM call (never made)."""


def _agent(old: OpenAICompatibleModel2) -> KISSAgent:
    agent = KISSAgent("HandOffHook")
    agent.model = old
    agent.model_name = old.model_name
    agent.printer = None
    agent.function_map = {"finish": agent.finish}
    agent.max_steps = 5
    agent.max_budget = 1.0
    agent._llm_hook_conversation_index = _BOUNDARY
    return agent


def test_the_next_call_hook_sees_the_switching_steps_tool_results() -> None:
    agent = _agent(_old_model())
    new = GeminiModel("gemini-2.5-flash", api_key="k")
    new.initialize("")
    seen: list[list[dict[str, Any]]] = []

    def hook(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen.append(messages)
        raise _HookReachedError

    agent.llm_call_hook = hook
    agent._reasoning_tokens = agent._newest_reasoning_tokens = 4_000
    agent.switch_model(new, "gemini-2.5-flash")

    assert agent.model is new and agent.model_name == "gemini-2.5-flash"
    # The conversion dropped the reasoning items, so none is counted as carried.
    assert (agent._reasoning_tokens, agent._newest_reasoning_tokens) == (0, 0)
    assert agent._cached_tools_schema == new._build_openai_tools_schema(agent.function_map)
    with pytest.raises(_HookReachedError):
        agent._execute_step()
    assert seen == [_TOOL_RESULTS]


def test_a_gemini_to_gemini_switch_carries_the_thought_signatures() -> None:
    old = GeminiModel("gemini-2.5-flash", api_key="k")
    old.conversation = [{"role": "user", "content": "hi"}]
    old._thought_signatures = {"call_1": b"sig-1"}
    agent = _agent(_old_model())
    agent.model = old
    new = GeminiModel("gemini-3-flash-preview", api_key="k")
    new.initialize("")
    agent._reasoning_tokens = agent._newest_reasoning_tokens = 4_000

    agent.switch_model(new, "gemini-3-flash-preview")

    assert new._thought_signatures == {"call_1": b"sig-1"}
    assert new.conversation == old.conversation
    # Kept verbatim, the conversation still carries its reasoning.
    assert (agent._reasoning_tokens, agent._newest_reasoning_tokens) == (4_000, 4_000)

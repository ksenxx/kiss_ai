# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the keep-K reasoning policy applied through the providers.

The retained context holds the model's reasoning for the last
``Config.reasoning_keep_turns`` turns and the inputs of the last
``Config.context_edit_keep_tool_uses`` tool calls.  On the Anthropic API
the requests carry the server-side context edits and the agent starts a
reasoning turn every ``Config.reasoning_turn_steps`` steps with a user
note; on the OpenAI Responses API the request asks for the current
turn's reasoning only (GPT-5.6 and later).  The Anthropic run goes
against the scripted local stand-in, which records every request.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.context_compaction import COMPACTION_START_TOKENS, STUB_PREFIX
from kiss.core.kiss_agent import REASONING_TURN_NOTE, KISSAgent
from kiss.core.models.anthropic_model import CONTEXT_MANAGEMENT_BETA, context_edits
from kiss.core.models.model_info import model as build_model
from kiss.core.models.openai_compatible_model2 import (
    OpenAICompatibleModel2,
    supports_reasoning_context,
)
from kiss.core.printer import Printer
from kiss.tests.core import local_anthropic_server as anthropic_server
from kiss.tests.core.test_context_compaction import Write, big_output

APPLIED = [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 3,
            "cleared_input_tokens": 1234}]
"""What the stand-in reports as applied on one turn."""


class _Recorder(Printer):
    """A printer that keeps every event it is given."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        self.events.append({"type": type, "content": content, **kwargs})

    def token_callback(self, token: str) -> None:
        pass


def test_anthropic_requests_carry_the_context_edits_and_the_turn_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Twelve tool turns, the seventh reporting applied edits, then one at
    # the client-side compaction threshold and the finish.
    script = [anthropic_server.tool_use_message("big_output", {}, 3000, f"toolu_{i}")
              for i in range(12)]
    script[6] = anthropic_server.tool_use_message("big_output", {}, 3000, "toolu_6", APPLIED)
    script.append(anthropic_server.tool_use_message(
        "big_output", {}, COMPACTION_START_TOKENS, "toolu_12"))
    script.append(anthropic_server.tool_use_message(
        "finish", {"result": "<p>done</p>"}, 60_000, "toolu_13"))
    recorder = _Recorder()
    hook_calls: list[int] = []
    agent = KISSAgent("context-policy-test")

    def on_reset() -> None:
        hook_calls.append(agent.step_count)

    agent.context_reset_hook = on_reset
    # the keep-K thinking policy, switched on for the test (off by default: each
    # clearing rewrites the cached prefix, see Config.reasoning_keep_turns)
    monkeypatch.setattr(DEFAULT_CONFIG, "reasoning_keep_turns", 2)
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_trigger_tokens", 100_000)
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_keep_tool_uses", 10)
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_clear_at_least", 20_000)
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        agent.run(
            model_name=anthropic_server.MODEL, prompt_template="Call big_output repeatedly.",
            tools=[big_output, Write], max_steps=len(script) + 5, max_budget=50.0,
            printer=recorder,
        )
    assert agent.model.server_side_context_editing()
    assert agent.model.keeps_reasoning_turns()
    streamed = [r for r in requests if r.get("stream")]
    assert len(streamed) == len(script)
    # Every request: the beta header and the two edits, thinking first.
    edits = context_edits()
    assert [e["type"] for e in edits] == ["clear_thinking_20251015", "clear_tool_uses_20250919"]
    assert edits[0]["keep"] == {"type": "thinking_turns", "value": 2}
    assert edits[1]["trigger"] == {"type": "input_tokens", "value": 100_000}
    assert edits[1]["keep"] == {"type": "tool_uses", "value": 10}
    assert edits[1]["clear_at_least"] == {"type": "input_tokens", "value": 20_000}
    assert edits[1]["clear_tool_inputs"] is True
    for request in streamed:
        assert CONTEXT_MANAGEMENT_BETA in request["_beta"]
        assert request["context_management"] == {"edits": edits}
    # A turn note after the tool results of steps 5 and 10, in the user
    # message that carries those results (a message of tool results only
    # would continue the provider's turn).
    notes = [m for m in agent.model.conversation
             if m.get("role") == "user" and isinstance(m.get("content"), str)
             and m["content"].startswith("Harness note at step")]
    assert [m["content"].split(":")[0] for m in notes] == [
        "Harness note at step 5", "Harness note at step 10"]
    assert notes[0]["content"].startswith(REASONING_TURN_NOTE.format(step=5, keep=2))
    sixth = streamed[5]["messages"][-1]
    assert sixth["role"] == "user"
    assert [b["type"] for b in sixth["content"]] == ["tool_result", "text"]
    assert sixth["content"][1]["text"].startswith("Harness note at step 5")
    assert [b["type"] for b in streamed[4]["messages"][-1]["content"]] == ["tool_result"]
    # The applied edits of the seventh turn are reported as an event.
    cleared = [e for e in recorder.events if e["type"] == "context_edits"]
    assert len(cleared) == 1
    assert cleared[0]["step"] == 7
    assert cleared[0]["edits"] == APPLIED
    assert cleared[0]["cleared_input_tokens"] == 1234
    assert agent.model.last_context_edits == []
    # Tool results left the context: the Read dedupe forgets what it showed.
    assert hook_calls == [7]
    # No client-side compaction, even past its threshold: the provider edits.
    assert not [e for e in recorder.events if e["type"] == "compaction"]
    assert agent._next_compaction_at == COMPACTION_START_TOKENS
    assert not any(str(m.get("content", "")).startswith(STUB_PREFIX)
                   for m in agent.model.conversation)


def test_a_request_without_thinking_carries_only_the_tool_use_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On a budget-thinking model a ``max_tokens`` too small for the budget
    # (the task classifier's case) keeps thinking off; the API rejects the
    # thinking edit on such a request, so only the tool-use edit goes out.
    script = [anthropic_server.tool_use_message("finish", {"result": "<p>done</p>"}, 50, "t0")]
    agent = KISSAgent("context-policy-no-thinking")
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        agent.run(
            model_name="claude-haiku-4-5", prompt_template="Finish at once.",
            tools=[big_output], max_steps=3, max_budget=5.0, model_config={"max_tokens": 1000},
        )
    (request,) = [r for r in requests if r.get("stream")]
    assert "thinking" not in request
    assert CONTEXT_MANAGEMENT_BETA in request["_beta"]
    assert [e["type"] for e in request["context_management"]["edits"]] == [
        "clear_tool_uses_20250919"]
    assert [e["type"] for e in context_edits(thinking=False)] == ["clear_tool_uses_20250919"]


def test_keep_k_without_thinking_sends_no_edits_and_leaves_the_compaction_to_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The keep-K thinking policy on, the tool-use clearing off, and the
    # model's thinking switched off in model_config: the request carries no
    # edit at all (the API rejects the thinking edit without thinking), so
    # the model must report no server-side editing and no kept reasoning
    # turns; the agent then adds no turn note and compacts the conversation
    # itself past its threshold.
    script = [
        anthropic_server.tool_use_message("big_output", {}, 3000, f"toolu_{i}") for i in range(30)
    ]
    script.append(
        anthropic_server.tool_use_message("big_output", {}, COMPACTION_START_TOKENS, "toolu_30")
    )
    script.append(
        anthropic_server.tool_use_message("finish", {"result": "<p>done</p>"}, 60_000, "toolu_31")
    )
    recorder = _Recorder()
    monkeypatch.setattr(DEFAULT_CONFIG, "reasoning_keep_turns", 2)
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_trigger_tokens", 0)
    agent = KISSAgent("context-policy-thinking-off")
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        agent.run(
            model_name=anthropic_server.MODEL,
            prompt_template="Call big_output repeatedly.",
            tools=[big_output, Write],
            max_steps=len(script) + 5,
            max_budget=50.0,
            printer=recorder,
            model_config={"thinking": {"type": "disabled"}},
        )
    streamed = [r for r in requests if r.get("stream")]
    assert len(streamed) == len(script)
    for request in streamed:
        assert request["thinking"] == {"type": "disabled"}
        assert "context_management" not in request
        assert CONTEXT_MANAGEMENT_BETA not in request["_beta"]
    assert not agent.model.server_side_context_editing()
    assert not agent.model.keeps_reasoning_turns()
    assert not [
        m
        for m in agent.model.conversation
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("Harness note at step")
    ]
    compactions = [e for e in recorder.events if e["type"] == "compaction"]
    assert len(compactions) == 1
    assert compactions[0]["step"] == 32
    assert compactions[0]["results"] > 0
    stubbed = [
        block
        for m in agent.model.conversation
        if isinstance(m.get("content"), list)
        for block in m["content"]
        if block.get("type") == "tool_result"
        and str(block.get("content", "")).startswith(STUB_PREFIX)
    ]
    assert len(stubbed) == compactions[0]["results"]


def test_context_edits_follow_the_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # the defaults: no thinking clearing (no turn notes either), a rare and bulk
    # tool-use clearing past 200k input tokens that keeps the last 20 tool calls
    # and must remove 100k tokens to be applied
    (default,) = context_edits()
    assert default["type"] == "clear_tool_uses_20250919"
    assert default["trigger"] == {"type": "input_tokens", "value": 200_000}
    assert default["keep"] == {"type": "tool_uses", "value": 20}
    assert default["clear_at_least"] == {"type": "input_tokens", "value": 100_000}
    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
    default_model = build_model(anthropic_server.MODEL)
    assert default_model.server_side_context_editing()
    assert not default_model.keeps_reasoning_turns()
    monkeypatch.setattr(DEFAULT_CONFIG, "reasoning_keep_turns", 0)
    assert [e["type"] for e in context_edits()] == ["clear_tool_uses_20250919"]
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_clear_at_least", 0)
    assert "clear_at_least" not in context_edits()[0]
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_trigger_tokens", 0)
    assert context_edits() == []
    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
    model = build_model(anthropic_server.MODEL)
    assert not model.server_side_context_editing()
    assert not model.keeps_reasoning_turns()


def test_responses_requests_render_the_current_turns_reasoning_only_on_gpt_5_6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "reasoning_keep_turns", 2)
    assert supports_reasoning_context("gpt-5.6-luna")
    assert supports_reasoning_context("openrouter/openai/gpt-6-sol")
    assert not supports_reasoning_context("gpt-5.5-sol")
    assert not supports_reasoning_context("gpt-4.1-mini-2025-04-14")
    assert not supports_reasoning_context("deepseek-ai/DeepSeek-V4-Pro")
    config: dict[str, Any] = {
        "use_responses_api": True, "base_url": "http://127.0.0.1:9/v1", "api_key": "local",
    }
    os.environ.setdefault("OPENAI_API_KEY", "local")
    new = build_model("gpt-5.6-luna", model_config=config)
    assert isinstance(new, OpenAICompatibleModel2)
    new.initialize("task")
    assert new.keeps_reasoning_turns()
    assert new._build_request_kwargs(tools=None)["reasoning"]["context"] == "current_turn"
    assert not new.compacts_whole_turns()
    # A caller's own setting stands.
    pinned = build_model("gpt-5.6-luna", model_config={
        **config, "reasoning": {"context": "all_turns"}})
    assert isinstance(pinned, OpenAICompatibleModel2)
    pinned.initialize("task")
    assert not pinned.keeps_reasoning_turns()
    assert pinned._build_request_kwargs(tools=None)["reasoning"]["context"] == "all_turns"
    # An earlier model: no such field.
    old = build_model("gpt-4.1-mini-2025-04-14", model_config=config)
    assert isinstance(old, OpenAICompatibleModel2)
    old.initialize("task")
    assert not old.keeps_reasoning_turns()
    assert "context" not in old._build_request_kwargs(tools=None).get("reasoning", {})

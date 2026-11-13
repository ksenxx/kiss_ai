# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the retained-context policy applied through the providers.

The retained context holds the inputs and results of the last
``Config.context_edit_keep_tool_uses`` tool calls and, under the keep-K
policy, the model's reasoning for the last ``Config.reasoning_keep_turns``
turns.  For a Claude model on the Anthropic API the agent clears the old
tool calls in its own conversation, rarely and in bulk, so that the
provider's prompt cache is re-written only then; the reasoning edit goes
out with the requests and the agent starts a reasoning turn every
``Config.reasoning_turn_steps`` steps with a user note.  On the OpenAI
Responses API the request asks for the current turn's reasoning only
(GPT-5.6 and later).  The Anthropic runs go against the local stand-in,
which records every request and, served ``as_provider``, bills them from
a prompt cache and checks the thinking blocks they send back.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.context_compaction import (
    COMPACTION_MIN_STEP_GAP,
    COMPACTION_START_TOKENS,
    STUB_PREFIX,
    clear_thresholds,
)
from kiss.core.kiss_agent import REASONING_TURN_NOTE, KISSAgent
from kiss.core.models.anthropic_model import (
    CONTEXT_MANAGEMENT_BETA,
    AnthropicModel,
    context_edits,
)
from kiss.core.models.model_info import get_cache_write_ratio
from kiss.core.models.model_info import model as build_model
from kiss.core.models.openai_compatible_model2 import (
    OpenAICompatibleModel2,
    supports_reasoning_context,
)
from kiss.core.printer import Printer
from kiss.tests.core import local_anthropic_provider as provider
from kiss.tests.core import local_anthropic_server as anthropic_server
from kiss.tests.core.test_context_compaction import Write, big_output

APPLIED = [{"type": "clear_thinking_20251015", "cleared_thinking_turns": 3,
            "cleared_input_tokens": 1234}]
"""What the stand-in reports as applied on one turn."""
THINKING_EDIT = {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 2}}
TOOL_USE_EDIT = {
    "type": "clear_tool_uses_20250919",
    "trigger": {"type": "input_tokens", "value": 40_000},
    "keep": {"type": "tool_uses", "value": 5},
    "clear_at_least": {"type": "input_tokens", "value": 20_000},
}
"""The provider's own clearing, under the rule :func:`_session` gives the agent."""


class _Recorder(Printer):
    """A printer that keeps every event it is given."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        self.events.append({"type": type, "content": content, **kwargs})

    def token_callback(self, token: str) -> None:
        pass

    def steps(self, event_type: str) -> list[int]:
        """Return the steps of the recorded events of *event_type*, in order."""
        return [e["step"] for e in self.events if e["type"] == event_type]


def note() -> str:
    """Return a tool output too short to be stubbed."""
    return "ok"


REASONED = {"name": "note", "thinking": "r" * 8_000, "output_tokens": 2_030}
"""A scripted reply that is 2,000 tokens of reasoning and a call with a short result."""


def _session(
    monkeypatch: pytest.MonkeyPatch,
    steps: int,
    model: str = anthropic_server.MODEL,
    trigger: int = 40_000,
    clear_at_least: int = 20_000,
    reply: dict[str, Any] | None = None,
    replies: dict[int, dict[str, Any]] | None = None,
    **run_options: Any,
) -> tuple[KISSAgent, list[dict[str, Any]], _Recorder]:
    """Run *steps* large tool calls and a finish against the stand-in served as the provider.

    The clearing rule is scaled down so that a short session clears several
    times: past *trigger* tokens of context the last 5 tool calls are kept,
    when that removes *clear_at_least* tokens (both for a model whose cache
    write costs fifty cache reads, as the stand-in's default model does).
    *reply* replaces fields of every scripted reply, *replies* those of single
    ones by index.  A call of the agent's context-reset hook is recorded as a
    ``context_reset`` event.
    """
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_trigger_tokens", trigger)
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_keep_tool_uses", 5)
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_clear_at_least", clear_at_least)
    script = [{**anthropic_server.tool_use_message("big_output", {}, 0, f"toolu_{i}"),
               **(reply or {}), **(replies or {}).get(i, {})} for i in range(steps)]
    script.append(anthropic_server.tool_use_message(
        "finish", {"result": "<p>done</p>"}, 0, "toolu_end"))
    recorder = _Recorder()
    agent = KISSAgent("retained-context-session")

    def on_reset() -> None:
        recorder.print("", type="context_reset", step=agent.step_count)

    agent.context_reset_hook = on_reset
    with anthropic_server.serve(script, as_provider=True) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        agent.run(
            model_name=model, prompt_template="Call big_output repeatedly.",
            tools=[big_output, Write, note], max_steps=steps + 5, max_budget=500.0,
            printer=recorder, **run_options,
        )
    assert len(recorder.steps("llm_call")) == steps + 1
    return agent, requests, recorder


def _cache_rewrites(recorder: _Recorder) -> tuple[list[int], list[int]]:
    """Return the steps of the calls that re-wrote the prompt cache, and those that freed nothing.

    A call re-writes the cache when it reads less from it than the call
    before left there (what that call read and what it wrote).  The
    re-write leaves the context the same size when the call's context is
    not smaller than the context of the call before.
    """
    calls = [e for e in recorder.events if e["type"] == "llm_call"]
    context = [c["input_tokens"] + c["cache_read"] + c["cache_write"] for c in calls]
    rewrites = [
        i for i in range(1, len(calls))
        if calls[i]["cache_read"] < calls[i - 1]["cache_read"] + calls[i - 1]["cache_write"]
    ]
    same_size = [i for i in rewrites if context[i] >= context[i - 1]]
    return [calls[i]["step"] for i in rewrites], [calls[i]["step"] for i in same_size]


def _blocks(request: dict[str, Any], block_type: str) -> list[dict[str, Any]]:
    """Return the content blocks of *block_type* in the messages of *request*."""
    return [
        block for message in request["messages"] if isinstance(message["content"], list)
        for block in message["content"] if block["type"] == block_type
    ]


def _clears(recorder: _Recorder) -> list[dict[str, Any]]:
    """Return the ``compaction`` events of a session, each with the context of the call after it."""
    context = {
        e["step"]: e["input_tokens"] + e["cache_read"] + e["cache_write"]
        for e in recorder.events if e["type"] == "llm_call"
    }
    return [
        {**e, "floor": context[e["step"]]} for e in recorder.events if e["type"] == "compaction"
    ]


def test_a_long_session_rewrites_the_prompt_cache_only_at_its_own_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, requests, recorder = _session(monkeypatch, steps=120)
    # The provider is asked for no edit; the agent clears its own conversation.
    for request in requests:
        assert "context_management" not in request
        assert CONTEXT_MANAGEMENT_BETA not in request["_beta"]
    assert agent.model.clears_tool_uses_in_bulk()
    rewrites, same_size = _cache_rewrites(recorder)
    # No call re-wrote the cache and left the context the same size ...
    assert same_size == []
    # ... because the cached prefix changed only at the agent's own clears.
    clears = [e for e in recorder.events if e["type"] == "compaction"]
    assert rewrites == [e["step"] for e in clears] == recorder.steps("context_reset")
    assert len(clears) >= 5
    for event in clears:
        assert event["context_tokens"] >= 40_000 and event["dropped_tokens"] >= 20_000
        assert event["results"] >= 10 and event["target_tokens"] is None
        # The request after a clear: the last 5 results the model saw and the
        # one it has not seen yet are whole, and no thinking block is sent
        # back (each was written after messages the clear has changed).
        request = requests[event["step"] - 1]
        results = _blocks(request, "tool_result")
        assert sum(not r["content"].startswith(STUB_PREFIX) for r in results) == 6
        assert event["reasoning_blocks"] > 0 and not _blocks(request, "thinking")
    # Between two clears the thinking is kept: the last request sends back
    # that of every turn since the last clear.
    assert len(_blocks(requests[-1], "thinking")) == len(requests) - clears[-1]["step"]
    # What the provider answers when a result is stubbed and the thinking kept.
    stale = copy.deepcopy(requests[clears[0]["step"] - 2])
    assert provider.stale_thinking(stale) is None
    _blocks(stale, "tool_result")[0]["content"] = "stubbed"
    assert "bound to a different conversation" in str(provider.stale_thinking(stale))


def test_the_providers_own_clearing_rewrites_the_cache_without_freeing_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The session the agent's own clearing replaces: the provider clears
    # (asked here through the caller's model_config, the agent's clearing
    # switched off) and, as on 2026-11-11, its boundary is one result short
    # on some calls.  Such a call re-writes the cache while the context only
    # grows, and so does the next one, on which the boundary moves back.
    _agent, requests, recorder = _session(
        monkeypatch, steps=60, compaction=False,
        model_config={"extra_body": {"context_management": {"edits": [TOOL_USE_EDIT]}}},
    )
    assert all(r["context_management"] == {"edits": [TOOL_USE_EDIT]} for r in requests)
    assert not recorder.steps("compaction")
    _rewrites, same_size = _cache_rewrites(recorder)
    assert len(same_size) >= 10
    # What the provider cleared is reported, and the Read dedupe forgets with it.
    edits = [e for e in recorder.events if e["type"] == "context_edits"]
    assert edits and all(e["edits"][0]["cleared_tool_uses"] > 0 for e in edits)
    assert recorder.steps("context_reset") == [e["step"] for e in edits]
    # The agent's conversation was not edited: every turn's thinking is sent back.
    assert len(_blocks(requests[-1], "thinking")) == 60


def test_a_model_that_does_not_bind_its_thinking_keeps_only_its_newest_through_a_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Claude Haiku 4.5 signs a thinking block over the block alone, so its
    # blocks would survive a clear; the old ones go with the old results all
    # the same, and the reply the pending tool result answers keeps its own.
    _agent, requests, recorder = _session(monkeypatch, steps=30, model="claude-haiku-4-5")
    clears = _clears(recorder)
    assert len(clears) >= 2 and all(e["reasoning_blocks"] > 0 for e in clears)
    for event in clears:
        request = requests[event["step"] - 1]
        assert len(_blocks(request, "thinking")) == 1
        assert _blocks(request, "thinking") == [request["messages"][-2]["content"][0]]
    assert _cache_rewrites(recorder) == ([e["step"] for e in clears], [])


def test_the_clears_follow_the_models_cache_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    # The settings hold at fifty cache reads a cache write; a cheaper write
    # is worth making sooner, by the square root of the ratio.  A step adds
    # about 2,000 tokens, so the first clear comes within one step of the
    # scaled trigger.
    dropped = []
    for model, write_reads in (
        ("claude-fable-5-1", 50.0), ("claude-opus-5-5", 25.0), ("claude-haiku-4-5", 12.5)
    ):
        assert get_cache_write_ratio(model) == write_reads
        _agent, _requests, recorder = _session(
            monkeypatch, steps=45, model=model, trigger=60_000, clear_at_least=10_000
        )
        first = _clears(recorder)[0]
        trigger = 60_000 * (write_reads / 50) ** 0.5
        assert trigger <= first["context_tokens"] < trigger + 2_200
        assert _cache_rewrites(recorder)[1] == []
        dropped.append(first["dropped_tokens"])
    # The dearer the write, the later the clear and the more it removes.
    assert dropped[0] > dropped[1] > dropped[2] >= 5_000


def test_a_small_window_is_cleared_before_the_hand_off_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Claude Haiku 4.5 hands off at 140k of its 200k window, below its
    # trigger under the default settings (250k, 125k at its prices): the
    # trigger is 80% of the hand-off limit instead.
    _agent, _requests, recorder = _session(
        monkeypatch, steps=64, model="claude-haiku-4-5", trigger=250_000, clear_at_least=50_000
    )
    clears = _clears(recorder)
    assert len(clears) == 1
    assert 112_000 <= clears[0]["context_tokens"] < 125_000 and clears[0]["floor"] < 30_000


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-haiku-4-5"])
def test_reasoning_is_cleared_with_the_tool_calls_and_the_floor_does_not_rise(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    # A session of reasoning: every reply is 2,000 tokens of thinking and a
    # call whose result is too short to stub, so nothing but the reasoning
    # can be cleared.  It counts toward the minimum and it goes at the clear,
    # so from one clear to the next the size the context falls back to grows
    # by the calls and their results alone, under 150 tokens a step.
    _agent, requests, recorder = _session(monkeypatch, steps=70, model=model, reply=REASONED)
    clears = _clears(recorder)
    assert len(clears) >= 3
    for event in clears:
        assert event["results"] == event["arguments"] == 0
        assert event["reasoning_blocks"] >= 9 and event["dropped_tokens"] >= 10_000
        assert len(_blocks(requests[event["step"] - 1], "thinking")) <= 1
    steps = clears[-1]["step"] - clears[0]["step"]
    assert 0 < clears[-1]["floor"] - clears[0]["floor"] < 150 * steps
    assert _cache_rewrites(recorder) == ([e["step"] for e in clears], [])


def test_the_newest_reply_an_earlier_model_keeps_is_not_counted_as_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Haiku 4.5 keeps the thinking of its newest reply through a clear.  One
    # reply of 30,000 reasoning tokens among short ones: right after it
    # nothing else can go, so a clear would re-write the cache and leave the
    # context as large.  It is made one step later, when that reply is an old
    # one, and takes the reasoning with it.
    big = {"thinking": "r" * 120_000, "output_tokens": 30_030}
    _agent, _requests, recorder = _session(
        monkeypatch, steps=16, model="claude-haiku-4-5", reply={"name": "note"},
        replies={11: big},
    )
    grown = next(e["step"] for e in recorder.events
                 if e["type"] == "llm_call" and e["output_tokens"] == 30_030)
    (clear,) = _clears(recorder)
    assert clear["step"] == grown + 2
    assert clear["dropped_tokens"] >= 30_000 and clear["floor"] < 10_000
    assert _cache_rewrites(recorder) == ([clear["step"]], [])


def test_a_clearing_without_a_minimum_waits_for_the_step_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No minimum to remove and a trigger the kept tool calls alone exceed:
    # every step is past the trigger, and the step gap of the compaction is
    # what keeps the clears, each a re-write of the cache, apart.
    _agent, _requests, recorder = _session(monkeypatch, steps=45, trigger=8_000, clear_at_least=0)
    clears = recorder.steps("compaction")
    assert len(clears) >= 3
    assert {b - a for a, b in zip(clears, clears[1:])} == {COMPACTION_MIN_STEP_GAP}
    assert _cache_rewrites(recorder)[0] == clears


def test_anthropic_requests_carry_the_thinking_edit_and_the_turn_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Twelve tool turns, the seventh reporting an applied edit, then one
    # past the clearing trigger and the finish.
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
    assert agent.model.keeps_reasoning_turns()
    streamed = [r for r in requests if r.get("stream")]
    assert len(streamed) == len(script)
    # Every request: the beta header and the thinking edit, the only one.
    assert context_edits() == [THINKING_EDIT]
    for request in streamed:
        assert CONTEXT_MANAGEMENT_BETA in request["_beta"]
        assert request["context_management"] == {"edits": [THINKING_EDIT]}
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
    # The applied edit of the seventh turn is reported as an event.
    cleared = [e for e in recorder.events if e["type"] == "context_edits"]
    assert len(cleared) == 1
    assert cleared[0]["step"] == 7
    assert cleared[0]["edits"] == APPLIED
    assert cleared[0]["cleared_input_tokens"] == 1234
    assert agent.model.last_context_edits == []
    # No tool result left the context: the Read dedupe keeps what it showed.
    assert hook_calls == []
    # Past the trigger the results older than the last ten hold less than
    # the 20k tokens a clear must remove, so nothing is cleared.
    assert not recorder.steps("compaction")
    assert not any(r["content"].startswith(STUB_PREFIX)
                   for r in _blocks(streamed[-1], "tool_result"))


def test_keep_k_without_thinking_sends_no_edits_and_no_turn_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The keep-K thinking policy on and the model's thinking switched off in
    # model_config: the request carries no edit (the API rejects the thinking
    # edit without thinking), so the model reports no kept reasoning turns
    # and the agent adds no turn note.  With the clearing trigger at 0 the
    # conversation is compacted by the thresholds of the Compaction settings.
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
    assert not agent.model.clears_tool_uses_in_bulk()
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
    assert compactions[0]["results"] > 0 and compactions[0]["target_tokens"] == 150_000
    stubbed = [r for r in _blocks(streamed[-1], "tool_result")
               if r["content"].startswith(STUB_PREFIX)]
    assert len(stubbed) == compactions[0]["results"]
    # The compaction dropped the reasoning too: what the agent still counts
    # as carried is that of the one reply after it, not of all 32.
    assert agent._reasoning_tokens <= 30


def test_the_policy_follows_the_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # The defaults: no edit is sent (no thinking clearing, so no turn notes
    # either) and the agent clears a Claude model's old tool calls and
    # reasoning itself, rarely and in bulk: at fifty cache reads a cache
    # write past 250k tokens of context, the last 20 tool calls kept, at
    # least 50k tokens removed.
    assert context_edits() == []
    assert DEFAULT_CONFIG.context_edit_trigger_tokens == 250_000
    assert DEFAULT_CONFIG.context_edit_keep_tool_uses == 20
    assert DEFAULT_CONFIG.context_edit_clear_at_least == 50_000
    assert get_cache_write_ratio("claude-fable-5-2") == 50.0
    # A model without cache prices has no ratio; the settings then stand.
    assert get_cache_write_ratio("no-such-model") is None
    assert clear_thresholds(250_000, 50_000, None, None) == (250_000, 50_000)
    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
    model = build_model(anthropic_server.MODEL)
    assert model.clears_tool_uses_in_bulk()
    assert not model.keeps_reasoning_turns()
    # Bedrock keeps the thresholds of the Compaction settings.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "local")
    assert not build_model("bedrock/us.anthropic.claude-fable-5-2").clears_tool_uses_in_bulk()
    # So does the Anthropic API with the trigger at 0.
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_trigger_tokens", 0)
    assert not model.clears_tool_uses_in_bulk()
    # The keep-K policy: one edit, on a request whose thinking is on.
    monkeypatch.setattr(DEFAULT_CONFIG, "reasoning_keep_turns", 2)
    assert context_edits() == [THINKING_EDIT]
    assert context_edits(thinking=False) == []
    assert model.keeps_reasoning_turns()


@pytest.mark.parametrize(
    ("name", "binds"),
    [
        # The models the API documents as checking a thinking block against
        # the messages before it, on either route to them ...
        ("claude-fable-5-2", True),
        ("claude-fable-5-1", True),
        ("claude-opus-5-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-haiku-5-5", True),
        ("bedrock/us.anthropic.claude-fable-5-2", True),
        # ... and earlier ones, which sign a block over the block alone.
        ("claude-opus-5", False),
        ("claude-sonnet-4-5", False),
        ("claude-3-5-sonnet-20241022", False),
    ],
)
def test_only_a_model_that_binds_its_thinking_drops_that_of_its_newest_reply(
    name: str, binds: bool
) -> None:
    model = AnthropicModel(name, api_key="local", bedrock=name.startswith("bedrock/"))
    reply = [
        {"type": "thinking", "thinking": "...", "signature": "s"},
        {"type": "redacted_thinking", "data": "d"},
        {"type": "tool_use", "id": "toolu_0", "name": "big_output", "input": {}},
    ]
    result = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_0", "content": "out"}]}
    model.conversation = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "I start."},
        {"role": "assistant", "content": list(reply)},
        result,
        {"role": "assistant", "content": list(reply)},
        result,
    ]
    # The old reply loses its thinking on every model, the newest one only
    # where a block is bound to the messages before it.
    assert model.drop_stale_reasoning() == (4 if binds else 2)
    assert model.conversation[1]["content"] == "I start."
    assert model.conversation[2]["content"] == reply[2:]
    assert model.conversation[4]["content"] == (reply[2:] if binds else reply)


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

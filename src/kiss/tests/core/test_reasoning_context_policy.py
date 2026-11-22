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
import hashlib
import json
import logging
import os
import re
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import summary
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.context_compaction import (
    ARGUMENT_STUB_PREFIX,
    COMPACTION_MIN_STEP_GAP,
    COMPACTION_START_TOKENS,
    STUB_PREFIX,
    clear_thresholds,
    payback_floor,
)
from kiss.core.kiss_agent import REASONING_TURN_NOTE, KISSAgent
from kiss.core.models.anthropic_model import (
    CONTEXT_MANAGEMENT_BETA,
    AnthropicModel,
    context_edits,
)
from kiss.core.models.model_info import MODEL_INFO, get_cache_discount, get_cache_write_ratio
from kiss.core.models.model_info import model as build_model
from kiss.core.models.openai_compatible_model2 import (
    OpenAICompatibleModel2,
    supports_reasoning_context,
)
from kiss.core.printer import Printer
from kiss.tests.agents.sorcar.local_model_server import MODEL as CHAT_MODEL
from kiss.tests.agents.sorcar.local_model_server import serve as serve_chat
from kiss.tests.agents.sorcar.local_model_server import tool_call_body
from kiss.tests.core import local_anthropic_provider as provider
from kiss.tests.core import local_anthropic_server as anthropic_server
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer
from kiss.tests.core.test_context_compaction import PASSBACK_MODEL, Write, big_output

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


def long_output() -> str:
    """Return a large tool output: 4,800 characters, which the stand-in counts as 1,920 tokens."""
    return "word " * 960


REASONED = {"name": "note", "thinking": "r" * 4_800, "output_tokens": 2_030}
"""A scripted reply that is 2,000 tokens of reasoning and a call with a short result."""
RECAP = "- Read the parser and its tests, then fixed the off-by-one in the tokenizer.\n" * 20
"""A ten-step recap as an agent writes it: 1,540 characters, the median of those recorded."""


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
    script = [{**anthropic_server.tool_use_message("long_output", {}, 0, f"toolu_{i}"),
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
            model_name=model, prompt_template="Call long_output repeatedly.",
            tools=[long_output, Write, note, summary], max_steps=steps + 5, max_budget=500.0,
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
    """Return the ``compaction`` events of a session, each with what the provider counted.

    ``floor`` is the context of the call after the clear and ``freed`` the
    tokens that left: what that call would have held, the context of the
    call before it grown by a step like the last one, less ``floor``.
    """
    context = {
        e["step"]: e["input_tokens"] + e["cache_read"] + e["cache_write"]
        for e in recorder.events if e["type"] == "llm_call"
    }
    return [
        {**e, "floor": context[e["step"]],
         "freed": 2 * context[e["step"] - 1] - context[e["step"] - 2] - context[e["step"]]}
        for e in recorder.events if e["type"] == "compaction"
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
    # by the calls and their results alone, under 200 tokens a step.
    _agent, requests, recorder = _session(monkeypatch, steps=70, model=model, reply=REASONED)
    clears = _clears(recorder)
    assert len(clears) >= 3
    for event in clears:
        assert event["results"] == event["arguments"] == 0
        assert event["reasoning_blocks"] >= 9 and event["dropped_tokens"] >= 10_000
        assert len(_blocks(requests[event["step"] - 1], "thinking")) <= 1
    steps = clears[-1]["step"] - clears[0]["step"]
    assert 0 < clears[-1]["floor"] - clears[0]["floor"] < 200 * steps
    assert _cache_rewrites(recorder) == ([e["step"] for e in clears], [])


def test_the_newest_reply_an_earlier_model_keeps_is_not_counted_as_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Haiku 4.5 keeps the thinking of its newest reply through a clear.  One
    # reply of 30,000 reasoning tokens among short ones: right after it
    # nothing else can go, so a clear would re-write the cache and leave the
    # context as large.  It is made one step later, when that reply is an old
    # one, and takes the reasoning with it.
    big = {"thinking": "r" * 75_000, "output_tokens": 30_030}
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


def test_a_clear_that_would_not_pay_for_its_cache_write_waits_until_it_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A session whose first 20 results cannot be stubbed (a Write's result is
    # protected), long outputs after them, and a minimum of a fifth of the
    # trigger, as in the default settings.  Past the trigger the old outputs
    # soon hold that minimum, but they are a sixth of the context: the clear
    # would write the other five sixths to the cache again, for more than
    # the later calls save by reading less.  It is skipped, step after step,
    # until the old outputs are the share that pays the write back.
    writes = {i: {"name": "Write", "arguments": {"text": "t"}} for i in range(20)}
    _agent, _requests, recorder = _session(
        monkeypatch, steps=80, clear_at_least=8_000, replies=writes
    )
    share = payback_floor(get_cache_discount(anthropic_server.MODEL))
    assert share == pytest.approx(0.438, abs=0.001)
    clears = _clears(recorder)
    # Two clears, where the minimum alone made five.
    assert len(clears) == 2
    for event in clears:
        # Each was judged to free the share, and the provider saw it leave.
        assert event["dropped_tokens"] >= share * event["context_tokens"]
        assert event["freed"] >= share * event["context_tokens"]
    # Until the first one the context grew to twice the trigger, and no call
    # before it or between the two wrote the cache again.
    assert clears[0]["context_tokens"] > 80_000
    assert _cache_rewrites(recorder) == ([e["step"] for e in clears], [])


def test_a_claude_models_text_is_sized_as_the_provider_counts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The agent sizes a text from its length where the provider has not
    # counted it yet: a step's tool output, shown in the footer of its
    # result, and what a clear would remove.  The stand-in counts as Claude
    # does, a token per 2.5 characters; at the 4 characters of the other
    # model families both sizes came out 37% short.  Thinking is off, so a
    # step is a call and its result and nothing else.
    _agent, requests, recorder = _session(
        monkeypatch, steps=30, replies={1: {"name": "note"}},
        model_config={"thinking": {"type": "disabled"}},
    )
    context = [e["input_tokens"] + e["cache_read"] + e["cache_write"]
               for e in recorder.events if e["type"] == "llm_call"]
    # What the long output of the first step added beyond the short one of the second.
    counted = (context[1] - context[0]) - (context[2] - context[1])
    footer = re.search(
        r"This step's tool output: ~([\d,]+) tokens",
        _blocks(requests[1], "tool_result")[-1]["content"],
    )
    assert footer is not None
    assert int(footer[1].replace(",", "")) == pytest.approx(counted, rel=0.05)
    first = _clears(recorder)[0]
    assert first["dropped_tokens"] == pytest.approx(first["freed"], rel=0.05)


def test_a_recap_survives_the_clear_that_stubs_the_steps_it_covers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every ten steps a Sorcar agent recaps its work in a call of its
    # ``summary`` tool.  Once a clear has stubbed the outputs of those steps
    # the recap is what the model still knows of them, so it stays whole; a
    # Write's content of the same length, which the file keeps, is stubbed.
    replies = {
        3: {"name": "summary", "arguments": {"description": RECAP}},
        4: {"name": "Write", "arguments": {"text": RECAP}},
    }
    _agent, requests, recorder = _session(monkeypatch, steps=30, replies=replies)
    first = _clears(recorder)[0]
    sent = {b["name"]: b["input"] for b in _blocks(requests[first["step"] - 1], "tool_use")}
    assert sent["summary"] == {"description": RECAP}
    assert sent["Write"]["text"].startswith(ARGUMENT_STUB_PREFIX)
    assert first["arguments"] == 1


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


def test_another_adapter_drops_what_claude_signed_and_nothing_else() -> None:
    # An edit made while another adapter holds the conversation takes Claude's
    # blocks and items, redacted ones too.  Another vendor's reasoning and an
    # item that is no message (the edit's planner reads past one) stay.
    model = build_model(CHAT_MODEL, model_config={"base_url": "http://localhost:9", "api_key": "x"})
    gemini = {"type": "reasoning", "format": "google-gemini-v1", "encrypted_content": "kept"}
    reply = [{"type": "redacted_thinking", "data": "gone"}, {"type": "text", "text": "I start."}]
    model.conversation = [
        "junk", {"role": "assistant", "content": reply}, gemini,
        {"type": "reasoning", "format": "anthropic-claude-v1", "signature": "gone"},
    ]
    assert model.drop_stale_reasoning() == 2
    assert model.conversation == ["junk", {"role": "assistant", "content": reply[1:]}, gemini]


CLAUDE = "claude-fable-5-2"
"""A model whose provider rejects a thinking block sent back after its conversation changed."""


class _Switch:
    """The ``set_model`` tool of a session: continues it on another model, as Sorcar's does."""

    def __init__(self, agent: KISSAgent, configs: dict[str, dict[str, Any]]) -> None:
        self.agent = agent
        self.configs = configs

    def set_model(self, model_name: str) -> str:
        """Continue the session on another model.

        Args:
            model_name: The model to continue on.
        """
        new = build_model(model_name, model_config=dict(self.configs[model_name]))
        new.initialize("")
        self.agent.switch_model(new, model_name, self.configs[model_name])
        return f"Model changed to {model_name}."


@pytest.mark.parametrize(("other", "removed"), [(CHAT_MODEL, "results"), (PASSBACK_MODEL, "turns")])
def test_a_compaction_on_another_model_takes_the_claude_thinking_it_invalidates(
    monkeypatch: pytest.MonkeyPatch, other: str, removed: str
) -> None:
    # The session that ended in three 400s on 2026-11-20: steps on Claude
    # Fable 5.2, a set_model to a model on the Chat Completions API, and one
    # back.  The context Claude left is past the other model's compaction
    # threshold, so before its first call the old outputs are stubbed (the
    # old turns removed whole, on a DeepSeek model).  That adapter leaves
    # Claude's thinking out of its requests only: every block was still in
    # the conversation, signed over messages the compaction had changed.
    monkeypatch.setattr(DEFAULT_CONFIG, "compaction_start_tokens", 20_000)
    script = [anthropic_server.tool_use_message("long_output", {}, 0, f"toolu_{i}")
              for i in range(24)]
    script.append(anthropic_server.tool_use_message(
        "set_model", {"model_name": other}, 0, "toolu_away"))
    script.append(anthropic_server.tool_use_message(
        "finish", {"result": "<p>done</p>"}, 0, "toolu_end"))
    back = [tool_call_body("set_model", {"model_name": CLAUDE}, 30_000)]
    recorder = _Recorder()
    agent = KISSAgent("round-trip-session")
    with (
        anthropic_server.serve(script, as_provider=True) as (url, requests),
        serve_chat(back) as (chat_url, chat_requests),
    ):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        switch = _Switch(agent, {other: {"base_url": chat_url, "api_key": "local"}, CLAUDE: {}})
        result = agent.run(
            model_name=CLAUDE, prompt_template="Call long_output repeatedly.",
            tools=[long_output, switch.set_model], max_steps=30, max_budget=500.0,
            printer=recorder,
        )
    assert "done" in result
    # One compaction, before the other model's one call ...
    (event,) = [e for e in recorder.events if e["type"] == "compaction"]
    assert event["step"] == 26 and len(chat_requests) == 1 and event[removed] > 0
    # ... which took the thinking of the 25 Claude replies (of the ten turns
    # a whole-turn removal keeps) with it.
    assert event["reasoning_blocks"] == (25 if removed == "results" else 10)
    # No request was answered 400: the one after the switch back sends no thinking.
    assert [provider.stale_thinking(r) for r in requests] == [None] * 26
    assert len(_blocks(requests[24], "thinking")) == 24 and not _blocks(requests[25], "thinking")


def _claude_on_responses(request: Request) -> Reply:
    """Answer as Claude behind a Responses endpoint, where its reasoning is bound too.

    A reply is a ``reasoning`` item, as OpenRouter returns Claude's, and a
    call of ``long_output``, after 14 results one of ``finish``.  The item is
    signed over the tools and the input items before it, reasoning left out,
    as the thinking of :func:`~kiss.tests.core.local_anthropic_provider.signatures`;
    a request that sends an item back after any of those changed gets the 400
    of the upstream as OpenRouter passes it on, inside its own.
    """
    body = request.body
    digest = hashlib.sha256(json.dumps(body.get("tools"), sort_keys=True).encode())
    for item in body["input"]:
        if item.get("type") != "reasoning":
            digest.update(json.dumps(item, sort_keys=True).encode())
        elif item["signature"] != digest.hexdigest():
            raw = {"type": "error", "error": {
                "type": "invalid_request_error",
                "message": "Invalid `signature` in `thinking` block. The block is bound to a "
                           "different conversation."}}
            error = {"message": "Provider returned error", "code": 400,
                     "metadata": {"raw": json.dumps(raw), "provider_name": "Google"}}
            return Reply(status=400, json_body={"error": error})
    done = sum(item.get("type") == "function_call_output" for item in body["input"])
    name, arguments = ("long_output", "{}") if done < 14 else ("finish", '{"result": "<p>ok</p>"}')
    output = [
        {"type": "reasoning", "id": f"rs_{done}", "summary": [], "format": "anthropic-claude-v1",
         "signature": digest.hexdigest()},
        {"type": "function_call", "id": f"fc_{done}", "call_id": f"call_{done}", "name": name,
         "arguments": arguments},
    ]
    return Reply(json_body={
        "id": f"resp_{done}", "object": "response", "created_at": 0, "model": body["model"],
        "output": output,
        "usage": {"input_tokens": 20_000, "output_tokens": 30, "total_tokens": 20_030},
    })


def _claude_session_on_responses(agent: KISSAgent) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Run *agent* against :func:`_claude_on_responses` until its ``finish``.

    Returns:
        The ``input`` of every request, and what the ``llm_call_hook`` was
        shown, in order: the ``call_id`` of a tool result, the ``role`` of
        any other message.
    """
    shown: list[str] = []

    def hook(messages: list[Any]) -> list[Any]:
        shown.extend(m.get("call_id", m.get("role")) for m in messages)
        return messages

    with ScriptedOpenAIServer(_claude_on_responses) as server:
        result = agent.run(
            model_name="openrouter/anthropic/claude-fable-5.2",
            prompt_template="Call long_output repeatedly.", tools=[long_output], max_steps=20,
            max_budget=500.0, llm_call_hook=hook, verbose=False,
            model_config={"use_responses_api": True, "base_url": server.base_url,
                          "api_key": "local"},
        )
        sent = [request.body["input"] for request in server.requests]
    assert "ok" in result
    return sent, shown


def test_claude_reasoning_on_the_responses_api_goes_with_the_outputs_a_compaction_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The sessions that ended in three 400s on 2026-11-16: Claude Fable 5.2
    # through OpenRouter's Responses API, whose signed reasoning items the
    # adapter sends back as they are.  A compaction stubs outputs that
    # precede them, and the upstream rejects the first item it finds stale.
    monkeypatch.setattr(DEFAULT_CONFIG, "compaction_start_tokens", 20_000)
    sent, shown = _claude_session_on_responses(KISSAgent("claude-on-responses"))
    # Every request was answered: none had to be sent again.
    assert len(sent) == 15
    reasoning = [sum(item.get("type") == "reasoning" for item in items) for items in sent]
    first = next(i for i, items in enumerate(sent)
                 if any(str(item.get("output")).startswith(STUB_PREFIX) for item in items))
    # Until the compaction a request sends back the reasoning of every reply
    # before it; the first one with a stubbed output sends none, and the
    # later ones that of the replies since.
    assert 0 < first < 14 and reasoning == [*range(first), *range(15 - first)]
    # The items removed had all been shown to the llm_call_hook, whose
    # boundary moved back with them: after the prompt it is shown every
    # result once.
    assert shown == ["user", *(f"call_{i}" for i in range(14))]


@pytest.mark.parametrize("name", [CLAUDE, "claude-fable-5-2-latest"])
def test_a_request_rejected_for_stale_thinking_is_sent_again_without_it(
    name: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A session of reasoning, where nothing else can be cleared, with four
    # requests answered 400.  The third for another reason: an error like
    # any other, and the thinking stays.  The sixth after an edit of an
    # earlier message that removes no reasoning (the agent's own leave none
    # behind; a hook makes this one): the provider names its remedy, so the
    # thinking goes and the next step sends the request again, at the price
    # of the prompt cache instead of the session.  The eighth and the ninth
    # in the same words whatever they hold: the first takes the thinking of
    # the one reply since, the second finds nothing left to remove and is
    # an error like any other again.  All of it holds under a name of the
    # user's own catalog that the adapter reads no version from, and so
    # takes for a model that keeps its newest thinking through an edit: the
    # rejection tells it otherwise.
    entry = copy.copy(MODEL_INFO[CLAUDE])
    entry.extended_thinking = entry.adaptive_thinking = True
    monkeypatch.setitem(MODEL_INFO, "claude-fable-5-2-latest", entry)
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_trigger_tokens", 1_000)
    monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_clear_at_least", 9_000)
    script = [{**anthropic_server.tool_use_message("note", {}, 0, f"toolu_{i}"), **REASONED}
              for i in range(14)]
    script[2] = {"error": "max_tokens: 1 is less than the minimum"}
    script[7] = script[8] = {"error": "The block is bound to a different conversation."}
    script.append(anthropic_server.tool_use_message(
        "finish", {"result": "<p>done</p>"}, 0, "toolu_end"))
    recorder = _Recorder()
    agent = KISSAgent("rejected-thinking-session")

    def edit(model: Any) -> None:
        if agent.step_count == 6:
            _blocks({"messages": model.conversation}, "tool_result")[0]["content"] = "edited"

    agent.pre_step_hook = edit
    with anthropic_server.serve(script, as_provider=True) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        with caplog.at_level(logging.WARNING, logger="kiss.core.kiss_agent"):
            result = agent.run(
                model_name=name, prompt_template="Call note repeatedly.", tools=[note],
                max_steps=20, max_budget=500.0, printer=recorder,
            )
    # Each 400 cost its step and no more: the rejected sixth request is
    # answered in the seventh step.
    assert "done" in result and len(requests) == 15
    assert recorder.steps("llm_call") == [1, 2, 4, 5, 7, 10, 11, 12, 13, 14, 15]
    # The sixth request held the thinking of four replies, the second signed
    # over the result the hook changed.  The seventh is that request without
    # any, the newest reply's included, and with nothing added: a "Failed to
    # get response" message follows the third and the ninth request only.
    assert "messages.3.content.0" in str(provider.stale_thinking(requests[5]))
    thinking = [len(_blocks(request, "thinking")) for request in requests]
    assert thinking == [0, 1, 2, 2, 3, 4, 0, 1, 0, 0, 1, 2, 3, 4, 0]
    failures = [json.dumps(r["messages"]).count("Failed to get response") for r in requests]
    assert failures == [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    # A warning for each removal, with the count and the provider's message,
    # which names the block.
    first, second = [r.getMessage() for r in caplog.records if r.name == "kiss.core.kiss_agent"]
    assert "4 reasoning blocks" in first and "messages.3.content.0" in first
    assert "1 reasoning blocks" in second
    # The agent counts the reasoning that is left, none after a removal.  So
    # the one clear of the session comes when the five replies since the
    # last one hold the minimum, and removes what it reports; counted as
    # before the first removal, the reply after it would have been cleared
    # as five.
    (clear,) = _clears(recorder)
    assert clear["step"] == 15 and clear["reasoning_blocks"] == 5
    assert 9_000 <= clear["freed"] == pytest.approx(clear["dropped_tokens"], rel=0.05)


def test_a_rejection_on_the_responses_api_takes_claudes_reasoning_items_the_same_way() -> None:
    # The same edit by a hook on the route of OpenRouter's Responses API,
    # before the sixth request: the upstream rejects the reasoning items
    # signed over the result as it was, and the seventh request is the sixth
    # without those of all five replies.  A message the user sends in
    # between follows it, and the llm_call_hook, whose boundary moved back
    # with the items removed, is shown that message.
    agent = KISSAgent("claude-on-responses")

    def edit(model: Any) -> None:
        if agent.step_count == 6:
            next(i for i in model.conversation if i.get("type") == "function_call_output")[
                "output"] = "edited"
        elif agent.step_count == 7:
            model.add_message_to_conversation("user", "User says: go on.")

    agent.pre_step_hook = edit
    sent, shown = _claude_session_on_responses(agent)
    reasoning = [sum(item.get("type") == "reasoning" for item in items) for items in sent]
    assert reasoning == [*range(6), *range(10)]
    *resent, _message = sent[6]
    assert resent == [item for item in sent[5] if item.get("type") != "reasoning"]
    calls = [f"call_{i}" for i in range(14)]
    assert shown == ["user", *calls[:5], "user", *calls[5:]]


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

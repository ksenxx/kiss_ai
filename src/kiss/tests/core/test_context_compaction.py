# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for batched tool-output compaction (WP2c of the cost levers).

The module-level tests exercise :func:`compact_tool_results` on the three
conversation shapes; the agent tests run a real :class:`KISSAgent` against
the scripted local model server and check when compaction fires, what it
leaves alone, and that the ``context_reset_hook`` is called.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.context_compaction import (
    COMPACTION_START_TOKENS,
    COMPACTION_STEP_TOKENS,
    STUB_PREFIX,
    apply_compaction,
    compact_tool_results,
    dropped_chars,
    make_stub,
    plan_compaction,
    should_compact,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

BIG = "x" * 3000
SMALL = "ok"


def _generic(n: int, name: str = "Bash", prefix: str = "c") -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = [{"role": "user", "content": "task"}]
    for i in range(n):
        conversation.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"{prefix}{i}", "type": "function",
                            "function": {"name": name, "arguments": "{}"}}],
        })
        conversation.append({"role": "tool", "tool_call_id": f"{prefix}{i}", "content": BIG})
    return conversation


def _anthropic(n: int, name: str = "Read") -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = [{"role": "user", "content": "task"}]
    for i in range(n):
        conversation.append({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": {}}],
        })
        content: Any = BIG if i % 2 == 0 else [{"type": "text", "text": BIG}, {"type": "image"}]
        conversation.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": content}],
        })
    return conversation


def _responses(n: int, name: str = "Bash") -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = [{"role": "user", "content": "task"}]
    for i in range(n):
        conversation.append({"type": "function_call", "call_id": f"r{i}", "name": name,
                             "arguments": "{}"})
        conversation.append({"type": "function_call_output", "call_id": f"r{i}", "output": BIG})
    return conversation


def _texts(conversation: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for m in conversation:
        if m.get("role") == "tool":
            out.append(m["content"])
        elif m.get("type") == "function_call_output":
            out.append(m["output"])
        elif m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    c = b["content"]
                    out.append(c if isinstance(c, str) else c[0]["text"])
    return out


@pytest.mark.parametrize("build", [_generic, _anthropic, _responses])
def test_compacts_only_old_large_results(build) -> None:
    conversation = build(25)
    before = len(conversation)
    # The 25th result follows the last assistant turn (unseen by the model):
    # 24 seen results minus the 20 newest leaves 4 candidates.
    assert compact_tool_results(conversation, keep_recent=20) == 4
    assert len(conversation) == before
    texts = _texts(conversation)
    assert all(t.startswith(STUB_PREFIX) for t in texts[:4])
    assert all(t == BIG for t in texts[4:])
    assert "3,000 chars" in texts[0] and texts[0].endswith("x" * 200)
    # Idempotent: stubs are never re-compacted.
    assert compact_tool_results(conversation, keep_recent=20) == 0


def test_small_and_protected_results_are_kept() -> None:
    conversation = _generic(3, name="Bash") + _generic(3, name="Write", prefix="w")[1:]
    conversation[2]["content"] = SMALL
    conversation.append({"role": "assistant", "content": "seen everything"})
    assert compact_tool_results(conversation, keep_recent=0) == 2
    texts = _texts(conversation)
    assert texts[0] == SMALL
    assert texts[1].startswith(STUB_PREFIX) and texts[2].startswith(STUB_PREFIX)
    assert texts[3:] == [BIG, BIG, BIG]  # Write results are protected


def test_nothing_to_do_and_unknown_messages() -> None:
    assert compact_tool_results([], keep_recent=20) == 0
    conversation: list[Any] = ["junk", {"role": "system", "content": BIG}, {"type": "reasoning"}]
    assert compact_tool_results(conversation, keep_recent=0) == 0
    assert make_stub("abc", preview_chars=2).endswith("\nab")


def test_plan_drop_and_gate() -> None:
    conversation = _generic(10)
    # 10 results, the last one unseen (after the last assistant turn), 4 kept.
    plan = plan_compaction(conversation, keep_recent=4)
    assert len(plan) == 5
    # Nothing was edited by planning.
    assert all(t == BIG for t in _texts(conversation))
    expected_drop = 5 * (len(BIG) - len(make_stub(BIG)))
    assert dropped_chars(plan) == expected_drop
    assert dropped_chars([]) == 0
    assert apply_compaction(plan) == 5
    assert sum(t.startswith(STUB_PREFIX) for t in _texts(conversation)) == 5
    # Gate: the drop must be >= 25 % of the context ...
    assert should_compact(100_000, 25_000, None) is True
    assert should_compact(100_000, 24_999, None) is False
    # ... and the session must be below 80 % of its hand-off limit.
    assert should_compact(200_000, 60_000, 350_000) is True
    assert should_compact(280_000, 84_000, 350_000) is False
    assert should_compact(280_000, 84_000, None) is True


def big_output() -> str:
    """Return a large tool output (a stand-in for a file read or test log)."""
    return "line\n" * 1600


def Write(text: str) -> str:  # noqa: N802
    """Pretend to write a file and return a large confirmation (a protected tool name).

    Args:
        text: Ignored.
    """
    return "written " + "y" * 3000


def _run_agent(base_url: str, script_len: int, hook_calls: list[int]) -> KISSAgent:
    agent = KISSAgent("compaction-test")

    def on_reset() -> None:
        hook_calls.append(agent.step_count)

    agent.context_reset_hook = on_reset
    agent.run(
        model_name=MODEL,
        prompt_template="Call big_output repeatedly.",
        tools=[big_output, Write],
        max_steps=script_len + 5,
        max_budget=50.0,
        verbose=False,
        model_config={"base_url": base_url, "api_key": "local"},
    )
    return agent


def _stubbed(agent: KISSAgent) -> list[bool]:
    return [
        str(m.get("content", "")).startswith(STUB_PREFIX)
        for m in agent.model.conversation if m.get("role") == "tool"
    ]


def test_agent_compaction_skipped_while_the_drop_is_under_a_quarter() -> None:
    # Step 7 reports 100k context with 7 results seen and 6 kept: the one
    # candidate (8k chars ≈ 2k tokens) is far below 25 % of the context, so
    # the compaction is skipped, the hook is not called and the threshold
    # stays put so the plan is re-evaluated on every later step.
    script = [tool_call_body("big_output", {}, 5000 * i) for i in range(1, 7)]
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    with serve(script) as (url, requests):
        agent = _run_agent(url, len(script), hook_calls)
    assert hook_calls == []
    assert len(requests) == len(script)
    assert agent._next_compaction_at == COMPACTION_START_TOKENS
    assert not any(_stubbed(agent))


def test_agent_compaction_skipped_near_the_handoff_limit() -> None:
    # 50 big outputs would drop ~84k tokens, more than 25 % of the reported
    # 290k context, but 290k is past 80 % of the 350k hand-off limit
    # (70 % of the 500k window): too few steps remain to amortise the miss.
    script = [tool_call_body("big_output", {}, 3000) for _ in range(50)]
    script.append(tool_call_body("big_output", {}, 290_000))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    with serve(script) as (url, _requests):
        agent = _run_agent(url, len(script), hook_calls)
    assert hook_calls == []
    assert agent._next_compaction_at == COMPACTION_START_TOKENS
    assert not any(_stubbed(agent))


def test_agent_compaction_replaces_old_outputs_but_not_writes() -> None:
    # 30 big outputs; step 31 reports 100k context, so compaction runs before
    # step 32: 31 results exist, the 31st not yet seen by the model, the
    # newest 6 seen ones kept, the Write protected, so results 0..23 minus
    # index 5 (23 x ~7.6k chars ≈ 44k tokens) go, well over 25 % of the context.
    script = [tool_call_body("big_output", {}, 3000) for _ in range(30)]
    script[5] = tool_call_body("Write", {"text": "w"}, 3000)
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    with serve(script) as (url, requests):
        agent = _run_agent(url, len(script), hook_calls)
    assert hook_calls == [32]
    # The last reported context (100k prompt + 100 completion tokens) + 100k.
    assert agent._next_compaction_at == 100_100 + COMPACTION_STEP_TOKENS
    tool_msgs = [m for m in agent.model.conversation if m.get("role") == "tool"]
    assert len(tool_msgs) == 31  # the finish result never enters the conversation
    for i, m in enumerate(tool_msgs):
        content = str(m["content"])
        if i < 24 and i != 5:
            assert content.startswith(STUB_PREFIX), i
        else:
            assert not content.startswith(STUB_PREFIX), i
    # The finish request carried the compacted conversation.
    last_request = requests[-1]
    sent = [m for m in last_request["messages"] if m.get("role") == "tool"]
    assert sent[0]["content"].startswith(STUB_PREFIX)
    assert sent[5]["content"].startswith("written ")
    # The agent's own trajectory keeps the full text.
    assert any("line\nline\n" in str(m.get("content", "")) for m in agent.messages)


def test_compaction_disabled_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "tool_output_compaction", False)
    script = [tool_call_body("big_output", {}, 3000) for _ in range(22)]
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    with serve(script) as (url, _requests):
        agent = _run_agent(url, len(script), hook_calls)
    assert hook_calls == []
    assert not any(
        str(m.get("content", "")).startswith(STUB_PREFIX)
        for m in agent.model.conversation if m.get("role") == "tool"
    )


def test_results_after_the_last_assistant_turn_are_never_compacted() -> None:
    """One turn issuing 25 calls: its 25 fresh results are unseen and stay whole."""
    conversation: list[dict[str, Any]] = _generic(3)
    turn: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}
    conversation.append(turn)
    for i in range(25):
        turn["tool_calls"].append({"id": f"n{i}", "type": "function",
                                   "function": {"name": "Bash", "arguments": "{}"}})
        conversation.append({"role": "tool", "tool_call_id": f"n{i}", "content": BIG})
    # keep_recent=0 makes every SEEN result a candidate: only the 3 old ones go.
    assert compact_tool_results(conversation, keep_recent=0) == 3
    texts = _texts(conversation)
    assert all(t.startswith(STUB_PREFIX) for t in texts[:3])
    assert all(t == BIG for t in texts[3:])
    assert "do not repeat a command that has side effects" in texts[0]

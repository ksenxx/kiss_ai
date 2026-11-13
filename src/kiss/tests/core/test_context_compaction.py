# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for batched tool-output compaction (WP2c of the cost levers).

The module-level tests exercise :func:`plan_compaction` + :func:`apply_compaction` on the three
conversation shapes; the agent tests run a real :class:`KISSAgent` against
the scripted local model server and check when compaction fires, what it
leaves alone, and that the ``context_reset_hook`` is called.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.context_compaction import (
    ARGUMENT_PREVIEW_CHARS,
    ARGUMENT_STUB_PREFIX,
    COMPACTION_MIN_STEP_GAP,
    COMPACTION_START_TOKENS,
    COMPACTION_STEP_TOKENS,
    PAYBACK_STEPS,
    STUB_PREFIX,
    TURN_REMOVAL_NOTE,
    CompactionTable,
    apply_compaction,
    apply_turn_removal,
    compaction_target_tokens,
    dropped_chars,
    expand_stubs,
    make_argument_stub,
    make_stub,
    payback_floor,
    plan_compaction,
    plan_turn_removal,
    should_compact,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import get_cache_discount
from kiss.core.printer import Printer
from kiss.core.tool_verdict import ALLOW, Verdict
from kiss.core.utils import finish
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.core import local_anthropic_server as anthropic_server

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
    assert apply_compaction(plan_compaction(conversation, keep_recent=20)) == 4
    assert len(conversation) == before
    texts = _texts(conversation)
    assert all(t.startswith(STUB_PREFIX) for t in texts[:4])
    assert all(t == BIG for t in texts[4:])
    assert "3,000 chars" in texts[0] and texts[0].endswith("x" * 200)
    # Idempotent: stubs are never re-compacted.
    assert apply_compaction(plan_compaction(conversation, keep_recent=20)) == 0


def test_small_and_protected_results_are_kept() -> None:
    conversation = _generic(3, name="Bash") + _generic(3, name="Write", prefix="w")[1:]
    conversation[2]["content"] = SMALL
    conversation.append({"role": "assistant", "content": "seen everything"})
    assert apply_compaction(plan_compaction(conversation, keep_recent=0)) == 2
    texts = _texts(conversation)
    assert texts[0] == SMALL
    assert texts[1].startswith(STUB_PREFIX) and texts[2].startswith(STUB_PREFIX)
    assert texts[3:] == [BIG, BIG, BIG]  # Write results are protected


def test_nothing_to_do_and_unknown_messages() -> None:
    assert apply_compaction(plan_compaction([], keep_recent=20)) == 0
    conversation: list[Any] = ["junk", {"role": "system", "content": BIG}, {"type": "reasoning"}]
    assert apply_compaction(plan_compaction(conversation, keep_recent=0)) == 0
    assert make_stub("abc").endswith("\nabc")


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


WRITTEN: list[str] = []


def Write(text: str) -> str:  # noqa: N802
    """Pretend to write a file and return a large confirmation (a protected tool name).

    Args:
        text: Recorded in :data:`WRITTEN`, otherwise ignored.
    """
    WRITTEN.append(text)
    return "written " + "y" * 3000


class _ArgumentsSeen:
    """A ``tool_call_hook`` (the instance) or ``tool_result_hook`` (:meth:`result`)
    recording each ``text`` argument it is given."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, name: str, args: dict[str, Any]) -> Verdict:
        self.result(name, args, "", False)
        return ALLOW

    def result(self, name: str, args: dict[str, Any], response: str, refused: bool) -> None:
        if "text" in args:
            self.texts.append(str(args["text"]))


def _run_agent(
    base_url: str,
    script_len: int,
    hook_calls: list[int],
    compaction: bool = True,
    tool_call_hook: _ArgumentsSeen | None = None,
    tool_result_hook: _ArgumentsSeen | None = None,
) -> KISSAgent:
    agent = KISSAgent("compaction-test")

    def on_reset() -> None:
        hook_calls.append(agent.step_count)

    agent.context_reset_hook = on_reset
    if tool_result_hook is not None:
        agent.tool_result_hook = tool_result_hook.result
    agent.run(
        model_name=MODEL,
        prompt_template="Call big_output repeatedly.",
        # The scripted ``finish`` has the structured contract of ``finish_body``.
        tools=[big_output, Write, finish],
        max_steps=script_len + 5,
        max_budget=50.0,
        verbose=False,
        model_config={"base_url": base_url, "api_key": "local"},
        compaction=compaction,
        tool_call_hook=tool_call_hook,
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


def bigger_output() -> str:
    """Return a 20k-character tool output."""
    return "line\n" * 4000


def test_second_compaction_waits_for_the_step_gap() -> None:
    # The first compaction fires before step 32 (as above).  From step 32
    # on every response reports 210k context, past the 150k steady-state
    # target (30 % of the 500k window), so every step is a candidate; the
    # step gap spaces the compactions 10 steps apart (before steps 42, 52
    # and 62), each dropping the results that aged out meanwhile (11 then
    # 10 outputs, 37k-50k tokens: over the 10 % floor above the target,
    # under the 25 % floor that applies below it).
    script = [tool_call_body("big_output", {}, 3000) for _ in range(30)]
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.extend(tool_call_body("bigger_output", {}, 210_000) for _ in range(35))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    agent = KISSAgent("compaction-gap-test")
    agent.context_reset_hook = lambda: hook_calls.append(agent.step_count)
    with serve(script) as (url, _requests):
        agent.run(
            model_name=MODEL,
            prompt_template="Call the outputs repeatedly.",
            tools=[big_output, bigger_output, finish],
            max_steps=len(script) + 5,
            max_budget=50.0,
            verbose=False,
            model_config={"base_url": url, "api_key": "local"},
        )
    assert hook_calls == [32 + i * COMPACTION_MIN_STEP_GAP for i in range(4)]
    stubbed = _stubbed(agent)
    # Steps 1..24 went in the first batch, 25..34, 35..44 and 45..54 in the
    # next three; the newest six seen results (55..60) and the unseen 61st
    # stay whole.
    assert all(stubbed[:54])
    assert not any(stubbed[54:])


def test_compaction_disabled_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "tool_output_compaction", False)
    script = [tool_call_body("big_output", {}, 3000) for _ in range(22)]
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    with serve(script) as (url, _requests):
        agent = _run_agent(url, len(script), hook_calls)
    assert hook_calls == []
    assert not any(_stubbed(agent))
    # The run's own option was on: the global setting alone turned it off.
    assert agent.compaction is True


def test_compaction_disabled_by_run_option() -> None:
    """``run(compaction=False)`` leaves every output whole while the global setting is on.

    The script is the one ``test_agent_compaction_replaces_old_outputs_but_not_writes``
    compacts at step 32, so the option is the only difference.
    """
    assert DEFAULT_CONFIG.tool_output_compaction is True
    script = [tool_call_body("big_output", {}, 3000) for _ in range(30)]
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    with serve(script) as (url, requests):
        agent = _run_agent(url, len(script), hook_calls, compaction=False)
    assert agent.compaction is False
    assert hook_calls == []
    assert agent._next_compaction_at == COMPACTION_START_TOKENS
    assert len(_stubbed(agent)) == 31
    assert not any(_stubbed(agent))
    # The finish request still carried every output in full (a stub keeps 200 chars).
    sent = [m for m in requests[-1]["messages"] if m.get("role") == "tool"]
    assert len(sent) == 31
    assert all(big_output() in str(m["content"]) for m in sent)
    # The option is per run: the next run of the same agent compacts again.
    with serve(script) as (url, _requests):
        agent.run(
            model_name=MODEL,
            prompt_template="Call big_output repeatedly.",
            tools=[big_output, Write, finish],
            max_steps=len(script) + 5,
            max_budget=50.0,
            verbose=False,
            model_config={"base_url": url, "api_key": "local"},
        )
    assert agent.compaction is True
    assert hook_calls == [32]
    assert any(_stubbed(agent))


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
    assert apply_compaction(plan_compaction(conversation, keep_recent=0)) == 3
    texts = _texts(conversation)
    assert all(t.startswith(STUB_PREFIX) for t in texts[:3])
    assert all(t == BIG for t in texts[3:])
    assert "do not repeat a command that has side effects" in texts[0]


def test_keep_recent_counts_tool_results_not_their_text_parts() -> None:
    """A result made of blocks is one result, kept or compacted whole with its call.

    An Anthropic result with a picture is a list of parts, and each text
    part is compacted on its own.  Counted as a result each, the two texts
    of a result filled two places of the kept window: ``keep_recent=6``
    kept the three newest of these eight results and stubbed five.  A
    result with no text at all (the ninth) still takes no place in it.
    """
    conversation: list[dict[str, Any]] = [{"role": "user", "content": "task"}]
    for i in range(8):
        conversation.append(_write_call("anthropic", f"t{i}", "Bash", {"command": BIG}))
        parts = [{"type": "text", "text": BIG}, {"type": "image"}, {"type": "text", "text": BIG}]
        conversation.append(_result("anthropic", f"t{i}", parts))
    conversation.append(_write_call("anthropic", "shot", "Bash", {"command": "ls"}))
    conversation.append(_result("anthropic", "shot", [{"type": "image"}]))
    conversation.append({"role": "assistant", "content": "seen everything"})
    plan = plan_compaction(conversation, keep_recent=6)
    assert [r.kind for r in plan] == ["result"] * 4 + ["argument"] * 2
    assert apply_compaction(plan) == 6
    stubbed = [
        [part["text"].startswith(STUB_PREFIX) for part in m["content"][0]["content"][::2]]
        for m in conversation[2:17:2]
    ]
    assert stubbed == [[True, True]] * 2 + [[False, False]] * 6
    commands = [_arguments(m)["command"] for m in conversation[1:17:2]]
    assert [c.startswith(ARGUMENT_STUB_PREFIX) for c in commands] == [True] * 2 + [False] * 6


FILE_TEXT = "def f():\n    return 1\n" * 100  # 2,300 characters


def _write_call(shape: str, call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """One assistant turn calling *name* with *args* in the given provider *shape*."""
    if shape == "anthropic":
        return {"role": "assistant",
                "content": [{"type": "tool_use", "id": call_id, "name": name, "input": args}]}
    if shape == "responses":
        return {"type": "function_call", "call_id": call_id, "name": name,
                "arguments": json.dumps(args)}
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def _result(shape: str, call_id: str, text: str | list[dict[str, Any]]) -> dict[str, Any]:
    if shape == "anthropic":
        return {"role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call_id, "content": text}]}
    if shape == "responses":
        return {"type": "function_call_output", "call_id": call_id, "output": text}
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _arguments(message: dict[str, Any]) -> dict[str, Any]:
    """The arguments of the one tool call in *message*, whatever the shape."""
    if message.get("type") == "function_call":
        return dict(json.loads(message["arguments"]))
    if message.get("tool_calls"):
        return dict(json.loads(message["tool_calls"][0]["function"]["arguments"]))
    return dict(message["content"][0]["input"])


@pytest.mark.parametrize("shape", ["generic", "anthropic", "responses"])
def test_old_write_and_edit_arguments_are_stubbed_and_point_at_the_file(shape: str) -> None:
    """The content of old Writes/Edits goes; the stub says which file and that Read shows it."""
    conversation: list[dict[str, Any]] = [{"role": "user", "content": "task"}]
    calls: list[tuple[str, str, dict[str, Any]]] = [
        ("w1", "Write", {"file_path": "/app/a.py", "content": FILE_TEXT}),
        ("e1", "Edit", {"file_path": "/app/a.py", "old_string": FILE_TEXT[:600],
                        "new_string": "x" * 700}),
        ("b1", "Bash", {"command": "cat <<'EOF' > /app/b.txt\n" + FILE_TEXT + "EOF",
                        "description": "make b"}),
        ("w2", "Write", {"file_path": "/app/c.py", "content": FILE_TEXT}),
        ("f1", "finish", {"success": True, "summary_in_html": FILE_TEXT}),
        ("w3", "Write", {"file_path": "/app/a.py", "content": FILE_TEXT}),
    ]
    for call_id, name, args in calls:
        conversation.append(_write_call(shape, call_id, name, args))
        conversation.append(_result(shape, call_id, "ok"))
    conversation.append(_write_call(shape, "last", "Bash", {"command": "ls"}))
    conversation.append(_result(shape, "last", "ok"))
    # Results are all short ("ok"), so only arguments are planned; the newest
    # result (keep_recent=1) keeps its call's arguments.
    plan = plan_compaction(conversation, keep_recent=1)
    assert {r.kind for r in plan} == {"argument"}
    # Two Write contents, a command and two Edit strings go, minus the stubs.
    assert dropped_chars(plan) > 3 * len(FILE_TEXT) - 5 * 300
    assert apply_compaction(plan) == 4
    turns = [m for m in conversation if m.get("role") == "assistant"
             or m.get("type") == "function_call"]
    w1, e1, b1, w2, f1, w3, last = (_arguments(m) for m in turns)
    # The first Write of a.py was superseded by the Edit and the later Write.
    assert w1["file_path"] == "/app/a.py"
    assert w1["content"].startswith(ARGUMENT_STUB_PREFIX)
    assert "Write.content to /app/a.py; the file was changed again later" in w1["content"]
    assert "Read /app/a.py to see it as it is now" in w1["content"]
    assert w1["content"].endswith(FILE_TEXT[:ARGUMENT_PREVIEW_CHARS])
    # Both strings of the Edit are long enough to stub; the file path stays.
    assert e1["old_string"].startswith(ARGUMENT_STUB_PREFIX)
    assert e1["new_string"].startswith(ARGUMENT_STUB_PREFIX)
    assert e1["file_path"] == "/app/a.py"
    # A long shell command is stubbed with its tool and argument named; the
    # short description is left alone.
    command_chars = len(str(calls[2][2]["command"]))
    assert b1["command"].startswith(ARGUMENT_STUB_PREFIX + " #")
    assert f": {command_chars:,} chars of Bash.command;" in b1["command"]
    assert b1["description"] == "make b"
    # The last Write of c.py is the file's current content.
    assert "this call made the file's current content" in w2["content"]
    # finish is never touched; the newest call (kept window) neither.
    assert f1["summary_in_html"] == FILE_TEXT
    assert w3["content"] == FILE_TEXT
    assert last == {"command": "ls"}
    # A second plan finds nothing more to do.
    assert plan_compaction(conversation, keep_recent=1) == []


def test_json_arguments_keep_non_ascii_text_as_the_model_wrote_it() -> None:
    """A compacted call's JSON arguments are not rewritten with ASCII escapes.

    Escaped, every CJK character of the stub's preview and of the untouched
    ``file_path`` took six characters: the call got longer than it was and
    the planned drop was overstated.  A lone surrogate keeps its escape, the
    one spelling of it a UTF-8 request body can carry.
    """
    raw = '{"file_path": "/tmp/日本語.md", "note": "lone \\ud83d", "content": "' + "漢" * 600 + '"}'
    function = {"name": "Write", "arguments": raw}
    conversation: list[dict[str, Any]] = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "w1", "type": "function", "function": function}]},
        _result("generic", "w1", "ok"),
        _write_call("generic", "last", "Bash", {"command": "ls"}),
        _result("generic", "last", "ok"),
    ]
    plan = plan_compaction(conversation, keep_recent=0)
    assert apply_compaction(plan) == 1
    after = function["arguments"]
    # What went is what was planned (the stub's one newline is two characters of JSON).
    assert len(raw) - len(after) == dropped_chars(plan) - 1 > 0
    assert '"/tmp/日本語.md"' in after and '"lone \\ud83d"' in after
    assert json.loads(after)["content"].endswith("漢" * ARGUMENT_PREVIEW_CHARS)
    assert after.encode("utf-8").decode("utf-8") == after


def test_stubs_are_references_the_table_expands() -> None:
    # One id per distinct text, whatever the kind of stub; re-planning a
    # skipped compaction (the same texts again) adds nothing to the table.
    table = CompactionTable()
    result_stub = make_stub(BIG, table=table)
    argument_stub = make_argument_stub("Bash", "command", BIG, "", False, table)
    other = make_argument_stub("Write", "content", FILE_TEXT, "/app/a.py", True, table)
    assert result_stub.startswith(f"{STUB_PREFIX} #1: {len(BIG):,} chars;")
    assert argument_stub.startswith(
        f"{ARGUMENT_STUB_PREFIX} #1: {len(BIG):,} chars of Bash.command;")
    assert other.startswith(f"{ARGUMENT_STUB_PREFIX} #2:")
    assert make_stub(BIG, table=table) == result_stub
    assert table.originals == {1: BIG, 2: FILE_TEXT}
    # A whole-argument stub (the model copied what it saw) is expanded, with a
    # note; a preview the model cut short is still the stub.
    args, notes, refusal = expand_stubs(
        {"command": argument_stub, "description": "run", "timeout_seconds": 5}, table)
    assert (args, refusal) == ({"command": BIG, "description": "run", "timeout_seconds": 5}, None)
    assert notes == ["argument 'command' (#1) restored from the compaction table"]
    args, notes, refusal = expand_stubs({"content": other[:-50]}, table)
    assert (args["content"], refusal) == (FILE_TEXT, None)
    # The whole stub as the model saw it in a result (a tool output stub).
    args, notes, refusal = expand_stubs({"content": result_stub}, table)
    assert (args["content"], notes) == (
        BIG, ["argument 'content' (#1) restored from the compaction table"])
    # Refused before the tool runs: a stub inside a longer text, a stub the
    # table does not hold, a stub followed by text that is not its preview,
    # and the older stub format without an id (its table is long gone).
    for bad, reason in [
        ("header\n" + argument_stub, "inside a longer text"),
        ("<p>The model saw this:</p>\n" + result_stub, "inside a longer text"),
        (other.replace("#2:", "#9:"), "does not hold"),
        (argument_stub + "\nmore that is not the original", "not its preview"),
        (f"{ARGUMENT_STUB_PREFIX}: 2,000 chars of Bash.command]\npartial", "does not hold"),
    ]:
        args, notes, refusal = expand_stubs({"content": bad, "file_path": "/x"}, table)
        assert refusal is not None and reason in refusal, bad[:40]
        assert (args, notes) == ({"content": bad, "file_path": "/x"}, [])
    # A stub is a whole head line.  Text that merely carries a stub's opening
    # words (a source line of the compaction module, a grep for them, prose
    # naming them) or a head that does not end its line is content: it passes
    # untouched, as the very dict the caller gave.
    for plain in [
        "ls",
        f'STUB_PREFIX = "{STUB_PREFIX}"\n',
        f"grep -n '{ARGUMENT_STUB_PREFIX}' src/kiss/core/context_compaction.py",
        f"<p>Each old result becomes a {STUB_PREFIX} #n: ...] stub.</p>",
        f"{STUB_PREFIX} #1: 3,000 chars",
        argument_stub.replace("]\n", "] EXTRA CONTENT\n", 1),
    ]:
        given = {"command": plain, "n": 3}
        args, notes, refusal = expand_stubs(given, table)
        assert args is given and (notes, refusal) == ([], None), plain[:40]
    # A plan fed the table records every original under the ids its stubs carry.
    conversation = _generic(3)
    plan = plan_compaction(conversation, keep_recent=0, table=table)
    assert apply_compaction(plan) == 2  # the newest result is not seen by the model yet
    assert all(text.startswith(f"{STUB_PREFIX} #1:") for text in _texts(conversation)[:2])
    assert table.originals[1] == BIG


def test_agent_expands_a_stub_the_model_passes_back_and_refuses_an_embedded_one() -> None:
    # 30 Writes of the same 10k-character text; step 31 reports 100k context,
    # so the compaction before step 32 stubs the arguments of the old Writes
    # (their results are protected): one text, one stub id.  The model then
    # passes that stub back as a Write argument: the tool runs on the
    # original text and says so; a stub inside a longer text is refused
    # before the tool runs; then the model finishes.
    text = "data\n" * 2000
    stub = make_argument_stub("Write", "text", text, "", False)
    script = [tool_call_body("Write", {"text": text}, 3000) for _ in range(30)]
    script.append(tool_call_body("Write", {"text": text}, COMPACTION_START_TOKENS))
    script.append(tool_call_body("Write", {"text": stub}, 100_000))
    script.append(tool_call_body("Write", {"text": "# header\n" + stub}, 100_000))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    call_hook, result_hook = _ArgumentsSeen(), _ArgumentsSeen()
    WRITTEN.clear()
    with serve(script) as (url, requests):
        agent = _run_agent(
            url, len(script), hook_calls, tool_call_hook=call_hook, tool_result_hook=result_hook
        )
    assert hook_calls == [32]
    assert agent._compaction_table.originals == {1: text}
    tool_msgs = [m for m in agent.model.conversation if m.get("role") == "tool"]
    expanded, refused = str(tool_msgs[-2]["content"]), str(tool_msgs[-1]["content"])
    assert expanded.startswith("written ")
    assert "[argument 'text' (#1) restored from the compaction table]" in expanded
    assert "carries a compaction stub inside a longer text" in refused
    assert not refused.startswith("written ")
    # The call hook, the tool and the result hook all see the restored
    # original, so a result is booked under the arguments that ran; the
    # refused call (no hook consulted, the tool not run) keeps what was sent.
    assert call_hook.texts == [text] * 32
    assert WRITTEN == [text] * 32
    assert result_hook.texts == [text] * 32 + ["# header\n" + stub]
    # The conversation's own record of the call keeps the stub as the model sent it.
    sent = [m for m in requests[-1]["messages"] if m.get("role") == "assistant"]
    assert json.loads(sent[-2]["tool_calls"][0]["function"]["arguments"]) == {"text": stub}


def test_agent_runs_tools_on_text_that_merely_names_the_stub_prefix() -> None:
    # A Write of a source line of the compaction module and a finish summary
    # that mentions the stubs are content, not stubs: the tool runs on the
    # text as sent and the finish is accepted at the first attempt.
    source_line = f'STUB_PREFIX = "{STUB_PREFIX}"\n'
    script = [
        tool_call_body("Write", {"text": source_line}, 3000),
        finish_body(f"<p>Old results are {STUB_PREFIX} #n: ...] stubs.</p>"),
    ]
    WRITTEN.clear()
    with serve(script) as (url, requests):
        agent = _run_agent(url, len(script), [])
    assert WRITTEN == [source_line] and len(requests) == len(script)
    tool_msgs = [m for m in agent.model.conversation if m.get("role") == "tool"]
    assert str(tool_msgs[0]["content"]).startswith("written ")


@pytest.mark.parametrize("path", [
    "/app/[id]/page.tsx", "/app/[...slug]/page.tsx", "/tmp/a b;c#1:(x)'y\".md",
    "/tmp/odd\nname].md",
])
def test_argument_stubs_round_trip_for_any_file_path(path: str) -> None:
    """A stub the generator makes for a file of any name is one its parser expands."""
    table = CompactionTable()
    shown = path.replace("\n", "\\n")  # the head line stays one line
    for name, key in [("Write", "content"), ("Edit", "old_string"), ("Edit", "new_string")]:
        stub = make_argument_stub(name, key, FILE_TEXT, path, name == "Edit", table)
        head, preview = stub.split("\n", 1)
        assert head.endswith("follow]") and preview == FILE_TEXT[:ARGUMENT_PREVIEW_CHARS]
        assert f"{name}.{key} to {shown};" in head and f"Read {shown} to see it" in head
        assert expand_stubs({key: stub, "file_path": path}, table) == (
            {key: FILE_TEXT, "file_path": path},
            [f"argument {key!r} (#1) restored from the compaction table"], None)


def test_malformed_arguments_are_skipped() -> None:
    conversation: list[dict[str, Any]] = [
        {"role": "user", "content": "task"},
        {"type": "function_call", "call_id": "r0", "name": "Write", "arguments": "{not json"},
        {"type": "function_call_output", "call_id": "r0", "output": "ok"},
        {"type": "function_call", "call_id": "r1", "name": "Bash", "arguments": "[1, 2]"},
        {"type": "function_call_output", "call_id": "r1", "output": "ok"},
        {"type": "function_call", "call_id": "r2", "name": "Bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "r2", "output": "ok"},
    ]
    assert plan_compaction(conversation, keep_recent=0) == []


DEEP = 60_000
"""Nesting no interpreter follows to the end: ``json.loads`` gives up on it (Python 3.13),
or ``json.dumps`` once the decoder accepted it (3.14)."""


def test_arguments_nested_too_deep_to_follow_are_left_alone() -> None:
    """Stored arguments can nest deeper than the interpreter follows.  ``json.loads`` then
    raises ``RecursionError``, not ``ValueError`` (Python 3.13); on 3.14 the decoder accepts
    them and ``json.dumps`` raises when the stubbed arguments are written back; ``str`` of a
    ``file_path`` nested that deep raises too.  One such call made every plan of the session
    raise.  Its arguments are opaque, like text that is not JSON: the plan leaves them alone
    and stubs the other calls, and a whole-turn removal counts nothing for an item it cannot
    measure.  (Which of the three guards a call reaches depends on the interpreter; the
    write-back guard is reached on 3.14 only, where the decoder follows deeper than the
    encoder.)"""
    nested: dict[str, Any] = {}
    for _ in range(DEEP):
        nested = {"a": nested}
    deep_json = '{"a":' * DEEP + "1" + "}" * DEEP
    long_text = "x" * 600
    conversation: list[dict[str, Any]] = [
        {"role": "user", "content": "task"},
        _write_call("generic", "plain", "Bash", {"command": long_text}),
        _result("generic", "plain", "ok"),
        {"type": "function_call", "call_id": "deep", "name": "Bash",
         "arguments": '{"payload": ' + deep_json + ', "command": "' + long_text + '"}'},
        _result("responses", "deep", "ok"),
        _write_call("anthropic", "path", "Write", {"file_path": nested, "content": long_text}),
        _result("anthropic", "path", "ok"),
        {"role": "assistant", "content": "Done."},
    ]
    plan = plan_compaction(conversation, keep_recent=0)
    assert [replacement.holder for replacement in plan] == [
        conversation[1]["tool_calls"][0]["function"]
    ]
    assert apply_compaction(plan) == 1
    assert _arguments(conversation[1])["command"].startswith(ARGUMENT_STUB_PREFIX)
    turns: list[dict[str, Any]] = [{"role": "user", "content": "task"}]
    for index in range(12):
        turns.append(_write_call("anthropic", f"t{index}", "Bash", {"payload": nested}))
        turns.append(_result("anthropic", f"t{index}", "ok"))
    removal = plan_turn_removal(turns)
    assert removal is not None and (removal.start, removal.stop, removal.turns) == (1, 5, 2)
    # The two results are measured; the two calls, too deep to measure, count nothing.
    assert removal.dropped == 2 * len(json.dumps(turns[2])) - len(removal.note)


def test_target_and_the_lower_floor_above_it() -> None:
    assert compaction_target_tokens(500_000, 100_000, 0.3) == 150_000
    # The start threshold is the floor of the target ...
    assert compaction_target_tokens(200_000, 100_000, 0.3) == 100_000
    # ... and an unknown window or fraction leaves the start threshold.
    assert compaction_target_tokens(None, 100_000, 0.3) == 100_000
    assert compaction_target_tokens(500_000, 100_000, 0.0) == 100_000
    # Below the target a quarter of the context must go; above it a tenth.
    assert should_compact(140_000, 20_000, None, 150_000) is False
    assert should_compact(160_000, 16_000, None, 150_000) is True
    assert should_compact(160_000, 15_999, None, 150_000) is False
    # The hand-off proximity rule still applies above the target.
    assert should_compact(300_000, 60_000, 350_000, 150_000) is False


def test_payback_floor_follows_the_cache_discount() -> None:
    """The floor is the share whose drop pays the miss back within PAYBACK_STEPS steps."""
    # No cache discount: a compaction costs no miss, so nothing is added to the floor.
    assert payback_floor(None) == 0.0
    assert payback_floor(0.0) == 0.0
    assert payback_floor(1.0) == 0.0
    assert payback_floor(1.5) == 0.0
    # r = 1/discount - 1; floor = r / (r + PAYBACK_STEPS).
    assert payback_floor(0.05) == pytest.approx(19 / (19 + PAYBACK_STEPS))
    assert payback_floor(0.1) == pytest.approx(9 / (9 + PAYBACK_STEPS))
    # The dearer the miss, the larger the drop a compaction needs ...
    assert payback_floor(0.025) > payback_floor(0.05) > payback_floor(0.1)
    # ... and above the target the floor is the larger of the two rules.
    assert should_compact(200_000, 20_000, None, 150_000) is True
    assert should_compact(200_000, 20_000, None, 150_000, cache_discount=0.1) is False
    assert should_compact(200_000, 31_000, None, 150_000, cache_discount=0.1) is True
    assert should_compact(200_000, 31_000, None, 150_000, cache_discount=0.05) is False
    assert should_compact(200_000, 56_000, None, 150_000, cache_discount=0.05) is True
    # Below the target the quarter rule already exceeds OpenAI's payback floor.
    assert should_compact(100_000, 25_000, None, 150_000, cache_discount=0.1) is True
    assert should_compact(100_000, 25_000, None, 150_000, cache_discount=0.05) is False
    # The catalogue knows the discount of a priced model and not of an unknown one.
    assert get_cache_discount("claude-opus-5-5") == pytest.approx(0.05)
    assert get_cache_discount("gpt-5.6-sol-xhigh") == pytest.approx(0.1)
    assert get_cache_discount("no-such-model-2026") is None


class _Recorder(Printer):
    """A printer that keeps every event it is given."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        self.events.append({"type": type, "content": content, **kwargs})

    def token_callback(self, token: str) -> None:
        pass


def test_agent_emits_a_compaction_event_with_the_tokens_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The applied compaction is published as a ``compaction`` event (proposal 10).

    A printer makes the agent stream, so the scripted Anthropic stand-in
    (which serves server-sent events) plays the model; its tool-use turns
    are the Anthropic shape, so a Write's content is compacted as an
    argument too.  The server-side context edits are off (the provider
    would otherwise do the clearing and the client-side compaction
    stays out of its way).
    """
    script = [anthropic_server.tool_use_message("big_output", {}, 3000, f"toolu_{i}")
              for i in range(30)]
    script[5] = anthropic_server.tool_use_message(
        "Write", {"file_path": "/tmp/x.txt", "content": "z" * 4000}, 3000, "toolu_5")
    script.append(anthropic_server.tool_use_message(
        "big_output", {}, COMPACTION_START_TOKENS, "toolu_30"))
    script.append(anthropic_server.tool_use_message(
        "finish", {"result": "<p>done</p>"}, 60_000, "toolu_31"))
    recorder = _Recorder()
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        monkeypatch.setattr(DEFAULT_CONFIG, "reasoning_keep_turns", 0)
        monkeypatch.setattr(DEFAULT_CONFIG, "context_edit_trigger_tokens", 0)
        agent = KISSAgent("compaction-event-test")
        agent.run(
            model_name=anthropic_server.MODEL, prompt_template="Call big_output repeatedly.",
            tools=[big_output, Write], max_steps=len(script) + 5, max_budget=50.0,
            printer=recorder,
        )
    assert not any("context_management" in r for r in requests)
    events = [e for e in recorder.events if e["type"] == "compaction"]
    assert len(events) == 1
    event = events[0]
    # Compacted before the 32nd model call (the step counter has advanced).
    assert event["step"] == 32
    assert event["context_tokens"] == 100_000 + 30
    # 24 old outputs less the protected Write's, plus that Write's content.
    assert event["results"] == 23
    assert event["arguments"] == 1
    assert event["target_tokens"] == 150_000
    # 23 outputs of 8,000 chars and a 4,000-char argument, less their stubs.
    assert 40_000 < event["dropped_tokens"] < 48_000
    write_turn = agent.model.conversation[11]
    assert write_turn["content"][0]["name"] == "Write"
    assert write_turn["content"][0]["input"]["content"].startswith(ARGUMENT_STUB_PREFIX)
    # The stop reason of every model call is in its llm_call event.
    calls = [e for e in recorder.events if e["type"] == "llm_call"]
    assert len(calls) == len(script)
    assert all(e["stop_reason"] == "tool_use" for e in calls)


# ---------------------------------------------------------------------------
# Whole-turn removal: the compaction of the families whose API wants every
# earlier turn's reasoning back (DeepSeek), where a turn goes out entire.
# ---------------------------------------------------------------------------

PASSBACK_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
"""A catalog name of the DeepSeek family; the local server answers for it."""


@pytest.mark.parametrize("build", [_generic, _anthropic, _responses])
def test_turn_removal_keeps_the_task_and_the_newest_turns(build) -> None:
    conversation = build(12)
    removal = plan_turn_removal(conversation)
    assert removal is not None
    assert (removal.start, removal.turns) == (1, 2)
    assert removal.dropped > 0
    kept = list(conversation[removal.stop:])
    assert apply_turn_removal(conversation, removal) == removal.stop - 1
    assert conversation[0] == {"role": "user", "content": "task"}
    assert conversation[1] == {"role": "user", "content": TURN_REMOVAL_NOTE.format(turns=2)}
    assert conversation[2:] == kept
    # Three more turns: the next removal absorbs the earlier note, so one
    # note states the cumulative count, and the newest ten turns stay.
    conversation.extend(build(3)[1:])
    again = plan_turn_removal(conversation)
    assert again is not None
    assert (again.start, again.turns) == (1, 5)
    apply_turn_removal(conversation, again)
    assert conversation[1] == {"role": "user", "content": TURN_REMOVAL_NOTE.format(turns=5)}
    assert sum(1 for m in conversation if m.get("role") == "user") == 2 + (
        10 if build is _anthropic else 0
    )
    assert plan_turn_removal(conversation) is None
    assert plan_turn_removal(build(2)) is None


def test_agent_removes_whole_old_turns_for_a_reasoning_passback_family() -> None:
    # The same 30 outputs as the stub test, on a DeepSeek-named model: at
    # the 100k context the oldest 21 of the 31 turns go out whole (their
    # calls and results together, no stubs), one user note stands where
    # they were, and the reset hook tells the harness the context changed.
    script = [tool_call_body("big_output", {}, 3000) for _ in range(30)]
    script.append(tool_call_body("big_output", {}, COMPACTION_START_TOKENS))
    script.append(finish_body("<p>done</p>", prompt_tokens=60_000))
    hook_calls: list[int] = []
    agent = KISSAgent("whole-turn-test")

    def on_reset() -> None:
        hook_calls.append(agent.step_count)

    agent.context_reset_hook = on_reset
    with serve(script) as (url, requests):
        agent.run(
            model_name=PASSBACK_MODEL, prompt_template="Call big_output repeatedly.",
            tools=[big_output, Write, finish], max_steps=len(script) + 5, max_budget=50.0,
            verbose=False, model_config={"base_url": url, "api_key": "local"},
        )
    assert agent.model.compacts_whole_turns()
    assert hook_calls == [32]
    conversation = agent.model.conversation
    note = conversation[1]
    assert note["role"] == "user"
    assert note["content"].startswith(TURN_REMOVAL_NOTE.format(turns=21))
    assistants = [m for m in conversation if m.get("role") == "assistant"]
    tool_msgs = [m for m in conversation if m.get("role") == "tool"]
    # The ten kept turns and the finish call (whose result never enters).
    assert (len(assistants), len(tool_msgs)) == (11, 10)
    assert not any(_stubbed(agent))
    # The finish request carried the shortened conversation, the note first.
    sent = requests[-1]["messages"]
    users = [m for m in sent if m.get("role") == "user"]
    assert users[1]["content"].startswith(TURN_REMOVAL_NOTE.format(turns=21))
    assert sum(1 for m in sent if m.get("role") == "assistant") == 10
    # The agent's own trajectory keeps every turn.
    assert sum(1 for m in agent.messages if m.get("role") == "model") == len(script)

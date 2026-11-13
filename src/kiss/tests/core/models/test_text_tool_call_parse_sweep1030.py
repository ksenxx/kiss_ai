# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the text-based tool-call parser survives hostile JSON.

The CLI-backed models (Claude Code, Codex) and the streaming filter that
hides tool-call JSON from the chat panel both run every model turn through
:func:`_parse_text_based_tool_calls`.  A ``name`` that is a list or an
object used to raise ``TypeError`` (unhashable dedup key), and JSON nested
deeper than the interpreter's recursion limit, which ``json.loads``
accepts, used to raise ``RecursionError`` in the recursive walk.  Either
error aborted the turn instead of being ignored as "not a tool call".
"""

from __future__ import annotations

from typing import Any

from kiss.core.context_compaction import (
    KEEP_RECENT_TOOL_RESULTS,
    apply_compaction,
    plan_compaction,
)
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model import _parse_text_based_tool_calls, _ToolCallFilteredStream

_DEPTH = 3000


def _deep(depth: int, leaf: str) -> str:
    """Return *leaf* wrapped in *depth* single-key objects."""
    return '{"a":' * depth + leaf + "}" * depth


def test_entries_whose_name_is_not_a_string_are_skipped() -> None:
    content = (
        '{"tool_calls": [{"name": ["Bash"]}, {"name": {"x": 1}}, {"name": 7},'
        ' {"name": "Bash", "arguments": {"command": "ls"}}]}'
    )
    calls = _parse_text_based_tool_calls(content)
    assert [(c["name"], c["arguments"]) for c in calls] == [("Bash", {"command": "ls"})]


def test_deeply_nested_prose_json_is_not_a_tool_call() -> None:
    assert _parse_text_based_tool_calls(_deep(_DEPTH, "1")) == []


def test_a_call_nested_deep_inside_a_wrapper_is_found() -> None:
    call = '{"tool_calls": [{"name": "Read", "arguments": {"file_path": "x"}}]}'
    calls = _parse_text_based_tool_calls(_deep(_DEPTH, call))
    assert [(c["name"], c["arguments"]) for c in calls] == [("Read", {"file_path": "x"})]


def test_a_call_with_deeply_nested_arguments_is_found() -> None:
    content = '{"tool_calls": [{"name": "Write", "arguments": ' + _deep(_DEPTH, "1") + "}]}"
    calls = _parse_text_based_tool_calls(content)
    assert [c["name"] for c in calls] == ["Write"]
    depth = 0
    node: Any = calls[0]["arguments"]
    while isinstance(node, dict):
        node = node["a"]
        depth += 1
    assert (depth, node) == (_DEPTH, 1)


def test_arguments_too_deep_to_serialize_do_not_raise() -> None:
    """``json.loads`` accepts nesting that ``json.dumps`` (the dedup key) cannot follow.

    Such an entry could not be recorded in the conversation either, so
    it is not a tool call; the exact depth limits depend on the C stack,
    so either outcome is accepted as long as nothing raises.
    """
    for depth in (40_000, 60_000, 80_000):
        content = '{"tool_calls": [{"name": "Write", "arguments": ' + _deep(depth, "1") + "}]}"
        assert [c["name"] for c in _parse_text_based_tool_calls(content)] in ([], ["Write"])


_HUGE_INT_OBJECT = '{"n": ' + "9" * 4301 + "}"
"""An object the decoder refuses with a plain ``ValueError`` (the integer digit limit)."""


def test_an_undecodable_object_before_a_call_does_not_abort_the_parse() -> None:
    """A decoder ``ValueError`` that is not a ``JSONDecodeError`` is a skipped candidate.

    Python's decoder raises a plain ``ValueError`` for an integer beyond
    its digit limit; model text is arbitrary, so such an object before a
    valid ``tool_calls`` block must not end the parse, the stream filter,
    the arguments fallback or the CLI event stream.
    """
    from kiss.core.models.model import _iter_jsonl, parse_tool_arguments

    call = '{"tool_calls": [{"name": "Read", "arguments": {}}]}'
    text = "Note " + _HUGE_INT_OBJECT + " then " + call
    assert [c["name"] for c in _parse_text_based_tool_calls(text)] == ["Read"]
    tokens: list[str] = []
    model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
    with _ToolCallFilteredStream(model):
        feed = model.token_callback
        assert feed is not None
        feed(text)
    assert "tool_calls" not in "".join(tokens)
    assert "".join(tokens).startswith("Note " + _HUGE_INT_OBJECT)
    assert parse_tool_arguments(_HUGE_INT_OBJECT) == {}
    assert list(_iter_jsonl([_HUGE_INT_OBJECT, '{"type": "ok"}'])) == [{"type": "ok"}]


_TOO_DEEP = "[" * 100_000
"""Text the decoder refuses with ``RecursionError``: nesting past its recursion limit."""


def _count(n: int) -> str:
    """Count to *n*."""
    return "ok"


def test_json_too_deep_for_the_decoder_is_undecodable_not_fatal() -> None:
    """Past its nesting limit ``json.loads`` raises ``RecursionError``, not ``ValueError``.

    The arguments fallback, the CLI event stream, the typing of an
    ``<invoke>`` argument and the transcript of a handed-off call caught
    only ``ValueError``, so such text ended the turn.
    """
    from kiss.core.models.model import _iter_jsonl, parse_tool_arguments

    assert parse_tool_arguments(_TOO_DEEP) == {}
    assert list(_iter_jsonl([_TOO_DEEP, '{"type": "ok"}'])) == [{"type": "ok"}]
    model = ClaudeCodeModel("cc/opus")
    calls = [{"id": "c1", "name": "_count", "arguments": {"n": _TOO_DEEP}}]
    model._decode_text_arguments(calls, {"_count": _count})
    assert calls[0]["arguments"] == {"n": _TOO_DEEP}
    handed_off = {"name": "Bash", "arguments": _TOO_DEEP}
    model.conversation = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": handed_off}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    assert f"[Assistant]: [Tool Call] Bash({_TOO_DEEP})" in model._build_prompt()


def test_json_too_deep_for_the_encoder_is_not_fatal_to_the_transcript() -> None:
    """``json.dumps`` gives up on nesting that ``json.loads`` accepts (Python 3.14).

    The transcript compares a handed-off call with the call its turn's
    text quotes, and writes a turn's JSON object back once a compaction
    stubbed a call in it: stored arguments, or a value next to the call,
    nested that deep raised ``RecursionError`` out of the prompt builder.
    Where the decoder refuses the nesting as well, nothing is decoded and
    the turns read the same.
    """
    deep = _deep(60_000, "1")
    stored = '{"payload": ' + deep + "}"
    quote = '{"tool_calls": [{"name": "tool", "arguments": {"payload": {}}}]}'
    handed_off = {"name": "tool", "arguments": stored}
    model = ClaudeCodeModel("cc/opus")
    model.conversation = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": quote,
            "tool_calls": [{"id": "c1", "type": "function", "function": handed_off}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    assert model._build_prompt() == (
        f"[User]: Go\n\n[Assistant]: {quote}\n[Tool Call] tool({stored})\n\n[Tool Result]: ok"
    )
    write = '{"name": "Write", "arguments": {"content": "' + "x" * 600 + '"}}'
    own = '{"tool_calls": [' + write + '], "note": ' + deep + "}"
    ls = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
    model.conversation = [{"role": "user", "content": "Go"}]
    for text in [own] + [ls] * (KEEP_RECENT_TOOL_RESULTS + 1):
        calls = _parse_text_based_tool_calls(text)
        model.add_message_to_conversation("assistant", text)
        model._replace_last_assistant_with_tool_calls(text, calls)
        model.add_function_results_to_conversation_and_return(
            [(call["name"], {"result": "ok"}) for call in calls]
        )
    apply_compaction(plan_compaction(model.conversation))
    assert model._build_prompt().startswith(f"[User]: Go\n\n[Assistant]: {own}\n\n")


def test_the_stream_filter_hides_a_deeply_wrapped_call() -> None:
    tokens: list[str] = []
    model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
    call = '{"tool_calls": [{"name": "Read", "arguments": {}}]}'
    with _ToolCallFilteredStream(model):
        feed = model.token_callback
        assert feed is not None
        feed("Reading. " + _deep(_DEPTH, call) + " done")
    shown = "".join(tokens)
    assert "tool_calls" not in shown
    assert shown.startswith("Reading. ")
    assert shown.endswith(" done")

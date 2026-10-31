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

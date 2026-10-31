# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a tool call whose brace a prose quote swallowed is still hidden.

The chat panel must never show a tool call's JSON.  When prose puts an
unmatched ``{"`` shortly before the call (``Use {"x {"tool_calls": ...}``),
the stream filter's candidate object read the call's own ``{`` as text of
a key string.  Once the candidate was ruled out it released everything it
held and resumed after the failing character, so the call's ``{`` was
never a candidate: the raw JSON streamed to the panel while the post-turn
parser, which tries every brace, still ran the call.
"""

from __future__ import annotations

import time

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model import _parse_text_based_tool_calls, _ToolCallFilteredStream

_CALL = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
_PRETTY_CALL = '{\n  "tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]\n}'
_BRACED_CALL = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "echo {x}"}}]}'
_PWD_CALL = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "pwd"}}]}'


def _stream(text: str, chunk: int) -> str:
    """Stream *text* through a real adapter's filter in *chunk*-sized tokens (0: one token)."""
    shown: list[str] = []
    model = ClaudeCodeModel("cc/opus", token_callback=shown.append)
    size = chunk or len(text)
    with _ToolCallFilteredStream(model):
        feed = model.token_callback
        assert feed is not None
        for start in range(0, len(text), size):
            feed(text[start : start + size])
    return "".join(shown)


@pytest.mark.parametrize("chunk", [0, 1, 3, 7])
@pytest.mark.parametrize(
    ("prefix", "call"),
    [
        ('Use {"x ', _CALL),
        ('Set {"a": "b ', _CALL),
        ('Try {"k": ["v ', _CALL),
        ('Use {"x ', _PRETTY_CALL),
        ('Odd {"a": "{"} then {"x {"y ', _CALL),
        ('Esc {"x \\', _CALL),
    ],
)
def test_a_call_inside_a_mistaken_string_is_hidden(prefix: str, call: str, chunk: int) -> None:
    text = prefix + call + " done"
    assert [c["name"] for c in _parse_text_based_tool_calls(text)] == ["Bash"]
    assert _stream(text, chunk) == prefix + " done"


@pytest.mark.parametrize("chunk", [0, 1, 3, 7])
@pytest.mark.parametrize(
    ("text", "commands", "shown"),
    [
        ('Note {"a": ' + _BRACED_CALL + " oops", ["echo {x}"], "Note oops"),
        ('Two {"a": ' + _CALL + ', "b": "x ' + _PWD_CALL + " done", ["ls", "pwd"], "Two  done"),
    ],
    ids=["brace-in-arguments", "second-call-swallowed"],
)
def test_a_call_complete_inside_a_rejected_candidate_is_hidden(
    text: str, commands: list[str], shown: str, chunk: int
) -> None:
    """A wrapper the prose never closes holds a whole call: the string brace inside it is no cut."""
    calls = _parse_text_based_tool_calls(text)
    assert [c["arguments"]["command"] for c in calls] == commands
    assert _stream(text, chunk) == shown


@pytest.mark.parametrize("chunk", [0, 1, 2, 3, 5])
@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ('Run:\n```json\n{"a": ' + _CALL + "\n```\nok", "Run:\nok"),
        ('```{"a": ' + _CALL + "``", ""),
    ],
    ids=["closing-fence", "partial-closing-fence"],
)
def test_a_dropped_fenced_wrapper_takes_its_closing_fence(
    text: str, shown: str, chunk: int
) -> None:
    """The opener goes with the call, so the closer must too, however the stream is chunked.

    A closer left behind opens a code block over the rest of the message.
    """
    assert [c["name"] for c in _parse_text_based_tool_calls(text)] == ["Bash"]
    assert _stream(text, chunk) == shown


@pytest.mark.parametrize("chunk", [0, 1, 5])
@pytest.mark.parametrize(
    "text",
    [
        'json like {"a": "{b", c} here',
        'print("{") and {"k": "v {"} ok',
        '{"x {"y {"z',
        '{"a": "{", "b": "{" oops',
    ],
)
def test_prose_with_braces_in_quotes_streams_unchanged(text: str, chunk: int) -> None:
    assert _parse_text_based_tool_calls(text) == []
    assert _stream(text, chunk) == text


@pytest.mark.parametrize(
    "text",
    [
        '{"{' * 50_000,
        '{"k{": ' * 20_000 + "x",
        '{"a{": [' + "1, " * 50_000 + "x",
        '{"' + "{" * 150_000,
        '{"a": "{", "b": {"c": "{", "d": [' * 5_000 + "x",
        '{"\\{' * 50_000,
        '{"a": {"b": "{"} ' * 20_000 + "x",
    ],
    ids=["brace-quote", "nested-keys", "long-array", "brace-run", "mixed", "escaped", "closed"],
)
def test_rescans_stay_linear(text: str) -> None:
    started = time.monotonic()
    shown = _stream(text, 0)
    assert time.monotonic() - started < 10.0
    assert shown == text

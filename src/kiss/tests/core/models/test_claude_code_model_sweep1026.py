# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Claude Code streamed ``tool_calls`` early stop (sweep 2026-10-26).

``_ObjectSpans`` tracked one string parity and toggled it only while a
candidate ``{`` was open, so a prose brace followed by a quote (``printf("{");``)
flipped the parity for the rest of the stream and the real ``tool_calls``
object after it was never reported.  The CLI then kept streaming, the
hallucinated continuation was kept in the content, and
``_parse_text_based_tool_calls`` executed the hallucinated call as well.
The scanner now mirrors ``_brace_matches`` (both parities, every quote).
"""

from __future__ import annotations

import json
import random

from kiss.core.models.claude_code_model import ClaudeCodeModel, _ObjectSpans
from kiss.core.models.model import _brace_matches, _parse_text_based_tool_calls

_REAL_CALL = json.dumps({"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]})
_HALLUCINATED_CALL = json.dumps(
    {"tool_calls": [{"name": "Read", "arguments": {"file_path": "/etc/passwd"}}]}
)
_HALLUCINATED_TAIL = "\n(no output)\nNow I will read the file.\n" + _HALLUCINATED_CALL


def _drop_token(token: str) -> None:
    pass


def _model() -> ClaudeCodeModel:
    return ClaudeCodeModel("cc/opus", model_config={}, token_callback=_drop_token)


def _delta_lines(deltas: list[str]) -> list[str]:
    """Stream *deltas* as ``content_block_delta`` events, then a ``result`` event."""
    lines = [json.dumps({"type": "content_block_start"})]
    for text in deltas:
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
        lines.append(json.dumps(event))
    lines.append(json.dumps({"type": "content_block_stop"}))
    lines.append(json.dumps({"type": "result", "result": "".join(deltas)}))
    return lines


def _stream(prefix: str) -> tuple[ClaudeCodeModel, str]:
    m = _model()
    content, _result = m._parse_stream_events(
        iter(_delta_lines([prefix, _REAL_CALL, _HALLUCINATED_TAIL])), stop_on_tool_calls=True
    )
    return m, content


class TestProseBraceBeforeQuote:
    """A prose ``{`` followed by a quote must not hide the tool_calls block after it."""

    def test_plain_prefix_stops_at_first_block(self) -> None:
        m, content = _stream("Listing first.\n")
        assert m._stopped_for_tool_calls is True
        assert content == "Listing first.\n" + _REAL_CALL
        assert [c["name"] for c in _parse_text_based_tool_calls(content)] == ["Bash"]

    def test_printf_brace_prefix_stops_at_first_block(self) -> None:
        m, content = _stream('In C you write printf("{");\n')
        assert m._stopped_for_tool_calls is True
        assert content == 'In C you write printf("{");\n' + _REAL_CALL
        assert [c["name"] for c in _parse_text_based_tool_calls(content)] == ["Bash"]

    def test_unclosed_quote_after_brace_stops_at_first_block(self) -> None:
        m, content = _stream('The block starts with {" and ends with a brace.\n')
        assert m._stopped_for_tool_calls is True
        assert content.endswith(_REAL_CALL)
        assert _HALLUCINATED_CALL not in content

    def test_brace_heavy_write_argument_is_still_one_call(self) -> None:
        body = 'function f() { return {"a": "{", "b": "}"}; }\n' * 50
        call = json.dumps(
            {
                "tool_calls": [
                    {"name": "Write", "arguments": {"file_path": "/tmp/x.js", "content": body}}
                ]
            }
        )
        m = _model()
        deltas = [call[i : i + 7] for i in range(0, len(call), 7)]
        content, _result = m._parse_stream_events(
            iter(_delta_lines(deltas + [_HALLUCINATED_TAIL])), stop_on_tool_calls=True
        )
        assert m._stopped_for_tool_calls is True
        assert content == call
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 1 and calls[0]["arguments"]["content"] == body


def _parses(text: str) -> bool:
    try:
        json.loads(text, strict=False)
    except json.JSONDecodeError:
        return False
    return True


class TestObjectSpansMatchesBraceMatches:
    """For any delta split, the spans ``feed`` reports that parse as JSON are exactly the
    pairings of ``_brace_matches`` that parse: no object can hide, none is invented.

    (A dropped prose candidate may shift the pairing of a span that does not parse;
    only the parser's ``_is_tool_calls_object`` check acts on a span, so that is moot.)
    """

    def test_fuzz_any_split_matches_whole_text_scan(self) -> None:
        rng = random.Random(1026)
        alphabet = '{}"\\[]:, a1\n'
        parsed_total = 0
        for _ in range(3000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            spans = _ObjectSpans()
            reported: set[tuple[int, int]] = set()
            pos = 0
            while pos < len(text):
                step = rng.randint(1, 5)
                reported.update(spans.feed(text[pos : pos + step]))
                pos += step
            paired = {(start, end) for start, (end, _q) in _brace_matches(text).items()}
            parsed = {(start, end) for start, end in paired if _parses(text[start:end])}
            parsed_total += len(parsed)
            assert parsed == {(s, e) for s, e in reported if _parses(text[s:e])}, text
        assert parsed_total > 100

    def test_prose_candidate_is_dropped_without_hiding_the_object(self) -> None:
        spans = _ObjectSpans()
        text = 'x = {a: 1}; y = {"b": {}}'
        reported = spans.feed(text)
        outer = text.index('{"b"')
        assert reported == [(outer + 6, len(text) - 1), (outer, len(text))]

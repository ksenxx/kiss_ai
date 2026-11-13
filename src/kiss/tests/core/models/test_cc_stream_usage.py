# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Tests for Claude Code stream usage accounting and the tool-call stop.

The ``claude`` CLI runs as a bare model (no native tools; see
``ClaudeCodeModel._build_cli_args``).  These tests pin the parts of the
stream-parsing contract that outlive the turn: a KISS ``tool_calls`` block
ends the turn at once and nothing after it is consumed, and the usage of
an interrupted or tool-stopped turn follows the Messages API's per-message
``message_start`` / cumulative ``message_delta`` counters.

Event shapes below match real ``claude --print --output-format stream-json
--verbose --include-partial-messages`` output (v2.1.229).
"""

import json
from pathlib import Path

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model_info import calculate_cost
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli


class _Recorder:
    """Collects (kind, text) pairs, tracking thinking start/end boundaries."""

    def __init__(self) -> None:
        self.tokens: list[tuple[str, str]] = []
        self.thinking_events: list[bool] = []
        self._in_thinking = False

    def token_cb(self, text: str) -> None:
        self.tokens.append(("thinking" if self._in_thinking else "text", text))

    def thinking_cb(self, is_start: bool) -> None:
        self._in_thinking = is_start
        self.thinking_events.append(is_start)

    def thinking_tokens(self) -> list[str]:
        return [t for kind, t in self.tokens if kind == "thinking"]

    def text_tokens(self) -> list[str]:
        return [t for kind, t in self.tokens if kind == "text"]


def _make_model(rec: _Recorder) -> ClaudeCodeModel:
    m = ClaudeCodeModel(
        "cc/haiku", token_callback=rec.token_cb, thinking_callback=rec.thinking_cb
    )
    m.initialize("test")
    return m


def _parse(m: ClaudeCodeModel, events: list[dict]) -> tuple[str, dict]:
    return m._parse_stream_events(iter(json.dumps(e) for e in events))

class TestNoDrainAfterKissToolCallStop:
    """Once a KISS tool_calls block ends the turn, nothing more is consumed."""

    def test_events_after_stop_are_not_consumed(self) -> None:
        """Events queued after the stop must never surface, and usage comes
        from the message_delta events seen before the stop."""
        rec = _Recorder()
        m = _make_model(rec)
        tc = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        events = [
            {"type": "stream_event", "event": {
                "type": "message_delta", "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 11, "output_tokens": 7}}},
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": tc}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": " rambling on"}}},
            # Everything below must be ignored — the turn already ended.
            {"type": "assistant", "message": {"id": "m9", "content": [
                {"type": "text", "text": "post-stop output"}]}},
            {"type": "result", "result": "SHOULD NOT BE USED",
             "usage": {"input_tokens": 999999, "output_tokens": 999999}},
        ]
        content, result_json = m._parse_stream_events(
            iter(json.dumps(e) for e in events), stop_on_tool_calls=True
        )
        assert content == tc
        assert m._stopped_for_tool_calls is True
        assert result_json == {
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "api_billed": False,
            "requests": 1,
            "request_usages": [{"input_tokens": 11, "output_tokens": 7}],
        }
        assert "post-stop output" not in "".join(rec.text_tokens())

def _se(event: dict) -> dict:
    return {"type": "stream_event", "event": event}


class TestPerMessageUsageAccounting:
    """Usage of an interrupted or tool-stopped turn follows the Messages API
    stream contract: ``message_start`` carries the input/cache counts and
    every ``message_delta`` snapshot is cumulative for its message
    (https://platform.claude.com/docs/en/build-with-claude/streaming)."""

    _START = {
        "input_tokens": 100,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 1000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 1,
    }

    def test_cumulative_deltas_replace_and_message_start_counts_survive(self) -> None:
        m = _make_model(_Recorder())
        events = [
            _se({"type": "message_start", "message": {"usage": dict(self._START)}}),
            _se({"type": "message_delta", "usage": {"output_tokens": 5}}),
            _se({"type": "message_delta", "usage": {"output_tokens": 10}}),
            _se({"type": "message_stop"}),
        ]
        _parse(m, events)
        response = m.take_partial_usage_response()
        assert response == {
            "usage": {**self._START, "output_tokens": 10},
            "api_billed": False,
            "requests": 1,
            "request_usages": [{**self._START, "output_tokens": 10}],
        }
        assert m.extract_input_output_token_counts_from_response(response) == (
            100, 10, 400, 1000, 0,
        )
        assert m.take_partial_usage_response() is None

    def test_messages_are_summed_and_an_unfinished_one_is_billed_once(self) -> None:
        m = _make_model(_Recorder())
        events = [
            _se({"type": "message_start", "message": {"usage": dict(self._START)}}),
            _se({"type": "message_delta", "usage": {
                "input_tokens": 100, "cache_read_input_tokens": 400,
                "cache_creation_input_tokens": 1000, "output_tokens": 20}}),
            _se({"type": "message_stop"}),
            # Second message cut short: no message_stop before the stream dies.
            _se({"type": "message_start", "message": {"usage": {
                "input_tokens": 3, "cache_read_input_tokens": 1500,
                "cache_creation_input_tokens": 0, "output_tokens": 1}}}),
            _se({"type": "message_delta", "usage": {"output_tokens": 4}}),
            _se({"type": "message_delta", "usage": {"output_tokens": 9}}),
        ]
        _parse(m, events)
        response = m.take_partial_usage_response()
        assert m.extract_input_output_token_counts_from_response(response) == (
            103, 29, 1900, 1000, 0,
        )
        assert m.take_partial_usage_response() is None

    def test_tool_stopped_turn_bills_the_message_cut_short(self) -> None:
        """The CLI is killed at the KISS tool_calls block, so the stream ends
        with or without the message's ``message_stop``; either way the open
        message is billed from ``message_start`` plus its last snapshot."""
        tc = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        head = [
            _se({"type": "message_start", "message": {"usage": dict(self._START)}}),
            _se({"type": "content_block_start", "content_block": {"type": "text", "text": ""}}),
            _se({"type": "content_block_delta", "delta": {"type": "text_delta", "text": tc}}),
            _se({"type": "message_delta", "usage": {"output_tokens": 42}}),
        ]
        for tail in ([], [_se({"type": "message_stop"})]):
            m = _make_model(_Recorder())
            content, result_json = m._parse_stream_events(
                iter(json.dumps(e) for e in head + tail), stop_on_tool_calls=True
            )
            assert content == tc
            assert m._stopped_for_tool_calls is True
            assert m.extract_input_output_token_counts_from_response(result_json) == (
                100, 42, 400, 1000, 0,
            ), tail


# Streams one complete message, then the line in KISS_TEST_RESULT_LINE.
_CLAUDE_STREAMS_ONE_MESSAGE = """
    import json
    import os
    import sys

    sys.stdin.read()
    for event in (
        {"type": "message_start", "message": {"model": "claude-haiku-4-5", "usage": {
            "input_tokens": 11, "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 30,
            "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 20},
            "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ):
        print(json.dumps({"type": "stream_event", "event": event}), flush=True)
    print(os.environ["KISS_TEST_RESULT_LINE"], flush=True)
"""


def _one_message_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result_line: str
) -> ClaudeCodeModel:
    """A ``cc/haiku`` model whose CLI streams one message, then *result_line*."""
    install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_STREAMS_ONE_MESSAGE)
    monkeypatch.setenv("KISS_TEST_RESULT_LINE", result_line)
    # An API key pays, so the turn has a price to get wrong.
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "local-only")
    model = ClaudeCodeModel("cc/haiku")
    model.initialize("hi")
    return model


@pytest.mark.parametrize(
    ("result_line", "output_tokens"),
    [
        ('{"type": "result", "result": "hello"}', 7),
        ('{"type": "result", "result": "hello", "usage": null}', 7),
        ('{"type": "result", "result": "hello", "usage": {}}', 7),
        ('{"type": "result", "result": "hello", "usage": {"output_tokens": 9}}', 9),
        (
            '{"type": "result", "result": "hello", "usage": {"output_tokens": 9, '
            '"input_tokens": null, "cache_creation": {"ephemeral_5m_input_tokens": 10}}}',
            9,
        ),
    ],
)
def test_turn_keeps_the_usage_its_result_does_not_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result_line: str, output_tokens: int
) -> None:
    """The CLI reported each message's usage as it streamed.  A turn whose
    terminal ``result`` has a ``usage`` that is missing, empty or partial is
    still billed what the stream reported: field by field, the result's own
    fields first, and once."""
    model = _one_message_model(tmp_path, monkeypatch, result_line)

    content, response = model.generate()

    counts = (11, output_tokens, 400, 10, 20)
    assert content == "hello"
    assert model.extract_input_output_token_counts_from_response(response) == counts
    assert model.extract_cost_from_response(response) == calculate_cost("claude-haiku-4-5", *counts)
    assert model.take_partial_usage_response() is None


def test_a_turn_without_its_result_is_refused_and_still_billed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI that exits 0 with no terminal ``result`` (or one whose line did
    not decode) has not finished its turn, as a stream of an API model
    without its terminal event has not.  What the stream reported is billed
    all the same, once."""
    model = _one_message_model(tmp_path, monkeypatch, "")

    with pytest.raises(KISSError, match="ended without a result event"):
        model.generate()

    response = model.take_partial_usage_response()
    counts = (11, 7, 400, 10, 20)
    assert [message["role"] for message in model.conversation] == ["user"]
    assert model.extract_input_output_token_counts_from_response(response) == counts
    assert model.extract_cost_from_response(response) == calculate_cost("claude-haiku-4-5", *counts)
    assert model.take_partial_usage_response() is None

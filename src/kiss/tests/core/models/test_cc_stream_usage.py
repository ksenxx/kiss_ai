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

from kiss.core.models.claude_code_model import ClaudeCodeModel


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
        assert result_json == {"usage": {"input_tokens": 11, "output_tokens": 7}}
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
        assert response == {"usage": {**self._START, "output_tokens": 10}}
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

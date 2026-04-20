"""Tests for cc/* model streaming and multiple-assistant-message handling.

Reproduces and verifies fixes for:
1. Blocking mode (no token_callback) not streaming — should always stream.
2. Multiple assistant messages being accumulated — should stop at second.
"""

import json

from kiss.core.models.claude_code_model import ClaudeCodeModel


class TestAlwaysStreaming:
    """Verify that generate() always uses streaming, even without a token_callback."""

    def test_build_cli_args_always_streaming(self) -> None:
        """CLI args should always include stream-json flags."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("test")
        args = m._build_cli_args()
        assert "--output-format" in args
        idx = args.index("--output-format")
        assert args[idx + 1] == "stream-json"
        assert "--verbose" in args
        assert "--include-partial-messages" in args

    def test_generate_without_callback_uses_streaming(self) -> None:
        """generate() should use streaming even when token_callback is None."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("test")
        assert m.token_callback is None
        # The model should not have a _generate_blocking method anymore
        assert not hasattr(m, "_generate_blocking")


class TestStopAtSecondAssistant:
    """Verify that _generate_streaming stops when a second assistant message appears."""

    def test_parse_single_assistant_event(self) -> None:
        """A single assistant event should be fully captured."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("test")

        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}},
            {"type": "result", "result": "Hello",
             "usage": {"input_tokens": 10, "output_tokens": 5}},
        ]

        content, result_json = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "Hello"
        assert result_json.get("result") == "Hello"

    def test_stop_at_second_assistant_event(self) -> None:
        """Content from the second assistant event should be excluded."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("test")

        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "First"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Second"}]}},
            {"type": "result", "result": "First\nSecond",
             "usage": {"input_tokens": 10, "output_tokens": 10}},
        ]

        content, result_json = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "First"
        assert "Second" not in content

    def test_token_callback_not_called_for_second_assistant(self) -> None:
        """Token callback should only be invoked for the first assistant message."""
        tokens: list[str] = []
        m = ClaudeCodeModel("cc/haiku", token_callback=tokens.append)
        m.initialize("test")

        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "A"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "B"}]}},
            {"type": "result", "result": "AB", "usage": {}},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "A"
        assert tokens == ["A"]

    def test_empty_stream(self) -> None:
        """An empty stream should return empty content."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("test")
        content, result_json = m._parse_stream_events(iter([]))
        assert content == ""
        assert result_json == {}

    def test_result_before_second_assistant_is_used(self) -> None:
        """If result comes before any second assistant, use result content."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("test")

        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi"}]}},
            {"type": "result", "result": "Hi there", "usage": {}},
        ]

        content, result_json = m._parse_stream_events(iter(json.dumps(e) for e in events))
        # result event overwrites content when there's only one assistant message
        assert content == "Hi there"

    def test_malformed_json_lines_skipped(self) -> None:
        """Non-JSON lines should be silently skipped."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("test")

        lines = [
            "not json",
            "",
            json.dumps({"type": "assistant", "message": {
                "content": [{"type": "text", "text": "OK"}]}}),
            "also not json",
            json.dumps({"type": "result", "result": "OK", "usage": {}}),
        ]

        content, _ = m._parse_stream_events(iter(lines))
        assert content == "OK"

    def test_multiple_text_blocks_in_one_assistant(self) -> None:
        """Multiple text blocks within a single assistant event should all be captured."""
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("test")

        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Part1"},
                {"type": "text", "text": "Part2"},
            ]}},
            {"type": "result", "result": "Part1Part2", "usage": {}},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "Part1Part2"

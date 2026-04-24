"""Tests for cc/* model streaming and multiple-assistant-message handling.

Reproduces and verifies fixes for:
1. Blocking mode (no token_callback) not streaming — should always stream.
2. Multiple assistant messages being accumulated — should stop at second.
3. Thinking blocks not emitted to the thoughts panel — should invoke thinking_callback.
4. generate_and_process_with_tools must relay thinking tokens to the printer.
"""

import json
import subprocess
from typing import Any

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


class TestThinkingBlocks:
    """Verify that thinking blocks trigger thinking_callback and stream tokens."""

    def test_thinking_block_in_assistant_event(self) -> None:
        """Thinking blocks in assistant events should trigger thinking_callback."""
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = ClaudeCodeModel(
            "cc/haiku",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.initialize("test")

        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "Let me reason..."},
                {"type": "text", "text": "Answer"},
            ]}},
            {"type": "result", "result": "Answer", "usage": {}},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "Answer"
        assert tokens == ["Let me reason...", "Answer"]
        assert thinking_events == [True, False]

    def test_thinking_via_content_block_events(self) -> None:
        """content_block_start/delta/stop events should trigger thinking_callback."""
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = ClaudeCodeModel(
            "cc/haiku",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.initialize("test")

        events = [
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "delta": {
                "type": "thinking_delta", "thinking": "Step 1..."}},
            {"type": "content_block_delta", "delta": {
                "type": "thinking_delta", "thinking": "Step 2..."}},
            {"type": "content_block_stop"},
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {
                "type": "text_delta", "text": "Result"}},
            {"type": "content_block_stop"},
            {"type": "result", "result": "Result", "usage": {}},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "Result"
        assert tokens == ["Step 1...", "Step 2...", "Result"]
        assert thinking_events == [True, False]

    def test_no_thinking_callback_does_not_crash(self) -> None:
        """Model without thinking_callback should handle thinking blocks gracefully."""
        tokens: list[str] = []
        m = ClaudeCodeModel("cc/haiku", token_callback=tokens.append)
        m.initialize("test")
        assert m.thinking_callback is None

        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "Hmm..."},
                {"type": "text", "text": "Done"},
            ]}},
            {"type": "result", "result": "Done", "usage": {}},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "Done"
        assert tokens == ["Hmm...", "Done"]

    def test_empty_thinking_block_no_callback(self) -> None:
        """Empty thinking blocks should not trigger callbacks."""
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = ClaudeCodeModel(
            "cc/haiku",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.initialize("test")

        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "Hello"},
            ]}},
            {"type": "result", "result": "Hello", "usage": {}},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "Hello"
        assert tokens == ["Hello"]
        assert thinking_events == []

    def test_thinking_only_no_text(self) -> None:
        """Assistant message with only a thinking block and no text."""
        tokens: list[str] = []
        thinking_events: list[bool] = []
        m = ClaudeCodeModel(
            "cc/haiku",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        m.initialize("test")

        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "Deep thought"},
            ]}},
            {"type": "result", "result": "", "usage": {}},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == ""
        assert tokens == ["Deep thought"]
        assert thinking_events == [True, False]

    def test_content_block_text_delta_accumulates(self) -> None:
        """text_delta events should accumulate into content."""
        tokens: list[str] = []
        m = ClaudeCodeModel("cc/haiku", token_callback=tokens.append)
        m.initialize("test")

        events = [
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "B"}},
            {"type": "content_block_stop"},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == "AB"
        assert tokens == ["A", "B"]

    def test_empty_content_block_deltas_skipped(self) -> None:
        """Empty deltas should not produce token callbacks."""
        tokens: list[str] = []
        m = ClaudeCodeModel("cc/haiku", token_callback=tokens.append)
        m.initialize("test")

        events = [
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": ""}},
            {"type": "content_block_stop"},
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": ""}},
            {"type": "content_block_stop"},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))
        assert content == ""
        assert tokens == []


class TestInterleavedPartialAndAssistant:
    """Reproduce the BUG where --include-partial-messages emits BOTH granular
    ``content_block_*`` events AND a final ``assistant`` event containing the
    same content, causing thinking callbacks to be fired twice.

    Real Claude CLI output (with ``--include-partial-messages``):
    - granular ``content_block_start`` / ``content_block_delta`` / ``content_block_stop``
      events stream tokens incrementally (wrapped in ``stream_event``),
    - then a final ``assistant`` event arrives with the SAME accumulated content.

    Without de-duplication the parser fires ``thinking_callback(True/False)``
    twice — the second call races through ``thinking_start → full text →
    thinking_end`` instantly, causing the UI to collapse the thinking panel
    into the "Thinking (click to expand)" bar, so the user never sees the
    streamed thinking tokens.
    """

    def test_interleaved_partial_and_final_assistant_no_duplicates(self) -> None:
        """Thinking callbacks must fire exactly once when both event kinds appear."""
        tokens: list[tuple[str, str]] = []
        thinking_events: list[bool] = []
        in_thinking = False

        def token_cb(text: str) -> None:
            tokens.append(("thinking" if in_thinking else "text", text))

        def thinking_cb(is_start: bool) -> None:
            nonlocal in_thinking
            in_thinking = is_start
            thinking_events.append(is_start)

        m = ClaudeCodeModel(
            "cc/haiku",
            token_callback=token_cb,
            thinking_callback=thinking_cb,
        )
        m.initialize("test")

        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "Let me "}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "think..."}}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hi"}}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "assistant", "message": {
                "id": "msg_abc",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Hi"},
                ]}},
            {"type": "result", "result": "Hi", "usage": {}},
        ]

        content, _ = m._parse_stream_events(iter(json.dumps(e) for e in events))

        assert thinking_events == [True, False], (
            f"Duplicate thinking boundaries: {thinking_events}"
        )
        thinking_tokens = [t for bt, t in tokens if bt == "thinking"]
        assert thinking_tokens == ["Let me ", "think..."], (
            f"Thinking tokens wrong or duplicated: {thinking_tokens}"
        )
        text_tokens = [t for bt, t in tokens if bt == "text"]
        assert text_tokens == ["Hi"], f"Text tokens duplicated: {text_tokens}"
        assert content == "Hi"

    def test_partial_assistant_snapshots_not_duplicated(self) -> None:
        """Multiple partial assistant events (same msg_id) must not replay content."""
        tokens: list[str] = []
        thinking_events: list[bool] = []
        in_thinking = False

        def token_cb(text: str) -> None:
            if in_thinking:
                tokens.append(text)

        def thinking_cb(is_start: bool) -> None:
            nonlocal in_thinking
            in_thinking = is_start
            thinking_events.append(is_start)

        m = ClaudeCodeModel(
            "cc/haiku",
            token_callback=token_cb,
            thinking_callback=thinking_cb,
        )
        m.initialize("test")

        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking"}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "abc"}}},
            {"type": "assistant", "message": {
                "id": "msg_X",
                "content": [{"type": "thinking", "thinking": "abc"}]}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "def"}}},
            {"type": "assistant", "message": {
                "id": "msg_X",
                "content": [{"type": "thinking", "thinking": "abcdef"}]}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "", "usage": {}},
        ]

        m._parse_stream_events(iter(json.dumps(e) for e in events))

        assert thinking_events == [True, False], thinking_events
        assert tokens == ["abc", "def"], tokens


class TestThinkingInToolMode:
    """Verify that generate_and_process_with_tools relays thinking tokens to the printer.

    The bug: generate_and_process_with_tools replaces token_callback with a
    buffer to capture tokens for tool-call JSON stripping.  But thinking
    tokens go to the buffer too, so the printer never receives them and the
    thoughts panel is empty.
    """

    def test_thinking_tokens_reach_printer_during_tool_generation(self) -> None:
        """Thinking tokens must reach the original callback, not just the buffer."""
        text_tokens: list[str] = []
        thinking_tokens: list[str] = []
        thinking_events: list[bool] = []

        in_thinking = False

        def token_cb(token: str) -> None:
            if in_thinking:
                thinking_tokens.append(token)
            else:
                text_tokens.append(token)

        def thinking_cb(is_start: bool) -> None:
            nonlocal in_thinking
            in_thinking = is_start
            thinking_events.append(is_start)

        m = ClaudeCodeModel(
            "cc/haiku",
            token_callback=token_cb,
            thinking_callback=thinking_cb,
        )
        m.initialize("test")

        events = [
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "delta": {
                "type": "thinking_delta", "thinking": "Let me think..."}},
            {"type": "content_block_stop"},
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {
                "type": "text_delta", "text": "The answer is 42"}},
            {"type": "content_block_stop"},
            {"type": "result", "result": "The answer is 42", "usage": {}},
        ]

        stream_data = "\n".join(json.dumps(e) for e in events) + "\n"

        original_popen = subprocess.Popen

        class FakePopen:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.returncode = 0
                self.stdin = _FakeStdin()
                self.stdout = _FakeStdout(stream_data)
                self.stderr = _FakeStdout("")

            def wait(self, timeout: float | None = None) -> int:
                return 0

        class _FakeStdin:
            def write(self, s: str) -> None:
                pass

            def close(self) -> None:
                pass

        class _FakeStdout:
            def __init__(self, data: str) -> None:
                self._lines = data.splitlines(keepends=True)
                self._pos = 0

            def __iter__(self) -> "_FakeStdout":
                return self

            def __next__(self) -> str:
                if self._pos >= len(self._lines):
                    raise StopIteration
                line = self._lines[self._pos]
                self._pos += 1
                return line

            def read(self) -> str:
                return "".join(self._lines[self._pos:])

        subprocess.Popen = FakePopen  # type: ignore[assignment,misc]
        try:
            function_calls, content, _ = m.generate_and_process_with_tools(
                {"dummy_tool": lambda: "ok"}
            )
        finally:
            subprocess.Popen = original_popen  # type: ignore[assignment,misc]

        assert thinking_events == [True, False], (
            f"Expected [True, False], got {thinking_events}"
        )
        assert thinking_tokens == ["Let me think..."], (
            f"Thinking tokens not delivered: {thinking_tokens}"
        )
        assert "42" in "".join(text_tokens)

    def test_thinking_callback_restored_after_tool_generation(self) -> None:
        """thinking_callback must be restored even if generate() raises."""
        thinking_events: list[bool] = []
        m = ClaudeCodeModel(
            "cc/haiku",
            token_callback=lambda t: None,
            thinking_callback=thinking_events.append,
        )
        m.initialize("test")
        original_thinking_cb = m.thinking_callback

        original_popen = subprocess.Popen

        class FailPopen:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise OSError("fake failure")

        subprocess.Popen = FailPopen  # type: ignore[assignment,misc]
        try:
            try:
                m.generate_and_process_with_tools({"dummy_tool": lambda: "ok"})
            except Exception:
                pass
        finally:
            subprocess.Popen = original_popen  # type: ignore[assignment,misc]

        assert m.token_callback is not None
        assert m.thinking_callback == original_thinking_cb

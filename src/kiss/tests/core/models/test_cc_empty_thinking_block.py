"""Integration test: cc/* model must NOT emit thinking UI events when the
thinking block contains no actual thinking text (only signature deltas).

Reproduces the bug: Claude opus sends ``content_block_start`` with
``type: "thinking"`` followed by ``signature_delta`` events (no
``thinking_delta``).  The parser emits ``thinking_start`` /
``thinking_end`` anyway, causing the browser UI to show an empty
collapsible "Thinking" bar with no content.

The fix: defer ``thinking_start`` until actual thinking content arrives.
If the block ends with only signature deltas, suppress both boundaries.
"""

import json

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.models.claude_code_model import ClaudeCodeModel


class TestEmptyThinkingBlockSuppressed:
    """Thinking blocks with no text content (signature-only) must not produce UI events."""

    def test_signature_only_thinking_block_no_ui_events(self) -> None:
        """A thinking block with only signature_delta must NOT produce
        thinking_start/thinking_delta/thinking_end events.
        """
        printer = BaseBrowserPrinter()
        printer.start_recording()

        model = ClaudeCodeModel(
            "cc/opus",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        model.initialize("test")

        # Real Claude opus output: thinking block with only signature_delta
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta",
                          "signature": "EuABClkIDBgCKk..."}}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "The answer is 42"}}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "The answer is 42", "usage": {}},
        ]

        content, _ = model._parse_stream_events(
            iter(json.dumps(e) for e in events)
        )

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]

        # No thinking events at all — the block had no readable content
        assert "thinking_start" not in types, (
            f"Empty thinking block should not emit thinking_start: {types}"
        )
        assert "thinking_delta" not in types, (
            f"Empty thinking block should not emit thinking_delta: {types}"
        )
        assert "thinking_end" not in types, (
            f"Empty thinking block should not emit thinking_end: {types}"
        )

        # Text content should still be delivered
        assert content == "The answer is 42"
        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        assert text_deltas

    def test_signature_only_thinking_block_raw_callbacks(self) -> None:
        """Raw callback test: no thinking_callback fires for signature-only blocks."""
        tokens: list[str] = []
        thinking_events: list[bool] = []

        model = ClaudeCodeModel(
            "cc/opus",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        model.initialize("test")

        events = [
            {"type": "content_block_start",
             "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta",
             "delta": {"type": "signature_delta", "signature": "abc123"}},
            {"type": "content_block_stop"},
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_stop"},
            {"type": "result", "result": "Hello", "usage": {}},
        ]

        content, _ = model._parse_stream_events(
            iter(json.dumps(e) for e in events)
        )

        assert content == "Hello"
        assert tokens == ["Hello"], f"Expected only text token, got {tokens}"
        assert thinking_events == [], (
            f"No thinking boundaries for empty block: {thinking_events}"
        )

    def test_assistant_event_empty_thinking_no_callbacks(self) -> None:
        """Empty thinking in assistant events already works — verify it still does."""
        thinking_events: list[bool] = []

        model = ClaudeCodeModel(
            "cc/opus",
            token_callback=lambda t: None,
            thinking_callback=thinking_events.append,
        )
        model.initialize("test")

        events = [
            {"type": "assistant", "message": {
                "id": "msg_1",
                "content": [
                    {"type": "thinking", "thinking": ""},
                    {"type": "text", "text": "Answer"},
                ]}},
            {"type": "result", "result": "Answer", "usage": {}},
        ]

        model._parse_stream_events(iter(json.dumps(e) for e in events))
        assert thinking_events == [], thinking_events


class TestRealThinkingBlockStillWorks:
    """Blocks with actual thinking_delta content must still produce full UI events."""

    def test_thinking_block_with_content_still_streams(self) -> None:
        """A thinking block with thinking_delta events must stream normally."""
        printer = BaseBrowserPrinter()
        printer.start_recording()

        model = ClaudeCodeModel(
            "cc/sonnet",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        model.initialize("test")

        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "Let me "}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "reason..."}}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""}}},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Answer"}}},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "Answer", "usage": {}},
        ]

        model._parse_stream_events(iter(json.dumps(e) for e in events))

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]

        assert types.count("thinking_start") == 1
        assert types.count("thinking_end") == 1
        thinking_deltas = [e for e in recorded if e["type"] == "thinking_delta"]
        full_thought = "".join(d["text"] for d in thinking_deltas)
        assert full_thought == "Let me reason..."

    def test_deferred_thinking_start_emitted_on_first_delta(self) -> None:
        """thinking_start must be emitted just before the first thinking_delta."""
        thinking_events: list[bool] = []
        tokens: list[str] = []

        model = ClaudeCodeModel(
            "cc/sonnet",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        model.initialize("test")

        events = [
            {"type": "content_block_start",
             "content_block": {"type": "thinking"}},
            # First real thinking content:
            {"type": "content_block_delta",
             "delta": {"type": "thinking_delta", "thinking": "Step 1"}},
            {"type": "content_block_delta",
             "delta": {"type": "thinking_delta", "thinking": "Step 2"}},
            {"type": "content_block_stop"},
            {"type": "result", "result": "", "usage": {}},
        ]

        model._parse_stream_events(iter(json.dumps(e) for e in events))

        assert thinking_events == [True, False]
        assert tokens == ["Step 1", "Step 2"]

    def test_signature_then_thinking_delta_still_works(self) -> None:
        """A block with signature_delta followed by thinking_delta should still show."""
        thinking_events: list[bool] = []
        tokens: list[str] = []

        model = ClaudeCodeModel(
            "cc/sonnet",
            token_callback=tokens.append,
            thinking_callback=thinking_events.append,
        )
        model.initialize("test")

        events = [
            {"type": "content_block_start",
             "content_block": {"type": "thinking"}},
            {"type": "content_block_delta",
             "delta": {"type": "signature_delta", "signature": "abc"}},
            {"type": "content_block_delta",
             "delta": {"type": "thinking_delta", "thinking": "Real thought"}},
            {"type": "content_block_stop"},
            {"type": "result", "result": "", "usage": {}},
        ]

        model._parse_stream_events(iter(json.dumps(e) for e in events))

        assert thinking_events == [True, False]
        assert tokens == ["Real thought"]


class TestToolModeWithEmptyThinking:
    """generate_and_process_with_tools must also suppress empty thinking blocks."""

    def test_tool_mode_suppresses_empty_thinking(self) -> None:
        """Tool mode with signature-only thinking should NOT emit thinking events."""
        import subprocess
        from typing import Any

        thinking_events: list[bool] = []
        text_tokens: list[str] = []
        in_thinking = False

        def token_cb(token: str) -> None:
            if in_thinking:
                pass  # thinking tokens
            else:
                text_tokens.append(token)

        def thinking_cb(is_start: bool) -> None:
            nonlocal in_thinking
            in_thinking = is_start
            thinking_events.append(is_start)

        model = ClaudeCodeModel(
            "cc/opus",
            token_callback=token_cb,
            thinking_callback=thinking_cb,
        )
        model.initialize("test")

        events = [
            {"type": "content_block_start",
             "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta",
             "delta": {"type": "signature_delta", "signature": "sig123"}},
            {"type": "content_block_stop"},
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": "result text"}},
            {"type": "content_block_stop"},
            {"type": "result", "result": "result text", "usage": {}},
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
            function_calls, content, _ = model.generate_and_process_with_tools(
                {"dummy_tool": lambda: "ok"}
            )
        finally:
            subprocess.Popen = original_popen  # type: ignore[assignment,misc]

        # No thinking events for signature-only block
        assert thinking_events == [], (
            f"Expected no thinking events, got {thinking_events}"
        )

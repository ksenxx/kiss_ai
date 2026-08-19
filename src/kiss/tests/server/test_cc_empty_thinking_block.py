# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.server.json_printer import JsonPrinter
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401


class TestEmptyThinkingBlockSuppressed:
    """Thinking blocks with no text content (signature-only) must not produce UI events."""

    def test_signature_only_thinking_block_no_ui_events(self) -> None:
        """A thinking block with only signature_delta must NOT produce
        thinking_start/thinking_delta/thinking_end events.
        """
        printer = JsonPrinter()
        printer._thread_local.task_id = "test"
        printer.start_recording()

        model = ClaudeCodeModel(
            "cc/opus",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        model.initialize("test")

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

        assert "thinking_start" not in types, (
            f"Empty thinking block should not emit thinking_start: {types}"
        )
        assert "thinking_delta" not in types, (
            f"Empty thinking block should not emit thinking_delta: {types}"
        )
        assert "thinking_end" not in types, (
            f"Empty thinking block should not emit thinking_end: {types}"
        )

        assert content == "The answer is 42"
        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        assert text_deltas


class TestRealThinkingBlockStillWorks:
    """Blocks with actual thinking_delta content must still produce full UI events."""

    def test_thinking_block_with_content_still_streams(self) -> None:
        """A thinking block with thinking_delta events must stream normally."""
        printer = JsonPrinter()
        printer._thread_local.task_id = "test"
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

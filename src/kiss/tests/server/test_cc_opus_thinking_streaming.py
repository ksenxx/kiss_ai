# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: cc/opus text tokens must stream incrementally during tool mode.

Reproduces the bug where ``generate_and_process_with_tools`` buffers ALL
text tokens and dumps them to the UI as a single event after generation
completes, instead of streaming them incrementally during generation.

The root cause: ``generate_and_process_with_tools`` replaces the
``token_callback`` with a wrapper that buffers non-thinking tokens,
then re-sends the entire buffer as one callback after ``generate()``
returns.  This defeats streaming for the text panel.
"""

import json
import subprocess
from typing import Any

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.server.json_printer import JsonPrinter
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_cc_opus_thinking_streaming import (  # noqa: F401
    _build_fake_popen_class,
    _FakeStdin,
    _FakeStdout,
    _make_stream_events_with_text_deltas,
)


def _make_stream_events_with_tool_call() -> list[dict[str, Any]]:
    """Build cc/opus stream with text + tool call JSON."""
    tool_call_json = json.dumps(
        {"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}
    )
    return [
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "I'll list "},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "the files.\n\n"},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": tool_call_json},
            },
        },
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {
            "type": "result",
            "result": "I'll list the files.\n\n" + tool_call_json,
            "usage": {"input_tokens": 10, "output_tokens": 15},
        },
    ]


def _make_stream_events_with_thinking_and_text() -> list[dict[str, Any]]:
    """Build cc/sonnet-like stream: thinking_delta events + text_delta events."""
    return [
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "I need to "},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "think about this."},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": "sig"},
            },
        },
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "The answer "},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "is 42."},
            },
        },
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {
            "type": "result",
            "result": "The answer is 42.",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        },
    ]


class TestCCOpusTextStreamingInToolMode:
    """Text tokens must stream incrementally during generate_and_process_with_tools.

    This tests the BUG: generate_and_process_with_tools buffers all text
    tokens and emits them as a single text_delta event after generation
    completes, instead of streaming them incrementally.
    """

    def test_text_tokens_stream_incrementally_during_tool_generation(self) -> None:
        """Each text_delta from the CLI must produce a separate text_delta broadcast.

        BUG: The wrapper callback buffers non-thinking tokens and sends them
        as ONE text_delta after generate() returns. This test fails if text
        tokens are batched into fewer events than the CLI produced.

        We check the raw recording (before coalescing) because stop_recording
        merges consecutive text_delta events for storage efficiency.
        """
        printer = JsonPrinter()
        printer._thread_local.task_id = "test-task-1"
        printer.start_recording()

        m = ClaudeCodeModel(
            "cc/opus",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        m.initialize("test")

        events = _make_stream_events_with_text_deltas()
        fake_popen = _build_fake_popen_class(events)

        original_popen = subprocess.Popen
        subprocess.Popen = fake_popen  # type: ignore[assignment,misc]
        try:
            m.generate_and_process_with_tools({"dummy": lambda: "ok"})
        finally:
            subprocess.Popen = original_popen  # type: ignore[assignment,misc]

        key = printer._task_key()
        with printer._lock:
            raw = list(printer._recordings.get(key, []))
        raw_text_deltas = [e for e in raw if e.get("type") == "text_delta"]

        assert len(raw_text_deltas) >= 3, (
            f"Expected ≥3 raw text_delta events (one per CLI chunk), "
            f"got {len(raw_text_deltas)}: "
            f"{[d.get('text', '')[:40] for d in raw_text_deltas]}"
        )

        recorded = printer.stop_recording()
        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        full_text = "".join(d["text"] for d in text_deltas)
        assert full_text == "Let me analyze this problem."

    def test_thinking_tokens_still_stream_in_tool_mode(self) -> None:
        """Thinking tokens must stream incrementally during tool generation."""
        printer = JsonPrinter()
        printer._thread_local.task_id = "test-task-2"
        printer.start_recording()

        m = ClaudeCodeModel(
            "cc/sonnet",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        m.initialize("test")

        events = _make_stream_events_with_thinking_and_text()
        fake_popen = _build_fake_popen_class(events)

        original_popen = subprocess.Popen
        subprocess.Popen = fake_popen  # type: ignore[assignment,misc]
        try:
            m.generate_and_process_with_tools({"dummy": lambda: "ok"})
        finally:
            subprocess.Popen = original_popen  # type: ignore[assignment,misc]

        key = printer._task_key()
        with printer._lock:
            raw = list(printer._recordings.get(key, []))
        raw_types = [e.get("type") for e in raw]

        assert "thinking_start" in raw_types, f"No thinking_start: {raw_types}"
        assert "thinking_end" in raw_types, f"No thinking_end: {raw_types}"

        raw_thinking = [e for e in raw if e.get("type") == "thinking_delta"]
        assert len(raw_thinking) >= 2, (
            f"Expected ≥2 raw thinking_delta events, got {len(raw_thinking)}"
        )

        raw_text = [e for e in raw if e.get("type") == "text_delta"]
        assert len(raw_text) >= 2, (
            f"Expected ≥2 raw text_delta events, got {len(raw_text)}: "
            f"{[d.get('text', '')[:40] for d in raw_text]}"
        )

        recorded = printer.stop_recording()
        thinking_deltas = [e for e in recorded if e["type"] == "thinking_delta"]
        full_thinking = "".join(d["text"] for d in thinking_deltas)
        assert full_thinking == "I need to think about this."
        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        full_text = "".join(d["text"] for d in text_deltas)
        assert full_text == "The answer is 42."

    def test_text_tokens_stream_when_tool_calls_present(self) -> None:
        """Text tokens must stream even when the response contains tool calls."""
        printer = JsonPrinter()
        printer._thread_local.task_id = "test-task-3"
        printer.start_recording()

        m = ClaudeCodeModel(
            "cc/opus",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        m.initialize("test")

        events = _make_stream_events_with_tool_call()
        fake_popen = _build_fake_popen_class(events)

        original_popen = subprocess.Popen
        subprocess.Popen = fake_popen  # type: ignore[assignment,misc]
        try:
            function_calls, content, _ = m.generate_and_process_with_tools(
                {"Bash": lambda command: "ok"}
            )
        finally:
            subprocess.Popen = original_popen  # type: ignore[assignment,misc]

        key = printer._task_key()
        with printer._lock:
            raw = list(printer._recordings.get(key, []))
        raw_text_deltas = [e for e in raw if e.get("type") == "text_delta"]

        assert len(raw_text_deltas) >= 2, (
            f"Expected ≥2 raw text_delta events, got {len(raw_text_deltas)}: "
            f"{[d.get('text', '')[:40] for d in raw_text_deltas]}"
        )

        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"

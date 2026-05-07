"""Regression test: tool calls must be extracted from pre-result content.

The ``result`` event from the Claude CLI can replace accumulated text
content (from ``content_block_delta`` events) with a different or empty
string.  When the post-result content yields no tool calls, the parser
must fall back to the pre-result accumulated content which may still
contain the ``tool_calls`` JSON that was streamed via text deltas.

Without this fallback, the agent stalls: the model produces tool calls
via streaming text deltas, the ``result`` event replaces the content
(e.g. with an empty string or a stripped version), and the framework
never sees the tool calls.
"""

import json
import subprocess
from typing import Any

from kiss.core.models.claude_code_model import ClaudeCodeModel


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
        rest = "".join(self._lines[self._pos:])
        self._pos = len(self._lines)
        return rest


def _build_fake_popen_class(events: list[dict[str, Any]]) -> type:
    """Build a FakePopen class that returns the given events as stdout."""
    stream_data = "\n".join(json.dumps(e) for e in events) + "\n"

    class FakePopen:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.returncode = 0
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout(stream_data)
            self.stderr = _FakeStdout("")

        def wait(self, timeout: float | None = None) -> int:
            return 0

    return FakePopen


def _run_with_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, ClaudeCodeModel]:
    """Run generate_and_process_with_tools with fake CLI events."""
    m = ClaudeCodeModel("cc/opus")
    m.initialize("test")
    fake_popen = _build_fake_popen_class(events)
    original_popen = subprocess.Popen
    subprocess.Popen = fake_popen  # type: ignore[assignment,misc]
    try:
        function_calls, content, _ = m.generate_and_process_with_tools(
            {"go_to_url": lambda url: "ok", "Bash": lambda command: "ok"}
        )
    finally:
        subprocess.Popen = original_popen  # type: ignore[assignment,misc]
    return function_calls, content, m


class TestPreResultContentFallback:
    """Tool calls must be recovered from pre-result content when result is empty."""

    def test_result_event_empty_but_text_deltas_have_tool_calls(self) -> None:
        """Regression: result event with empty result must not lose tool calls.

        When the CLI's result event has an empty ``result`` field but the
        streaming ``content_block_delta`` events carried valid tool_calls
        JSON, the parser must fall back to the pre-result accumulated
        content and extract the tool calls.
        """
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": json.dumps(
                    {"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://example.com"}}]}
                )},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "", "usage": {"input_tokens": 10, "output_tokens": 10}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert content == ""
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "go_to_url"
        assert function_calls[0]["arguments"] == {"url": "https://example.com"}
        assert "tool_calls" in m._pre_result_content

    def test_result_event_strips_tool_calls_json(self) -> None:
        """Result event may contain prose without the tool_calls JSON.

        If the CLI post-processes the result and strips the JSON, the
        pre-result content still has the full text including tool_calls.
        """
        tool_json = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "ls -la"}}]}
        )
        full_text = f"I will list the files.\n{tool_json}"

        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": full_text},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            # Result event has only the prose, tool_calls JSON was stripped
            {"type": "result", "result": "I will list the files.",
             "usage": {"input_tokens": 10, "output_tokens": 10}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert content == "I will list the files."
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"
        assert function_calls[0]["arguments"] == {"command": "ls -la"}

    def test_result_matches_content_normal_case(self) -> None:
        """When result matches accumulated content, normal parsing works."""
        tool_json = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "pwd"}}]}
        )

        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": tool_json},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": tool_json,
             "usage": {"input_tokens": 10, "output_tokens": 10}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert content == tool_json
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"
        # Pre-result content should also match
        assert m._pre_result_content == tool_json

    def test_no_result_event_uses_accumulated_content(self) -> None:
        """When there's no result event, accumulated content is used directly."""
        tool_json = json.dumps(
            {"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://test.com"}}]}
        )

        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": tool_json},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            # No result event
        ]

        function_calls, content, m = _run_with_events(events)

        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "go_to_url"

    def test_thinking_fallback_still_works(self) -> None:
        """Thinking block fallback works even when pre-result is empty."""
        tool_json = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "echo hi"}}]}
        )

        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": tool_json},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "", "usage": {"input_tokens": 10, "output_tokens": 10}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert content == ""
        assert m._pre_result_content == ""
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"
        assert "tool_calls" in m._last_thinking_content

    def test_combined_empty_result_and_thinking(self) -> None:
        """Both pre-result and thinking fallbacks can be exercised.

        Thinking has tool calls, text deltas also have tool calls, but result
        is empty.  Pre-result should be preferred over thinking because it
        was in the visible text block.
        """
        text_tool = json.dumps(
            {"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://real.com"}}]}
        )
        thinking_tool = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "WRONG"}}]}
        )

        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": thinking_tool},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": text_tool},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "", "usage": {"input_tokens": 10, "output_tokens": 10}},
        ]

        function_calls, content, m = _run_with_events(events)

        # Pre-result content should win (text block had go_to_url)
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "go_to_url"
        assert function_calls[0]["arguments"] == {"url": "https://real.com"}

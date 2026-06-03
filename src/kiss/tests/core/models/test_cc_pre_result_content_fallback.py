# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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
            self._terminated = False

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def poll(self) -> int | None:
            return 0 if self._terminated else None

        def terminate(self) -> None:
            self._terminated = True

        def kill(self) -> None:
            self._terminated = True

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
        """Tool calls in text deltas are extracted even when result event is empty.

        The early-stop mechanism terminates the CLI as soon as the first
        complete ``tool_calls`` JSON block is found in the streaming text,
        so the ``result`` event is never processed.  The content retains
        the tool_calls JSON from the deltas.
        """
        tool_json = json.dumps(
            {"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://example.com"}}]}
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
            {"type": "result", "result": "", "usage": {"input_tokens": 10, "output_tokens": 10}},
        ]

        function_calls, content, m = _run_with_events(events)

        # Early stop truncates content to the first tool_calls block; the
        # result event is never reached.
        assert content == tool_json
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "go_to_url"
        assert function_calls[0]["arguments"] == {"url": "https://example.com"}
        assert m._stopped_for_tool_calls is True

    def test_result_event_strips_tool_calls_json(self) -> None:
        """Early-stop captures tool calls even when result would strip them.

        The early-stop mechanism terminates the CLI as soon as the first
        complete ``tool_calls`` block is detected.  The result event
        (which might strip the JSON) is never processed.
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

        # Content is truncated to end of tool_calls block
        assert content == full_text
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"
        assert function_calls[0]["arguments"] == {"command": "ls -la"}
        assert m._stopped_for_tool_calls is True

    def test_result_matches_content_normal_case(self) -> None:
        """Early-stop captures tool calls; result event is never processed."""
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
        assert m._stopped_for_tool_calls is True

    def test_no_result_event_uses_accumulated_content(self) -> None:
        """Tool calls extracted even when there is no result event."""
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
        assert m._stopped_for_tool_calls is True

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
        """Tool calls from both text and thinking blocks are merged.

        Thinking has tool calls and text deltas also have tool calls.
        The early-stop mechanism stops at the first text-block tool call,
        but the merged parsing collects tool calls from all sources
        (content + thinking) so both are extracted.
        """
        text_tool = json.dumps(
            {"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://real.com"}}]}
        )
        thinking_tool = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "echo merged"}}]}
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

        # Both tool calls are extracted: go_to_url from text + Bash from thinking
        names = {fc["name"] for fc in function_calls}
        assert len(function_calls) == 2
        assert "go_to_url" in names
        assert "Bash" in names
        assert m._stopped_for_tool_calls is True

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test: cc/opus early-stop when tool_calls detected in streaming text.

When reasoning models (cc/opus) generate a tool_calls JSON block followed
by hallucinated tool results in a single response, the streaming parser
must detect the first complete tool_calls block and terminate the CLI
process early.  This prevents the model from generating an unbounded
response that simulates the entire agentic loop.
"""

import json
import subprocess
from typing import Any

from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_consecutive_tool_calls_end


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
    function_map: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, ClaudeCodeModel]:
    m = ClaudeCodeModel("cc/opus")
    m.initialize("test")
    if function_map is None:
        function_map = {
            "Bash": lambda command, **kw: "ok",
            "go_to_url": lambda url: "page content",
        }
    fake_popen = _build_fake_popen_class(events)
    original_popen = subprocess.Popen
    subprocess.Popen = fake_popen  # type: ignore[assignment,misc]
    try:
        function_calls, content, _ = m.generate_and_process_with_tools(function_map)
    finally:
        subprocess.Popen = original_popen  # type: ignore[assignment,misc]
    return function_calls, content, m


def _run_with_events_and_response(
    events: list[dict[str, Any]],
    function_map: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], ClaudeCodeModel]:
    """Same as _run_with_events but also returns the raw response dict."""
    m = ClaudeCodeModel("cc/opus")
    m.initialize("test")
    if function_map is None:
        function_map = {
            "Bash": lambda command, **kw: "ok",
            "go_to_url": lambda url: "page content",
        }
    fake_popen = _build_fake_popen_class(events)
    original_popen = subprocess.Popen
    subprocess.Popen = fake_popen  # type: ignore[assignment,misc]
    try:
        function_calls, content, response = m.generate_and_process_with_tools(function_map)
    finally:
        subprocess.Popen = original_popen  # type: ignore[assignment,misc]
    return function_calls, content, response, m


class TestFindConsecutiveToolCallsEnd:
    """Unit tests for _find_consecutive_tool_calls_end helper."""

    def test_no_tool_calls(self) -> None:
        assert _find_consecutive_tool_calls_end("just some text") == -1

    def test_incomplete_json(self) -> None:
        assert _find_consecutive_tool_calls_end('{"tool_calls": [{"name": "Bash"') == -1

    def test_single_complete_block(self) -> None:
        text = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        assert _find_consecutive_tool_calls_end(text) == len(text)

    def test_text_before_block(self) -> None:
        prefix = "I will list files.\n"
        tool = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        text = prefix + tool
        end = _find_consecutive_tool_calls_end(text)
        assert end == len(text)

    def test_text_after_block_returns_block_end(self) -> None:
        tool = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        text = tool + "\n\nHallucinated result"
        end = _find_consecutive_tool_calls_end(text)
        assert end == len(tool)

    def test_two_blocks_with_text_between_returns_first_end(self) -> None:
        tool1 = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        tool2 = '{"tool_calls": [{"name": "go_to_url", "arguments": {"url": "x"}}]}'
        text = tool1 + "\nfake result\n" + tool2
        end = _find_consecutive_tool_calls_end(text)
        assert end == len(tool1)

    def test_two_consecutive_blocks_returns_last_end(self) -> None:
        """Consecutive blocks (only whitespace between) → end of last block."""
        tool1 = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        tool2 = '{"tool_calls": [{"name": "go_to_url", "arguments": {"url": "x"}}]}'
        text = tool1 + "\n" + tool2
        end = _find_consecutive_tool_calls_end(text)
        assert end == len(text)

    def test_three_consecutive_blocks(self) -> None:
        """Three consecutive blocks all captured."""
        t1 = '{"tool_calls": [{"name": "A", "arguments": {}}]}'
        t2 = '{"tool_calls": [{"name": "B", "arguments": {}}]}'
        t3 = '{"tool_calls": [{"name": "C", "arguments": {}}]}'
        text = t1 + "\n" + t2 + "\n" + t3
        end = _find_consecutive_tool_calls_end(text)
        assert end == len(text)

    def test_non_tool_calls_json_after_block_stops(self) -> None:
        """A non-tool-calls JSON object after tool_calls stops the sequence."""
        tool = '{"tool_calls": [{"name": "Bash", "arguments": {}}]}'
        other = '{"result": "ok"}'
        text = tool + "\n" + other
        end = _find_consecutive_tool_calls_end(text)
        assert end == len(tool)


class TestEarlyStopOnToolCalls:
    """The streaming parser must stop early when tool_calls are detected."""

    def test_stops_before_hallucinated_result(self) -> None:
        """Model generates tool call + hallucinated result; only tool call extracted."""
        tool_json = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "mkdir -p /tmp"}}]}
        )
        hallucinated = '\n\n(no output)\n\n' + json.dumps(
            {"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://fake.com"}}]}
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
            # The model continues with hallucinated results (but we should stop)
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": hallucinated},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert m._stopped_for_tool_calls is True
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"
        # go_to_url from hallucinated result should NOT be extracted
        names = {fc["name"] for fc in function_calls}
        assert "go_to_url" not in names
        # Content should be truncated to end of first tool_calls block
        assert "fake.com" not in content

    def test_multiple_tool_calls_in_one_block_all_extracted(self) -> None:
        """Multiple tool calls in a SINGLE tool_calls array are all kept."""
        tool_json = json.dumps(
            {"tool_calls": [
                {"name": "Bash", "arguments": {"command": "mkdir -p /tmp"}},
                {"name": "go_to_url", "arguments": {"url": "https://real.com"}},
            ]}
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
        ]

        function_calls, content, m = _run_with_events(events)

        assert len(function_calls) == 2
        names = {fc["name"] for fc in function_calls}
        assert "Bash" in names
        assert "go_to_url" in names

    def test_tool_calls_in_chunks(self) -> None:
        """Tool call JSON split across multiple text_delta chunks."""
        full = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        mid = len(full) // 2
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": full[:mid]},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": full[mid:]},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
        ]

        function_calls, content, m = _run_with_events(events)

        # No trailing content after tool_calls → model finishes normally
        # (early stop is only triggered by trailing hallucinated text)
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"

    def test_simulated_agentic_loop_halted(self) -> None:
        """The model simulates an entire agentic loop; only first tool call extracted.

        This reproduces the exact bug from task 941: the cc/opus model
        generates tool call → hallucinated result → more tool calls → ...
        all in a single response.
        """
        first_tool = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {
                "command": "mkdir -p /tmp && echo 'hello'",
                "description": "Create file",
            }}]}
        )
        hallucinated_session = (
            '\n\n```\n(no output)\n```\n\n'
            + json.dumps({"tool_calls": [{"name": "go_to_url", "arguments": {
                "url": "https://www.nps.gov/yose/"
            }}]})
            + '\n\n[Tool Result]: Page: Yosemite National Park\n...\n\n'
            + 'Steps: 2/100, Tokens: 11148/200000\n\n'
            + json.dumps({"tool_calls": [{"name": "go_to_url", "arguments": {
                "url": "https://www.travelyosemite.com/lodging/"
            }}]})
            + '\n\n[Tool Result]: Page: Lodging\n...'
        )
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta",
                          "text": "I'll research.\n\n" + first_tool},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": hallucinated_session},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert m._stopped_for_tool_calls is True
        # Only the first tool call (Bash) should be extracted
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"
        # The hallucinated URLs should not appear in the content
        assert "nps.gov" not in content
        assert "travelyosemite" not in content

    def test_no_tool_calls_does_not_stop_early(self) -> None:
        """When there are no tool_calls, the full response is captured."""
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello, I can help you."},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "Hello, I can help you.",
             "usage": {"input_tokens": 10, "output_tokens": 5}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert m._stopped_for_tool_calls is False
        assert function_calls == []
        assert content == "Hello, I can help you."

    def test_two_consecutive_blocks_both_extracted(self) -> None:
        """Two consecutive tool_calls blocks (only whitespace between) both collected."""
        tool1 = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}
        )
        tool2 = json.dumps(
            {"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://x.com"}}]}
        )
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": tool1},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "\n" + tool2},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "\nDone."},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert m._stopped_for_tool_calls is True
        assert len(function_calls) == 2
        names = {fc["name"] for fc in function_calls}
        assert "Bash" in names
        assert "go_to_url" in names
        assert "Done." not in content

    def test_consecutive_blocks_without_trailing_text(self) -> None:
        """Consecutive blocks followed by content_block_stop (no trailing text)."""
        tool1 = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}
        )
        tool2 = json.dumps(
            {"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://x.com"}}]}
        )
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": tool1 + "\n" + tool2},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": tool1 + "\n" + tool2,
             "usage": {"input_tokens": 10, "output_tokens": 20}},
        ]

        function_calls, content, m = _run_with_events(events)

        assert len(function_calls) == 2
        names = {fc["name"] for fc in function_calls}
        assert "Bash" in names
        assert "go_to_url" in names

    def test_usage_captured_after_early_stop(self) -> None:
        """After early stop for tool_calls, usage data from result event is
        captured so cost is not reported as $0."""
        tool_json = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}
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
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "\n\n(no output)\n"},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": tool_json + "\n\n(no output)",
             "usage": {"input_tokens": 1234, "output_tokens": 56,
                       "cache_read_input_tokens": 0}},
        ]

        function_calls, content, response, m = _run_with_events_and_response(events)

        assert m._stopped_for_tool_calls is True
        assert len(function_calls) == 1
        assert isinstance(response, dict)
        usage = response.get("usage", {})
        assert usage.get("input_tokens") == 1234
        assert usage.get("output_tokens") == 56

    def test_usage_captured_with_simulated_agentic_loop(self) -> None:
        """Same as test_simulated_agentic_loop_halted but includes the
        result event with usage after the hallucinated content."""
        first_tool = json.dumps(
            {"tool_calls": [{"name": "Bash", "arguments": {
                "command": "mkdir -p /tmp && echo 'hello'",
            }}]}
        )
        hallucinated = (
            '\n\n```\n(no output)\n```\n\n'
            + json.dumps({"tool_calls": [{"name": "go_to_url", "arguments": {
                "url": "https://x.com"
            }}]})
        )
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta",
                          "text": "I'll help.\n\n" + first_tool},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": hallucinated},
            }},
            {"type": "stream_event", "event": {"type": "content_block_stop"}},
            {"type": "result", "result": "...",
             "usage": {"input_tokens": 5000, "output_tokens": 1200,
                       "cache_read_input_tokens": 200,
                       "cache_creation": {"ephemeral_5m_input_tokens": 300,
                                          "ephemeral_1h_input_tokens": 0}}},
        ]

        function_calls, content, response, m = _run_with_events_and_response(events)

        assert m._stopped_for_tool_calls is True
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"
        usage = response.get("usage", {})
        assert usage.get("input_tokens") == 5000
        assert usage.get("output_tokens") == 1200
        assert usage.get("cache_read_input_tokens") == 200
        cache_creation = usage.get("cache_creation", {})
        assert cache_creation.get("ephemeral_5m_input_tokens") == 300

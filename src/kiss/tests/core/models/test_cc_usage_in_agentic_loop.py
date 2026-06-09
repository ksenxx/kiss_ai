# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: Claude Code CLI agentic calls report correct costs.

When the streaming parser stops early because tool_calls were detected,
the "result" event (which carries usage: input/output/cache token counts)
must still be captured.  Otherwise ``extract_input_output_token_counts``
returns all zeros and ``calculate_cost`` reports $0 for every agentic
step, causing the budget / dashboard to massively undercount real API
spend (issue #34).
"""

import json
import subprocess
from typing import Any

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model_info import calculate_cost


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
) -> tuple[list[dict[str, Any]], str, dict[str, Any], ClaudeCodeModel]:
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


class TestAgenticCost:
    def test_early_stop_reports_correct_cost(self) -> None:
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
             "usage": {"input_tokens": 10000, "output_tokens": 500,
                       "cache_read_input_tokens": 0}},
        ]

        _, _, response, m = _run_with_events(events)

        assert m._stopped_for_tool_calls is True

        counts = m.extract_input_output_token_counts_from_response(response)
        input_t, output_t, cache_read, cache_write_5m, cache_write_1h = counts

        assert input_t == 10000
        assert output_t == 500

        # calculate_cost must not raise KISSError — cc/opus pricing is
        # $0/$0 (billed via subscription), so cost == 0 is expected.
        calculate_cost(
            m.model_name,
            input_t,
            output_t,
            cache_read,
            cache_write_5m,
            cache_write_1h,
        )

    def test_agentic_loop_cost_with_cache(self) -> None:
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
             "usage": {"input_tokens": 50000, "output_tokens": 2000,
                       "cache_read_input_tokens": 10000,
                       "cache_creation": {"ephemeral_5m_input_tokens": 5000,
                                          "ephemeral_1h_input_tokens": 2000}}},
        ]

        _, _, response, m = _run_with_events(events)

        assert m._stopped_for_tool_calls is True

        counts = m.extract_input_output_token_counts_from_response(response)
        input_t, output_t, cache_read, cache_write_5m, cache_write_1h = counts

        assert input_t == 50000
        assert output_t == 2000
        assert cache_read == 10000
        assert cache_write_5m == 5000
        assert cache_write_1h == 2000

        # calculate_cost must not raise KISSError — cc/opus pricing is
        # $0/$0 (billed via subscription), so cost == 0 is expected.
        calculate_cost(
            m.model_name,
            input_t,
            output_t,
            cache_read,
            cache_write_5m,
            cache_write_1h,
        )

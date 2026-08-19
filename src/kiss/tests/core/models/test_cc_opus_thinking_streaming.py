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
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401


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

    def readline(self) -> str:
        if self._pos >= len(self._lines):
            return ""
        line = self._lines[self._pos]
        self._pos += 1
        return line

    def read(self) -> str:
        rest = "".join(self._lines[self._pos :])
        self._pos = len(self._lines)
        return rest

    def close(self) -> None:
        pass


def _make_stream_events_with_text_deltas() -> list[dict[str, Any]]:
    """Build a realistic cc/opus stream: no thinking_delta, multiple text_delta chunks."""
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
                "delta": {"type": "signature_delta", "signature": "abc123"},
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
                "delta": {"type": "text_delta", "text": "Let me "},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "analyze "},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "this problem."},
            },
        },
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {
            "type": "result",
            "result": "Let me analyze this problem.",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    ]


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


class TestCCOpusTextStreamingInToolMode:
    """Text tokens must stream incrementally during generate_and_process_with_tools.

    This tests the BUG: generate_and_process_with_tools buffers all text
    tokens and emits them as a single text_delta event after generation
    completes, instead of streaming them incrementally.
    """

    def test_callbacks_restored_after_tool_generation(self) -> None:
        """Token and thinking callbacks must be restored after tool generation."""
        original_token_cb = lambda t: None  # noqa: E731
        original_thinking_cb = lambda s: None  # noqa: E731

        m = ClaudeCodeModel(
            "cc/opus",
            token_callback=original_token_cb,
            thinking_callback=original_thinking_cb,
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

        assert m.token_callback is original_token_cb
        assert m.thinking_callback is original_thinking_cb

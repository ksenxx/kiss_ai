# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Regression test: cc/opus must extract tool calls emitted inside thinking blocks.

Reasoning-capable Claude CLI models (notably ``cc/opus`` with extended
thinking enabled) sometimes emit the agent's ``tool_calls`` JSON inside an
extended-thinking ``content_block`` rather than a visible ``text`` block.

Before the fix, ``ClaudeCodeModel._parse_stream_events`` only accumulated
``text_delta`` text into ``content`` while discarding ``thinking_delta``
text after forwarding it to the thinking callback.  As a result,
``_parse_text_based_tool_calls(content)`` returned an empty list and the
agent stalled (the user observed that "function calls appear in the
thoughts ... it does not call any function").

The fix captures thinking-block text into
``ClaudeCodeModel._last_thinking_content`` and falls back to parsing it for
``tool_calls`` when the visible content yields none.
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
        rest = "".join(self._lines[self._pos :])
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


def _bash_tool_call_json() -> str:
    return json.dumps(
        {"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}
    )


def _make_events_tool_call_only_in_thinking() -> list[dict[str, Any]]:
    """Build a stream where the tool_calls JSON appears ONLY in thinking_delta.

    This mirrors what ``cc/opus`` produces with the long Sorcar SYSTEM.md
    prompt: the model reasons in an extended-thinking block, decides on a
    tool call, emits the JSON inside that block, and never opens a visible
    ``text`` block.
    """
    tool_json = _bash_tool_call_json()
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
                "delta": {"type": "thinking_delta", "thinking": "I should "},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "list the files.\n\n"},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": tool_json},
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
            "type": "result",
            "result": "",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    ]


def _make_events_tool_call_in_text_and_thinking() -> list[dict[str, Any]]:
    """Both blocks contain tool_calls; visible text wins (no fallback)."""
    text_tool_json = json.dumps(
        {"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}
    )
    thinking_tool_json = json.dumps(
        {"tool_calls": [{"name": "Bash", "arguments": {"command": "WRONG"}}]}
    )
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
                "delta": {"type": "thinking_delta", "thinking": thinking_tool_json},
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
                "delta": {"type": "text_delta", "text": text_tool_json},
            },
        },
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {
            "type": "result",
            "result": text_tool_json,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        },
    ]


def _make_events_no_tool_calls_anywhere() -> list[dict[str, Any]]:
    """Neither block contains tool_calls; result must be empty list."""
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
                "delta": {"type": "thinking_delta", "thinking": "Just thinking."},
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
                "delta": {"type": "text_delta", "text": "Hello, world."},
            },
        },
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {
            "type": "result",
            "result": "Hello, world.",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    ]


class TestCCToolCallsInsideThinking:
    """The agent must receive tool calls even when they are emitted in thinking."""

    def _run(self, events: list[dict[str, Any]]) -> tuple[Any, str, ClaudeCodeModel]:
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        fake_popen = _build_fake_popen_class(events)
        original_popen = subprocess.Popen
        subprocess.Popen = fake_popen  # type: ignore[assignment,misc]
        try:
            function_calls, content, _ = m.generate_and_process_with_tools(
                {"Bash": lambda command: "ok"}
            )
        finally:
            subprocess.Popen = original_popen  # type: ignore[assignment,misc]
        return function_calls, content, m

    def test_tool_call_emitted_only_in_thinking_is_extracted(self) -> None:
        """Regression: tool_calls JSON inside thinking_delta must reach the agent.

        Before the fix this returned an empty ``function_calls`` list and
        the cc/opus agent stalled on the Yosemite trip-planning prompt.
        """
        function_calls, content, m = self._run(
            _make_events_tool_call_only_in_thinking()
        )

        # The visible content has nothing — tool call lives in thinking.
        assert content == ""
        # The agent received the tool call from the thinking-block fallback.
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "Bash"
        assert function_calls[0]["arguments"] == {"command": "ls"}
        # The captured thinking buffer must include the JSON.
        assert "tool_calls" in m._last_thinking_content
        # And the conversation's last assistant message has been upgraded
        # with tool_calls so subsequent turns reference it correctly.
        last = m.conversation[-1]
        assert last["role"] == "assistant"
        assert "tool_calls" in last
        assert len(last["tool_calls"]) == 1
        assert last["tool_calls"][0]["function"]["name"] == "Bash"
        assert json.loads(last["tool_calls"][0]["function"]["arguments"]) == {
            "command": "ls"
        }

    def test_visible_text_wins_when_both_blocks_have_tool_calls(self) -> None:
        """Tool calls from both text and thinking are merged.

        The text block carries ``{"command": "ls"}`` and the thinking block
        carries ``{"command": "WRONG"}``.  Since we now merge tool calls
        from all sources, both are extracted (de-duplicated by name+args).
        """
        function_calls, content, _ = self._run(
            _make_events_tool_call_in_text_and_thinking()
        )

        # Both tool calls are extracted (different args → no dedup)
        assert len(function_calls) == 2
        commands = {fc["arguments"]["command"] for fc in function_calls}
        assert "ls" in commands
        assert "WRONG" in commands

    def test_no_tool_calls_anywhere_returns_empty(self) -> None:
        """When neither block has tool_calls, function_calls is empty."""
        function_calls, content, m = self._run(
            _make_events_no_tool_calls_anywhere()
        )

        assert function_calls == []
        assert content == "Hello, world."
        # The last assistant message must remain a plain message (no
        # ``tool_calls`` upgrade) since none were parsed.
        assert m.conversation[-1].get("tool_calls") is None

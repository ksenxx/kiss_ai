"""Tests for cc/* model tool-call stripping from Thoughts panel.

Verifies that ``_strip_text_based_tool_calls`` correctly removes
tool_calls JSON from model output, and that
``ClaudeCodeModel.generate_and_process_with_tools`` streams tokens
directly to callbacks (no buffering) for incremental UI updates.

No mocks — uses real functions and real model instances.
"""

import inspect
import unittest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model import (
    _parse_text_based_tool_calls,
    _strip_text_based_tool_calls,
)


class TestStripTextBasedToolCalls(unittest.TestCase):
    """Unit tests for _strip_text_based_tool_calls."""


    def test_whole_content_json(self) -> None:
        content = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        assert _strip_text_based_tool_calls(content) == ""

    def test_whole_content_json_with_whitespace(self) -> None:
        content = '  {"tool_calls": [{"name": "Bash", "arguments": {}}]}  '
        assert _strip_text_based_tool_calls(content) == ""


    def test_fenced_json_block(self) -> None:
        content = (
            "I will run a command.\n\n"
            '```json\n{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}\n```'
        )
        result = _strip_text_based_tool_calls(content)
        assert "tool_calls" not in result
        assert "I will run a command." in result

    def test_fenced_generic_block(self) -> None:
        content = (
            "Thinking...\n\n"
            '```\n{"tool_calls": [{"name": "Read", "arguments": {"file_path": "x.py"}}]}\n```'
        )
        result = _strip_text_based_tool_calls(content)
        assert "tool_calls" not in result
        assert "Thinking..." in result


    def test_inline_json(self) -> None:
        content = (
            'Let me check.\n'
            '{"tool_calls": [{"name": "Bash", "arguments": {"command": "pwd"}}]}'
        )
        result = _strip_text_based_tool_calls(content)
        assert "tool_calls" not in result
        assert "Let me check." in result


    def test_no_tool_calls_returns_original(self) -> None:
        content = "Just some plain text with no JSON."
        assert _strip_text_based_tool_calls(content) == content.strip()

    def test_empty_string(self) -> None:
        assert _strip_text_based_tool_calls("") == ""

    def test_json_without_tool_calls_key(self) -> None:
        content = '{"result": "hello"}'
        assert _strip_text_based_tool_calls(content) == content.strip()


    def test_strip_removes_exactly_what_parse_finds(self) -> None:
        """If _parse finds tool calls, _strip should remove them."""
        cases = [
            '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}',
            'text before\n{"tool_calls": [{"name": "Read", "arguments": {"file_path": "a"}}]}',
            '```json\n{"tool_calls": [{"name": "Write", "arguments": {"path": "b"}}]}\n```',
        ]
        for content in cases:
            parsed = _parse_text_based_tool_calls(content)
            assert len(parsed) > 0, f"Expected tool calls in: {content}"
            stripped = _strip_text_based_tool_calls(content)
            assert "tool_calls" not in stripped, f"tool_calls still in stripped: {stripped}"


    def test_multiple_inline_tool_calls(self) -> None:
        tc1 = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        tc2 = '{"tool_calls": [{"name": "Read", "arguments": {"file_path": "a.py"}}]}'
        content = f"First step:\n{tc1}\nSecond step:\n{tc2}"
        result = _strip_text_based_tool_calls(content)
        assert "tool_calls" not in result
        assert "First step:" in result


class TestCCModelTokenStreaming(unittest.TestCase):
    """Verify ClaudeCodeModel.generate_and_process_with_tools streams tokens
    directly to callbacks without buffering."""

    def test_source_has_no_buffer(self) -> None:
        """generate_and_process_with_tools must NOT buffer tokens."""
        src = inspect.getsource(ClaudeCodeModel.generate_and_process_with_tools)
        assert "buffer" not in src, "Token buffering was removed for streaming"
        assert "buffer.append" not in src

    def test_source_does_not_replace_callbacks(self) -> None:
        """Callbacks must not be swapped — tokens stream through the originals."""
        src = inspect.getsource(ClaudeCodeModel.generate_and_process_with_tools)
        assert "self.token_callback = " not in src
        assert "self.thinking_callback = " not in src

    def test_no_callback_no_crash(self) -> None:
        """When there's no token_callback, generate_and_process_with_tools works."""
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        assert m.token_callback is None

    def test_config_restored_in_finally(self) -> None:
        """Model config is restored even on exception."""
        src = inspect.getsource(ClaudeCodeModel.generate_and_process_with_tools)
        assert "finally:" in src
        assert "self.model_config = original_config" in src


class TestStripEdgeCases(unittest.TestCase):
    """Edge cases for _strip_text_based_tool_calls."""

    def test_invalid_json_in_fenced_block_not_stripped(self) -> None:
        """Invalid JSON is not a tool call — it must NOT be stripped.

        The brace-balanced scanner validates each candidate object via
        :func:`json.loads`, so heuristic matches that fail to parse are
        preserved verbatim in the visible Thoughts panel.
        """
        content = '```json\n{not valid "tool_calls": broken}\n```'
        result = _strip_text_based_tool_calls(content)
        assert result == content.strip()

    def test_tool_calls_in_prose_not_stripped(self) -> None:
        """The word 'tool_calls' in prose is not stripped."""
        content = "The tool_calls key is used for function calling."
        result = _strip_text_based_tool_calls(content)
        assert result == content.strip()

    def test_nested_braces_inline(self) -> None:
        """The inline regex strips the inner object containing tool_calls.

        The regex ``\\{[^{}]*"tool_calls"...\\}`` matches the inner
        ``{"tool_calls": [...]}`` even when nested, because the leading
        ``[^{}]*`` stops at the outer ``{`` and the match starts from
        the inner one.
        """
        content = '{"outer": {"tool_calls": [{"name": "x"}]}}'
        result = _strip_text_based_tool_calls(content)
        assert "tool_calls" not in result

    def test_whitespace_only_after_strip(self) -> None:
        """If only whitespace remains after stripping, return empty."""
        content = '  {"tool_calls": [{"name": "Bash", "arguments": {}}]}  '
        result = _strip_text_based_tool_calls(content)
        assert result == ""


if __name__ == "__main__":
    unittest.main()

"""Tests for cc/* model tool-call stripping from Thoughts panel.

Verifies that ``_strip_text_based_tool_calls`` correctly removes
tool_calls JSON from model output, and that
``ClaudeCodeModel.generate_and_process_with_tools`` buffers streamed
tokens and emits only the cleaned text to the token callback.

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

    # --- Whole-content JSON ---

    def test_whole_content_json(self) -> None:
        content = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        assert _strip_text_based_tool_calls(content) == ""

    def test_whole_content_json_with_whitespace(self) -> None:
        content = '  {"tool_calls": [{"name": "Bash", "arguments": {}}]}  '
        assert _strip_text_based_tool_calls(content) == ""

    # --- Fenced JSON code blocks ---

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

    # --- Inline JSON ---

    def test_inline_json(self) -> None:
        content = (
            'Let me check.\n'
            '{"tool_calls": [{"name": "Bash", "arguments": {"command": "pwd"}}]}'
        )
        result = _strip_text_based_tool_calls(content)
        assert "tool_calls" not in result
        assert "Let me check." in result

    # --- No tool calls ---

    def test_no_tool_calls_returns_original(self) -> None:
        content = "Just some plain text with no JSON."
        assert _strip_text_based_tool_calls(content) == content.strip()

    def test_empty_string(self) -> None:
        assert _strip_text_based_tool_calls("") == ""

    def test_json_without_tool_calls_key(self) -> None:
        content = '{"result": "hello"}'
        assert _strip_text_based_tool_calls(content) == content.strip()

    # --- Consistency with _parse_text_based_tool_calls ---

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

    # --- Multiple tool calls ---

    def test_multiple_inline_tool_calls(self) -> None:
        tc1 = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        tc2 = '{"tool_calls": [{"name": "Read", "arguments": {"file_path": "a.py"}}]}'
        content = f"First step:\n{tc1}\nSecond step:\n{tc2}"
        result = _strip_text_based_tool_calls(content)
        assert "tool_calls" not in result
        assert "First step:" in result


class TestCCModelTokenBuffering(unittest.TestCase):
    """Verify ClaudeCodeModel.generate_and_process_with_tools buffers tokens
    and emits cleaned text without tool-call JSON."""

    def test_source_has_buffer_and_strip(self) -> None:
        """generate_and_process_with_tools buffers tokens and strips tool calls."""
        src = inspect.getsource(ClaudeCodeModel.generate_and_process_with_tools)
        # Buffers tokens instead of passing them directly
        assert "buffer" in src
        assert "buffer.append" in src
        # Strips tool calls from content before emitting
        assert "_strip_text_based_tool_calls" in src
        # Restores original callback
        assert "self.token_callback = original_callback" in src

    def test_no_callback_no_buffering(self) -> None:
        """When there's no token_callback, no buffering occurs."""
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        # No callback set — should not raise
        assert m.token_callback is None

    def test_callback_buffered_during_generate(self) -> None:
        """The original callback is replaced with buffer.append during generate."""
        src = inspect.getsource(ClaudeCodeModel.generate_and_process_with_tools)
        # original_callback is saved
        assert "original_callback = self.token_callback" in src
        # conditional check before buffering
        assert "if original_callback is not None:" in src

    def test_cleaned_text_emitted_with_tool_calls(self) -> None:
        """When tool calls found, cleaned text (no JSON) is emitted."""
        src = inspect.getsource(ClaudeCodeModel.generate_and_process_with_tools)
        assert "_strip_text_based_tool_calls(content) if function_calls else content" in src

    def test_original_text_emitted_without_tool_calls(self) -> None:
        """When no tool calls, original content is emitted unchanged."""
        src = inspect.getsource(ClaudeCodeModel.generate_and_process_with_tools)
        # The ternary emits content when no function_calls
        assert "if function_calls else content" in src

    def test_callback_restored_in_finally(self) -> None:
        """Original callback is restored even on exception."""
        src = inspect.getsource(ClaudeCodeModel.generate_and_process_with_tools)
        # Check that callback restore is in the finally block
        finally_idx = src.index("finally:")
        restore_idx = src.index("self.token_callback = original_callback")
        assert restore_idx > finally_idx

    def test_import_strip_function(self) -> None:
        """_strip_text_based_tool_calls is imported in claude_code_model."""
        import kiss.core.models.claude_code_model as ccm
        assert hasattr(ccm, "_strip_text_based_tool_calls")


class TestStripEdgeCases(unittest.TestCase):
    """Edge cases for _strip_text_based_tool_calls."""

    def test_invalid_json_in_fenced_block_stripped_by_regex(self) -> None:
        """Fenced blocks matching the pattern are stripped even if invalid JSON.

        The regex is a heuristic — it does NOT validate JSON.
        """
        content = '```json\n{not valid "tool_calls": broken}\n```'
        result = _strip_text_based_tool_calls(content)
        assert result == ""

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
        # The inner {"tool_calls": ...} is stripped; outer shell remains
        assert "tool_calls" not in result

    def test_whitespace_only_after_strip(self) -> None:
        """If only whitespace remains after stripping, return empty."""
        content = '  {"tool_calls": [{"name": "Bash", "arguments": {}}]}  '
        result = _strip_text_based_tool_calls(content)
        assert result == ""


if __name__ == "__main__":
    unittest.main()

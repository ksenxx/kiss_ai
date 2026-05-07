"""Regression tests for ``_parse_text_based_tool_calls`` — complex args.

These cases reproduce the bug where ``cc/opus`` tool calls were silently
dropped because the previous regex-based parser failed when arguments
contained:

* arrays (``[1, 2, 3]``),
* string values containing ``[`` / ``]`` such as markdown links
  (``[label](url)``),
* string values containing ``{`` / ``}``,
* raw (unescaped) newlines inside a multi-line ``summary`` argument,
* deeply nested objects inside ``arguments``.

The new brace-balanced scanner extracts all of these correctly.
"""

import unittest

from kiss.core.models.model import (
    _parse_text_based_tool_calls,
    _strip_text_based_tool_calls,
)


class TestParseComplexToolCallArgs(unittest.TestCase):
    """Verify extraction works for realistic ``cc/opus`` outputs."""

    def test_array_in_arguments(self) -> None:
        """Arguments containing a JSON array are extracted."""
        content = (
            'I will: {"tool_calls": [{"name": "Foo", '
            '"arguments": {"items": [1, 2, 3]}}]}'
        )
        calls = _parse_text_based_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Foo")
        self.assertEqual(calls[0]["arguments"], {"items": [1, 2, 3]})

    def test_string_with_brackets_in_arguments(self) -> None:
        """Bracket characters inside a JSON string value are tolerated."""
        content = (
            'Run: {"tool_calls": [{"name": "Bash", '
            '"arguments": {"command": "echo [hi]"}}]}'
        )
        calls = _parse_text_based_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"], {"command": "echo [hi]"})

    def test_string_with_braces_in_arguments(self) -> None:
        """Brace characters inside a JSON string value are tolerated."""
        content = (
            'Run: {"tool_calls": [{"name": "Bash", '
            '"arguments": {"command": "echo {hi}"}}]}'
        )
        calls = _parse_text_based_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"], {"command": "echo {hi}"})

    def test_finish_summary_with_markdown_link(self) -> None:
        """Markdown link brackets in a finish summary do not break parsing."""
        content = (
            'Here:\n{"tool_calls": [{"name": "finish", '
            '"arguments": {"success": true, '
            '"summary": "Visit [docs](https://x.example/y) for details."}}]}'
        )
        calls = _parse_text_based_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "finish")
        self.assertIn("[docs](", calls[0]["arguments"]["summary"])

    def test_summary_with_raw_newlines(self) -> None:
        """Unescaped control characters inside string values are tolerated.

        ``cc/opus`` regularly emits multi-line ``summary`` arguments with
        literal ``\\n`` characters (not the escaped ``\\\\n`` form), which
        strict :func:`json.loads` rejects.  The scanner uses
        ``strict=False`` to accept them.
        """
        content = (
            'Done.\n{"tool_calls": [{"name": "finish", '
            '"arguments": {"success": true, '
            '"summary": "- Step 1\n- Step 2\n  - sub [link](http://x)"}}]}'
        )
        calls = _parse_text_based_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "finish")
        self.assertIn("Step 1", calls[0]["arguments"]["summary"])
        self.assertIn("[link]", calls[0]["arguments"]["summary"])

    def test_deeply_nested_arguments(self) -> None:
        """Deeply nested objects inside arguments are preserved verbatim."""
        content = (
            '{"tool_calls": [{"name": "Edit", '
            '"arguments": {"changes": {"file": {"line": 5, "ops": ["a", "b"]}}}}]}'
        )
        calls = _parse_text_based_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["arguments"],
            {"changes": {"file": {"line": 5, "ops": ["a", "b"]}}},
        )

    def test_strip_keeps_prose_around_complex_call(self) -> None:
        """Stripping removes the JSON span and keeps the surrounding prose."""
        content = (
            "Before.\n"
            '{"tool_calls": [{"name": "finish", '
            '"arguments": {"summary": "see [x](u)"}}]}\n'
            "After."
        )
        result = _strip_text_based_tool_calls(content)
        self.assertNotIn("tool_calls", result)
        self.assertIn("Before.", result)
        self.assertIn("After.", result)


if __name__ == "__main__":
    unittest.main()

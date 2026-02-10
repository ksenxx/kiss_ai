"""Tests for compact_formatter _extract_parts and format_message."""

import unittest

from kiss.core import config as config_module
from kiss.core.compact_formatter import (
    LINE_LENGTH,
    CompactFormatter,
    _extract_parts,
    _strip_markdown,
)


def _msg(content: str, role: str = "model") -> dict:
    return {"role": role, "content": content}


FULL_CONTENT = (
    "I need to list the files.\n"
    "```python\n"
    "Bash(command='ls -la', description='list project files')\n"
    "```\n"
    "#### Usage Information\n"
    "  - [Token usage: 2450/1048576]\n"
    "  - [Step 2/100]\n"
)


class TestExtractParts(unittest.TestCase):
    def test_full_message(self) -> None:
        thought, tool_desc, usage = _extract_parts(FULL_CONTENT)
        assert thought == "I need to list the files."
        assert tool_desc == "list project files"
        assert "[usage]:" in usage and "[Step 2/100]" in usage

    def test_description_non_string_falls_back(self) -> None:
        c = "T.\n```python\nFunc(description=42, path='/foo')\n```\n"
        assert _extract_parts(c)[1] == "Func(path=/foo)"

    def test_empty(self) -> None:
        assert _extract_parts("") == ("", "", "")

    def test_int_arg(self) -> None:
        assert _extract_parts("T.\n```python\nfinish(result=42)\n```\n")[1] == "finish(result=42)"

    def test_long_arg_truncated(self) -> None:
        c = f"T.\n```python\nWriteFile(path='{'a' * (LINE_LENGTH + 10)}')\n```\n"
        desc = _extract_parts(c)[1]
        assert "WriteFile(" in desc and "..." in desc

    def test_not_a_call(self) -> None:
        assert _extract_parts("T.\n```python\nsome_name\n```\n")[1] == "some_name"

    def test_syntax_error(self) -> None:
        assert _extract_parts("T.\n```python\nbad syntax (((\n```\n")[1] == "bad syntax ((("


class TestStripMarkdown(unittest.TestCase):
    def test_fenced_code_block(self) -> None:
        result = _strip_markdown('before\n```json\n{"a": 1}\n```\nafter')
        assert "before" in result and "after" in result


class TestFormatMessage(unittest.TestCase):
    def setUp(self) -> None:
        self.formatter = CompactFormatter()
        self.orig_verbose = config_module.DEFAULT_CONFIG.agent.verbose
        config_module.DEFAULT_CONFIG.agent.verbose = True

    def tearDown(self) -> None:
        config_module.DEFAULT_CONFIG.agent.verbose = self.orig_verbose

    def test_verbose_false(self) -> None:
        config_module.DEFAULT_CONFIG.agent.verbose = False
        assert self.formatter.format_message(_msg("anything")) == ""

    def test_full_message(self) -> None:
        result = self.formatter.format_message(_msg(FULL_CONTENT))
        assert "[model]: I need to list the files." in result
        assert "..." not in result.split("\n")[0]
        assert "[action]: list project files" in result
        assert "[usage]:" in result

    def test_empty_and_missing(self) -> None:
        assert self.formatter.format_message(_msg("")) == "[model]: (empty)"
        assert "[unknown]:" in self.formatter.format_message({"content": "hello"})
        assert self.formatter.format_message({"role": "user"}) == "[user]: (empty)"

    def test_thought_truncation(self) -> None:
        content = f"{'x' * 200}\n```python\nBash(description='list')\n```\n"
        line = self.formatter.format_message(_msg(content)).split("\n")[0]
        assert line.endswith("...") and len(line) <= LINE_LENGTH + len("...")

    def test_format_messages(self) -> None:
        result = self.formatter.format_messages([_msg("Hello"), _msg("World", "user")])
        assert "[model]: Hello" in result and "[user]: World" in result
        assert "..." not in result

    def test_format_messages_not_verbose(self) -> None:
        config_module.DEFAULT_CONFIG.agent.verbose = False
        assert self.formatter.format_messages([_msg("Hello")]) == ""


if __name__ == "__main__":
    unittest.main()

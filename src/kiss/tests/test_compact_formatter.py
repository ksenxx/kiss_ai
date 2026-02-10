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


class TestExtractParts(unittest.TestCase):
    def test_description_non_string_falls_back(self) -> None:
        c = "T.\n```python\nFunc(description=42, path='/foo')\n```\n"
        assert _extract_parts(c)[1] == "Func(path=/foo)"

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

    def test_thought_truncation(self) -> None:
        content = f"{'x' * 200}\n```python\nBash(description='list')\n```\n"
        line = self.formatter.format_message(_msg(content)).split("\n")[0]
        assert line.endswith("...") and len(line) <= LINE_LENGTH + len("...")


if __name__ == "__main__":
    unittest.main()

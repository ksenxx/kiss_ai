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

    def test_description_double_quotes(self) -> None:
        c = 'T.\n```python\nBash(command="pwd", description="show cwd")\n```\n'
        _, desc, _ = _extract_parts(c)
        assert desc == "show cwd"

    def test_description_not_first_kwarg(self) -> None:
        c = "T.\n```python\nBash(command='ls', timeout=30, description='list')\n```\n"
        assert _extract_parts(c)[1] == "list"

    def test_description_non_string_falls_back(self) -> None:
        c = "T.\n```python\nFunc(description=42, path='/foo')\n```\n"
        assert _extract_parts(c)[1] == "Func(path=/foo)"

    def test_no_description_summarizes(self) -> None:
        c = "T.\n```python\nReadFile(path='/tmp/test.py')\n```\n"
        assert _extract_parts(c)[1] == "ReadFile(path=/tmp/test.py)"

    def test_no_description_multiple_kwargs(self) -> None:
        c = "T.\n```python\nWriteFile(path='/tmp/out.py', content='hello')\n```\n"
        assert _extract_parts(c)[1] == "WriteFile(path=/tmp/out.py, content=hello)"

    def test_long_arg_truncated(self) -> None:
        c = f"T.\n```python\nWriteFile(path='{'a' * (LINE_LENGTH + 10)}')\n```\n"
        desc = _extract_parts(c)[1]
        assert "WriteFile(" in desc and "..." in desc

    def test_no_args(self) -> None:
        assert _extract_parts("T.\n```python\nfinish()\n```\n")[1] == "finish"

    def test_int_arg(self) -> None:
        assert _extract_parts("T.\n```python\nfinish(result=42)\n```\n")[1] == "finish(result=42)"

    def test_no_tool_call(self) -> None:
        thought, desc, usage = _extract_parts(
            "Just thinking.\n#### Usage Information\n  - [Step 1/100]\n"
        )
        assert thought == "Just thinking." and desc == "" and "[usage]:" in usage

    def test_no_usage(self) -> None:
        thought, desc, usage = _extract_parts(
            "A thought.\n```python\nBash(command='ls', description='list')\n```\n"
        )
        assert thought == "A thought." and desc == "list" and usage == ""

    def test_plain_text(self) -> None:
        assert _extract_parts("Just a simple response.") == ("Just a simple response.", "", "")

    def test_empty(self) -> None:
        assert _extract_parts("") == ("", "", "")

    def test_multiline_thought(self) -> None:
        thought, desc, _ = _extract_parts(
            "Line one.\nLine two.\n```python\nBash(description='hi')\n```\n"
        )
        assert "Line one" in thought and "Line two" in thought and desc == "hi"

    def test_three_hash_usage(self) -> None:
        c = "T.\n```python\nBash(description='x')\n```\n### Usage Information\n  - [Step 1/10]\n"
        assert "[Step 1/10]" in _extract_parts(c)[2]

    def test_not_a_call(self) -> None:
        assert _extract_parts("T.\n```python\nsome_name\n```\n")[1] == "some_name"

    def test_syntax_error(self) -> None:
        assert _extract_parts("T.\n```python\nbad syntax (((\n```\n")[1] == "bad syntax ((("


class TestStripMarkdown(unittest.TestCase):
    def test_formatting_stripped(self) -> None:
        assert _strip_markdown("**bold** and *italic*") == "bold and italic"
        assert _strip_markdown("__bold__ and _italic_") == "bold and italic"
        assert _strip_markdown("~~removed~~") == "removed"
        assert _strip_markdown("run `ls -la` now") == "run ls -la now"

    def test_structure_stripped(self) -> None:
        assert _strip_markdown("# Heading\nsome text") == "Heading some text"
        assert _strip_markdown("- item one\n- item two") == "item one item two"
        assert _strip_markdown("1. first\n2. second") == "first second"
        assert _strip_markdown("> quoted text") == "quoted text"
        assert _strip_markdown("[click](http://x.com)") == "click"

    def test_fenced_code_block(self) -> None:
        result = _strip_markdown('before\n```json\n{"a": 1}\n```\nafter')
        assert "before" in result and "after" in result

    def test_plain_text_unchanged(self) -> None:
        assert _strip_markdown("no markdown here") == "no markdown here"

    def test_in_extract_parts(self) -> None:
        content = "## Analysis\nI need to **read** the `config` file.\n"
        content += "```python\nBash(description='read config')\n```\n"
        thought, desc, _ = _extract_parts(content)
        assert "##" not in thought and "**" not in thought and "`" not in thought
        assert "Analysis" in thought and "read" in thought and "config" in thought
        assert desc == "read config"


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

    def test_user_message(self) -> None:
        result = self.formatter.format_message(_msg("Tool output.", "user"))
        assert result == "[user]: Tool output."

    def test_user_message_truncated(self) -> None:
        long = "x" * 200
        result = self.formatter.format_message(_msg(long, "user"))
        assert result.endswith("...") and len(result) <= LINE_LENGTH + len("...")

    def test_user_message_no_extract_parts(self) -> None:
        content = "result.\n```python\nBash(description='sneaky')\n```\n"
        result = self.formatter.format_message(_msg(content, "user"))
        assert "[action]:" not in result
        assert result.startswith("[user]: ")

    def test_user_message_empty(self) -> None:
        assert self.formatter.format_message(_msg("", "user")) == "[user]: (empty)"

    def test_format_messages(self) -> None:
        result = self.formatter.format_messages([_msg("Hello"), _msg("World", "user")])
        assert "[model]: Hello" in result and "[user]: World" in result
        assert "..." not in result

    def test_format_messages_not_verbose(self) -> None:
        config_module.DEFAULT_CONFIG.agent.verbose = False
        assert self.formatter.format_messages([_msg("Hello")]) == ""


if __name__ == "__main__":
    unittest.main()

"""Tests for _set_prompt handling of curly braces in argument values."""

import unittest

from kiss.core.kiss_agent import KISSAgent
from kiss.tests.conftest import requires_gemini_api_key

TEST_MODEL = "gemini-3-flash-preview"


@requires_gemini_api_key
class TestSetPromptCurlyBraces(unittest.TestCase):
    def _run_set_prompt(self, prompt_template: str, arguments: dict[str, str] | None = None) -> str:
        agent = KISSAgent("CurlyBraceTest")
        agent._reset(
            model_name=TEST_MODEL,
            is_agentic=False,
            max_steps=1,
            max_budget=1.0,
            model_config=None,
        )
        agent._set_prompt(prompt_template, arguments)
        return str(agent.messages[-1]["content"])

    def test_argument_with_curly_braces(self) -> None:
        result = self._run_set_prompt(
            "Review this code:\n{code}",
            {"code": "fn main() { println!(); }"},
        )
        self.assertIn("fn main() { println!(); }", result)

    def test_argument_with_nested_curly_braces(self) -> None:
        result = self._run_set_prompt(
            "Code:\n{code}",
            {"code": "if x { if y { z } }"},
        )
        self.assertIn("if x { if y { z } }", result)

    def test_argument_with_json_value(self) -> None:
        result = self._run_set_prompt(
            "Parse this JSON:\n{data}",
            {"data": '{"key": "value", "nested": {"a": 1}}'},
        )
        self.assertIn('{"key": "value", "nested": {"a": 1}}', result)

    def test_argument_with_only_braces(self) -> None:
        result = self._run_set_prompt(
            "Content: {val}",
            {"val": "{}"},
        )
        self.assertIn("{}", result)

    def test_argument_without_curly_braces(self) -> None:
        result = self._run_set_prompt(
            "Hello {name}!",
            {"name": "Alice"},
        )
        self.assertIn("Hello Alice!", result)

    def test_no_arguments(self) -> None:
        result = self._run_set_prompt("Plain prompt with no placeholders")
        self.assertIn("Plain prompt with no placeholders", result)

    def test_multiple_arguments_with_braces(self) -> None:
        result = self._run_set_prompt(
            "File1:\n{file1}\nFile2:\n{file2}",
            {"file1": "func() { return 1; }", "file2": "func() { return 2; }"},
        )
        self.assertIn("func() { return 1; }", result)
        self.assertIn("func() { return 2; }", result)

    def test_argument_with_unbalanced_braces(self) -> None:
        result = self._run_set_prompt(
            "Content: {val}",
            {"val": "open { but no close"},
        )
        self.assertIn("open { but no close", result)

    def test_template_with_literal_braces_and_arguments(self) -> None:
        result = self._run_set_prompt(
            "Use {literal} braces and {arg}",
            {"arg": "value with { braces }"},
        )
        self.assertIn("{literal}", result)
        self.assertIn("value with { braces }", result)

    def test_template_with_curly_braces_no_placeholders(self) -> None:
        result = self._run_set_prompt("function() { return 42; }")
        self.assertIn("function() { return 42; }", result)


if __name__ == "__main__":
    unittest.main()

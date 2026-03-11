"""Tests for GEPA _sanitize_prompt_template placeholder validation."""

import unittest

from kiss.agents.gepa import GEPA


def dummy_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
    return "result", []


class TestSanitizePromptTemplate(unittest.TestCase):
    """Tests for relaxed placeholder validation in _sanitize_prompt_template."""

    def _make_gepa(self, template: str = "Solve {task} with {input}") -> GEPA:
        return GEPA(
            agent_wrapper=dummy_wrapper,
            initial_prompt_template=template,
            max_generations=1,
            population_size=1,
        )

    def test_malformed_braces_returns_fallback(self):
        """Prompt with unparseable format strings returns fallback."""
        gepa = self._make_gepa()
        result = gepa._sanitize_prompt_template("Do {task} with {bad", fallback="fallback")
        self.assertEqual(result, "fallback")

    def test_single_placeholder_template(self):
        """Works with a single-placeholder initial template."""
        gepa = self._make_gepa("Answer {question}")
        # Subset (empty) is OK
        result = gepa._sanitize_prompt_template("Just answer", fallback="fallback")
        self.assertEqual(result, "Just answer")
        # Original is OK
        result = gepa._sanitize_prompt_template("Answer {question} now", fallback="fallback")
        self.assertEqual(result, "Answer {question} now")
        # New placeholder rejected
        result = gepa._sanitize_prompt_template("Answer {other}", fallback="fallback")
        self.assertEqual(result, "fallback")


if __name__ == "__main__":
    unittest.main()

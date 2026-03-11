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

    def test_all_placeholders_present_accepted(self):
        """Prompt with all original placeholders is accepted."""
        gepa = self._make_gepa()
        result = gepa._sanitize_prompt_template("Do {task} using {input}", fallback="fallback")
        self.assertEqual(result, "Do {task} using {input}")

    def test_subset_of_placeholders_accepted(self):
        """Prompt with only a subset of placeholders is accepted (the key change)."""
        gepa = self._make_gepa()
        # Only {task} present, {input} removed — should now be accepted
        result = gepa._sanitize_prompt_template("Do {task} only", fallback="fallback")
        self.assertEqual(result, "Do {task} only")

    def test_no_placeholders_accepted(self):
        """Prompt with zero placeholders is accepted (empty set is subset of any set)."""
        gepa = self._make_gepa()
        result = gepa._sanitize_prompt_template("Do something", fallback="fallback")
        self.assertEqual(result, "Do something")

    def test_new_placeholder_rejected(self):
        """Prompt with a new placeholder not in the original is rejected."""
        gepa = self._make_gepa()
        result = gepa._sanitize_prompt_template("Do {task} with {new_thing}", fallback="fallback")
        self.assertEqual(result, "fallback")

    def test_superset_placeholders_rejected(self):
        """Prompt with all original + extra placeholder is rejected."""
        gepa = self._make_gepa()
        result = gepa._sanitize_prompt_template(
            "Do {task} with {input} and {extra}", fallback="fallback"
        )
        self.assertEqual(result, "fallback")

    def test_malformed_braces_returns_fallback(self):
        """Prompt with unparseable format strings returns fallback."""
        gepa = self._make_gepa()
        result = gepa._sanitize_prompt_template("Do {task} with {bad", fallback="fallback")
        self.assertEqual(result, "fallback")

    def test_quoted_placeholders_normalized(self):
        """Quoted placeholders like {'task'} are normalized to {task}."""
        gepa = self._make_gepa()
        result = gepa._sanitize_prompt_template(
            "Do {'task'} with {'input'}", fallback="fallback"
        )
        self.assertEqual(result, "Do {task} with {input}")

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

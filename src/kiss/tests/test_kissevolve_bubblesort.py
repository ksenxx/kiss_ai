# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for kissevolve_bubblesort helper functions.

These tests verify the evaluation and analysis functions without running
the full evolution process, making them fast to execute.
"""

import unittest

import pytest

from kiss.agents.kiss_evolve.kiss_evolve import KISSEvolve
from kiss.scripts.kissevolve_bubblesort import (
    INITIAL_CODE,
    analyze_complexity,
    evaluate_correctness_of_code,
    evaluate_performance_of_code,
)
from kiss.tests.conftest import requires_gemini_api_key


class TestEvaluateCorrectnessOfCode(unittest.TestCase):
    """Tests for evaluate_correctness_of_code function."""

    def test_incorrect_sorting_code(self):
        """Test incorrect sorting implementation fails."""
        incorrect_code = """
def sort_array(arr):
    return arr  # Does not sort
"""
        result = evaluate_correctness_of_code(incorrect_code)
        self.assertFalse(result["correctness"])
        self.assertIn("Incorrect result", result["error"])

    def test_missing_function(self):
        """Test code without sort_array function fails."""
        no_function_code = """
def other_function(arr):
    return arr
"""
        result = evaluate_correctness_of_code(no_function_code)
        self.assertFalse(result["correctness"])
        self.assertIn("not found", result["error"])

    def test_syntax_error(self):
        """Test code with syntax error fails."""
        syntax_error_code = """
def sort_array(arr)  # Missing colon
    return arr
"""
        result = evaluate_correctness_of_code(syntax_error_code)
        self.assertFalse(result["correctness"])
        self.assertIn("Syntax error", result["error"])

    def test_runtime_error(self):
        """Test code with runtime error fails."""
        runtime_error_code = """
def sort_array(arr):
    raise ValueError("Intentional error")
"""
        result = evaluate_correctness_of_code(runtime_error_code)
        self.assertFalse(result["correctness"])
        self.assertIn("Exception", result["error"])


class TestEvaluatePerformanceOfCode(unittest.TestCase):
    """Tests for evaluate_performance_of_code function."""

    def test_missing_function_returns_zero_fitness(self):
        """Test code without sort_array function returns zero fitness."""
        no_function_code = """
def other_function(arr):
    return arr
"""
        result = evaluate_performance_of_code(no_function_code)
        self.assertEqual(result["fitness"], 0.0)
        self.assertIn("not found", result["error"])

    def test_syntax_error_returns_zero_fitness(self):
        """Test code with syntax error returns zero fitness."""
        syntax_error_code = """
def sort_array(arr)
    return arr
"""
        result = evaluate_performance_of_code(syntax_error_code)
        self.assertEqual(result["fitness"], 0.0)
        self.assertIn("Syntax error", result["error"])


class TestAnalyzeComplexity(unittest.TestCase):
    """Tests for analyze_complexity function."""

    def test_invalid_times_by_size(self):
        """Test with invalid times_by_size returns Unknown."""
        metrics = {"times_by_size": "not a dict"}
        self.assertEqual(analyze_complexity(metrics), "Unknown")

        metrics = {"times_by_size": {100: 0.1}}  # Only one size
        self.assertEqual(analyze_complexity(metrics), "Unknown")

    def test_quadratic_complexity_detection(self):
        """Test detection of O(n²) complexity."""
        metrics = {
            "times_by_size": {
                100: 0.01,
                200: 0.04,  # 4x time for 2x size (n²)
            }
        }
        result = analyze_complexity(metrics)
        self.assertIn("n²", result)

    def test_nlogn_complexity_detection(self):
        """Test detection of O(n log n) complexity."""
        import math

        base_time = 0.01
        metrics = {
            "times_by_size": {
                100: base_time,
                200: base_time * 2 * math.log2(2),  # ~2.0x for 2x size
            }
        }
        result = analyze_complexity(metrics)
        self.assertIn("n log n", result)


@requires_gemini_api_key
class TestKISSEvolveIntegration(unittest.TestCase):
    """Integration tests for KISSEvolve with sorting algorithms using real LLM."""

    def _test_correctness(self, code: str) -> bool:
        result = evaluate_correctness_of_code(code)
        return bool(result.get("correctness", False))

    @pytest.mark.timeout(300)  # 5 minutes for LLM-based evolution
    def test_kiss_evolve_improves_performance(self):
        """Test that KISSEvolve improves performance over the initial code."""
        from kiss.agents.kiss import get_run_simple_coding_agent

        # Evaluate initial code performance
        initial_result = evaluate_performance_of_code(INITIAL_CODE)
        initial_fitness = initial_result["fitness"]

        # Create KISSEvolve optimizer with real model
        optimizer = KISSEvolve(
            code_agent_wrapper=get_run_simple_coding_agent(self._test_correctness),
            initial_code=INITIAL_CODE,
            evaluation_fn=evaluate_performance_of_code,
            model_names=[("gemini-2.5-flash", 1.0)],
            extra_coding_instructions="""
- You **MUST NOT** use any builtin or standard library functions to sort.
- You **MUST** implement your best algorithm to sort the array.
""",
            population_size=3,  # Small population for fast test
            max_generations=2,  # Few generations for fast test
            mutation_rate=0.8,
            elite_size=1,
        )

        # Run evolution
        best_variant = optimizer.evolve()

        # Verify that the best variant has improved fitness
        self.assertGreater(
            best_variant.fitness,
            initial_fitness,
            f"Expected evolved fitness ({best_variant.fitness:.4f}) to be greater "
            f"than initial fitness ({initial_fitness:.4f})",
        )

        # Verify the evolved code is correct
        correctness_result = evaluate_correctness_of_code(best_variant.code)
        self.assertTrue(
            correctness_result["correctness"],
            f"Evolved code is not correct: {correctness_result.get('error', 'Unknown error')}",
        )

    @pytest.mark.timeout(300)  # 5 minutes for LLM-based evolution
    def test_kiss_evolve_population_stats(self):
        """Test that KISSEvolve maintains valid population statistics."""
        from kiss.agents.kiss import get_run_simple_coding_agent

        optimizer = KISSEvolve(
            code_agent_wrapper=get_run_simple_coding_agent(self._test_correctness),
            initial_code=INITIAL_CODE,
            evaluation_fn=evaluate_performance_of_code,
            model_names=[("gemini-2.5-flash", 1.0)],
            extra_coding_instructions="""
- You **MUST NOT** use any builtin or standard library functions to sort.
- You **MUST** implement your best algorithm to sort the array.
""",
            population_size=4,
            max_generations=1,
            mutation_rate=0.7,
            elite_size=1,
        )

        # Run evolution
        optimizer.evolve()

        # Check population stats
        stats = optimizer.get_population_stats()
        self.assertGreater(stats["size"], 0)
        self.assertGreater(stats["best_fitness"], 0.0)
        self.assertGreaterEqual(stats["best_fitness"], stats["avg_fitness"])
        self.assertLessEqual(stats["worst_fitness"], stats["avg_fitness"])


if __name__ == "__main__":
    unittest.main()

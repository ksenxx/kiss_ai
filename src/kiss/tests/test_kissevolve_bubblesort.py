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


class TestEvaluateCorrectnessOfCode(unittest.TestCase):
    """Tests for evaluate_correctness_of_code function."""

    def test_initial_code_is_correct(self):
        """Test that the initial bubble sort code is correct."""
        result = evaluate_correctness_of_code(INITIAL_CODE)
        self.assertTrue(result["correctness"])
        self.assertEqual(result["error"], "None")

    def test_correct_sorting_code(self):
        """Test correct sorting implementation passes.

        Verifies that a correct sorting implementation using Python's
        built-in sorted() passes correctness evaluation.

        Returns:
            None
        """
        correct_code = """
def sort_array(arr):
    return sorted(arr)
"""
        result = evaluate_correctness_of_code(correct_code)
        self.assertTrue(result["correctness"])

    def test_incorrect_sorting_code(self):
        """Test incorrect sorting implementation fails.

        Verifies that code that doesn't sort correctly fails evaluation
        with an appropriate error message.

        Returns:
            None
        """
        incorrect_code = """
def sort_array(arr):
    return arr  # Does not sort
"""
        result = evaluate_correctness_of_code(incorrect_code)
        self.assertFalse(result["correctness"])
        self.assertIn("Incorrect result", result["error"])

    def test_missing_function(self):
        """Test code without sort_array function fails.

        Verifies that code missing the required sort_array function
        fails with a 'not found' error message.

        Returns:
            None
        """
        no_function_code = """
def other_function(arr):
    return arr
"""
        result = evaluate_correctness_of_code(no_function_code)
        self.assertFalse(result["correctness"])
        self.assertIn("not found", result["error"])

    def test_syntax_error(self):
        """Test code with syntax error fails.

        Verifies that code with Python syntax errors fails evaluation
        with a 'Syntax error' message.

        Returns:
            None
        """
        syntax_error_code = """
def sort_array(arr)  # Missing colon
    return arr
"""
        result = evaluate_correctness_of_code(syntax_error_code)
        self.assertFalse(result["correctness"])
        self.assertIn("Syntax error", result["error"])

    def test_runtime_error(self):
        """Test code with runtime error fails.

        Verifies that code that raises an exception at runtime fails
        evaluation with an 'Exception' error message.

        Returns:
            None
        """
        runtime_error_code = """
def sort_array(arr):
    raise ValueError("Intentional error")
"""
        result = evaluate_correctness_of_code(runtime_error_code)
        self.assertFalse(result["correctness"])
        self.assertIn("Exception", result["error"])


class TestEvaluatePerformanceOfCode(unittest.TestCase):
    """Tests for evaluate_performance_of_code function."""

    def test_initial_code_performance(self):
        """Test that initial code returns valid performance metrics.

        Verifies that INITIAL_CODE performance evaluation returns positive
        fitness and includes expected metric keys.

        Returns:
            None
        """
        result = evaluate_performance_of_code(INITIAL_CODE)
        self.assertGreater(result["fitness"], 0.0)
        self.assertIn("total_time_seconds", result["metrics"])
        self.assertIn("times_by_size", result["metrics"])
        self.assertIn("growth_ratio", result["metrics"])
        self.assertIn("test_sizes", result["artifacts"])

    def test_faster_algorithm_has_higher_fitness(self):
        """Test that a faster algorithm gets higher fitness.

        Verifies that Python's built-in sorted() achieves higher fitness
        than the O(n²) bubble sort implementation.

        Returns:
            None
        """
        # Python's built-in sort is O(n log n) and highly optimized
        fast_code = """
def sort_array(arr):
    return sorted(arr)
"""
        fast_result = evaluate_performance_of_code(fast_code)
        slow_result = evaluate_performance_of_code(INITIAL_CODE)

        # The built-in sort should be significantly faster
        self.assertGreater(fast_result["fitness"], slow_result["fitness"])

    def test_missing_function_returns_zero_fitness(self):
        """Test code without sort_array function returns zero fitness.

        Verifies that code missing sort_array gets zero fitness with
        a 'not found' error message.

        Returns:
            None
        """
        no_function_code = """
def other_function(arr):
    return arr
"""
        result = evaluate_performance_of_code(no_function_code)
        self.assertEqual(result["fitness"], 0.0)
        self.assertIn("not found", result["error"])

    def test_syntax_error_returns_zero_fitness(self):
        """Test code with syntax error returns zero fitness.

        Verifies that code with syntax errors gets zero fitness with
        a 'Syntax error' message.

        Returns:
            None
        """
        syntax_error_code = """
def sort_array(arr)
    return arr
"""
        result = evaluate_performance_of_code(syntax_error_code)
        self.assertEqual(result["fitness"], 0.0)
        self.assertIn("Syntax error", result["error"])


class TestAnalyzeComplexity(unittest.TestCase):
    """Tests for analyze_complexity function."""

    def test_empty_metrics(self):
        """Test with empty metrics returns Unknown.

        Verifies that analyze_complexity handles empty or None metrics
        gracefully by returning 'Unknown'.

        Returns:
            None
        """
        self.assertEqual(analyze_complexity({}), "Unknown")
        self.assertEqual(analyze_complexity(None), "Unknown")

    def test_missing_times_by_size(self):
        """Test with missing times_by_size returns Unknown.

        Verifies that analyze_complexity returns 'Unknown' when the
        required times_by_size key is missing from metrics.

        Returns:
            None
        """
        metrics = {"total_time_seconds": 1.0}
        self.assertEqual(analyze_complexity(metrics), "Unknown")

    def test_invalid_times_by_size(self):
        """Test with invalid times_by_size returns Unknown.

        Verifies that analyze_complexity returns 'Unknown' when
        times_by_size is not a valid dict or has insufficient data.

        Returns:
            None
        """
        metrics = {"times_by_size": "not a dict"}
        self.assertEqual(analyze_complexity(metrics), "Unknown")

        metrics = {"times_by_size": {100: 0.1}}  # Only one size
        self.assertEqual(analyze_complexity(metrics), "Unknown")

    def test_quadratic_complexity_detection(self):
        """Test detection of O(n²) complexity.

        Verifies that analyze_complexity correctly identifies quadratic
        time complexity based on timing growth patterns.

        Returns:
            None
        """
        # Simulate O(n²) growth: time grows as square of size
        metrics = {
            "times_by_size": {
                100: 0.01,
                200: 0.04,  # 4x time for 2x size (n²)
            }
        }
        result = analyze_complexity(metrics)
        self.assertIn("n²", result)

    def test_nlogn_complexity_detection(self):
        """Test detection of O(n log n) complexity.

        Verifies that analyze_complexity correctly identifies n log n
        time complexity based on timing growth patterns.

        Returns:
            None
        """
        # Simulate O(n log n) growth
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


class TestInitialCode(unittest.TestCase):
    """Tests for the INITIAL_CODE constant."""

    def test_initial_code_contains_sort_array(self):
        """Test that INITIAL_CODE defines sort_array function.

        Verifies that INITIAL_CODE contains the required function signature
        for the sort_array function.

        Returns:
            None
        """
        self.assertIn("def sort_array", INITIAL_CODE)

    def test_initial_code_is_bubble_sort(self):
        """Test that INITIAL_CODE implements bubble sort.

        Verifies that INITIAL_CODE contains characteristic bubble sort
        patterns: nested loops and element comparison/swap logic.

        Returns:
            None
        """
        self.assertIn("for i in range", INITIAL_CODE)
        self.assertIn("for j in range", INITIAL_CODE)
        self.assertIn("arr[j] > arr[j + 1]", INITIAL_CODE)


class TestKISSEvolveIntegration(unittest.TestCase):
    """Integration tests for KISSEvolve with sorting algorithms using real LLM."""

    def _test_correctness(self, code: str) -> bool:
        """Test if code is correct, returning a boolean.

        Args:
            code: Python code string containing sort_array function.

        Returns:
            True if code passes correctness evaluation, False otherwise.
        """
        result = evaluate_correctness_of_code(code)
        return bool(result.get("correctness", False))

    @pytest.mark.timeout(300)  # 5 minutes for LLM-based evolution
    def test_kiss_evolve_improves_performance(self):
        """Test that KISSEvolve improves performance over the initial code.

        Verifies that after evolution, the best variant has higher fitness
        than the initial code and remains correct.

        Returns:
            None
        """
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
        """Test that KISSEvolve maintains valid population statistics.

        Verifies that population stats (size, best/avg/worst fitness) are
        valid after evolution completes.

        Returns:
            None
        """
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

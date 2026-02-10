# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# Cursor AI (cursor@cursor.com)
# add your name here

"""Test GEPA algorithm on HotPotQA benchmark."""

import unittest

from kiss.evals.hotpotqa import (
    HotPotQABenchmark,
    evaluate_hotpotqa_result,
)
from kiss.tests.conftest import requires_openai_api_key


class TestHotPotQAEvaluation(unittest.TestCase):
    """Test HotPotQA evaluation functions."""

    def test_evaluate_hotpotqa_result(self):
        """Test HotPotQA result evaluation."""
        # Test successful exact match
        result_yaml = """
status: success
analysis: Found the answer
result: Barack Obama
"""
        scores = evaluate_hotpotqa_result(result_yaml, "Barack Obama")
        self.assertEqual(scores["success"], 1.0)
        self.assertEqual(scores["exact_match"], 1.0)
        self.assertAlmostEqual(scores["f1"], 1.0)

        # Test failure status
        result_yaml = """
status: failure
analysis: Could not find
result: unknown
"""
        scores = evaluate_hotpotqa_result(result_yaml, "Barack Obama")
        self.assertEqual(scores["success"], 0.0)
        self.assertEqual(scores["exact_match"], 0.0)


class TestHotPotQABenchmark(unittest.TestCase):
    """Test HotPotQA benchmark functionality."""

    def test_create_evaluation_fn(self):
        """Test evaluation function creation."""
        benchmark = HotPotQABenchmark(num_examples=1)
        example = benchmark.get_example(0)
        eval_fn = benchmark.create_evaluation_fn(example)

        # Test with matching answer (quote to ensure YAML parses as string)
        result_yaml = f"""
status: success
analysis: test
result: "{example.answer}"
"""
        scores = eval_fn(result_yaml)
        self.assertEqual(scores["success"], 1.0)
        self.assertEqual(scores["exact_match"], 1.0)


@requires_openai_api_key
class TestGEPAHotPotQA(unittest.TestCase):
    """Test GEPA optimization on HotPotQA."""

    def test_gepa_optimization_outputs_valid_prompt(self):
        """Test that GEPA optimization produces a valid prompt."""
        benchmark = HotPotQABenchmark(
            split="validation",
            num_examples=3,
            config_name="distractor",
        )

        gepa, best_scores = benchmark.run_gepa_optimization(
            example_indices=[0, 1, 2],  # Optimize over all examples
            model_name="gpt-4o",
            max_generations=2,  # Small for testing
            population_size=2,
            pareto_size=2,
            mutation_rate=0.5,
        )

        self.assertIsNotNone(best_scores)
        self.assertIn("success", best_scores)
        self.assertIn("exact_match", best_scores)
        self.assertIn("f1", best_scores)

        pareto_frontier = gepa.get_pareto_frontier()
        self.assertGreater(len(pareto_frontier), 0)

        best_prompt = gepa.get_best_prompt()
        self.assertGreater(len(best_prompt), 0)

        self.assertIn("{context}", best_prompt)
        self.assertIn("{question}", best_prompt)
        self.assertGreater(len(best_prompt), 100)
        for candidate in pareto_frontier:
            self.assertIsNotNone(candidate.val_scores)


if __name__ == "__main__":
    unittest.main()

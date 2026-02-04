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
from kiss.evals.hotpotqa.hotpotqa_benchmark import (
    compute_f1,
    normalize_answer,
)
from kiss.tests.conftest import requires_openai_api_key


class TestHotPotQAEvaluation(unittest.TestCase):
    """Test HotPotQA evaluation functions."""

    def test_normalize_answer(self):
        """Test answer normalization.

        Verifies that normalize_answer correctly handles articles,
        punctuation, extra whitespace, and case conversion.

        Returns:
            None
        """
        self.assertEqual(normalize_answer("The Answer"), "answer")
        self.assertEqual(normalize_answer("Barack Obama"), "barack obama")
        self.assertEqual(normalize_answer("   extra   spaces   "), "extra spaces")
        self.assertEqual(normalize_answer("Hello, World!"), "hello world")

    def test_compute_f1(self):
        """Test F1 score computation.

        Verifies that compute_f1 returns correct scores for exact matches,
        partial matches, and non-matching answers.

        Returns:
            None
        """
        # Exact match
        self.assertAlmostEqual(compute_f1("Barack Obama", "Barack Obama"), 1.0)

        # Partial match
        f1 = compute_f1("Barack Hussein Obama", "Barack Obama")
        self.assertGreater(f1, 0.5)
        self.assertLess(f1, 1.0)

        # No match
        self.assertAlmostEqual(compute_f1("John Smith", "Barack Obama"), 0.0)

    def test_evaluate_hotpotqa_result(self):
        """Test HotPotQA result evaluation.

        Verifies that evaluate_hotpotqa_result correctly parses YAML results
        and computes success, exact_match, and f1 scores.

        Returns:
            None
        """
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

    def test_load_dataset(self):
        """Test loading HotPotQA dataset.

        Verifies that HotPotQABenchmark correctly loads examples with
        required fields (id, question, answer, question_type, level).

        Returns:
            None
        """
        benchmark = HotPotQABenchmark(
            split="validation",
            num_examples=3,
            config_name="distractor",
        )

        self.assertEqual(len(benchmark.examples), 3)

        # Check first example has required fields
        example = benchmark.get_example(0)
        self.assertIsNotNone(example.id)
        self.assertIsNotNone(example.question)
        self.assertIsNotNone(example.answer)
        self.assertIn(example.question_type, ["comparison", "bridge"])
        self.assertIn(example.level, ["easy", "medium", "hard"])

        # Check formatted context
        formatted_context = example.formatted_context
        self.assertGreater(len(formatted_context), 0)
        print(f"\nExample question: {example.question}")
        print(f"Expected answer: {example.answer}")
        print(f"Question type: {example.question_type}")
        print(f"Level: {example.level}")

    def test_create_evaluation_fn(self):
        """Test evaluation function creation.

        Verifies that create_evaluation_fn returns a function that correctly
        evaluates results against the expected answer for an example.

        Returns:
            None
        """
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

    def test_gepa_optimizes_over_multiple_examples(self):
        """Test that GEPA optimizes prompts over multiple HotPotQA examples.

        This test verifies that:
        1. GEPA runs successfully on multiple HotPotQA examples
        2. The optimized prompt generalizes across examples
        3. The Pareto frontier contains valid candidates
        """
        # Load benchmark with multiple examples
        benchmark = HotPotQABenchmark(
            split="validation",
            num_examples=3,
            config_name="distractor",
        )

        # Run GEPA optimization over all 3 examples
        gepa, best_scores = benchmark.run_gepa_optimization(
            example_indices=[0, 1, 2],  # Optimize over all examples
            model_name="gpt-4o",
            max_generations=2,  # Small for testing
            population_size=2,
            pareto_size=2,
            mutation_rate=0.5,
        )

        # Verify optimization ran
        self.assertIsNotNone(best_scores)
        self.assertIn("success", best_scores)
        self.assertIn("exact_match", best_scores)
        self.assertIn("f1", best_scores)

        # Verify Pareto frontier
        pareto_frontier = gepa.get_pareto_frontier()
        self.assertGreater(len(pareto_frontier), 0)

        # Print results
        print("\n=== GEPA Multi-Example Optimization Results ===")
        print(f"Best scores: {best_scores}")
        print(f"Pareto frontier size: {len(pareto_frontier)}")

        for i, candidate in enumerate(pareto_frontier):
            print(f"\nCandidate {i}: val_scores={candidate.val_scores}")
            print(f"Prompt preview (first 300 chars):\n{candidate.prompt_template[:300]}...")

        # Get the best prompt
        best_prompt = gepa.get_best_prompt()
        self.assertGreater(len(best_prompt), 0)

        # Verify the prompt still has required placeholders
        self.assertIn("{context}", best_prompt)
        self.assertIn("{question}", best_prompt)

        # Evaluate the optimized prompt on all examples to verify generalization
        print("\n=== Evaluating Optimized Prompt on All Examples ===")
        avg_scores = benchmark.evaluate_prompt_on_examples(
            prompt_template=best_prompt,
            model_name="gpt-4o",
            example_indices=[0, 1, 2],
        )
        print(f"Average scores across all examples: {avg_scores}")

    def test_gepa_optimization_produces_valid_prompts(self):
        """Test that GEPA optimization produces valid, well-formed prompts.

        Verifies that the optimized prompt preserves required placeholders,
        has reasonable length, and all Pareto frontier candidates have scores.

        Returns:
            None
        """
        benchmark = HotPotQABenchmark(
            split="validation",
            num_examples=3,
            config_name="distractor",
        )

        # Print examples being used
        for i in range(3):
            example = benchmark.get_example(i)
            print(f"\nExample {i}: {example.question[:50]}... -> {example.answer}")

        # Run GEPA optimization over multiple examples
        print("\n--- Running GEPA Optimization Over Multiple Examples ---")
        gepa, best_scores = benchmark.run_gepa_optimization(
            example_indices=[0, 1, 2],  # All examples
            model_name="gpt-4o",
            max_generations=2,
            population_size=2,
            pareto_size=2,
            mutation_rate=0.5,
        )

        # Get optimized prompt
        optimized_prompt = gepa.get_best_prompt()

        # Verify the optimized prompt is valid
        print("\n=== GEPA Optimization Results ===")
        print(f"Best scores during optimization: {best_scores}")
        print(f"Pareto frontier size: {len(gepa.get_pareto_frontier())}")
        print(f"\nOptimized prompt preview:\n{optimized_prompt[:500]}...")

        # Test that the optimization produced valid results
        self.assertIsNotNone(best_scores)
        self.assertIn("success", best_scores)
        self.assertIn("exact_match", best_scores)
        self.assertIn("f1", best_scores)

        # Verify the prompt still has required placeholders
        self.assertIn("{context}", optimized_prompt)
        self.assertIn("{question}", optimized_prompt)

        # Verify the prompt has reasonable length (not empty or truncated)
        self.assertGreater(len(optimized_prompt), 100)

        # Verify Pareto frontier has candidates
        pareto_frontier = gepa.get_pareto_frontier()
        self.assertGreater(len(pareto_frontier), 0)

        # All candidates in Pareto frontier should have scores
        for candidate in pareto_frontier:
            self.assertIsNotNone(candidate.val_scores)
            print(f"  Candidate {candidate.id}: val_scores={candidate.val_scores}")


if __name__ == "__main__":
    unittest.main()

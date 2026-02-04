# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Comprehensive integration tests for GEPA - all tests call gepa.optimize()."""

import unittest

import kiss.core.utils as utils
from kiss.agents.gepa import GEPA, PromptCandidate
from kiss.core.kiss_agent import KISSAgent


def create_agent_wrapper_with_expected(model_name: str = "gpt-4o"):
    """Create an agent wrapper that embeds expected answer for evaluation.

    Args:
        model_name: Name of the LLM model to use for agent calls.

    Returns:
        Tuple of (agent_wrapper function, call_counter list for tracking calls).
    """
    import json

    call_counter = [0]

    def agent_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
        """Run agent with real LLM call, embedding expected answer and capturing trajectory.

        Args:
            prompt_template: The prompt template to use for the agent.
            arguments: Dict of arguments including '_expected' for the expected answer.

        Returns:
            Tuple of (result string with expected/actual, trajectory list).
        """
        expected = arguments.get("_expected", "")
        # Remove _expected from arguments passed to agent
        agent_args = {k: v for k, v in arguments.items() if not k.startswith("_")}

        call_counter[0] += 1
        agent = KISSAgent(f"Test Agent {call_counter[0]}")
        result = agent.run(
            model_name=model_name,
            prompt_template=prompt_template,
            arguments=agent_args,
            tools=[utils.finish],
        )

        # Capture trajectory for better reflection
        trajectory = json.loads(agent.get_trajectory())

        return f"EXPECTED:{expected}\nRESULT:{result}", trajectory

    return agent_wrapper, call_counter


def create_evaluation_fn():
    """Create evaluation function that extracts expected and compares.

    Returns:
        Evaluation function that takes a result string and returns metric scores.
    """
    import yaml

    def evaluation_fn(result: str) -> dict[str, float]:
        """Evaluate result by comparing expected and actual answers.

        Args:
            result: Result string in format 'EXPECTED:...\nRESULT:...'

        Returns:
            Dict with 'success' and 'correct' scores (0.0 or 1.0).
        """
        try:
            if result.startswith("EXPECTED:"):
                parts = result.split("\nRESULT:", 1)
                expected = parts[0].replace("EXPECTED:", "").strip().lower()
                actual_result = parts[1] if len(parts) > 1 else ""

                result_dict = yaml.safe_load(actual_result) or {}
                actual = str(result_dict.get("result", "")).strip().lower()

                return {
                    "success": 1.0 if result_dict.get("status") == "success" else 0.0,
                    "correct": 1.0 if expected in actual or actual in expected else 0.0,
                }
        except Exception:
            pass
        return {"success": 0.0, "correct": 0.0}

    return evaluation_fn


class TestGEPAOptimizeBasic(unittest.TestCase):
    """Test basic GEPA optimization functionality."""

    def test_optimize_returns_prompt_candidate(self):
        """Test that optimize returns a valid PromptCandidate.

        Verifies that GEPA.optimize() returns a non-None PromptCandidate
        with preserved placeholders and non-empty prompt template.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = """Solve the math problem: {problem}
Call finish with status='success' and result=<answer>."""

        train_examples = [
            {"problem": "2 + 2", "_expected": "4"},
            {"problem": "5 - 3", "_expected": "2"},
            {"problem": "3 * 3", "_expected": "9"},
            {"problem": "8 / 2", "_expected": "4"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        best = gepa.optimize(train_examples)

        self.assertIsNotNone(best)
        self.assertIsInstance(best, PromptCandidate)
        self.assertIn("{problem}", best.prompt_template)
        self.assertGreater(len(best.prompt_template), 0)

    def test_optimize_with_dev_val_split(self):
        """Test that optimize correctly splits examples into dev/val.

        Verifies that training examples are properly divided between
        development and validation sets according to dev_val_split ratio.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Answer: {question}\nCall finish with result."

        train_examples = [
            {"question": "Capital of France?", "_expected": "paris"},
            {"question": "Capital of Japan?", "_expected": "tokyo"},
            {"question": "Capital of Italy?", "_expected": "rome"},
            {"question": "Capital of Spain?", "_expected": "madrid"},
            {"question": "Capital of Germany?", "_expected": "berlin"},
            {"question": "Capital of UK?", "_expected": "london"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            dev_val_split=0.5,
            mutation_rate=0.0,
        )

        best = gepa.optimize(train_examples)

        # Check split happened correctly
        total = len(gepa.dev_examples) + len(gepa.val_examples)
        self.assertEqual(total, len(train_examples))
        self.assertGreater(len(gepa.dev_examples), 0)
        self.assertGreater(len(gepa.val_examples), 0)
        self.assertIsNotNone(best)

    def test_optimize_with_minibatch_size(self):
        """Test optimize with dev minibatch size parameter.

        Verifies that optimization works correctly when using
        dev_minibatch_size to limit examples per evaluation batch.

        Returns:
            None
        """
        agent_wrapper, call_counter = create_agent_wrapper_with_expected()

        initial_prompt = "Compute: {expr}\nCall finish with the result."

        train_examples = [
            {"expr": "1+1", "_expected": "2"},
            {"expr": "2+2", "_expected": "4"},
            {"expr": "3+3", "_expected": "6"},
            {"expr": "4+4", "_expected": "8"},
            {"expr": "5+5", "_expected": "10"},
            {"expr": "6+6", "_expected": "12"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        best = gepa.optimize(train_examples, dev_minibatch_size=2)

        self.assertIsNotNone(best)
        self.assertGreater(call_counter[0], 0)


class TestGEPAOptimizeWithMutation(unittest.TestCase):
    """Test GEPA optimization with mutation/reflection enabled."""

    def test_optimize_with_mutation_creates_new_candidates(self):
        """Test that mutation creates new candidate prompts via reflection.

        Verifies that with high mutation rate, GEPA creates new candidate
        prompts through the reflection/mutation mechanism.

        Returns:
            None
        """
        agent_wrapper, call_counter = create_agent_wrapper_with_expected()

        initial_prompt = "Problem: {problem}\nSolve and call finish."

        train_examples = [
            {"problem": "What is 7 + 5?", "_expected": "12"},
            {"problem": "What is 9 - 4?", "_expected": "5"},
            {"problem": "What is 6 * 2?", "_expected": "12"},
            {"problem": "What is 10 / 2?", "_expected": "5"},
            {"problem": "What is 8 + 3?", "_expected": "11"},
            {"problem": "What is 15 - 6?", "_expected": "9"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=2,
            population_size=2,
            pareto_size=2,
            mutation_rate=0.8,  # High mutation rate
            dev_val_split=0.5,
        )

        best = gepa.optimize(train_examples, dev_minibatch_size=2)

        self.assertIsNotNone(best)
        self.assertIn("{problem}", best.prompt_template)
        # With mutation, we should have created new candidates
        self.assertGreaterEqual(gepa._candidate_id, 1)
        print(f"\nCandidates created: {gepa._candidate_id}")
        print(f"LLM calls: {call_counter[0]}")

    def test_optimize_multiple_generations(self):
        """Test optimization across multiple generations.

        Verifies that GEPA can run through multiple generations and
        maintain valid validation scores and Pareto frontier.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Q: {q}\nAnswer and call finish with result."

        train_examples = [
            {"q": "2*3", "_expected": "6"},
            {"q": "4*2", "_expected": "8"},
            {"q": "5*1", "_expected": "5"},
            {"q": "3*3", "_expected": "9"},
            {"q": "6*2", "_expected": "12"},
            {"q": "7*1", "_expected": "7"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=3,
            population_size=2,
            pareto_size=3,
            mutation_rate=0.5,
            dev_val_split=0.5,
        )

        best = gepa.optimize(train_examples, dev_minibatch_size=2)

        self.assertIsNotNone(best)
        self.assertIsInstance(best.val_scores, dict)
        print(f"\nBest val_scores: {best.val_scores}")
        print(f"Pareto frontier size: {len(gepa.get_pareto_frontier())}")


class TestGEPAOptimizeParetoFrontier(unittest.TestCase):
    """Test GEPA Pareto frontier management through optimize."""

    def test_optimize_builds_pareto_frontier(self):
        """Test that optimize builds a Pareto frontier.

        Verifies that optimization creates a valid Pareto frontier with
        PromptCandidate instances that preserve required placeholders.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Calculate {calc} and call finish with the answer."

        train_examples = [
            {"calc": "10+5", "_expected": "15"},
            {"calc": "20-8", "_expected": "12"},
            {"calc": "4*5", "_expected": "20"},
            {"calc": "18/3", "_expected": "6"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=2,
            population_size=2,
            pareto_size=3,
            mutation_rate=0.5,
        )

        gepa.optimize(train_examples)

        frontier = gepa.get_pareto_frontier()
        self.assertIsInstance(frontier, list)
        self.assertGreater(len(frontier), 0)
        self.assertLessEqual(len(frontier), gepa.pareto_size)

        # Verify frontier contains PromptCandidates
        for candidate in frontier:
            self.assertIsInstance(candidate, PromptCandidate)
            self.assertIn("{calc}", candidate.prompt_template)

    def test_optimize_pareto_respects_size_limit(self):
        """Test that Pareto frontier respects max size during optimize.

        Verifies that the Pareto frontier never exceeds the configured
        pareto_size limit during optimization.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Solve: {x}\nCall finish with answer."

        train_examples = [
            {"x": "1+1", "_expected": "2"},
            {"x": "2+2", "_expected": "4"},
            {"x": "3+3", "_expected": "6"},
            {"x": "4+4", "_expected": "8"},
            {"x": "5+5", "_expected": "10"},
            {"x": "6+6", "_expected": "12"},
        ]

        pareto_size = 2
        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=2,
            population_size=3,
            pareto_size=pareto_size,
            mutation_rate=0.6,
        )

        gepa.optimize(train_examples)

        self.assertLessEqual(len(gepa.pareto_frontier), pareto_size)


class TestGEPAOptimizeWithMerge(unittest.TestCase):
    """Test GEPA optimization with merge functionality."""

    def test_optimize_with_merge_enabled(self):
        """Test optimization with structural merge enabled.

        Verifies that optimization works correctly when use_merge is True,
        allowing candidates to be merged for potential improvement.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Task: {task}\nProvide answer via finish tool."

        train_examples = [
            {"task": "Add 3 and 7", "_expected": "10"},
            {"task": "Subtract 5 from 12", "_expected": "7"},
            {"task": "Multiply 4 by 3", "_expected": "12"},
            {"task": "Divide 16 by 4", "_expected": "4"},
            {"task": "Add 8 and 2", "_expected": "10"},
            {"task": "Subtract 3 from 9", "_expected": "6"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=2,
            population_size=2,
            pareto_size=3,
            mutation_rate=0.7,
            use_merge=True,
            merge_val_overlap_floor=1,
            max_merge_invocations=3,
        )

        best = gepa.optimize(train_examples, dev_minibatch_size=2)

        self.assertIsNotNone(best)
        self.assertIn("{task}", best.prompt_template)
        print(f"\nMerge invocations: {gepa._merge_invocations}")

    def test_optimize_merge_disabled(self):
        """Test optimization with merge explicitly disabled.

        Verifies that when use_merge is False, no merge operations
        are invoked during optimization.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Compute {op} and respond with finish."

        train_examples = [
            {"op": "5+5", "_expected": "10"},
            {"op": "6-2", "_expected": "4"},
            {"op": "3*4", "_expected": "12"},
            {"op": "9/3", "_expected": "3"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=2,
            population_size=2,
            mutation_rate=0.5,
            use_merge=False,
        )

        best = gepa.optimize(train_examples)

        self.assertIsNotNone(best)
        # Merge should not have been invoked
        self.assertEqual(gepa._merge_invocations, 0)


class TestGEPAOptimizeGetters(unittest.TestCase):
    """Test GEPA getter methods after optimization."""

    def test_get_best_prompt_after_optimize(self):
        """Test get_best_prompt returns valid prompt after optimize.

        Verifies that get_best_prompt() returns a string with preserved
        placeholders and reasonable length after optimization completes.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Question: {q}\nUse finish to answer."

        train_examples = [
            {"q": "What is 2+2?", "_expected": "4"},
            {"q": "What is 3+3?", "_expected": "6"},
            {"q": "What is 4+4?", "_expected": "8"},
            {"q": "What is 5+5?", "_expected": "10"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        gepa.optimize(train_examples)

        best_prompt = gepa.get_best_prompt()

        self.assertIsInstance(best_prompt, str)
        self.assertIn("{q}", best_prompt)
        self.assertGreater(len(best_prompt), 10)

    def test_get_pareto_frontier_is_copy(self):
        """Test get_pareto_frontier returns a copy.

        Verifies that get_pareto_frontier() returns a copy of the frontier
        list, so modifications don't affect internal state.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Solve {p} and finish."

        train_examples = [
            {"p": "1+1", "_expected": "2"},
            {"p": "2+2", "_expected": "4"},
            {"p": "3+3", "_expected": "6"},
            {"p": "4+4", "_expected": "8"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        gepa.optimize(train_examples)

        frontier = gepa.get_pareto_frontier()
        original_len = len(gepa.pareto_frontier)

        # Modify returned list
        frontier.clear()

        # Internal state should be unchanged
        self.assertEqual(len(gepa.pareto_frontier), original_len)


class TestGEPAOptimizeMultiplePlaceholders(unittest.TestCase):
    """Test GEPA with prompts containing multiple placeholders."""

    def test_optimize_with_two_placeholders(self):
        """Test optimization with prompt containing two placeholders.

        Verifies that GEPA preserves multiple placeholders (context, question)
        in the prompt template during optimization.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = """Context: {context}
Question: {question}
Answer using the finish tool."""

        train_examples = [
            {
                "context": "Paris is the capital of France.",
                "question": "What is the capital of France?",
                "_expected": "paris",
            },
            {
                "context": "Tokyo is in Japan.",
                "question": "Where is Tokyo?",
                "_expected": "japan",
            },
            {
                "context": "Water boils at 100 degrees Celsius.",
                "question": "At what temperature does water boil?",
                "_expected": "100",
            },
            {
                "context": "The sun is a star.",
                "question": "What is the sun?",
                "_expected": "star",
            },
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        best = gepa.optimize(train_examples)

        self.assertIsNotNone(best)
        # Both placeholders should be preserved
        self.assertIn("{context}", best.prompt_template)
        self.assertIn("{question}", best.prompt_template)

    def test_optimize_sanitizes_invalid_placeholders(self):
        """Test that optimize sanitizes prompts with invalid placeholders.

        Verifies that valid placeholders are preserved even when mutations
        might introduce changes to the prompt template.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        # Initial prompt has valid placeholder
        initial_prompt = "Input: {input}\nCall finish with result."

        train_examples = [
            {"input": "Add 5+5", "_expected": "10"},
            {"input": "Add 3+3", "_expected": "6"},
            {"input": "Add 2+2", "_expected": "4"},
            {"input": "Add 1+1", "_expected": "2"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=2,
            population_size=2,
            mutation_rate=0.5,
        )

        best = gepa.optimize(train_examples)

        # Valid placeholder should be preserved
        self.assertIn("{input}", best.prompt_template)


class TestGEPAOptimizeScoring(unittest.TestCase):
    """Test GEPA scoring behavior through optimization."""

    def test_optimize_tracks_val_scores(self):
        """Test that optimization tracks validation scores.

        Verifies that the best candidate has populated val_scores dict
        and per_item_val_scores list after optimization.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Evaluate: {expr}\nFinish with the answer."

        train_examples = [
            {"expr": "10+10", "_expected": "20"},
            {"expr": "15-5", "_expected": "10"},
            {"expr": "5*4", "_expected": "20"},
            {"expr": "30/6", "_expected": "5"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        best = gepa.optimize(train_examples)

        # Should have validation scores
        self.assertIsInstance(best.val_scores, dict)
        self.assertGreater(len(best.val_scores), 0)

        # Per-item scores should be tracked
        self.assertIsInstance(best.per_item_val_scores, list)

    def test_optimize_with_perfect_score_eval(self):
        """Test optimization behavior with perfect scores.

        Verifies that optimization completes successfully when the
        evaluation function always returns perfect scores.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Simple: {s}\nJust call finish with success."

        train_examples = [
            {"s": "test1", "_expected": ""},
            {"s": "test2", "_expected": ""},
            {"s": "test3", "_expected": ""},
            {"s": "test4", "_expected": ""},
        ]

        # Evaluation always returns perfect score
        def perfect_eval(result: str) -> dict[str, float]:
            return {"score": 1.0}

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=perfect_eval,
            max_generations=2,
            population_size=2,
            mutation_rate=0.5,
            perfect_score=1.0,
        )

        best = gepa.optimize(train_examples)

        self.assertIsNotNone(best)
        # With perfect scores, optimization should complete successfully


class TestGEPAOptimizeEdgeCases(unittest.TestCase):
    """Test GEPA optimization edge cases."""

    def test_optimize_with_minimal_examples(self):
        """Test optimization with minimal number of examples.

        Verifies that optimization works with only 2 examples (minimum
        needed for dev/val split to have at least one example each).

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Do: {x}\nCall finish."

        # Only 2 examples - minimum for dev/val split
        train_examples = [
            {"x": "1+1", "_expected": "2"},
            {"x": "2+2", "_expected": "4"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        best = gepa.optimize(train_examples)

        self.assertIsNotNone(best)
        # Both dev and val should have at least one example
        self.assertGreaterEqual(len(gepa.dev_examples), 1)
        self.assertGreaterEqual(len(gepa.val_examples), 1)

    def test_optimize_with_single_generation(self):
        """Test optimization with single generation.

        Verifies that with max_generations=1 and mutation_rate=0,
        the returned prompt equals the initial prompt.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Calc: {c}\nFinish with answer."

        train_examples = [
            {"c": "7+3", "_expected": "10"},
            {"c": "8-2", "_expected": "6"},
            {"c": "3*5", "_expected": "15"},
            {"c": "12/4", "_expected": "3"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        best = gepa.optimize(train_examples)

        self.assertIsNotNone(best)
        self.assertEqual(best.prompt_template, initial_prompt)

    def test_optimize_returns_best_from_frontier(self):
        """Test that optimize returns best candidate from frontier.

        Verifies that the returned best candidate is found in either
        the Pareto frontier or current candidates list.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Math: {m}\nAnswer via finish."

        train_examples = [
            {"m": "100+100", "_expected": "200"},
            {"m": "50-25", "_expected": "25"},
            {"m": "10*10", "_expected": "100"},
            {"m": "80/8", "_expected": "10"},
            {"m": "25+25", "_expected": "50"},
            {"m": "60-30", "_expected": "30"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=2,
            population_size=2,
            pareto_size=3,
            mutation_rate=0.5,
        )

        best = gepa.optimize(train_examples)

        # Best should be from frontier or candidates
        frontier = gepa.get_pareto_frontier()
        all_candidates = frontier + gepa.candidates

        found = any(c.id == best.id for c in all_candidates)
        self.assertTrue(found or best.prompt_template == initial_prompt)


class TestGEPAOptimizeUseBestPrompt(unittest.TestCase):
    """Test using the optimized prompt with an agent."""

    def test_use_optimized_prompt_with_agent(self):
        """Test that optimized prompt works with a real agent.

        Verifies that the optimized prompt can be used with a KISSAgent
        to produce valid results on new problems.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = """You are a helpful math assistant.
Problem: {problem}
Solve and call finish with status='success' and result=<answer>."""

        train_examples = [
            {"problem": "What is 5 + 7?", "_expected": "12"},
            {"problem": "What is 10 - 3?", "_expected": "7"},
            {"problem": "What is 4 * 6?", "_expected": "24"},
            {"problem": "What is 20 / 5?", "_expected": "4"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )

        gepa.optimize(train_examples)

        # Get the best prompt and use it
        best_prompt = gepa.get_best_prompt()

        # Use the optimized prompt with a new agent
        agent = KISSAgent("Verification Agent")
        result = agent.run(
            model_name="gpt-4o",
            prompt_template=best_prompt,
            arguments={"problem": "What is 3 + 3?"},
            tools=[utils.finish],
        )

        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        print(f"\nVerification result: {result[:300]}...")


if __name__ == "__main__":
    unittest.main()

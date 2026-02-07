# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Tests for GEPA progress callback functionality."""

import unittest

import kiss.core.utils as utils
from kiss.agents.gepa import GEPA, GEPAPhase, GEPAProgress
from kiss.core.kiss_agent import KISSAgent
from kiss.tests.conftest import requires_openai_api_key


def create_agent_wrapper_with_expected(model_name: str = "gpt-4o", max_steps: int = 10):
    """Create an agent wrapper that embeds expected answer for evaluation."""
    import json

    call_counter = [0]

    def agent_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
        """Run agent with real LLM call, embedding expected answer and capturing trajectory."""
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
            max_steps=max_steps,
        )

        # Capture trajectory for better reflection
        trajectory = json.loads(agent.get_trajectory())

        return f"EXPECTED:{expected}\nRESULT:{result}", trajectory

    return agent_wrapper, call_counter


def create_deterministic_agent_wrapper():
    """Create a deterministic agent wrapper for testing callback behavior."""
    call_counter = [0]

    def agent_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
        """Simple agent that returns deterministic results based on input."""
        expected = arguments.get("_expected", "unknown")
        call_counter[0] += 1
        # Build a simple trajectory showing the prompt was processed
        trajectory = [
            {"role": "user", "content": f"Prompt: {prompt_template[:50]}..."},
            {"role": "assistant", "content": f"Processing arguments: {list(arguments.keys())}"},
        ]
        # Return a result that matches expected (simulates successful agent)
        return f"EXPECTED:{expected}\nRESULT:result={expected}", trajectory

    return agent_wrapper, call_counter


def create_evaluation_fn():
    """Create evaluation function that extracts expected and compares."""
    import yaml

    def evaluation_fn(result: str) -> dict[str, float]:
        """Evaluate result by comparing expected and actual answers."""
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


def create_simple_evaluation_fn():
    """Create a simple evaluation function for testing callback behavior."""

    def evaluation_fn(result: str) -> dict[str, float]:
        """Evaluate result based on format and content matching."""
        try:
            if "EXPECTED:" in result and "RESULT:" in result:
                parts = result.split("\nRESULT:", 1)
                expected = parts[0].replace("EXPECTED:", "").strip().lower()
                actual = parts[1].strip().lower() if len(parts) > 1 else ""
                # Check if expected value appears in the result
                if expected in actual:
                    return {"accuracy": 1.0}
                elif "result=" in actual:
                    return {"accuracy": 0.8}
        except Exception:
            pass
        return {"accuracy": 0.2}

    return evaluation_fn


class TestGEPAProgressDataclass(unittest.TestCase):
    """Test GEPAProgress dataclass structure."""

    def test_progress_dataclass_has_required_fields(self):
        """Test that GEPAProgress has all required fields."""
        progress = GEPAProgress(
            generation=0,
            max_generations=10,
            phase=GEPAPhase.DEV_EVALUATION,
            candidate_id=1,
            candidate_index=0,
            population_size=4,
            best_val_accuracy=0.85,
            current_val_accuracy=0.80,
            pareto_frontier_size=2,
            num_candidates_evaluated=3,
            message="Testing progress",
        )

        self.assertEqual(progress.generation, 0)
        self.assertEqual(progress.max_generations, 10)
        self.assertEqual(progress.phase, GEPAPhase.DEV_EVALUATION)
        self.assertEqual(progress.candidate_id, 1)
        self.assertEqual(progress.candidate_index, 0)
        self.assertEqual(progress.population_size, 4)
        self.assertEqual(progress.best_val_accuracy, 0.85)
        self.assertEqual(progress.current_val_accuracy, 0.80)
        self.assertEqual(progress.pareto_frontier_size, 2)
        self.assertEqual(progress.num_candidates_evaluated, 3)
        self.assertEqual(progress.message, "Testing progress")


class TestGEPAProgressCallbackDeterministic(unittest.TestCase):
    """Test GEPA progress callback with deterministic agent (fast, no LLM calls)."""

    def test_callback_is_called_during_optimization(self):
        """Test that progress callback is called during optimization."""
        agent_wrapper, _ = create_deterministic_agent_wrapper()

        initial_prompt = "Solve: {problem}"

        train_examples = [
            {"problem": "2 + 2", "_expected": "4"},
            {"problem": "5 - 3", "_expected": "2"},
            {"problem": "3 * 3", "_expected": "9"},
            {"problem": "8 / 2", "_expected": "4"},
        ]

        # Collect progress updates
        progress_updates: list[GEPAProgress] = []

        def progress_callback(progress: GEPAProgress) -> None:
            progress_updates.append(progress)

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_simple_evaluation_fn(),
            max_generations=2,
            population_size=1,
            mutation_rate=0.0,  # No mutations for simpler test
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should have received progress updates
        self.assertGreater(len(progress_updates), 0)

    def test_callback_receives_all_phases(self):
        """Test that callback receives updates for dev and val phases."""
        agent_wrapper, _ = create_deterministic_agent_wrapper()

        initial_prompt = "Calculate: {expr}"

        train_examples = [
            {"expr": "1+1", "_expected": "2"},
            {"expr": "2+2", "_expected": "4"},
            {"expr": "3+3", "_expected": "6"},
            {"expr": "4+4", "_expected": "8"},
        ]

        phases_seen: set[GEPAPhase] = set()

        def progress_callback(progress: GEPAProgress) -> None:
            phases_seen.add(progress.phase)

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_simple_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should see dev and val evaluation phases
        self.assertIn(GEPAPhase.DEV_EVALUATION, phases_seen)
        self.assertIn(GEPAPhase.VAL_EVALUATION, phases_seen)

    def test_callback_tracks_generation_number(self):
        """Test that callback correctly reports generation numbers."""
        agent_wrapper, _ = create_deterministic_agent_wrapper()

        initial_prompt = "Answer: {q}"

        train_examples = [
            {"q": "A", "_expected": "a"},
            {"q": "B", "_expected": "b"},
            {"q": "C", "_expected": "c"},
            {"q": "D", "_expected": "d"},
        ]

        generations_seen: set[int] = set()

        def progress_callback(progress: GEPAProgress) -> None:
            generations_seen.add(progress.generation)
            # Verify max_generations is consistent
            self.assertEqual(progress.max_generations, 3)

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_simple_evaluation_fn(),
            max_generations=3,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should see all generations
        self.assertEqual(generations_seen, {0, 1, 2})

    def test_callback_receives_validation_accuracy(self):
        """Test that callback receives validation accuracy updates."""
        agent_wrapper, _ = create_deterministic_agent_wrapper()

        initial_prompt = "Eval: {x}"

        train_examples = [
            {"x": "1", "_expected": "1"},
            {"x": "2", "_expected": "2"},
            {"x": "3", "_expected": "3"},
            {"x": "4", "_expected": "4"},
        ]

        val_accuracies: list[float | None] = []
        best_accuracies: list[float | None] = []

        def progress_callback(progress: GEPAProgress) -> None:
            if progress.phase == GEPAPhase.VAL_EVALUATION:
                val_accuracies.append(progress.current_val_accuracy)
                best_accuracies.append(progress.best_val_accuracy)

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_simple_evaluation_fn(),
            max_generations=2,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should have received some accuracy values
        self.assertGreater(len(val_accuracies), 0)
        # After first evaluation, best_val_accuracy should be set
        non_none_best = [a for a in best_accuracies if a is not None]
        self.assertGreater(len(non_none_best), 0)

    def test_callback_not_called_when_none(self):
        """Test that no errors occur when callback is None."""
        agent_wrapper, _ = create_deterministic_agent_wrapper()

        initial_prompt = "Test: {t}"

        train_examples = [
            {"t": "a", "_expected": "a"},
            {"t": "b", "_expected": "b"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_simple_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=None,  # Explicitly None
        )

        # Should not raise any errors
        best = gepa.optimize(train_examples)
        self.assertIsNotNone(best)

    def test_callback_receives_pareto_frontier_size(self):
        """Test that callback reports Pareto frontier size."""
        agent_wrapper, _ = create_deterministic_agent_wrapper()

        initial_prompt = "Compute: {c}"

        train_examples = [
            {"c": "1", "_expected": "1"},
            {"c": "2", "_expected": "2"},
            {"c": "3", "_expected": "3"},
            {"c": "4", "_expected": "4"},
        ]

        pareto_sizes: list[int] = []

        def progress_callback(progress: GEPAProgress) -> None:
            pareto_sizes.append(progress.pareto_frontier_size)

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_simple_evaluation_fn(),
            max_generations=2,
            population_size=2,
            pareto_size=3,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Pareto size should be tracked
        self.assertGreater(len(pareto_sizes), 0)

    def test_callback_messages_are_descriptive(self):
        """Test that callback messages contain useful information."""
        agent_wrapper, _ = create_deterministic_agent_wrapper()

        initial_prompt = "Do: {d}"

        train_examples = [
            {"d": "x", "_expected": "x"},
            {"d": "y", "_expected": "y"},
            {"d": "z", "_expected": "z"},
            {"d": "w", "_expected": "w"},
        ]

        messages: list[str] = []

        def progress_callback(progress: GEPAProgress) -> None:
            messages.append(progress.message)

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_simple_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should have descriptive messages
        non_empty_messages = [m for m in messages if m]
        self.assertGreater(len(non_empty_messages), 0)

        # Messages should mention evaluation or candidates
        found_relevant = any(
            "eval" in m.lower() or "candidate" in m.lower() for m in messages
        )
        self.assertTrue(found_relevant)


def create_imperfect_evaluation_fn():
    """Create evaluation function that always returns imperfect scores."""

    def evaluation_fn(result: str) -> dict[str, float]:
        """Return imperfect scores to ensure reflection is triggered."""
        # Return scores that are good but not perfect, ensuring reflection happens
        return {"accuracy": 0.7, "completeness": 0.6}

    return evaluation_fn


@requires_openai_api_key
class TestGEPAProgressCallbackWithMutation(unittest.TestCase):
    """Test progress callback with mutation/reflection phases.

    These tests require API keys as mutation triggers LLM-based reflection.
    """

    def test_callback_receives_reflection_and_mutation_gating(self):
        """Test that callback receives reflection and mutation gating phases."""
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Problem: {p}\nCall finish with result."

        # Minimal examples to reduce LLM calls
        train_examples = [
            {"p": "1+1", "_expected": "2"},
            {"p": "2+2", "_expected": "4"},
        ]

        phases_seen: set[GEPAPhase] = set()

        def progress_callback(progress: GEPAProgress) -> None:
            phases_seen.add(progress.phase)

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_imperfect_evaluation_fn(),
            max_generations=2,
            population_size=1,
            pareto_size=1,
            mutation_rate=1.0,
            reflection_model="gpt-4o",
            use_merge=False,
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        self.assertIn(GEPAPhase.DEV_EVALUATION, phases_seen)
        self.assertIn(GEPAPhase.VAL_EVALUATION, phases_seen)
        self.assertIn(GEPAPhase.REFLECTION, phases_seen)
        self.assertIn(GEPAPhase.MUTATION_GATING, phases_seen)


@requires_openai_api_key
class TestGEPAProgressCallbackWithMerge(unittest.TestCase):
    """Test progress callback with merge functionality.

    These tests require API keys as merge tests use mutation which triggers LLM reflection.
    """

    def test_callback_receives_merge_phase(self):
        """Test that callback receives MERGE phase updates when merge is enabled."""
        # Use agent with varying results to build diverse pareto frontier
        call_count = [0]

        def varying_agent(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
            """Agent that returns different results to build diverse Pareto frontier."""
            expected = arguments.get("_expected", "unknown")
            call_count[0] += 1
            # Return slightly different results to create diverse candidates
            suffix = "a" if call_count[0] % 2 == 0 else "b"
            trajectory = [
                {"role": "user", "content": f"Processing: {expected}"},
                {"role": "assistant", "content": f"Result variant: {suffix}"},
            ]
            return f"EXPECTED:{expected}\nRESULT:result={expected}{suffix}", trajectory

        initial_prompt = "Calc: {c}"

        # Minimal examples - need at least 4 for dev/val split to have 2 each
        train_examples = [
            {"c": "1+1", "_expected": "2"},
            {"c": "2+2", "_expected": "4"},
            {"c": "3+3", "_expected": "6"},
            {"c": "4+4", "_expected": "8"},
        ]

        phases_seen: set[GEPAPhase] = set()

        def progress_callback(progress: GEPAProgress) -> None:
            phases_seen.add(progress.phase)

        # Create evaluation function that returns varying scores based on result
        # This helps create diverse candidates with different per-instance scores
        def varying_eval_fn(result: str) -> dict[str, float]:
            """Evaluation that returns varying scores to create diverse Pareto frontier."""
            if "a" in result:
                return {"accuracy": 0.8, "completeness": 0.5}
            elif "b" in result:
                return {"accuracy": 0.5, "completeness": 0.8}
            return {"accuracy": 0.6, "completeness": 0.6}

        # Use 3 generations with high mutation to build diverse pareto frontier
        # Merge requires at least 2 candidates in pareto frontier with val_overlap
        gepa = GEPA(
            agent_wrapper=varying_agent,
            initial_prompt_template=initial_prompt,
            evaluation_fn=varying_eval_fn,
            max_generations=3,
            population_size=3,
            pareto_size=4,  # Allow more candidates in pareto frontier
            mutation_rate=1.0,  # Always mutate to create diverse candidates
            use_merge=True,
            merge_val_overlap_floor=1,
            reflection_model="gpt-4o",  # Use OpenAI model for reflection
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should attempt merge phase when merge is enabled and pareto frontier >= 2
        self.assertIn(GEPAPhase.MERGE, phases_seen)


if __name__ == "__main__":
    unittest.main()

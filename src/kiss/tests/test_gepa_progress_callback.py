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
    """Create an agent wrapper that embeds expected answer for evaluation.

    Args:
        model_name: Name of the LLM model to use for agent calls.
        max_steps: Maximum steps for the agent (default: 10, keeps tests fast).

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
            max_steps=max_steps,
        )

        # Capture trajectory for better reflection
        trajectory = json.loads(agent.get_trajectory())

        return f"EXPECTED:{expected}\nRESULT:{result}", trajectory

    return agent_wrapper, call_counter


def create_mock_agent_wrapper():
    """Create a mock agent wrapper for fast testing without LLM calls.

    Returns:
        Tuple of (agent_wrapper function, call_counter list for tracking calls).
    """
    call_counter = [0]

    def agent_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
        """Mock agent that returns deterministic results.

        Args:
            prompt_template: The prompt template (ignored).
            arguments: Dict of arguments including '_expected' for the expected answer.

        Returns:
            Tuple of (result string, empty trajectory list).
        """
        expected = arguments.get("_expected", "unknown")
        call_counter[0] += 1
        # Return a result that sometimes matches expected
        return f"EXPECTED:{expected}\nRESULT:result={expected}", []

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


def create_mock_evaluation_fn():
    """Create a mock evaluation function for fast testing.

    Returns:
        Evaluation function that returns high scores for matching results.
    """

    def evaluation_fn(result: str) -> dict[str, float]:
        """Simple evaluation based on result format.

        Args:
            result: Result string.

        Returns:
            Dict with 'accuracy' score.
        """
        if "result=" in result:
            return {"accuracy": 0.8}
        return {"accuracy": 0.2}

    return evaluation_fn


class TestGEPAProgressDataclass(unittest.TestCase):
    """Test GEPAProgress dataclass structure."""

    def test_progress_dataclass_has_required_fields(self):
        """Test that GEPAProgress has all required fields.

        Verifies that the GEPAProgress dataclass can be instantiated
        with all expected fields.

        Returns:
            None
        """
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

    def test_progress_dataclass_default_values(self):
        """Test that GEPAProgress has correct default values.

        Verifies that optional fields have sensible defaults.

        Returns:
            None
        """
        progress = GEPAProgress(
            generation=1,
            max_generations=5,
            phase=GEPAPhase.REFLECTION,
        )

        self.assertIsNone(progress.candidate_id)
        self.assertIsNone(progress.candidate_index)
        self.assertEqual(progress.population_size, 0)
        self.assertIsNone(progress.best_val_accuracy)
        self.assertIsNone(progress.current_val_accuracy)
        self.assertEqual(progress.pareto_frontier_size, 0)
        self.assertEqual(progress.num_candidates_evaluated, 0)
        self.assertEqual(progress.message, "")


class TestGEPAPhaseEnum(unittest.TestCase):
    """Test GEPAPhase enum values."""

    def test_phase_enum_values(self):
        """Test that GEPAPhase has all expected values.

        Verifies that the enum contains all the phases of GEPA optimization.

        Returns:
            None
        """
        expected_phases = [
            "dev_evaluation",
            "val_evaluation",
            "reflection",
            "mutation_gating",
            "merge",
        ]

        for phase_value in expected_phases:
            phase = GEPAPhase(phase_value)
            self.assertIsNotNone(phase)

    def test_phase_enum_names(self):
        """Test that GEPAPhase enum names are accessible.

        Returns:
            None
        """
        self.assertEqual(GEPAPhase.DEV_EVALUATION.value, "dev_evaluation")
        self.assertEqual(GEPAPhase.VAL_EVALUATION.value, "val_evaluation")
        self.assertEqual(GEPAPhase.REFLECTION.value, "reflection")
        self.assertEqual(GEPAPhase.MUTATION_GATING.value, "mutation_gating")
        self.assertEqual(GEPAPhase.MERGE.value, "merge")


class TestGEPAProgressCallbackMock(unittest.TestCase):
    """Test GEPA progress callback with mock agent (no LLM calls)."""

    def test_callback_is_called_during_optimization(self):
        """Test that progress callback is called during optimization.

        Verifies that the callback function receives progress updates.

        Returns:
            None
        """
        agent_wrapper, _ = create_mock_agent_wrapper()

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
            evaluation_fn=create_mock_evaluation_fn(),
            max_generations=2,
            population_size=1,
            mutation_rate=0.0,  # No mutations for simpler test
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should have received progress updates
        self.assertGreater(len(progress_updates), 0)

    def test_callback_receives_all_phases(self):
        """Test that callback receives updates for dev and val phases.

        Verifies that DEV_EVALUATION and VAL_EVALUATION phases are reported.

        Returns:
            None
        """
        agent_wrapper, _ = create_mock_agent_wrapper()

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
            evaluation_fn=create_mock_evaluation_fn(),
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
        """Test that callback correctly reports generation numbers.

        Verifies that generation numbers increment correctly through optimization.

        Returns:
            None
        """
        agent_wrapper, _ = create_mock_agent_wrapper()

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
            evaluation_fn=create_mock_evaluation_fn(),
            max_generations=3,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should see all generations
        self.assertEqual(generations_seen, {0, 1, 2})

    def test_callback_receives_validation_accuracy(self):
        """Test that callback receives validation accuracy updates.

        Verifies that current_val_accuracy and best_val_accuracy are provided.

        Returns:
            None
        """
        agent_wrapper, _ = create_mock_agent_wrapper()

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
            evaluation_fn=create_mock_evaluation_fn(),
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
        """Test that no errors occur when callback is None.

        Verifies that optimization works without a callback.

        Returns:
            None
        """
        agent_wrapper, _ = create_mock_agent_wrapper()

        initial_prompt = "Test: {t}"

        train_examples = [
            {"t": "a", "_expected": "a"},
            {"t": "b", "_expected": "b"},
        ]

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_mock_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=None,  # Explicitly None
        )

        # Should not raise any errors
        best = gepa.optimize(train_examples)
        self.assertIsNotNone(best)

    def test_callback_receives_pareto_frontier_size(self):
        """Test that callback reports Pareto frontier size.

        Verifies that pareto_frontier_size is updated as candidates are evaluated.

        Returns:
            None
        """
        agent_wrapper, _ = create_mock_agent_wrapper()

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
            evaluation_fn=create_mock_evaluation_fn(),
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
        """Test that callback messages contain useful information.

        Verifies that progress messages describe what's happening.

        Returns:
            None
        """
        agent_wrapper, _ = create_mock_agent_wrapper()

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
            evaluation_fn=create_mock_evaluation_fn(),
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
    """Create evaluation function that always returns imperfect scores.

    This ensures that mutation/reflection is triggered (not skipped due to perfect scores).

    Returns:
        Evaluation function that returns scores < 1.0.
    """

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

    def test_callback_receives_reflection_phase(self):
        """Test that callback receives REFLECTION phase updates.

        Verifies that when mutation_rate > 0, reflection phases are reported.

        Returns:
            None
        """
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

        # Use imperfect eval to ensure reflection is triggered (not skipped due to perfect scores)
        # Use gpt-4o for reflection since test already requires OpenAI API key
        # Minimal config: 1 candidate, 2 generations to trigger reflection once
        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_imperfect_evaluation_fn(),
            max_generations=2,
            population_size=1,
            pareto_size=1,
            mutation_rate=1.0,  # Always mutate
            reflection_model="gpt-4o",  # Use OpenAI model for reflection
            use_merge=False,  # Disable merge to speed up test
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should see reflection phase with high mutation rate and imperfect scores
        self.assertIn(GEPAPhase.DEV_EVALUATION, phases_seen)
        self.assertIn(GEPAPhase.VAL_EVALUATION, phases_seen)
        self.assertIn(GEPAPhase.REFLECTION, phases_seen)

    def test_callback_receives_mutation_gating_phase(self):
        """Test that callback receives MUTATION_GATING phase updates.

        Verifies that mutation gating is reported during optimization.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = "Task: {t}\nCall finish with result."

        # Minimal examples to reduce LLM calls
        train_examples = [
            {"t": "1+1", "_expected": "2"},
            {"t": "2+2", "_expected": "4"},
        ]

        phases_seen: set[GEPAPhase] = set()

        def progress_callback(progress: GEPAProgress) -> None:
            phases_seen.add(progress.phase)

        # Use imperfect eval to ensure mutation gating is triggered
        # Use gpt-4o for reflection since test already requires OpenAI API key
        # Minimal config: 1 candidate, 2 generations to trigger mutation gating
        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_imperfect_evaluation_fn(),
            max_generations=2,
            population_size=1,
            mutation_rate=1.0,
            reflection_model="gpt-4o",  # Use OpenAI model for reflection
            use_merge=False,  # Disable merge to speed up test
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should see mutation gating phase
        self.assertIn(GEPAPhase.MUTATION_GATING, phases_seen)


@requires_openai_api_key
class TestGEPAProgressCallbackWithMerge(unittest.TestCase):
    """Test progress callback with merge functionality.

    These tests require API keys as merge tests use mutation which triggers LLM reflection.
    """

    def test_callback_receives_merge_phase(self):
        """Test that callback receives MERGE phase updates when merge is enabled.

        Verifies that merge attempts are reported via callback.

        Returns:
            None
        """
        # Use mock agent with varying results to build diverse pareto frontier
        call_count = [0]

        def varying_mock_agent(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
            """Mock agent that returns different results to build diverse pareto frontier."""
            expected = arguments.get("_expected", "unknown")
            call_count[0] += 1
            # Return slightly different results to create diverse candidates
            suffix = "a" if call_count[0] % 2 == 0 else "b"
            return f"EXPECTED:{expected}\nRESULT:result={expected}{suffix}", []

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
            """Evaluation that returns varying scores to create diverse pareto frontier."""
            if "a" in result:
                return {"accuracy": 0.8, "completeness": 0.5}
            elif "b" in result:
                return {"accuracy": 0.5, "completeness": 0.8}
            return {"accuracy": 0.6, "completeness": 0.6}

        # Use 3 generations with high mutation to build diverse pareto frontier
        # Merge requires at least 2 candidates in pareto frontier with val_overlap
        gepa = GEPA(
            agent_wrapper=varying_mock_agent,
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


@requires_openai_api_key
class TestGEPAProgressCallbackIntegration(unittest.TestCase):
    """Integration tests for GEPA progress callback with real LLM calls."""

    def test_callback_with_real_agent(self):
        """Test progress callback works with real LLM agent.

        Verifies that progress updates are received during actual optimization.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = """Solve: {problem}
Call finish with status='success' and result=<answer>."""

        train_examples = [
            {"problem": "2 + 2", "_expected": "4"},
            {"problem": "5 - 3", "_expected": "2"},
            {"problem": "3 * 3", "_expected": "9"},
            {"problem": "8 / 2", "_expected": "4"},
        ]

        progress_log: list[dict] = []

        def progress_callback(progress: GEPAProgress) -> None:
            progress_log.append(
                {
                    "generation": progress.generation,
                    "phase": progress.phase.value,
                    "best_val_accuracy": progress.best_val_accuracy,
                    "message": progress.message,
                }
            )

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )

        best = gepa.optimize(train_examples)

        # Should have progress updates
        self.assertGreater(len(progress_log), 0)
        self.assertIsNotNone(best)

        # Print progress for visibility
        print("\n--- Progress Log ---")
        for entry in progress_log:
            print(f"Gen {entry['generation']} | {entry['phase']} | "
                  f"Best: {entry['best_val_accuracy']} | {entry['message']}")

    def test_callback_accuracy_tracking(self):
        """Test that accuracy is tracked correctly through optimization.

        Verifies that best_val_accuracy updates appropriately.

        Returns:
            None
        """
        agent_wrapper, _ = create_agent_wrapper_with_expected()

        initial_prompt = """Problem: {problem}
Call finish with result."""

        train_examples = [
            {"problem": "What is 1+1?", "_expected": "2"},
            {"problem": "What is 2+2?", "_expected": "4"},
            {"problem": "What is 3+3?", "_expected": "6"},
            {"problem": "What is 4+4?", "_expected": "8"},
        ]

        best_accuracies: list[float | None] = []

        def progress_callback(progress: GEPAProgress) -> None:
            if progress.phase == GEPAPhase.VAL_EVALUATION:
                best_accuracies.append(progress.best_val_accuracy)

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_evaluation_fn(),
            max_generations=2,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        # Should have accuracy updates
        self.assertGreater(len(best_accuracies), 0)

        # Best accuracy should be set after first val evaluation
        non_none = [a for a in best_accuracies if a is not None]
        self.assertGreater(len(non_none), 0)
        print(f"\nAccuracy progression: {best_accuracies}")


class TestProgressCallbackRichExample(unittest.TestCase):
    """Example of using progress callback for Rich progress bars."""

    def test_progress_callback_for_rich_integration(self):
        """Demonstrate how to use callback with Rich progress bars.

        This test shows the pattern for integrating with Rich progress bars.
        Note: This doesn't actually use Rich, but shows the data flow.

        Returns:
            None
        """
        agent_wrapper, _ = create_mock_agent_wrapper()

        initial_prompt = "Solve: {problem}"

        train_examples = [
            {"problem": "a", "_expected": "a"},
            {"problem": "b", "_expected": "b"},
            {"problem": "c", "_expected": "c"},
            {"problem": "d", "_expected": "d"},
        ]

        # Simulate Rich progress tracking
        progress_state = {
            "current_gen": 0,
            "total_gens": 0,
            "phase": "",
            "best_accuracy": 0.0,
            "updates": 0,
        }

        def rich_style_callback(progress: GEPAProgress) -> None:
            """Callback that would update Rich progress bars.

            In real usage, this would call:
            - progress_bar.update(task_id, advance=1)
            - console.print(f"Generation {progress.generation}")
            - etc.
            """
            progress_state["current_gen"] = progress.generation
            progress_state["total_gens"] = progress.max_generations
            progress_state["phase"] = progress.phase.value
            if progress.best_val_accuracy is not None:
                progress_state["best_accuracy"] = progress.best_val_accuracy
            progress_state["updates"] += 1

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=initial_prompt,
            evaluation_fn=create_mock_evaluation_fn(),
            max_generations=3,
            population_size=2,
            mutation_rate=0.0,
            progress_callback=rich_style_callback,
        )

        gepa.optimize(train_examples)

        # Verify progress state was updated
        self.assertGreater(progress_state["updates"], 0)
        self.assertEqual(progress_state["total_gens"], 3)
        print(f"\nFinal progress state: {progress_state}")


if __name__ == "__main__":
    unittest.main()

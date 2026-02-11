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
        trajectory = [
            {"role": "user", "content": f"Prompt: {prompt_template[:50]}..."},
            {"role": "assistant", "content": f"Processing arguments: {list(arguments.keys())}"},
        ]
        return f"EXPECTED:{expected}\nRESULT:result={expected}", trajectory

    return agent_wrapper, call_counter


def create_simple_evaluation_fn():
    """Create a simple evaluation function for testing callback behavior."""

    def evaluation_fn(result: str) -> dict[str, float]:
        """Evaluate result based on format and content matching."""
        try:
            if "EXPECTED:" in result and "RESULT:" in result:
                parts = result.split("\nRESULT:", 1)
                expected = parts[0].replace("EXPECTED:", "").strip().lower()
                actual = parts[1].strip().lower() if len(parts) > 1 else ""
                if expected in actual:
                    return {"accuracy": 1.0}
                elif "result=" in actual:
                    return {"accuracy": 0.8}
        except Exception:
            pass
        return {"accuracy": 0.2}

    return evaluation_fn


def create_imperfect_evaluation_fn():
    """Create evaluation function that always returns imperfect scores."""

    def evaluation_fn(result: str) -> dict[str, float]:
        """Return imperfect scores to ensure reflection is triggered."""
        return {"accuracy": 0.7, "completeness": 0.6}

    return evaluation_fn


class TestGEPAProgressCallbackDeterministic(unittest.TestCase):
    """Test GEPA progress callback with deterministic agent (fast, no LLM calls)."""

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
            progress_callback=None,
        )

        best = gepa.optimize(train_examples)
        self.assertIsNotNone(best)


@requires_openai_api_key
class TestGEPAProgressCallbackWithMutation(unittest.TestCase):
    """Test progress callback with mutation/reflection phases.

    These tests require API keys as mutation triggers LLM-based reflection.
    """

    def test_callback_receives_reflection_and_mutation_gating(self):
        """Test that callback receives reflection and mutation gating phases."""
        agent_wrapper, _ = create_deterministic_agent_wrapper()

        initial_prompt = "Problem: {p}"

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
    """Test progress callback with merge functionality."""

    def test_callback_receives_merge_phase(self):
        """Test that callback receives MERGE phase updates when merge is enabled."""
        call_count = [0]

        def varying_agent(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
            """Agent that returns different results to build diverse Pareto frontier."""
            expected = arguments.get("_expected", "unknown")
            call_count[0] += 1
            suffix = "a" if call_count[0] % 2 == 0 else "b"
            trajectory = [
                {"role": "user", "content": f"Processing: {expected}"},
                {"role": "assistant", "content": f"Result variant: {suffix}"},
            ]
            return f"EXPECTED:{expected}\nRESULT:result={expected}{suffix}", trajectory

        initial_prompt = "Calc: {c}"

        train_examples = [
            {"c": "1+1", "_expected": "2"},
            {"c": "2+2", "_expected": "4"},
            {"c": "3+3", "_expected": "6"},
            {"c": "4+4", "_expected": "8"},
        ]

        phases_seen: set[GEPAPhase] = set()

        def progress_callback(progress: GEPAProgress) -> None:
            phases_seen.add(progress.phase)

        def varying_eval_fn(result: str) -> dict[str, float]:
            """Evaluation that returns varying scores to create diverse Pareto frontier."""
            if "a" in result:
                return {"accuracy": 0.8, "completeness": 0.5}
            elif "b" in result:
                return {"accuracy": 0.5, "completeness": 0.8}
            return {"accuracy": 0.6, "completeness": 0.6}

        gepa = GEPA(
            agent_wrapper=varying_agent,
            initial_prompt_template=initial_prompt,
            evaluation_fn=varying_eval_fn,
            max_generations=3,
            population_size=3,
            pareto_size=4,
            mutation_rate=1.0,
            use_merge=True,
            merge_val_overlap_floor=1,
            reflection_model="gpt-4o",
            progress_callback=progress_callback,
        )

        gepa.optimize(train_examples)

        self.assertIn(GEPAPhase.MERGE, phases_seen)


if __name__ == "__main__":
    unittest.main()

"""Tests for GEPA batched agent wrapper support."""

import unittest
from typing import Any

from kiss.agents.gepa import GEPA, GEPAPhase, GEPAProgress
from kiss.tests.conftest import requires_openai_api_key


def create_deterministic_sequential_wrapper():
    call_counter = [0]

    def agent_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
        expected = arguments.get("_expected", "unknown")
        call_counter[0] += 1
        trajectory = [{"role": "assistant", "content": f"result={expected}"}]
        return f"EXPECTED:{expected}\nRESULT:result={expected}", trajectory

    return agent_wrapper, call_counter


def create_deterministic_batched_wrapper():
    call_counter = [0]

    def batched_agent_wrapper(
        prompt_template: str, args_list: list[dict[str, str]]
    ) -> list[tuple[str, list]]:
        call_counter[0] += 1
        results = []
        for arguments in args_list:
            expected = arguments.get("_expected", "unknown")
            trajectory = [{"role": "assistant", "content": f"result={expected}"}]
            results.append((f"EXPECTED:{expected}\nRESULT:result={expected}", trajectory))
        return results

    return batched_agent_wrapper, call_counter


def simple_eval_fn(result: str) -> dict[str, float]:
    try:
        if "EXPECTED:" in result and "RESULT:" in result:
            parts = result.split("\nRESULT:", 1)
            expected = parts[0].replace("EXPECTED:", "").strip().lower()
            actual = parts[1].strip().lower() if len(parts) > 1 else ""
            if expected in actual:
                return {"accuracy": 1.0}
    except Exception:
        pass
    return {"accuracy": 0.2}


def imperfect_eval_fn(result: str) -> dict[str, float]:
    return {"accuracy": 0.7, "completeness": 0.6}


TRAIN_EXAMPLES = [
    {"t": "a", "_expected": "a"},
    {"t": "b", "_expected": "b"},
    {"t": "c", "_expected": "c"},
    {"t": "d", "_expected": "d"},
]

INITIAL_PROMPT = "Test: {t}"


class TestGEPABatchedBasic(unittest.TestCase):
    def test_batched_wrapper_produces_same_result_as_sequential(self):
        seq_wrapper, seq_counter = create_deterministic_sequential_wrapper()
        batch_wrapper, batch_counter = create_deterministic_batched_wrapper()

        gepa_seq = GEPA(
            agent_wrapper=seq_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=simple_eval_fn,
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )
        best_seq = gepa_seq.optimize(TRAIN_EXAMPLES)

        gepa_batch = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=batch_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=simple_eval_fn,
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )
        best_batch = gepa_batch.optimize(TRAIN_EXAMPLES)

        self.assertEqual(best_seq.prompt_template, best_batch.prompt_template)
        self.assertAlmostEqual(
            sum(best_seq.val_scores.values()),
            sum(best_batch.val_scores.values()),
            places=5,
        )
        # Sequential wrapper should NOT be called when batched is provided
        self.assertEqual(seq_counter[0], 4)  # only from first gepa_seq run (2 dev + 2 val)
        self.assertGreater(batch_counter[0], 0)

    def test_batched_wrapper_called_with_all_examples(self):
        received_batches: list[list[dict[str, str]]] = []

        def tracking_batch_wrapper(
            prompt_template: str, args_list: list[dict[str, str]]
        ) -> list[tuple[str, list]]:
            received_batches.append(args_list)
            results: list[tuple[str, Any]] = []
            for args in args_list:
                expected = args.get("_expected", "unknown")
                results.append(
                    (f"EXPECTED:{expected}\nRESULT:result={expected}", [])
                )
            return results

        seq_wrapper, _ = create_deterministic_sequential_wrapper()
        gepa = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=tracking_batch_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=simple_eval_fn,
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )
        gepa.optimize(TRAIN_EXAMPLES)

        # Should have received batches (dev + val evaluations)
        self.assertGreater(len(received_batches), 0)
        # Each batch should be a list of dicts
        for batch in received_batches:
            self.assertIsInstance(batch, list)
            for item in batch:
                self.assertIsInstance(item, dict)

    def test_batched_wrapper_not_used_when_none(self):
        seq_wrapper, seq_counter = create_deterministic_sequential_wrapper()
        gepa = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=None,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=simple_eval_fn,
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )
        gepa.optimize(TRAIN_EXAMPLES)
        self.assertGreater(seq_counter[0], 0)

    def test_batched_wrapper_single_example(self):
        batch_wrapper, batch_counter = create_deterministic_batched_wrapper()
        seq_wrapper, _ = create_deterministic_sequential_wrapper()

        gepa = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=batch_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=simple_eval_fn,
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )
        # With just 2 examples, dev=1, val=1 (after split)
        gepa.optimize([{"t": "x", "_expected": "x"}, {"t": "y", "_expected": "y"}])
        self.assertGreater(batch_counter[0], 0)


class TestGEPABatchedProgressCallback(unittest.TestCase):
    def test_progress_callback_works_with_batched_wrapper(self):
        batch_wrapper, _ = create_deterministic_batched_wrapper()
        seq_wrapper, _ = create_deterministic_sequential_wrapper()

        phases_seen: set[GEPAPhase] = set()

        def progress_callback(progress: GEPAProgress) -> None:
            phases_seen.add(progress.phase)

        gepa = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=batch_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=simple_eval_fn,
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
            progress_callback=progress_callback,
        )
        gepa.optimize(TRAIN_EXAMPLES)

        self.assertIn(GEPAPhase.DEV_EVALUATION, phases_seen)
        self.assertIn(GEPAPhase.VAL_EVALUATION, phases_seen)


class TestGEPABatchedMultiGeneration(unittest.TestCase):
    def test_batched_across_multiple_generations(self):
        batch_call_count = [0]

        def counting_batch_wrapper(
            prompt_template: str, args_list: list[dict[str, str]]
        ) -> list[tuple[str, list]]:
            batch_call_count[0] += 1
            results: list[tuple[str, Any]] = []
            for args in args_list:
                expected = args.get("_expected", "unknown")
                results.append(
                    (f"EXPECTED:{expected}\nRESULT:result={expected}", [])
                )
            return results

        seq_wrapper, seq_counter = create_deterministic_sequential_wrapper()

        gepa = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=counting_batch_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=simple_eval_fn,
            max_generations=2,
            population_size=1,
            mutation_rate=0.0,
            use_merge=False,
        )
        gepa.optimize(TRAIN_EXAMPLES)

        # Batched wrapper should be called for both generations
        self.assertGreater(batch_call_count[0], 2)
        # Sequential wrapper should not be called at all
        self.assertEqual(seq_counter[0], 0)

@requires_openai_api_key
class TestGEPABatchedWithMutation(unittest.TestCase):
    def test_batched_wrapper_with_mutation_and_reflection(self):
        batch_wrapper, batch_counter = create_deterministic_batched_wrapper()
        seq_wrapper, seq_counter = create_deterministic_sequential_wrapper()

        gepa = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=batch_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=imperfect_eval_fn,
            max_generations=2,
            population_size=1,
            pareto_size=1,
            mutation_rate=1.0,
            reflection_model="gpt-4o",
            use_merge=False,
        )
        best = gepa.optimize(TRAIN_EXAMPLES)
        self.assertIsNotNone(best)
        # Batched wrapper should have been used for dev, val, AND mutation gating
        self.assertGreater(batch_counter[0], 0)
        # Sequential should never be called
        self.assertEqual(seq_counter[0], 0)


class TestGEPABatchedEdgeCases(unittest.TestCase):
    def test_batched_with_varying_scores(self):
        call_count = [0]

        def varying_batch_wrapper(
            prompt_template: str, args_list: list[dict[str, str]]
        ) -> list[tuple[str, list]]:
            call_count[0] += 1
            results: list[tuple[str, Any]] = []
            for i, args in enumerate(args_list):
                expected = args.get("_expected", "unknown")
                suffix = "a" if (call_count[0] + i) % 2 == 0 else "b"
                results.append(
                    (f"EXPECTED:{expected}\nRESULT:result={expected}{suffix}", [])
                )
            return results

        def varying_eval_fn(result: str) -> dict[str, float]:
            if "a" in result:
                return {"accuracy": 0.8, "completeness": 0.5}
            elif "b" in result:
                return {"accuracy": 0.5, "completeness": 0.8}
            return {"accuracy": 0.6, "completeness": 0.6}

        seq_wrapper, _ = create_deterministic_sequential_wrapper()
        gepa = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=varying_batch_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=varying_eval_fn,
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )
        best = gepa.optimize(TRAIN_EXAMPLES)
        self.assertIsNotNone(best)
        self.assertTrue(len(best.val_scores) > 0)

    def test_prompt_template_passed_correctly_to_batched_wrapper(self):
        received_prompts: list[str] = []

        def tracking_batch_wrapper(
            prompt_template: str, args_list: list[dict[str, str]]
        ) -> list[tuple[str, list]]:
            received_prompts.append(prompt_template)
            return [
                (f"EXPECTED:{a.get('_expected', '')}\nRESULT:result={a.get('_expected', '')}", [])
                for a in args_list
            ]

        seq_wrapper, _ = create_deterministic_sequential_wrapper()
        gepa = GEPA(
            agent_wrapper=seq_wrapper,
            batched_agent_wrapper=tracking_batch_wrapper,
            initial_prompt_template=INITIAL_PROMPT,
            evaluation_fn=simple_eval_fn,
            max_generations=1,
            population_size=1,
            mutation_rate=0.0,
        )
        gepa.optimize(TRAIN_EXAMPLES)
        # All calls should receive the initial prompt template
        for prompt in received_prompts:
            self.assertEqual(prompt, INITIAL_PROMPT)


if __name__ == "__main__":
    unittest.main()

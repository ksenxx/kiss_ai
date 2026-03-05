"""Tests for GEPA batched agent wrapper support."""

import unittest
from typing import Any

from kiss.agents.gepa import GEPA


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
                results.append((f"EXPECTED:{expected}\nRESULT:result={expected}", []))
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


if __name__ == "__main__":
    unittest.main()

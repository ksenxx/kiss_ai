# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test GEPA algorithm on a HuggingFace dataset."""

import unittest

import yaml
from datasets import load_dataset

import kiss.core.utils as utils
from kiss.agents.gepa import GEPA
from kiss.core.kiss_agent import KISSAgent


def evaluate_qa_result(result: str, expected_answer: str) -> dict[str, float]:
    """Evaluate a QA result against expected answer.

    Args:
        result: Result string (YAML) from agent.run() with keys: status, analysis, result
        expected_answer: The expected answer string

    Returns:
        Dict of metric scores (higher is better)
    """
    # Parse YAML result string
    try:
        result_dict = yaml.safe_load(result) or {}
    except Exception:
        result_dict = {}

    scores = {}

    # Success score: 1.0 if status is success, 0.0 otherwise
    if result_dict.get("status") == "success":
        scores["success"] = 1.0
    else:
        scores["success"] = 0.0

    # Answer quality: check if expected answer appears in result
    result_text = result_dict.get("result", "").lower()
    expected_lower = expected_answer.lower()

    # Simple exact match score
    if expected_lower in result_text:
        scores["answer_match"] = 1.0
    else:
        # Partial match: check for word overlap
        expected_words = set(expected_lower.split())
        result_words = set(result_text.split())
        if expected_words:
            overlap = len(expected_words & result_words) / len(expected_words)
            scores["answer_match"] = overlap
        else:
            scores["answer_match"] = 0.0

    return scores


class TestGEPAHuggingFace(unittest.TestCase):
    """Test GEPA algorithm on HuggingFace datasets."""

    def test_gepa_on_squad_dataset(self):
        """Test GEPA on a small sample from SQuAD dataset."""
        # Load a small sample from SQuAD dataset
        print("Loading SQuAD dataset...")
        dataset = load_dataset("squad", split="validation[:5]")  # Just 5 examples for testing

        # Create evaluation function that works with dataset examples
        def create_evaluation_fn(example):
            def evaluation_fn(result: str) -> dict[str, float]:
                return evaluate_qa_result(result, example["answers"]["text"][0])

            return evaluation_fn

        # Initial prompt template for question answering
        prompt_template = """You are a helpful assistant that answers questions
based on given context.

## Context ##
{context}

## Question ##
{question}

## Task ##
Answer the question based on the context provided above. Be precise and concise.
Your answer should be extracted directly from the context when possible.

After you have your answer, you MUST call the 'finish' tool with your answer
as the 'result' argument.

"""

        # Get first example for testing
        example = dataset[0]
        context = example["context"]
        question = example["question"]
        expected_answer = example["answers"]["text"][0]

        print(f"\nTesting with question: {question}")
        print(f"Expected answer: {expected_answer}")

        def agent_wrapper(
            prompt_template: str, arguments: dict[str, str]
        ) -> tuple[str, list[dict[str, str]]]:
            agent = KISSAgent("SQuAD QA Agent")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template=prompt_template,
                arguments=arguments,
                tools=[utils.finish],
            )
            import json

            trajectory_json = agent.get_trajectory()
            trajectory_list = json.loads(trajectory_json)
            return result, trajectory_list

        # Create GEPA optimizer with limited generations for testing
        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=prompt_template,
            evaluation_fn=create_evaluation_fn(example),
            max_generations=2,
            population_size=3,
            pareto_size=2,
            mutation_rate=0.5,
        )

        # Run optimization
        print("\nRunning GEPA optimization...")
        arguments = {
            "context": context,
            "question": question,
        }

        best_candidate = gepa.optimize(
            arguments=arguments,
            rollouts_per_generation=1,
        )

        # Check results
        print(f"\nBest prompt scores: {best_candidate.scores}")
        print(f"\nBest prompt (first 500 chars):\n{best_candidate.prompt_template[:500]}...")

        # Verify that optimization ran
        self.assertIsNotNone(best_candidate)
        self.assertIsNotNone(best_candidate.scores)
        self.assertGreater(len(best_candidate.prompt_template), 0)

        # Check Pareto frontier
        pareto_frontier = gepa.get_pareto_frontier()
        self.assertGreater(len(pareto_frontier), 0)

        print(f"\nPareto frontier size: {len(pareto_frontier)}")
        for i, candidate in enumerate(pareto_frontier):
            print(f"  Candidate {i}: scores={candidate.scores}")


if __name__ == "__main__":
    unittest.main()

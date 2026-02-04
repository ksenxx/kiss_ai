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
from kiss.tests.conftest import requires_openai_api_key


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


@requires_openai_api_key
class TestGEPAHuggingFace(unittest.TestCase):
    """Test GEPA algorithm on HuggingFace datasets."""

    def test_gepa_on_squad_dataset(self):
        """Test GEPA on a small sample from SQuAD dataset.

        This test uses the new GEPA API with proper dev/val split:
        - Training examples are split into dev (feedback) and val (selection) sets
        - Candidates are evaluated on dev minibatches for trajectory feedback
        - Selection is based on val set scores

        Verifies that optimization runs successfully and produces valid
        candidates with scores and preserved placeholders.

        Returns:
            None
        """
        # Load a small sample from SQuAD dataset
        print("Loading SQuAD dataset...")
        dataset = load_dataset("squad", split="validation[:6]")  # 6 examples for dev/val split

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

        # Build train_examples list with expected answers embedded
        train_examples: list[dict[str, str]] = []
        for example in dataset:
            example = dict(example)  # type: ignore[assignment]
            train_examples.append(
                {
                    "context": example["context"],  # type: ignore[index]
                    "question": example["question"],  # type: ignore[index]
                    "_expected_answer": example["answers"]["text"][0],  # type: ignore[index]
                }
            )

        print(f"\nLoaded {len(train_examples)} examples for optimization")
        for i, ex in enumerate(train_examples[:3]):
            print(f"  [{i}] Q: {ex['question'][:50]}... -> A: {ex['_expected_answer']}")

        # Agent counter for unique names
        agent_counter = [0]

        def agent_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
            """Agent wrapper that handles expected answer embedding.

            Args:
                prompt_template: The prompt template to use for the agent.
                arguments: Dict with context, question, and _expected_answer keys.

            Returns:
                Tuple of (result string with expected/actual, empty trajectory list).
            """
            expected_answer = arguments.get("_expected_answer", "")
            agent_args = {
                "context": arguments["context"],
                "question": arguments["question"],
            }

            agent_counter[0] += 1
            agent = KISSAgent(f"SQuAD QA Agent {agent_counter[0]}")
            result = agent.run(
                model_name="gpt-4o",
                prompt_template=prompt_template,
                arguments=agent_args,
                tools=[utils.finish],
            )

            result_with_expected = f"EXPECTED:{expected_answer}\nRESULT:{result}"
            return result_with_expected, []

        def evaluation_fn(result: str) -> dict[str, float]:
            """Evaluation function that extracts expected answer from result.

            Args:
                result: Result string in format 'EXPECTED:...\nRESULT:...'

            Returns:
                Dict with 'success' and 'answer_match' scores.
            """
            try:
                if result.startswith("EXPECTED:"):
                    parts = result.split("\nRESULT:", 1)
                    expected = parts[0].replace("EXPECTED:", "").strip()
                    actual_result = parts[1] if len(parts) > 1 else ""
                    return evaluate_qa_result(actual_result, expected)
            except Exception:
                pass
            return {"success": 0.0, "answer_match": 0.0}

        # Create GEPA optimizer with dev/val split
        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=prompt_template,
            evaluation_fn=evaluation_fn,
            max_generations=2,
            population_size=2,
            pareto_size=2,
            mutation_rate=0.5,
            dev_val_split=0.5,  # 50% dev, 50% val
        )

        # Run optimization with dev/val split
        print("\nRunning GEPA optimization with dev/val split...")
        best_candidate = gepa.optimize(
            train_examples=train_examples,
            dev_minibatch_size=2,  # Use 2 dev examples per evaluation
        )

        # Check results
        print(f"\nBest prompt val_scores: {best_candidate.val_scores}")
        print(f"Best prompt dev_scores: {best_candidate.dev_scores}")
        print(f"\nBest prompt (first 500 chars):\n{best_candidate.prompt_template[:500]}...")

        # Verify that optimization ran
        self.assertIsNotNone(best_candidate)
        self.assertIsNotNone(best_candidate.val_scores)
        self.assertGreater(len(best_candidate.prompt_template), 0)

        # Check Pareto frontier
        pareto_frontier = gepa.get_pareto_frontier()
        self.assertGreater(len(pareto_frontier), 0)

        print(f"\nPareto frontier size: {len(pareto_frontier)}")
        for i, candidate in enumerate(pareto_frontier):
            print(f"  Candidate {i}: val_scores={candidate.val_scores}")


if __name__ == "__main__":
    unittest.main()

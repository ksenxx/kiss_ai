# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# Cursor AI (cursor@cursor.com)
# add your name here

"""HotPotQA benchmark for GEPA prompt optimization.

HotPotQA is a multi-hop question answering dataset that requires reasoning
over multiple supporting documents. This module provides utilities for
loading the dataset, evaluating results, and running GEPA optimization.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import yaml
from datasets import load_dataset

import kiss.core.utils as utils
from kiss.agents.gepa import GEPA, create_progress_callback
from kiss.core.kiss_agent import KISSAgent


@dataclass
class HotPotQAExample:
    """A single HotPotQA example."""

    id: str
    question: str
    answer: str
    question_type: str  # "comparison" or "bridge"
    level: str  # "easy", "medium", or "hard"
    supporting_facts: dict  # {"title": [...], "sent_id": [...]}
    context: dict  # {"title": [...], "sentences": [...]}

    @property
    def formatted_context(self) -> str:
        """Format the context paragraphs into a readable string.

        Returns:
            A formatted string with numbered paragraphs containing titles
            and their sentences.
        """
        paragraphs = []
        titles = self.context.get("title", [])
        sentences_list = self.context.get("sentences", [])

        for i, (title, sentences) in enumerate(zip(titles, sentences_list)):
            paragraph_text = " ".join(sentences)
            paragraphs.append(f"[{i + 1}] {title}:\n{paragraph_text}")

        return "\n\n".join(paragraphs)


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison.

    Lowercases, removes punctuation, articles, and extra whitespace.

    Args:
        s: The answer string to normalize.

    Returns:
        The normalized string with lowercase, no punctuation, no articles,
        and collapsed whitespace.
    """
    # Convert to string if not already (handles bool, int, etc.)
    s = str(s)
    # Lowercase
    s = s.lower()
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove extra whitespace
    s = " ".join(s.split())
    return s


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth.

    Args:
        prediction: The predicted answer string.
        ground_truth: The expected ground truth answer string.

    Returns:
        The F1 score as a float between 0.0 and 1.0.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0 if pred_tokens != truth_tokens else 1.0

    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def evaluate_hotpotqa_result(result: str, expected_answer: str) -> dict[str, float]:
    """Evaluate a HotPotQA result against expected answer.

    Args:
        result: Result string (YAML) from agent.run() with keys: status, analysis, result
        expected_answer: The expected answer string

    Returns:
        Dict of metric scores (higher is better):
        - success: 1.0 if status is success, 0.0 otherwise
        - exact_match: 1.0 if normalized answer matches exactly, 0.0 otherwise
        - f1: Token-level F1 score between predicted and expected answer
    """
    # Parse YAML result string
    try:
        result_dict = yaml.safe_load(result) or {}
    except Exception:
        result_dict = {}

    scores: dict[str, float] = {}

    # Success score: 1.0 if status is success, 0.0 otherwise
    if result_dict.get("status") == "success":
        scores["success"] = 1.0
    else:
        scores["success"] = 0.0

    # Get predicted answer (convert to string in case YAML parsed it as bool/int)
    predicted_answer = str(result_dict.get("result", ""))

    # Exact match score
    if normalize_answer(predicted_answer) == normalize_answer(expected_answer):
        scores["exact_match"] = 1.0
    else:
        scores["exact_match"] = 0.0

    # F1 score
    scores["f1"] = compute_f1(predicted_answer, expected_answer)

    return scores


# Initial prompt template for HotPotQA multi-hop QA
HOTPOTQA_INITIAL_PROMPT_TEMPLATE = """You are an expert question-answering assistant
specialized in multi-hop reasoning. Multi-hop questions require combining information
from multiple paragraphs to arrive at the correct answer.

## Context Paragraphs ##
{context}

## Question ##
{question}

## Instructions ##
1. Carefully read all the context paragraphs provided above.
2. Identify which paragraphs contain information relevant to the question.
3. Reason step-by-step, connecting information from different paragraphs.
4. Formulate a precise, concise answer based on the evidence.

## Important ##
- The answer should be short and factual (usually a name, date, number, or short phrase).
- Extract the answer directly from the context when possible.
- For comparison questions, identify the entities being compared and find the
  relevant attributes.
- You have ONLY one tool available: the 'finish' tool. Do NOT attempt to search
  the web or use any other tools.

After you have determined your answer, you MUST call the 'finish' tool with:
- status: 'success' if you found the answer, 'failure' otherwise
- analysis: Brief explanation of your reasoning steps
- result: Your final answer (keep it concise)

"""


class HotPotQABenchmark:
    """HotPotQA benchmark for GEPA prompt optimization."""

    def __init__(
        self,
        split: str = "validation",
        num_examples: int = 5,
        config_name: str = "distractor",
    ):
        """Initialize HotPotQA benchmark.

        Args:
            split: Dataset split to use ("train" or "validation")
            num_examples: Number of examples to load
            config_name: Dataset config ("distractor" or "fullwiki")
        """
        self.split = split
        self.num_examples = num_examples
        self.config_name = config_name
        self.examples: list[HotPotQAExample] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load HotPotQA dataset from HuggingFace.

        Loads the dataset using the configured split, config_name, and
        num_examples, then populates self.examples with HotPotQAExample
        instances.

        Returns:
            None. Populates self.examples in place.
        """
        print(f"Loading HotPotQA dataset ({self.config_name}, {self.split})...")
        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            self.config_name,
            split=f"{self.split}[:{self.num_examples}]",
        )

        for item in dataset:
            item = dict(item)  # type: ignore[assignment]
            example = HotPotQAExample(
                id=item["id"],  # type: ignore[index]
                question=item["question"],  # type: ignore[index]
                answer=item["answer"],  # type: ignore[index]
                question_type=item["type"],  # type: ignore[index]
                level=item["level"],  # type: ignore[index]
                supporting_facts={
                    "title": item["supporting_facts"]["title"],  # type: ignore[index]
                    "sent_id": item["supporting_facts"]["sent_id"],  # type: ignore[index]
                },
                context={
                    "title": item["context"]["title"],  # type: ignore[index]
                    "sentences": item["context"]["sentences"],  # type: ignore[index]
                },
            )
            self.examples.append(example)

        print(f"Loaded {len(self.examples)} examples")

    def get_example(self, index: int) -> HotPotQAExample:
        """Get a specific example by index.

        Args:
            index: The index of the example to retrieve.

        Returns:
            The HotPotQAExample at the specified index.
        """
        return self.examples[index]

    def create_evaluation_fn(self, example: HotPotQAExample) -> Callable[[str], dict[str, float]]:
        """Create an evaluation function for a specific example.

        Args:
            example: The HotPotQA example to evaluate against

        Returns:
            Evaluation function that takes result string and returns scores
        """

        def evaluation_fn(result: str) -> dict[str, float]:
            return evaluate_hotpotqa_result(result, example.answer)

        return evaluation_fn

    def run_gepa_optimization(
        self,
        example_indices: list[int] | None = None,
        model_name: str = "gpt-4o-mini",
        max_generations: int = 3,
        population_size: int = 3,
        pareto_size: int = 2,
        mutation_rate: float = 0.5,
        dev_minibatch_size: int | None = None,
        dev_val_split: float = 0.5,
        initial_prompt_template: str | None = None,
    ) -> tuple[GEPA, dict[str, float]]:
        """Run GEPA optimization over multiple HotPotQA examples.

        GEPA splits the examples into dev (for feedback/reflection) and val
        (for selection) sets. Candidates are evaluated on dev minibatches to
        collect trajectories for reflection, then scored on val set for selection.

        Args:
            example_indices: Indices of examples to use (uses all if None)
            model_name: Model to use for the QA agent
            max_generations: Number of GEPA generations
            population_size: GEPA population size
            pareto_size: GEPA Pareto frontier size
            mutation_rate: GEPA mutation rate
            dev_minibatch_size: Size of dev minibatch for each evaluation
            dev_val_split: Fraction of examples for dev set (default 0.5)
            initial_prompt_template: Initial prompt template (uses default if None)

        Returns:
            Tuple of (GEPA optimizer instance, best candidate val_scores)
        """
        indices = example_indices or list(range(len(self.examples)))
        print(f"\nOptimizing over {len(indices)} examples:")
        for idx in indices:
            example = self.get_example(idx)
            print(f"  [{idx}] {example.question[:60]}... -> {example.answer}")

        prompt_template = initial_prompt_template or HOTPOTQA_INITIAL_PROMPT_TEMPLATE
        self._agent_counter = 0

        def evaluation_fn(result: str) -> dict[str, float]:
            """Evaluation function that extracts expected answer from result prefix."""
            try:
                if result.startswith("EXPECTED:"):
                    parts = result.split("\nRESULT:", 1)
                    expected = parts[0].replace("EXPECTED:", "").strip()
                    actual_result = parts[1] if len(parts) > 1 else ""
                    return evaluate_hotpotqa_result(actual_result, expected)
            except Exception:
                pass
            return {"success": 0.0, "exact_match": 0.0, "f1": 0.0}

        train_examples: list[dict[str, str]] = []
        for idx in indices:
            example = self.get_example(idx)
            train_examples.append(
                {
                    "context": example.formatted_context,
                    "question": example.question,
                    "_expected_answer": example.answer,
                }
            )

        def agent_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
            """Agent wrapper that embeds expected answer for evaluation."""
            expected_answer = arguments.get("_expected_answer", "")
            agent_args = {
                "context": arguments["context"],
                "question": arguments["question"],
            }

            self._agent_counter += 1
            agent = KISSAgent(f"HotPotQA Agent {self._agent_counter}")
            result = agent.run(
                model_name=model_name,
                prompt_template=prompt_template,
                arguments=agent_args,
                tools=[utils.finish],
            )

            result_with_expected = f"EXPECTED:{expected_answer}\nRESULT:{result}"
            return result_with_expected, []

        gepa = GEPA(
            agent_wrapper=agent_wrapper,
            initial_prompt_template=prompt_template,
            evaluation_fn=evaluation_fn,
            max_generations=max_generations,
            population_size=population_size,
            pareto_size=pareto_size,
            mutation_rate=mutation_rate,
            dev_val_split=dev_val_split,
            progress_callback=create_progress_callback(verbose=True),
        )

        dev_count = int(len(train_examples) * dev_val_split)
        val_count = len(train_examples) - dev_count
        print(f"\nRunning GEPA optimization (dev={dev_count}, val={val_count} examples)...")

        best_candidate = gepa.optimize(
            train_examples=train_examples,
            dev_minibatch_size=dev_minibatch_size,
        )

        print(f"\nBest prompt val scores: {best_candidate.val_scores}")
        print(f"Best prompt dev scores: {best_candidate.dev_scores}")
        print(f"\nPareto frontier size: {len(gepa.get_pareto_frontier())}")

        return gepa, best_candidate.val_scores or {}

    def evaluate_prompt_on_examples(
        self,
        prompt_template: str,
        model_name: str = "gpt-4o-mini",
        example_indices: list[int] | None = None,
    ) -> dict[str, float]:
        """Evaluate a prompt template across multiple examples.

        Args:
            prompt_template: The prompt template to evaluate.
            model_name: The model to use for evaluation.
            example_indices: Specific indices to evaluate (uses all if None).

        Returns:
            Dictionary of average scores across all evaluated examples,
            with keys like 'success', 'exact_match', and 'f1'.
        """
        indices = example_indices or list(range(len(self.examples)))
        all_scores: list[dict[str, float]] = []
        call_counter = [0]

        for idx in indices:
            example = self.get_example(idx)
            call_counter[0] += 1
            agent = KISSAgent(f"HotPotQA Eval Agent {call_counter[0]}")
            result = agent.run(
                model_name=model_name,
                prompt_template=prompt_template,
                arguments={
                    "context": example.formatted_context,
                    "question": example.question,
                },
                tools=[utils.finish],
            )
            scores = evaluate_hotpotqa_result(result, example.answer)
            all_scores.append(scores)
            print(f"Example {idx}: {scores}")

        avg_scores: dict[str, float] = {}
        if all_scores:
            for key in all_scores[0]:
                avg_scores[key] = sum(s[key] for s in all_scores) / len(all_scores)

        print(f"\nAverage scores: {avg_scores}")
        return avg_scores


def main() -> None:
    """Main entry point for running HotPotQA benchmark with GEPA optimization.

    Runs GEPA optimization on HotPotQA examples and prints results.
    Configuration is hardcoded for simplicity - modify parameters directly
    in this function as needed.
    """
    print("=" * 70)
    print("HotPotQA Benchmark - GEPA Prompt Optimization")
    print("=" * 70)

    # Configuration - modify these as needed
    # Based on GEPA paper (arXiv:2507.19457) experimental settings:
    # - Paper uses 400-1200 rollouts for HotPotQA, with dev/val split
    # - pareto_size controls validation set size (D_pareto)
    # - minibatch_size is typically 8 for feedback per iteration
    num_examples = 100  # Total examples (split into dev/val sets)
    max_generations = 10  # GEPA iterations (paper uses 5-10)
    population_size = 5  # Candidates maintained per generation
    pareto_size = 5  # Max Pareto frontier size
    mutation_rate = 0.5  # Probability of mutation
    dev_val_split = 0.5  # 50% dev, 50% val (paper uses similar splits)
    model_name = "gpt-4o-mini"  # Model for QA agent

    # Initialize benchmark
    benchmark = HotPotQABenchmark(
        split="validation",
        num_examples=num_examples,
        config_name="distractor",
    )

    # Run GEPA optimization
    gepa, val_scores = benchmark.run_gepa_optimization(
        model_name=model_name,
        max_generations=max_generations,
        population_size=population_size,
        pareto_size=pareto_size,
        mutation_rate=mutation_rate,
        dev_val_split=dev_val_split,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    best_prompt = gepa.get_best_prompt()
    print(f"\nBest prompt (first 500 chars):\n{'-' * 40}")
    print(best_prompt[:500])
    print(f"{'-' * 40}")

    print(f"\nFinal validation scores: {val_scores}")
    print(f"Pareto frontier size: {len(gepa.get_pareto_frontier())}")

    # Show Pareto frontier candidates
    print("\nPareto frontier candidates:")
    for i, candidate in enumerate(gepa.get_pareto_frontier()):
        print(f"  [{i}] val_scores={candidate.val_scores}")


if __name__ == "__main__":
    main()

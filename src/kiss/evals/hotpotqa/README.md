# HotPotQA Benchmark for GEPA

This module provides a HotPotQA benchmark integration for GEPA (Genetic-Pareto) prompt optimization. HotPotQA is a multi-hop question answering dataset that requires reasoning over multiple supporting documents.

## Overview

HotPotQA challenges models to perform multi-hop reasoning by:

- Connecting information across multiple paragraphs
- Handling comparison questions between entities
- Supporting both "bridge" questions (chaining facts) and "comparison" questions

## Quick Start

```python
from kiss.evals.hotpotqa import HotPotQABenchmark

# Load benchmark with a few examples
benchmark = HotPotQABenchmark(
    split="validation",
    num_examples=5,
    config_name="distractor",  # or "fullwiki"
)

# Run GEPA optimization over multiple examples (prevents overfitting)
gepa, best_scores = benchmark.run_gepa_optimization(
    example_indices=[0, 1, 2, 3, 4],  # Optimize over all 5 examples
    model_name="gpt-4o-mini",
    max_generations=3,
    population_size=3,
    pareto_size=2,
)

print(f"Best scores: {best_scores}")
print(f"Optimized prompt: {gepa.get_best_prompt()[:500]}...")

# Evaluate the optimized prompt on all examples
avg_scores = benchmark.evaluate_prompt_on_examples(
    prompt_template=gepa.get_best_prompt(),
    model_name="gpt-4o-mini",
)
print(f"Average scores: {avg_scores}")
```

## Dataset Structure

Each HotPotQA example contains:

- `id`: Unique identifier
- `question`: The multi-hop question
- `answer`: Expected answer (usually a short phrase)
- `question_type`: "comparison" or "bridge"
- `level`: Difficulty level ("easy", "medium", "hard")
- `supporting_facts`: Titles and sentence IDs of supporting facts
- `context`: Multiple paragraphs with titles and sentences

## Evaluation Metrics

The benchmark evaluates three metrics:

1. **success**: 1.0 if the agent reports success, 0.0 otherwise
1. **exact_match**: 1.0 if normalized answer matches exactly
1. **f1**: Token-level F1 score between prediction and ground truth

## API Reference

### `HotPotQABenchmark`

```python
class HotPotQABenchmark:
    def __init__(
        self,
        split: str = "validation",
        num_examples: int = 5,
        config_name: str = "distractor",
    ):
        """Initialize HotPotQA benchmark.
        
        Args:
            split: Dataset split ("train" or "validation")
            num_examples: Number of examples to load
            config_name: Dataset config ("distractor" or "fullwiki")
        """
```

### Key Methods

- `get_example(index)`: Get a specific example by index
- `create_evaluation_fn(example)`: Create an evaluation function for a specific example
- `create_agent_wrapper(model_name)`: Create an agent wrapper for GEPA
- `run_gepa_optimization(example_indices, ...)`: Run GEPA optimization over multiple examples
- `evaluate_prompt_on_examples(...)`: Evaluate a prompt across multiple examples

### Multi-Example Optimization

The `run_gepa_optimization` method optimizes prompts over multiple examples to prevent overfitting:

```python
gepa, best_scores = benchmark.run_gepa_optimization(
    example_indices=[0, 1, 2],  # List of example indices to optimize over
    model_name="gpt-4o-mini",
    max_generations=3,
    population_size=3,
    pareto_size=2,
    mutation_rate=0.5,
)
```

Each candidate prompt is evaluated by cycling through all specified examples during rollouts. This ensures the optimized prompt generalizes across different question types rather than overfitting to a single example.

### `evaluate_hotpotqa_result`

```python
def evaluate_hotpotqa_result(result: str, expected_answer: str) -> dict[str, float]:
    """Evaluate a HotPotQA result against expected answer.
    
    Returns:
        Dict with 'success', 'exact_match', and 'f1' scores.
    """
```

## Initial Prompt Template

The module includes a carefully designed initial prompt template (`HOTPOTQA_INITIAL_PROMPT_TEMPLATE`) optimized for multi-hop reasoning:

- Instructions for step-by-step reasoning
- Guidance for comparison questions
- Requirement to use the `finish` tool

## Running Tests

```bash
# Run HotPotQA-specific tests
uv run pytest src/kiss/tests/test_gepa_hotpotqa.py -v

# Run just the evaluation tests (no API calls)
uv run pytest src/kiss/tests/test_gepa_hotpotqa.py::TestHotPotQAEvaluation -v
```

## Authors

- Koushik Sen (ksen@berkeley.edu)

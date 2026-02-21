"""Agent optimizer using GEPA for prompt optimization."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from kiss.agents.gepa.gepa import GEPA


def optimize_agent(
    agent_runner: Callable[[str, dict[str, str]], tuple[str, list[Any]]],
    train_data: list[dict[str, str]],
    initial_prompt_template: str,
    metric_name: str = "accuracy",
    num_generations: int = 5,
    population_size: int = 4,
    dev_minibatch_size: int | None = None,
    output_file: str = "optimized_agent.json",
) -> str:
    """Optimize agent prompts using GEPA.

    Args:
        agent_runner: Function (prompt_template, arguments) -> (result, trajectory)
        train_data: Training dataset (will be split into dev/val by GEPA)
        initial_prompt_template: The initial prompt template to optimize
        metric_name: Name of metric to optimize (default: "accuracy")
        num_generations: Number of GEPA generations
        population_size: Size of prompt population
        dev_minibatch_size: Dev examples per evaluation (default: all)
        output_file: File to save optimized prompts

    Returns:
        Optimized prompt template as a string
    """

    def evaluation_fn(result: str) -> dict[str, float]:
        """Evaluate a result string."""
        # Simple default: check for "success" keyword
        return {metric_name: 1.0 if "success" in result.lower() else 0.0}

    # Initialize GEPA with the new API
    gepa = GEPA(
        agent_wrapper=agent_runner,
        initial_prompt_template=initial_prompt_template,
        evaluation_fn=evaluation_fn,
        max_generations=num_generations,
        population_size=population_size,
    )

    # Run optimization
    print(f"Starting GEPA optimization for {num_generations} generations...")
    best_candidate = gepa.optimize(
        train_examples=train_data,
        dev_minibatch_size=dev_minibatch_size,
    )

    # Save results
    output = {
        "optimized_prompt_template": best_candidate.prompt_template,
        "dev_scores": best_candidate.dev_scores,
        "val_scores": best_candidate.val_scores,
        "config": {
            "num_generations": num_generations,
            "population_size": population_size,
            "dev_minibatch_size": dev_minibatch_size,
            "metric_name": metric_name,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("\nOptimization complete!")
    print(f"Results saved to {output_file}")
    print(f"Best validation scores: {best_candidate.val_scores}")

    return best_candidate.prompt_template


def main() -> None:
    """Example usage."""
    print("agent_optimizer.py - Use this module to optimize agent prompts with GEPA")
    print("Import and call optimize_agent() with your agent runner and data")


if __name__ == "__main__":
    main()

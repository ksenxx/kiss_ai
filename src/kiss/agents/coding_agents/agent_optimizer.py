"""Agent optimizer using GEPA for prompt optimization."""

from __future__ import annotations

import json
from typing import Any, Callable

from kiss.agents.gepa.gepa import GEPA


def optimize_agent(
    agent_runner: Callable[[dict[str, str]], dict[str, Any]],
    train_data: list[dict],
    val_data: list[dict],
    initial_prompts: dict[str, str],
    metric_name: str = "accuracy",
    minimize_metric: bool = False,
    num_generations: int = 5,
    population_size: int = 4,
    dev_batch_size: int = 2,
    output_file: str = "optimized_agent.json",
) -> dict[str, str]:
    """Optimize agent prompts using GEPA.
    
    Args:
        agent_runner: Function that takes prompts dict and returns results dict
        train_data: Training dataset (split into dev/val)
        val_data: Validation dataset
        initial_prompts: Dictionary of initial prompt templates
        metric_name: Name of metric to optimize
        minimize_metric: Whether to minimize (True) or maximize (False) the metric
        num_generations: Number of GEPA generations
        population_size: Size of prompt population
        dev_batch_size: Batch size for dev evaluation
        output_file: File to save optimized prompts
        
    Returns:
        Dictionary of optimized prompts
    """
    
    def system_runner(prompts: dict[str, str], examples: list[dict]) -> list[dict[str, Any]]:
        """Wrapper to run agent on examples."""
        results = []
        for example in examples:
            try:
                result = agent_runner(prompts, example)
                results.append(result)
            except Exception as e:
                print(f"Error running agent: {e}")
                results.append({"error": str(e), metric_name: 0.0})
        return results
    
    # Initialize GEPA
    gepa = GEPA(
        train_data=train_data,
        val_data=val_data,
        system_runner=system_runner,
        initial_prompts=initial_prompts,
        metric_names=[metric_name],
        minimize_metrics=[minimize_metric],
    )
    
    # Run optimization
    print(f"Starting GEPA optimization for {num_generations} generations...")
    optimized_prompts, stats = gepa.optimize(
        num_generations=num_generations,
        population_size=population_size,
        dev_batch_size=dev_batch_size,
        verbose=True,
    )
    
    # Save results
    output = {
        "optimized_prompts": optimized_prompts,
        "stats": stats,
        "config": {
            "num_generations": num_generations,
            "population_size": population_size,
            "dev_batch_size": dev_batch_size,
            "metric_name": metric_name,
            "minimize_metric": minimize_metric,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nOptimization complete!")
    print(f"Results saved to {output_file}")
    print(f"Best {metric_name}: {stats.get('best_val_' + metric_name, 'N/A')}")
    
    return optimized_prompts


def main():
    """Example usage."""
    print("agent_optimizer.py - Use this module to optimize agent prompts with GEPA")
    print("Import and call optimize_agent() with your agent runner and data")


if __name__ == "__main__":
    main()

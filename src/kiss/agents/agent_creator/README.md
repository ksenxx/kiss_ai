# Agent Creator

A module for evolving and improving AI agents through multi-objective optimization. It provides tools to automatically optimize existing agent code for **token efficiency** and **execution speed** using evolutionary algorithms with Pareto frontier maintenance.

## Overview

The Agent Creator module consists of two main components:

1. **ImproverAgent**: Takes existing agent source code and creates optimized versions through iterative improvement
1. **AgentEvolver**: Maintains a population of agent variants and evolves them using mutation and crossover operations

Both components use a **Pareto frontier** approach to track non-dominated solutions, optimizing for multiple objectives simultaneously without requiring a single combined metric.

## Key Features

- **Multi-Objective Optimization**: Optimizes for both token usage and execution time
- **Pareto Frontier Maintenance**: Keeps track of all non-dominated solutions
- **Evolutionary Operations**: Supports mutation (improving one variant) and crossover (combining ideas from two variants)
- **Automatic Pruning**: Removes dominated variants to manage memory and storage
- **Lineage Tracking**: Records parent relationships and improvement history
- **Configurable Parameters**: Extensive configuration options for generations, population size, thresholds, etc.

## Installation

The module is part of the `kiss` package. No additional installation required.

## Quick Start

### Improving an Existing Agent

```python
import anyio
from kiss.agents.agent_creator import ImproverAgent

async def improve_agent():
    improver = ImproverAgent(
        num_generations=5,
        pareto_frontier_size=4,
    )

    success, report = await improver.improve(
        source_folder="/path/to/agent",
        target_folder="/path/to/improved_agent",
    )

    if success:
        print(f"Token improvement: {report.token_improvement_pct:.1f}%")
        print(f"Time improvement: {report.time_improvement_pct:.1f}%")

anyio.run(improve_agent)
```

### Evolving a New Agent from Scratch

```python
import anyio
from kiss.agents.agent_creator import AgentEvolver

async def evolve_agent():
    evolver = AgentEvolver(
        task_description="Build a code analysis assistant that can parse and analyze large codebases",
        max_generations=10,
        population_size=8,
        pareto_size=6,
    )

    best_variant = await evolver.evolve()

    print(f"Best agent: {best_variant.folder_path}")
    print(f"Tokens used: {best_variant.tokens_used}")
    print(f"Execution time: {best_variant.execution_time:.2f}s")

anyio.run(evolve_agent)
```

## Components

### ImproverAgent

The `ImproverAgent` optimizes existing agent code through multiple generations of improvement.

**Parameters:**

- `model`: LLM model to use (default: `"claude-sonnet-4-5"`)
- `max_steps`: Maximum steps per generation (default: `150`)
- `max_budget`: Maximum USD budget per generation (default: `15.0`)
- `num_generations`: Number of improvement generations (default: `5`)
- `pareto_frontier_size`: Maximum Pareto frontier size (default: `4`)
- `mutation_probability`: Probability of mutation vs crossover (default: `0.5`)
- `min_improvement_threshold`: Minimum improvement to keep a variant (default: `0.01`)
- `prune_non_frontier`: Whether to prune non-frontier variants (default: `True`)

**Methods:**

- `improve(source_folder, target_folder, ...)`: Improve an agent over multiple generations
- `crossover_improve(primary_folder, ..., secondary_report_path, ...)`: Combine ideas from two agents

### AgentEvolver

The `AgentEvolver` creates and evolves agent populations from a task description.

**Parameters:**

- `task_description`: Description of the task the agent should solve
- `evaluation_fn`: Optional custom evaluation function `(folder_path) -> (tokens, time)`
- `model`: LLM model for orchestration (default: `"claude-sonnet-4-5"`)
- `max_generations`: Maximum evolutionary generations (default: `10`)
- `population_size`: Maximum population size (default: `8`)
- `pareto_size`: Maximum Pareto frontier size (default: `6`)
- `mutation_probability`: Probability of mutation vs crossover (default: `0.5`)
- `work_dir`: Working directory for variants

**Methods:**

- `evolve()`: Run the evolutionary optimization, returns the best variant
- `get_best_variant()`: Get the current best variant by combined metric
- `get_pareto_frontier()`: Get all variants in the Pareto frontier
- `save_state(path)`: Save evolver state to JSON

### Data Classes

**ImprovementReport**: Tracks improvements made to an agent

- Baseline and improved metrics (tokens, time)
- Implemented and failed optimization ideas
- Lineage tracking (parent folder, generation)
- Estimated improvements

**AgentVariant**: Represents an agent variant in the Pareto frontier

- Folder and report paths
- Measured metrics
- Generation and parent tracking
- Pareto dominance checking

**ImproverVariant**: Internal variant representation for the improver

- Folder path and report
- Estimated cumulative improvements
- Pareto dominance checking

## Configuration

Configuration can be provided via the global config system:

```python
from kiss.core.config import DEFAULT_CONFIG

# Access agent_creator config
cfg = DEFAULT_CONFIG.agent_creator

# Improver settings
cfg.improver.model = "claude-sonnet-4-5"
cfg.improver.num_generations = 5
cfg.improver.pareto_frontier_size = 4

# Evolver settings
cfg.evolver.max_generations = 10
cfg.evolver.population_size = 8
cfg.evolver.mutation_probability = 0.5
```

## How It Works

### Pareto Frontier

The module uses **Pareto dominance** to compare solutions. A solution A dominates solution B if:

- A is at least as good as B in all objectives
- A is strictly better than B in at least one objective

The Pareto frontier contains all non-dominated solutions, representing the best trade-offs between objectives.

### Evolutionary Operations

1. **Mutation**: Select one variant from the frontier and apply improvements
1. **Crossover**: Select two variants, use the better one as the base, and incorporate ideas from the other's improvement report

### Improvement Process

1. Copy source agent to target folder
1. Analyze code structure and existing optimizations
1. Apply optimizations (prompt reduction, caching, batching, etc.)
1. Generate improvement report with estimates
1. Update Pareto frontier and prune dominated variants
1. Repeat for configured number of generations

### Agent Creation

The `AgentEvolver` creates agents with these patterns:

- **Orchestrator Pattern**: Central coordinator managing workflow
- **Dynamic To-Do List**: Task tracking with dependencies and priorities
- **Dynamic Tool Creation**: On-the-fly tool generation for subtasks
- **Checkpointing**: State persistence for recovery
- **Sub-Agent Delegation**: Specialized agents for complex subtasks

## Output

### Improvement Report JSON

```json
{
    "baseline_tokens": 10000,
    "baseline_time": 30.0,
    "improved_tokens": 8000,
    "improved_time": 25.0,
    "token_improvement_pct": 20.0,
    "time_improvement_pct": 16.7,
    "implemented_ideas": [
        {"idea": "Reduced prompt verbosity", "expected_impact": "10% token reduction"}
    ],
    "failed_ideas": [
        {"idea": "Aggressive caching", "reason": "Caused correctness issues"}
    ],
    "generation": 5
}
```

### Evolver State JSON

```json
{
    "task_description": "Build a code analysis assistant...",
    "generation": 10,
    "variant_counter": 15,
    "all_variants": [...],
    "pareto_frontier_ids": [3, 7, 12]
}
```

## Optimization Strategies

The improver applies various optimization strategies:

- **Prompt Optimization**: Reduce verbosity while maintaining clarity
- **Caching**: Cache repeated operations and intermediate results
- **Batching**: Batch API calls and operations where possible
- **Algorithm Efficiency**: Use more efficient algorithms
- **Context Reduction**: Minimize unnecessary context in conversations
- **Early Termination**: Stop when goals are achieved
- **Incremental Processing**: Use streaming or incremental processing
- **Step Minimization**: Reduce agent steps while maintaining correctness

## API Reference

### ImproverAgent

```python
class ImproverAgent:
    async def improve(
        self,
        source_folder: str,
        target_folder: str,
        report_path: str | None = None,
        base_dir: str | None = None,
    ) -> tuple[bool, ImprovementReport | None]: ...

    async def crossover_improve(
        self,
        primary_folder: str,
        primary_report_path: str,
        secondary_report_path: str,
        target_folder: str,
        base_dir: str | None = None,
    ) -> tuple[bool, ImprovementReport | None]: ...
```

### AgentEvolver

```python
class AgentEvolver:
    async def evolve(self) -> AgentVariant: ...
    def get_best_variant(self) -> AgentVariant: ...
    def get_pareto_frontier(self) -> list[AgentVariant]: ...
    def save_state(self, path: str | None = None) -> None: ...
```

## License

See the main project LICENSE file.

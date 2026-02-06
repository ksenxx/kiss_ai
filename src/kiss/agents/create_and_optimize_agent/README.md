# Create and Optimize Agent

A module for evolving and improving AI agents through multi-objective optimization. It provides tools to automatically optimize existing agent code for **token efficiency** and **execution speed** using evolutionary algorithms with Pareto frontier maintenance.

## Overview

The Create and Optimize Agent module consists of two main components:

1. **ImproverAgent**: Takes existing agent source code and creates optimized versions through iterative improvement
1. **AgentEvolver**: Maintains a population of agent variants and evolves them using mutation and crossover operations

Both components use a **Pareto frontier** approach to track non-dominated solutions, optimizing for multiple objectives simultaneously without requiring a single combined metric.

## Key Features

- **Multi-Objective Optimization**: Optimizes for flexible metrics (e.g., success, token usage, execution time)
- **Pareto Frontier Maintenance**: Keeps track of all non-dominated solutions
- **Evolutionary Operations**: Supports mutation (improving one variant) and crossover (combining ideas from two variants)
- **Automatic Pruning**: Removes dominated variants to manage memory and storage
- **Lineage Tracking**: Records parent relationships and improvement history
- **Configurable Parameters**: Extensive configuration options for generations, frontier size, thresholds, etc.

## Installation

The module is part of the `kiss` package. No additional installation required.

## Quick Start

### Improving an Existing Agent

```python
from kiss.agents.create_and_optimize_agent import ImproverAgent

improver = ImproverAgent(
    max_steps=150,
    max_budget=15.0,
)

success, report = improver.improve(
    source_folder="/path/to/agent",
    work_dir="/path/to/improved_agent",
    task_description="Build a code analysis assistant that can parse and analyze large codebases",
)

if success and report:
    print(f"Improvement completed in {report.metrics.get('execution_time', 0):.2f}s")
    print(f"Tokens used: {report.metrics.get('tokens_used', 0)}")
```

### Evolving a New Agent from Scratch

```python
from kiss.agents.create_and_optimize_agent import AgentEvolver

evolver = AgentEvolver()

best_variant = evolver.evolve(
    task_description="Build a code analysis assistant that can parse and analyze large codebases",
    max_generations=10,
    max_frontier_size=6,
    mutation_probability=0.8,
)

print(f"Best agent: {best_variant.folder_path}")
print(f"Metrics: {best_variant.metrics}")
```

## Components

### ImproverAgent

The `ImproverAgent` optimizes existing agent code by analyzing and improving it for token efficiency and execution speed. Configuration (model, max_steps, max_budget) is read from `DEFAULT_CONFIG.create_and_optimize_agent.improver`.

**Methods:**

- `improve(source_folder, work_dir, task_description, report_path)`: Improve an agent's code
- `crossover_improve(primary_folder, primary_report_path, secondary_report_path, work_dir, task_description)`: Combine ideas from two agents

### AgentEvolver

The `AgentEvolver` creates and evolves agent populations from a task description. Configuration is read from `DEFAULT_CONFIG.create_and_optimize_agent.evolver`.

**Methods:**

- `evolve(task_description, max_generations, initial_frontier_size, max_frontier_size, mutation_probability)`: Run the evolutionary optimization, returns the best variant. All parameters except `task_description` are optional and fall back to config defaults.
- `get_best_variant()`: Get the current best variant by combined score
- `get_pareto_frontier()`: Get all variants in the Pareto frontier
- `save_state(path)`: Save evolver state to JSON

### Data Classes

**ImprovementReport**: Tracks improvements made to an agent

- `metrics`: Dictionary of metric values (e.g., tokens_used, cost, execution_time)
- `implemented_ideas`: List of successful optimizations with idea and source
- `failed_ideas`: List of failed optimizations with idea and reason
- `generation`: The generation number of this improvement (default: 0)
- `summary`: Summary of the improvement (default: "")

**AgentVariant**: Represents an agent variant in the Pareto frontier

- `folder_path`: Path to the variant's source code
- `report_path`: Path to the variant's improvement report
- `report`: The ImprovementReport instance
- `metrics`: Dictionary of metric values (e.g., success, tokens_used, execution_time)
- `parent_ids`: List of parent variant IDs
- `id`: Unique variant identifier (default: 0)
- `generation`: Generation when created (default: 0)

## Configuration

Configuration can be provided via the global config system:

```python
from kiss.core.config import DEFAULT_CONFIG

# Access create_and_optimize_agent config
cfg = DEFAULT_CONFIG.create_and_optimize_agent

# Improver settings
cfg.improver.max_steps = 150
cfg.improver.max_budget = 15.0

# Evolver settings
cfg.evolver.max_generations = 10
cfg.evolver.max_frontier_size = 6
cfg.evolver.mutation_probability = 0.8
```

## How It Works

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           Task Description                                               │
└─────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│         Initial Agent Creation (Relentless Coding Agent) + Web Search for Best Practices                 │
└─────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          Evolution Loop                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────◄──┐   │
│  │    Mutation (80%): Single parent, Targeted changes   │   Crossover (20%): Two parents, Combine │  │   │
│  └───────────────────────────────────────────────┬────────────────────────────────────────────────┘  │   │
│                                                  ▼                                                   │   │ 
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐  │   │
│  │                         Evaluation: Measure tokens_used, execution_time                        │  │   │
│  └───────────────────────────────────────────────┬────────────────────────────────────────────────┘  │   │
│                                                  ▼                                                   │   │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐  │   │
│  │            Pareto Frontier Update: Keep non-dominated solutions, Trim by crowding distance     │  │   │
│  └───────────────────────────────────────────────┬────────────────────────────────────────────────┘  │   │
│                                                  └───────────────── More generations? ───────────────┘   │
└─────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                      │ Done
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            Optimal Agent Output: Best trade-off on Pareto frontier                       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Algorithm Pseudocode

```
Inputs:
    - task_description: Description of the task the agent should perform
    - max_generations: Maximum number of improvement generations
    - initial_frontier_size: Number of initial agents to create
    - max_frontier_size: Maximum size of the Pareto frontier
    - mutation_probability: Probability of mutation vs crossover (0.0 to 1.0)

Data Structures:
    AgentVariant:
        - folder_path: Directory containing agent code
        - report_path: Path to improvement report JSON file
        - report: ImprovementReport tracking implemented/failed ideas
        - metrics: {success, tokens_used, execution_time, ...}
        - id, generation, parent_ids (for lineage tracking)
        - feedback: Feedback from evaluation

    dominates(A, B):
        # A dominates B if A is at least as good in all metrics
        # and strictly better in one
        # ALL metrics are minimized (lower is better)
        # Note: success is 0 for success, 1 for failure

    score(variant, weights=None):
        # Combined ranking score (lower is better)
        # Default: success * 1,000,000 + tokens_used * 1 + execution_time * 1000

Algorithm EVOLVE():
    1. INITIALIZE
       - Create temporary work_dir for variants
       - Set optimal_dir for storing best agent
       - Initialize empty pareto_frontier

    2. CREATE INITIAL AGENTS
       - WHILE len(pareto_frontier) < initial_frontier_size:
           - Use coding agent to generate agent files from task_description
           - Agent must implement agent_run(task) -> {metrics: {...}, feedback: "..."}
           - Evaluate agent by calling agent_run(task_description)
           - Update pareto_frontier (may reject if dominated)
           - Copy current best variant (min score) to optimal_dir

    3. FOR generation = 1 TO max_generations:
       a. SELECT OPERATION
          IF random() < mutation_probability OR frontier_size < 2:
              # MUTATION
              parent = sample_uniform(pareto_frontier)
              new_variant = ImproverAgent.improve(parent)
          ELSE:
              # CROSSOVER
              v1, v2 = sample_two(pareto_frontier)
              primary, secondary = order_by_score(v1, v2)  # better score first
              new_variant = ImproverAgent.crossover_improve(primary, secondary)

       b. IF new_variant created successfully:
          - Evaluate: load agent.py, call agent_run(task_description)
          - Store feedback from evaluation result
          - Update pareto_frontier:
              - Reject if dominated by any existing variant
              - Remove variants dominated by new_variant
              - Add new_variant
              - If frontier > max_size: trim using crowding distance
          - Copy best variant (min score) to optimal_dir

    4. RETURN best variant from pareto_frontier (min score)

    5. CLEANUP work_dir
```

### Pareto Frontier

The module uses **Pareto dominance** to compare solutions. A solution A dominates solution B if:

- A is at least as good as B in all objectives
- A is strictly better than B in at least one objective

The Pareto frontier contains all non-dominated solutions, representing the best trade-offs between objectives.

By default, `tokens_used` and `execution_time` are minimized.

### Scoring

Variants are ranked using a combined score (lower is better). The score is calculated as:

- `tokens_used` + (`execution_time` * 1000)

This gives higher weight to execution time improvements.

### Evolutionary Operations

1. **Mutation**: Select one variant from the frontier and apply improvements
1. **Crossover**: Select two variants, use the better one (by score) as the base, and incorporate ideas from the other's improvement report

### Improvement Process

1. Copy source agent to target folder
1. Analyze code structure and existing optimizations
1. Apply optimizations (prompt reduction, caching, batching, etc.)
1. Generate improvement report with metrics
1. Update Pareto frontier and prune dominated variants

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
    "metrics": {"tokens_used": 8000, "execution_time": 25.0, "cost": 0.5},
    "implemented_ideas": [
        {"idea": "Reduced prompt verbosity", "source": "improver"}
    ],
    "failed_ideas": [
        {"idea": "Aggressive caching", "reason": "Caused correctness issues"}
    ],
    "generation": 5,
    "summary": "Optimized prompts and added caching for repeated operations"
}
```

### Evolver State JSON

```json
{
    "task_description": "Build a code analysis assistant...",
    "generation": 10,
    "variant_counter": 15,
    "pareto_frontier": [
        {
            "folder_path": "/path/to/variant_3",
            "report_path": "/path/to/variant_3/improvement_report.json",
            "report": {
                "metrics": {"tokens_used": 5000, "execution_time": 12.5},
                "implemented_ideas": [...],
                "failed_ideas": [...],
                "generation": 4,
                "summary": "..."
            },
            "metrics": {"success": 0, "tokens_used": 5000, "execution_time": 12.5},
            "id": 3,
            "generation": 4,
            "parent_ids": [1]
        }
    ]
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
    def improve(
        self,
        source_folder: str,
        work_dir: str,
        task_description: str,
        report_path: str | None = None,
    ) -> tuple[bool, ImprovementReport | None]: ...

    def crossover_improve(
        self,
        primary_folder: str,
        primary_report_path: str,
        secondary_report_path: str,
        work_dir: str,
        task_description: str,
    ) -> tuple[bool, ImprovementReport | None]: ...
```

### AgentEvolver

```python
class AgentEvolver:
    def evolve(
        self,
        task_description: str,
        max_generations: int | None = None,
        initial_frontier_size: int | None = None,
        max_frontier_size: int | None = None,
        mutation_probability: float | None = None,
    ) -> AgentVariant: ...

    def get_best_variant(self) -> AgentVariant: ...
    def get_pareto_frontier(self) -> list[AgentVariant]: ...
    def save_state(self, path: str) -> None: ...
```

### ImprovementReport

```python
class ImprovementReport:
    def __init__(
        self,
        metrics: dict[str, float],
        implemented_ideas: list[dict[str, str]],
        failed_ideas: list[dict[str, str]],
        generation: int = 0,
        summary: str = "",
    ): ...
```

### AgentVariant

```python
@dataclass
class AgentVariant:
    folder_path: str
    report_path: str
    report: ImprovementReport
    metrics: dict[str, float]
    parent_ids: list[int]
    id: int = 0
    generation: int = 0
```

## License

See the main project LICENSE file.
